#include "feature_provider.h"

#include <cmath>
#include <cstring>

#include "audio_provider.h"
#include "model_settings.h"

// The DSP pipeline below (Hann window, radix-2 FFT, mel filterbank matmul,
// DCT-II) was developed and cross-checked separately against
// scripts/extract_features.py's compute_frame_mfcc on a desktop build before
// being transplanted here: fed an identical 640-sample test frame into both,
// matched to ~5 decimal places (independent FFT implementations, so not
// bit-exact -- float32 accumulation order differs -- but far tighter than
// the int8 quantization step downstream will ever care about). See the
// recipe comment in scripts/config.py next to N_FFT for the full reasoning
// behind every convention used here (window type, why top_db-based dynamic
// range clipping is deliberately absent, DCT normalization).

namespace {

constexpr float kPi = 3.14159265358979323846f;

float g_hann_window[kWindowSizeSamples];

float g_input_scale = 1.0f;
int32_t g_input_zero_point = 0;
bool g_quantization_set = false;

// Rolling feature window: kFeatureFrames frames x kFeatureCount MFCCs,
// int8-quantized, frame-major (frame 0 = oldest). Shifted left and appended
// to incrementally rather than recomputed from scratch every call -- a full
// recompute every ~20ms (49 frames x FFT+mel+DCT each) would be far more
// compute than a 64 MHz Cortex-M4F can sustain in real time; recomputing
// only the handful of genuinely new frames keeps this cheap.
int8_t g_feature_buffer[kFeatureElements];
int g_frames_valid = 0;  // how many of kFeatureFrames hold real data

// Drift-free frame timing: advanced by exact kWindowStrideMs multiples
// rather than snapped to whatever raw millis() PopulateFeatures happens to
// be called with, so frame boundaries stay locked to the same
// i * kWindowStrideSamples grid extract_features.py's frame i is defined by,
// indefinitely, rather than slowly drifting over a long-running device.
int32_t g_newest_frame_end_time_ms = 0;

struct Complex {
  float re;
  float im;
};

// In-place iterative radix-2 decimation-in-time FFT. kFftSize must be (and
// is, per model_settings.h) a power of 2. Forward-transform convention
// (negative exponent), matching numpy.fft / scripts/extract_features.py.
void FftRadix2(Complex* a, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      Complex tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  }

  for (int len = 2; len <= n; len <<= 1) {
    const float ang = -2.0f * kPi / (float)len;
    const Complex wlen = {cosf(ang), sinf(ang)};
    for (int i = 0; i < n; i += len) {
      Complex w = {1.0f, 0.0f};
      for (int j = 0; j < len / 2; j++) {
        const Complex u = a[i + j];
        const Complex t = a[i + j + len / 2];
        const Complex v = {t.re * w.re - t.im * w.im,
                            t.re * w.im + t.im * w.re};
        a[i + j] = {u.re + v.re, u.im + v.im};
        a[i + j + len / 2] = {u.re - v.re, u.im - v.im};
        const float new_wre = w.re * wlen.re - w.im * wlen.im;
        const float new_wim = w.re * wlen.im + w.im * wlen.re;
        w.re = new_wre;
        w.im = new_wim;
      }
    }
  }
}

// One frame (exactly kWindowSizeSamples raw int16 samples) -> kFeatureCount
// quantized int8 MFCCs, written to `out`. Pure function of this frame alone
// -- see scripts/config.py's note by N_FFT on why that matters (it's what
// makes this incremental design possible at all).
void ComputeFrameFeatures(const int16_t* frame, int8_t* out) {
  static Complex fft_buf[kFftSize];
  for (int i = 0; i < kWindowSizeSamples; i++) {
    // int16 full-scale is +/-32768; extract_features.py's load_clip reads
    // WAV samples the same library (soundfile, float32 PCM) normalizes to
    // [-1, 1], so dividing by 32768 here matches that scale exactly.
    const float sample = (float)frame[i] / 32768.0f;
    fft_buf[i] = {sample * g_hann_window[i], 0.0f};
  }
  for (int i = kWindowSizeSamples; i < kFftSize; i++) {
    fft_buf[i] = {0.0f, 0.0f};
  }

  FftRadix2(fft_buf, kFftSize);

  static float power[kSpectrumBins];
  for (int k = 0; k < kSpectrumBins; k++) {
    power[k] = fft_buf[k].re * fft_buf[k].re + fft_buf[k].im * fft_buf[k].im;
  }

  static float log_mel[kNumMels];
  for (int m = 0; m < kNumMels; m++) {
    float energy = 0.0f;
    const float* row = kMelFilterbank + m * kSpectrumBins;
    for (int k = 0; k < kSpectrumBins; k++) {
      energy += row[k] * power[k];
    }
    const float floored = energy < 1e-10f ? 1e-10f : energy;
    log_mel[m] = 10.0f * log10f(floored);
  }

  // DCT-II, norm='ortho', keep first kFeatureCount coefficients. Verified
  // against scipy.fft.dct(..., type=2, norm='ortho') to ~1e-14 in isolation.
  const float scale0 = sqrtf(1.0f / (4.0f * kNumMels));
  const float scalek = sqrtf(1.0f / (2.0f * kNumMels));
  for (int k = 0; k < kFeatureCount; k++) {
    float sum = 0.0f;
    for (int n = 0; n < kNumMels; n++) {
      sum += log_mel[n] *
             cosf(kPi / kNumMels * ((float)n + 0.5f) * (float)k);
    }
    const float dctScale = (k == 0) ? scale0 : scalek;
    const float mfcc = 2.0f * dctScale * sum;

    // Normalize (train.py's per-coefficient train-split mean/std), then
    // quantize to int8 using the model input tensor's own scale/zero-point
    // -- read at runtime via FeatureProviderSetQuantization, never hardcoded,
    // since a retrained model can pick a different scale/zero-point even
    // with the same architecture.
    const float normalized = (mfcc - kFeatureMean[k]) / kFeatureStd[k];
    int32_t quantized =
        (int32_t)lroundf(normalized / g_input_scale) + g_input_zero_point;
    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;
    out[k] = (int8_t)quantized;
  }
}

}  // namespace

bool FeatureProviderInit() {
  for (int i = 0; i < kWindowSizeSamples; i++) {
    // Periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n/N). Verified against
    // librosa.filters.get_window('hann', N, fftbins=True) to float
    // precision -- NOT the symmetric variant (sym=True), which differs by
    // up to 0.0038 and is the easy mistake to make here.
    g_hann_window[i] =
        0.5f - 0.5f * cosf(2.0f * kPi * (float)i / (float)kWindowSizeSamples);
  }
  g_frames_valid = 0;
  g_newest_frame_end_time_ms = 0;
  return true;
}

void FeatureProviderSetQuantization(float scale, int32_t zero_point) {
  g_input_scale = scale;
  g_input_zero_point = zero_point;
  g_quantization_set = true;
}

bool PopulateFeatures(int32_t last_time_ms, int32_t end_time_ms,
                      int8_t* features, int* slices_needed) {
  (void)last_time_ms;  // superseded by g_newest_frame_end_time_ms below --
                        // kept in the signature for the caller's bookkeeping
  *slices_needed = 0;

  if (features == nullptr || !g_quantization_set) {
    return false;
  }

  const int32_t elapsed_ms = end_time_ms - g_newest_frame_end_time_ms;
  int new_frames = elapsed_ms / kWindowStrideMs;
  if (new_frames <= 0) {
    // Nothing new since the last call -- the common case, since loop() calls
    // this far more often than a new 20ms frame actually becomes available.
    // Cheap: no audio fetch, no DSP.
    if (g_frames_valid == kFeatureFrames) {
      memcpy(features, g_feature_buffer, kFeatureElements);
      return true;
    }
    return false;  // window not fully populated yet (e.g. just after boot)
  }
  if (new_frames > kFeatureFrames) {
    new_frames = kFeatureFrames;  // cap: a full window refresh either way
  }

  const int32_t target_end_ms =
      g_newest_frame_end_time_ms + (int32_t)new_frames * kWindowStrideMs;
  const int needed_samples =
      (new_frames - 1) * kWindowStrideSamples + kWindowSizeSamples;

  const int16_t* samples = nullptr;
  int samples_read = 0;
  if (!GetAudioSamples(target_end_ms, needed_samples, &samples,
                       &samples_read) ||
      samples_read < needed_samples) {
    return g_frames_valid == kFeatureFrames;  // try again next call
  }

  // Shift the existing window left by new_frames, dropping the oldest ones.
  if (new_frames < kFeatureFrames) {
    memmove(g_feature_buffer,
            g_feature_buffer + new_frames * kFeatureCount,
            (kFeatureFrames - new_frames) * kFeatureCount);
  }

  // Compute the new_frames newly-available frames and append them. Frame j
  // (0 = oldest of the new ones) starts at this offset within `samples`,
  // which ends exactly at target_end_ms:
  for (int j = 0; j < new_frames; j++) {
    const int offset = samples_read - kWindowSizeSamples -
                       (new_frames - 1 - j) * kWindowStrideSamples;
    int8_t* dest = g_feature_buffer +
                   (kFeatureFrames - new_frames + j) * kFeatureCount;
    ComputeFrameFeatures(samples + offset, dest);
  }

  g_newest_frame_end_time_ms = target_end_ms;
  if (g_frames_valid < kFeatureFrames) {
    g_frames_valid += new_frames;
    if (g_frames_valid > kFeatureFrames) {
      g_frames_valid = kFeatureFrames;
    }
  }
  *slices_needed = new_frames;

  if (g_frames_valid < kFeatureFrames) {
    return false;  // window not fully populated yet
  }
  memcpy(features, g_feature_buffer, kFeatureElements);
  return true;
}
