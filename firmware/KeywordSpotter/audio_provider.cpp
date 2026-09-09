#include "audio_provider.h"

#include <Arduino.h>
#include <PDM.h>
#include <cstring>

#include "hardware_config.h"
#include "model_settings.h"

namespace {

// Ring buffer sized to hold one full clip plus headroom for the DMA callback
// to run ahead of the feature extractor.
constexpr int kBufferSampleCount = kClipSamples * 2;
int16_t g_audio_buffer[kBufferSampleCount];

// PDM's own scratch buffer, sized for its library default (512 bytes = 256
// samples) -- matches firmware/AudioCapture's AudioCapture.ino exactly,
// which is verified working on real hardware; not calling
// PDM.setBufferSize() here for the same reason it isn't called there.
int16_t g_pdm_scratch[256];

// Returned by GetAudioSamples: a linear (never-wrapping) copy of whatever
// span was requested, sized for the largest span feature_provider ever asks
// for (a full kClipSamples window, on the "catch up from nothing" path).
int16_t g_staging_buffer[kClipSamples];

volatile int g_write_pos = 0;             // next ring-buffer slot to write
volatile int32_t g_total_samples_written = 0;  // monotonic, never wraps
volatile int32_t g_latest_timestamp_ms = 0;

uint32_t g_init_time_ms = 0;
bool g_initialized = false;

// PDM ISR context: copy out and return. No MFCC work happens here -- see
// feature_provider.cpp for why that matters (compute budget).
void OnPdmData() {
  int bytes = PDM.available();
  if (bytes > (int)sizeof(g_pdm_scratch)) {
    bytes = sizeof(g_pdm_scratch);
  }
  PDM.read(g_pdm_scratch, bytes);

  const int n = bytes / 2;
  for (int i = 0; i < n; i++) {
    g_audio_buffer[g_write_pos] = g_pdm_scratch[i];
    g_write_pos++;
    if (g_write_pos >= kBufferSampleCount) {
      g_write_pos = 0;
    }
  }
  g_total_samples_written += n;
  g_latest_timestamp_ms =
      (int32_t)((int64_t)g_total_samples_written * 1000 / kAudioSampleRate);
}

}  // namespace

bool AudioProviderInit() {
  PDM.onReceive(OnPdmData);
  PDM.setGain(kPdmGain);  // MUST match firmware/AudioCapture's kPdmGain --
                           // that is what recorded the training data.
  if (!PDM.begin(1 /* mono */, kAudioSampleRate)) {
    return false;
  }

  // Capture starts immediately, not gated behind the settle window -- the
  // mic itself is what's unreliable early on (see AudioProviderReady()),
  // not this pipeline. By the time AudioProviderReady() first returns true
  // the ring buffer already holds several seconds of real audio.
  g_init_time_ms = millis();
  g_initialized = true;
  return true;
}

bool AudioProviderReady() {
  if (!g_initialized) {
    return false;
  }
  return (millis() - g_init_time_ms) >= kMicSettleMs;
}

bool GetAudioSamples(int32_t end_time_ms, int sample_count,
                     const int16_t** samples, int* samples_read) {
  *samples_read = 0;
  if (!AudioProviderReady() || sample_count <= 0 ||
      sample_count > kClipSamples) {
    return false;
  }

  // Snapshot the two counters the ISR updates together, so a mid-read
  // interrupt can't hand back a torn (inconsistent) pair.
  noInterrupts();
  const int32_t total_written = g_total_samples_written;
  const int write_pos = g_write_pos;
  interrupts();

  int32_t end_sample =
      (int32_t)((int64_t)end_time_ms * kAudioSampleRate / 1000);
  if (end_sample > total_written) {
    end_sample = total_written;  // never ask for samples that don't exist yet
  }
  const int32_t start_sample = end_sample - sample_count;
  if (start_sample < 0 ||
      (total_written - start_sample) > kBufferSampleCount) {
    return false;  // older than what the ring buffer still holds
  }

  // Map the absolute sample index start_sample to its physical ring slot:
  // the most recently written sample sits at index (write_pos - 1), i.e.
  // absolute index (total_written - 1).
  int start_pos =
      (int)(((int64_t)write_pos - (total_written - start_sample)) %
            kBufferSampleCount);
  if (start_pos < 0) {
    start_pos += kBufferSampleCount;
  }

  const int first_chunk = kBufferSampleCount - start_pos;
  if (first_chunk >= sample_count) {
    memcpy(g_staging_buffer, g_audio_buffer + start_pos,
           sample_count * sizeof(int16_t));
  } else {
    memcpy(g_staging_buffer, g_audio_buffer + start_pos,
           first_chunk * sizeof(int16_t));
    memcpy(g_staging_buffer + first_chunk, g_audio_buffer,
           (sample_count - first_chunk) * sizeof(int16_t));
  }

  *samples = g_staging_buffer;
  *samples_read = sample_count;
  return true;
}

int32_t LatestAudioTimestamp() { return g_latest_timestamp_ms; }
