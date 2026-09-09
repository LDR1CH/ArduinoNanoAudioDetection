#include "feature_provider.h"

#include "audio_provider.h"
#include "model_settings.h"

namespace {

// Rolling feature window: kFeatureFrames x kFeatureCount, oldest frame first.
int8_t g_feature_buffer[kFeatureElements];
bool g_is_first_run = true;

// Compute one frame of MFCCs from kWindowSizeMs of audio.
// Must stay numerically equivalent to scripts/extract_features.py --
// if the MFCC parameters there change, they change here too.
bool ComputeFrame(const int16_t* samples, int sample_count, int8_t* out) {
  // TODO: pre-emphasis -> Hann window -> FFT -> mel filterbank -> log -> DCT
  (void)samples;
  (void)sample_count;
  (void)out;
  return false;
}

}  // namespace

bool FeatureProviderInit() {
  // TODO: precompute Hann window and mel filterbank tables
  g_is_first_run = true;
  return false;
}

bool PopulateFeatures(int32_t last_time_ms, int32_t end_time_ms,
                      int8_t* features, int* slices_needed) {
  // TODO: work out which frames are new since last_time_ms, shift the older
  // frames down in g_feature_buffer, compute only the new ones, then copy
  // g_feature_buffer into features.
  (void)last_time_ms;
  (void)end_time_ms;
  (void)features;
  (void)slices_needed;
  return false;
}
