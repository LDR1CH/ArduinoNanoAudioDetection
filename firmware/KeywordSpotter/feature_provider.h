// feature_provider -- owns MFCC math. Turns raw int16 audio into the int8
// feature slab the model expects.
//
// Deliberately separate from audio_provider: this interface is the seam where
// on-device MFCC computation could be replaced by features streamed over
// serial from a host PC, without audio_provider or the model code changing.

#ifndef FEATURE_PROVIDER_H_
#define FEATURE_PROVIDER_H_

#include <cstdint>

#include "model_settings.h"

// Prepare any windowing/filterbank tables. Call once from setup().
bool FeatureProviderInit();

// Fill `features` (kFeatureElements int8 values, frame-major) with the MFCCs
// for the window ending at `end_time_ms`. Only frames not already covered by
// the previous call are recomputed; *slices_needed reports how many were.
bool PopulateFeatures(int32_t last_time_ms, int32_t end_time_ms,
                      int8_t* features, int* slices_needed);

#endif  // FEATURE_PROVIDER_H_
