// feature_provider -- owns MFCC math. Turns raw int16 audio into the int8
// feature slab the model expects.
//
// Deliberately separate from audio_provider: this interface is the seam where
// on-device MFCC computation could be replaced by features streamed over
// serial from a host PC, without audio_provider or the model code changing.
//
// Also deliberately knows nothing about the TFLM interpreter -- it needs
// exactly two numbers to quantize its output (the input tensor's scale and
// zero-point), supplied via FeatureProviderSetQuantization rather than by
// reaching into the interpreter itself.

#ifndef FEATURE_PROVIDER_H_
#define FEATURE_PROVIDER_H_

#include <cstdint>

#include "model_settings.h"

// Prepare any windowing/filterbank tables. Call once from setup().
bool FeatureProviderInit();

// The model input tensor's int8 quantization params (TfLiteTensor::params),
// read once after AllocateTensors() and handed in here -- see the file
// comment above for why this lives in KeywordSpotter.ino, not here.
void FeatureProviderSetQuantization(float scale, int32_t zero_point);

// Fill `features` (kFeatureElements int8 values, frame-major) with the MFCCs
// for the window ending at `end_time_ms`. Only frames not already covered by
// the previous call are recomputed; *slices_needed reports how many were.
bool PopulateFeatures(int32_t last_time_ms, int32_t end_time_ms,
                      int8_t* features, int* slices_needed);

#endif  // FEATURE_PROVIDER_H_
