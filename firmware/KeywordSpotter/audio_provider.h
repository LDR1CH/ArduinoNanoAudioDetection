// audio_provider -- owns PDM/DMA capture from the onboard MP34DT05 mic.
//
// This file knows about microphones and nothing about MFCCs or the model.
// feature_provider is the only consumer. Swapping in a different mic should
// mean rewriting audio_provider.cpp and nothing else.

#ifndef AUDIO_PROVIDER_H_
#define AUDIO_PROVIDER_H_

#include <cstdint>

// Start PDM capture at kAudioSampleRate. Returns false if the mic fails to
// initialize. Call once from setup().
bool AudioProviderInit();

// Hand back the most recent `sample_count` samples ending at `end_time_ms`.
// On success *samples points into the provider ring buffer (valid until the
// next call) and *samples_read holds how many samples are actually there.
//
// Returns false while the mic is still settling -- see AudioProviderReady().
bool GetAudioSamples(int32_t end_time_ms, int sample_count,
                     const int16_t** samples, int* samples_read);

// False for the first kMicSettleMs after init. The onboard PDM mic returns
// unreliable data for roughly 3.5 s after power-up or reset; trusting it
// during that window produces phantom detections the moment the board is
// plugged in.
bool AudioProviderReady();

// Timestamp of the newest sample captured so far, in ms since boot.
int32_t LatestAudioTimestamp();

#endif  // AUDIO_PROVIDER_H_
