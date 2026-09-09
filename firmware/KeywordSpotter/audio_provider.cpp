#include "audio_provider.h"

#include <Arduino.h>
#include <PDM.h>

#include "hardware_config.h"
#include "model_settings.h"

namespace {

// Ring buffer sized to hold one full clip plus headroom for the DMA callback
// to run ahead of the feature extractor.
constexpr int kBufferSampleCount = kClipSamples * 2;
int16_t g_audio_buffer[kBufferSampleCount];
volatile int32_t g_latest_timestamp_ms = 0;

uint32_t g_init_time_ms = 0;
bool g_initialized = false;

// PDM ISR context: copy out and return. No MFCC work happens here.
void OnPdmData() {
  // TODO: PDM.read() into g_audio_buffer at the current write offset,
  // advance the offset modulo kBufferSampleCount, update the timestamp.
}

}  // namespace

bool AudioProviderInit() {
  PDM.onReceive(OnPdmData);
  // TODO: PDM.setBufferSize(), PDM.begin(1 /* mono */, kAudioSampleRate)

  g_init_time_ms = millis();
  g_initialized = true;
  return g_initialized;
}

bool AudioProviderReady() {
  if (!g_initialized) {
    return false;
  }
  return (millis() - g_init_time_ms) >= kMicSettleMs;
}

bool GetAudioSamples(int32_t end_time_ms, int sample_count,
                     const int16_t** samples, int* samples_read) {
  if (!AudioProviderReady()) {
    *samples_read = 0;
    return false;
  }

  // TODO: map end_time_ms onto the ring buffer, return a contiguous span
  (void)end_time_ms;
  (void)sample_count;
  (void)samples;
  return false;
}

int32_t LatestAudioTimestamp() { return g_latest_timestamp_ms; }
