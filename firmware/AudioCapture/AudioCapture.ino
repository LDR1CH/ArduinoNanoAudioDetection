// AudioCapture -- turns the Nano into the recording device for the training
// set, so training data comes from the same microphone that will run inference.
//
// Flash this, then run `python scripts/record_samples.py` on the host. The
// board captures 1-second clips on demand and streams them back over USB
// serial as raw int16; the host writes them to data/raw/<label>/*.wav.
//
// Why not just record on the laptop: a laptop mic has a different frequency
// response and noise floor than the MP34DT05, and usually applies automatic
// gain control the Nano does not. A model trained on one and deployed on the
// other is the classic way a keyword spotter tests at 95% and then fails on
// the board.
//
// A record LED on D3 lights while the microphone is actually capturing, so you
// can see the exact 1-second window rather than guessing from the host's
// countdown. It is lit only during capture, not during the serial transfer
// that follows -- talking while the clip uploads does not matter.
//
// Protocol (host drives, board responds):
//   host -> 'P'   board -> "PONG\n"          liveness check
//   host -> 'R'   board -> "KS16" + N*2 raw bytes, little-endian int16
// On boot the board prints "READY\n" once the mic has settled. Nothing is
// captured before that.
//
// These constants intentionally duplicate the main firmware's headers -- an
// Arduino sketch cannot include from a sibling sketch folder. kPdmGain in
// particular MUST match hardware_config.h, or the training set and the
// deployed model see different audio.

#include <PDM.h>

constexpr int kSampleRate = 16000;      // must match model_settings.h
constexpr int kClipSamples = 16000;     // 1 second -- must match config.py CLIP_SAMPLES
constexpr int kSentenceSamples = 48000; // 3 seconds -- must match config.py SENTENCE_SAMPLES
constexpr int kPdmGain = 80;            // must match hardware_config.h
constexpr uint32_t kMicSettleMs = 4000; // must match hardware_config.h

// Record indicator, wired D3 -> 330R -> LED anode, cathode to GND. Active
// HIGH. Used only by this sketch; the KeywordSpotter firmware leaves D3 alone.
constexpr int kRecordLedPin = 3;

// Bytes per block in the paced transfer. Must match record_samples.py.
constexpr int kBlockBytes = 1000;

namespace {

// Sized for the longer of the two capture modes; the 1s capture just uses the
// front of it.
int16_t g_clip[kSentenceSamples];
int16_t g_scratch[256];  // PDM default buffer is 512 bytes = 256 samples

volatile bool g_capturing = false;
volatile int g_write_index = 0;
volatile int g_target_samples = kClipSamples;

void OnPdmData() {
  int bytes = PDM.available();
  if (bytes > (int)sizeof(g_scratch)) {
    bytes = sizeof(g_scratch);
  }
  PDM.read(g_scratch, bytes);

  // Always drain the buffer, even when idle, so the mic keeps running and the
  // first samples of a capture are not stale.
  if (!g_capturing) {
    return;
  }

  int samples = bytes / 2;
  int remaining = g_target_samples - g_write_index;
  if (samples > remaining) {
    samples = remaining;
  }
  memcpy(&g_clip[g_write_index], g_scratch, samples * 2);
  g_write_index += samples;
  if (g_write_index >= g_target_samples) {
    g_capturing = false;
  }
}

void CaptureClip(int sample_count) {
  // On for the live window only. SendClip() runs afterwards and takes longer
  // than the capture itself, so including it would make the LED lie about
  // when the mic is actually listening.
  digitalWrite(kRecordLedPin, HIGH);

  g_target_samples = sample_count;
  g_write_index = 0;
  g_capturing = true;
  while (g_capturing) {
    // The ISR fills g_clip. Nothing to do but wait for it to finish.
  }

  digitalWrite(kRecordLedPin, LOW);
}

// Wait for the host's 'G' before sending the next block. Returns false if the
// host went away, so a dead host cannot wedge the board forever.
bool WaitForGo() {
  const uint32_t start = millis();
  while (millis() - start < 3000) {
    if (Serial.available()) {
      return Serial.read() == 'G';
    }
  }
  return false;
}

void SendClip(const char* magic, int sample_count) {
  Serial.write(magic, 4);

  const uint8_t* const base = (const uint8_t*)g_clip;
  const size_t total = (size_t)sample_count * 2;
  size_t offset = 0;

  while (offset < total) {
    if (!WaitForGo()) {
      return;  // host aborted; it will resynchronise on the next "KS16"
    }
    size_t n = total - offset;
    if (n > (size_t)kBlockBytes) {
      n = kBlockBytes;
    }
    const uint8_t* p = base + offset;
    size_t remaining = n;
    while (remaining > 0) {
      // Serial.write can accept less than asked; advance by what it took.
      const size_t written = Serial.write(p, remaining);
      p += written;
      remaining -= written;
    }
    offset += n;
  }
  Serial.flush();
}

}  // namespace

void setup() {
  pinMode(kRecordLedPin, OUTPUT);
  digitalWrite(kRecordLedPin, LOW);

  Serial.begin(115200);
  while (!Serial) {
  }

  PDM.onReceive(OnPdmData);
  PDM.setGain(kPdmGain);
  if (!PDM.begin(1 /* mono */, kSampleRate)) {
    Serial.println("ERROR: PDM.begin failed");
    while (true) {
    }
  }

  // The mic needs a few seconds after reset before its output means anything.
  // Recording through this window would poison the training set with clips
  // that look nothing like what inference will see.
  delay(kMicSettleMs);
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) {
    return;
  }
  const int command = Serial.read();
  switch (command) {
    case 'P':
      Serial.println("PONG");
      break;
    case 'R':
      CaptureClip(kClipSamples);
      SendClip("KS16", kClipSamples);
      break;
    case 'L':
      CaptureClip(kSentenceSamples);
      SendClip("KS3S", kSentenceSamples);
      break;
    default:
      break;  // ignore stray bytes, including the host's line endings
  }
}
