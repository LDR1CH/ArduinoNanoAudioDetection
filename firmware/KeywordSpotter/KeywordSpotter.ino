// KeywordSpotter -- TinyML keyword spotting on the Arduino Nano 33 BLE Sense.
//
// Board: Arduino Nano 33 BLE. Libraries: Chirale_TensorFlowLite, PDM (core).
// Flash from the Arduino IDE normally -- no drivers or elevation.
//
// Chirale_TensorFlowLite replaces Arduino_TensorFlowLite, which Arduino no
// longer publishes -- it is generated from the same TFLite Micro source and
// keeps the same tensorflow/lite/... header paths. Only the umbrella header
// below differs.
//
// Loop: audio_provider captures PDM audio -> feature_provider turns it into
// MFCCs -> the INT8 DS-CNN classifies them -> command_responder drives the
// action LED on a confident, newly-seen keyword.
//
// Wiring is in docs/hardware.md. The short version: one LED on D2 through a
// 330R resistor to ground, and nothing else external.

#include <Chirale_TensorFlowLite.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "hardware_config.h"
#include "model_data.h"
#include "model_settings.h"

namespace {

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;

alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

// Note on RAM: while the interpreter below is still a TODO, nothing references
// this arena (or audio_provider's ring buffer), so the linker discards them and
// the compiler's "global variables use N bytes" figure is optimistic. Once the
// stubs are implemented, expect roughly 30 KB for the arena plus 64 KB for the
// 2-second audio ring buffer on top of the reported figure -- about 140 KB of
// the 256 KB total. It fits, but the headroom is smaller than a fresh build
// suggests. If AllocateTensors() fails later, that ring buffer is the first
// thing to shrink.

int32_t g_previous_time_ms = 0;
bool g_was_settling = true;

}  // namespace

void setup() {
  Serial.begin(115200);
  tflite::InitializeTarget();

  g_model = tflite::GetModel(g_model_data);
  // TODO: check g_model->version() against TFLITE_SCHEMA_VERSION and halt on
  // mismatch rather than running a model the interpreter cannot parse.

  // TODO: register only the ops the DS-CNN actually uses (conv2d,
  // depthwise_conv2d, fully_connected, softmax, reshape, average_pool_2d) --
  // a MicroMutableOpResolver keeps the flash cost down versus AllOpsResolver.

  // TODO: construct the MicroInterpreter over g_tensor_arena, call
  // AllocateTensors(), and cache g_interpreter->input(0) in g_input.

  if (!AudioProviderInit() || !FeatureProviderInit() ||
      !CommandResponderInit()) {
    Serial.println("init failed");
  }
  SetStatus(SpotterStatus::kSettling);
}

void loop() {
  const int32_t current_time_ms = LatestAudioTimestamp();

  // The mic needs a few seconds after reset before its data means anything.
  if (!AudioProviderReady()) {
    return;
  }
  if (g_was_settling) {
    g_was_settling = false;
    SetStatus(SpotterStatus::kListening);
    Serial.println("listening");
  }

  UpdateStatus(current_time_ms);

  int slices_needed = 0;
  if (!PopulateFeatures(g_previous_time_ms, current_time_ms,
                        g_input ? g_input->data.int8 : nullptr,
                        &slices_needed)) {
    return;
  }
  g_previous_time_ms = current_time_ms;
  if (slices_needed == 0) {
    return;  // nothing new to classify yet
  }

  // TODO: g_interpreter->Invoke(), then dequantize output(0) with its
  // scale/zero_point, take the argmax, and smooth across a few consecutive
  // windows before treating it as a detection.
  const int category = kBackgroundIndex;
  const float score = 0.0f;
  const bool is_new_command = false;

  RespondToCommand(current_time_ms, category, score, is_new_command);
}
