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
TfLiteTensor* g_output = nullptr;

alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

// The 9 ops this exact model actually uses -- read directly off
// models/tflite/ds_cnn_int8.tflite's flatbuffer, not assumed from the Keras
// architecture. The obvious guess (conv2d, depthwise_conv2d,
// fully_connected, softmax, reshape, average_pool_2d -- one op per Keras
// layer) was WRONG and cost a debugging session: Keras's Flatten(), with a
// symbolic batch dimension, doesn't always lower to a single static RESHAPE.
// Here it lowered to SHAPE + STRIDED_SLICE + PACK + RESHAPE, computing the
// flattened size at graph-run time instead of baking it in. Confirmed by
// the board itself: registering only the 6 "obvious" ops produced "Didn't
// find op for builtin opcode 'SHAPE'" at AllocateTensors() and silently
// hung with no other output (a MicroInterpreter op-resolution failure
// doesn't halt cleanly the way this file's own Halt() does).
//
// If you retrain and requantize, re-verify this list rather than trusting
// it still holds -- e.g. in Python:
//   interp = tf.lite.Interpreter(model_path="models/tflite/ds_cnn_int8.tflite")
//   interp.allocate_tensors()  # then inspect the flatbuffer's operator codes
//
// A MicroMutableOpResolver registering only ops actually present keeps flash
// cost down versus AllOpsResolver, which pulls in every kernel TFLM knows
// about whether this model uses it or not.
tflite::MicroMutableOpResolver<9> g_resolver;

int32_t g_previous_time_ms = 0;
bool g_was_settling = true;

// Detection smoothing: require the argmax category to be stable across
// kSmoothingWindow consecutive inferences (~60ms at one new frame per ~20ms)
// before treating it as a real detection, not single-frame noise. Resets
// once the category returns to background, so the same word can fire again
// on a later utterance.
constexpr int kSmoothingWindow = 3;
int g_recent_categories[kSmoothingWindow] = {0, 0, 0};  // kBackgroundIndex
int g_recent_write_pos = 0;
int g_last_responded_category = 0;  // kBackgroundIndex

bool RecentCategoriesAllMatch(int category) {
  for (int i = 0; i < kSmoothingWindow; i++) {
    if (g_recent_categories[i] != category) {
      return false;
    }
  }
  return true;
}

// Halts here -- there is no recovery path for a model/interpreter that
// failed to set up, and continuing would mean running inference against
// tensors that were never allocated. Matches the fail-loud convention
// already used elsewhere in this project (e.g. firmware/AudioCapture on a
// PDM.begin() failure) rather than silently limping on.
void Halt(const char* message) {
  Serial.print("FATAL: ");
  Serial.println(message);
  while (true) {
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  tflite::InitializeTarget();

  g_model = tflite::GetModel(g_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Halt("model schema version mismatch -- regenerate model_data.cc");
  }

  if (g_resolver.AddConv2D() != kTfLiteOk ||
      g_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      g_resolver.AddAveragePool2D() != kTfLiteOk ||
      g_resolver.AddShape() != kTfLiteOk ||
      g_resolver.AddStridedSlice() != kTfLiteOk ||
      g_resolver.AddPack() != kTfLiteOk ||
      g_resolver.AddReshape() != kTfLiteOk ||
      g_resolver.AddFullyConnected() != kTfLiteOk ||
      g_resolver.AddSoftmax() != kTfLiteOk) {
    Halt("op resolver registration failed");
  }

  // Static, not stack-local: MicroInterpreter's lifetime must outlive setup()
  // (loop() uses g_interpreter every iteration), and it's too large for a
  // comfortable stack frame regardless.
  static tflite::MicroInterpreter static_interpreter(
      g_model, g_resolver, g_tensor_arena, kTensorArenaSize);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Halt("AllocateTensors failed -- try shrinking kTensorArenaSize's "
        "companion buffers, or grow the arena if there's headroom");
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // The model's actual int8 scale/zero-point -- read from the tensor, never
  // hardcoded, since a retrained model can pick different quantization
  // parameters even with an identical architecture. feature_provider needs
  // these to quantize its MFCC output to match what this model expects.
  FeatureProviderSetQuantization(g_input->params.scale,
                                 g_input->params.zero_point);

  if (!AudioProviderInit() || !FeatureProviderInit() ||
      !CommandResponderInit()) {
    Halt("audio/feature/command provider init failed");
  }
  SetStatus(SpotterStatus::kSettling);
}

namespace {
int32_t g_last_heartbeat_ms = 0;
constexpr int32_t kHeartbeatIntervalMs = 2000;
}  // namespace

void loop() {
  const int32_t current_time_ms = LatestAudioTimestamp();

  // A silent device and a hung one look identical from outside. This costs
  // one print every 2s and settles that ambiguity at a glance, including
  // for whoever is debugging this on a desk months from now.
  if (current_time_ms - g_last_heartbeat_ms >= kHeartbeatIntervalMs) {
    g_last_heartbeat_ms = current_time_ms;
    Serial.print("heartbeat t=");
    Serial.println(current_time_ms);
  }

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

  if (g_interpreter->Invoke() != kTfLiteOk) {
    return;  // transient failure; try again next loop iteration
  }

  // Dequantize: real_value = (int8_value - zero_point) * scale.
  int category = kBackgroundIndex;
  float best_score = -1.0f;
  for (int i = 0; i < kCategoryCount; i++) {
    const float score = (g_output->data.int8[i] - g_output->params.zero_point) *
                        g_output->params.scale;
    if (score > best_score) {
      best_score = score;
      category = i;
    }
  }

  g_recent_categories[g_recent_write_pos] = category;
  g_recent_write_pos = (g_recent_write_pos + 1) % kSmoothingWindow;

  bool is_new_command = false;
  if (category == kBackgroundIndex) {
    // Once background returns, the next confident keyword is free to fire
    // again -- without this, saying the same word twice in a row would only
    // ever trigger the action once.
    g_last_responded_category = kBackgroundIndex;
  } else if (best_score >= kDetectionThreshold &&
            RecentCategoriesAllMatch(category) &&
            category != g_last_responded_category) {
    is_new_command = true;
    g_last_responded_category = category;
  }

  RespondToCommand(current_time_ms, category, best_score, is_new_command);
}
