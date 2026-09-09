#include "command_responder.h"

#include <Arduino.h>
#include <string.h>

#include "hardware_config.h"
#include "model_settings.h"

namespace {

SpotterStatus g_status = SpotterStatus::kSettling;
int32_t g_status_set_ms = 0;

// Active LOW: writing LOW lights the channel.
void WriteStatusLed(bool red, bool green, bool blue) {
  digitalWrite(kLedRedPin, red ? LOW : HIGH);
  digitalWrite(kLedGreenPin, green ? LOW : HIGH);
  digitalWrite(kLedBluePin, blue ? LOW : HIGH);
}

bool IsTransient(SpotterStatus status) {
  return status == SpotterStatus::kAccepted ||
         status == SpotterStatus::kRejected;
}

}  // namespace

bool CommandResponderInit() {
  pinMode(kActionLedPin, OUTPUT);
  digitalWrite(kActionLedPin, LOW);

  pinMode(kLedRedPin, OUTPUT);
  pinMode(kLedGreenPin, OUTPUT);
  pinMode(kLedBluePin, OUTPUT);
  WriteStatusLed(false, false, false);

  return true;
}

void SetStatus(SpotterStatus status) {
  g_status = status;
  g_status_set_ms = millis();

  switch (status) {
    case SpotterStatus::kSettling:
      WriteStatusLed(true, true, false);  // amber: mic not trustworthy yet
      break;
    case SpotterStatus::kListening:
      WriteStatusLed(false, false, true);  // blue: idle
      break;
    case SpotterStatus::kAccepted:
      WriteStatusLed(false, true, false);  // green
      break;
    case SpotterStatus::kRejected:
      WriteStatusLed(true, false, false);  // red
      break;
  }
}

void UpdateStatus(int32_t current_time_ms) {
  (void)current_time_ms;
  if (!IsTransient(g_status)) {
    return;
  }
  if ((millis() - (uint32_t)g_status_set_ms) >= kStatusFlashMs) {
    SetStatus(SpotterStatus::kListening);
  }
}

void RespondToCommand(int32_t current_time_ms, int category, float score,
                      bool is_new_command) {
  if (!is_new_command || category == kBackgroundIndex) {
    return;
  }

  Serial.print("heard ");
  Serial.print(kCategoryLabels[category]);
  Serial.print(" @ ");
  Serial.println(score);

  SetStatus(SpotterStatus::kAccepted);

  // Label comparisons rather than raw indices, so reordering the class list
  // in scripts/config.py cannot silently swap the actions.
  const char* label = kCategoryLabels[category];
  if (strcmp(label, "play") == 0) {
    digitalWrite(kActionLedPin, HIGH);
  } else if (strcmp(label, "pause") == 0) {
    digitalWrite(kActionLedPin, LOW);
  }
  // "music" and "skip" currently only report over serial and flash the status
  // LED. With a single LED there is no obvious mapping for them -- give them
  // one here (a blink pattern, say) if the demo needs it.

  (void)current_time_ms;
}
