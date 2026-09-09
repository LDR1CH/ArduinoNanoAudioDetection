// command_responder -- the physical action side. Everything that touches the
// action LED or the status LED lives here; the inference loop only reports
// which category won and how confidently.
//
// Pin numbers and timings come from hardware_config.h, never from literals.

#ifndef COMMAND_RESPONDER_H_
#define COMMAND_RESPONDER_H_

#include <cstdint>

// Shown on the onboard RGB LED. Distinct from the action LED on D2, which
// reflects the spotter's decision rather than its state.
enum class SpotterStatus {
  kSettling,   // mic still stabilizing after reset -- not yet listening
  kListening,  // idle, waiting for a keyword
  kAccepted,   // a keyword cleared the threshold
  kRejected,   // something was heard but scored too low
};

bool CommandResponderInit();

// Set the status LED. kAccepted and kRejected are transient: they revert to
// kListening after kStatusFlashMs, provided UpdateStatus is being called.
void SetStatus(SpotterStatus status);

// Call once per loop iteration. Expires transient statuses so the LED does not
// stay stuck on the last verdict.
void UpdateStatus(int32_t current_time_ms);

// Called once per inference. `category` indexes kCategoryLabels; `score` is
// the 0..1 softmax score. `is_new_command` is true only on the transition
// into a detection, so held speech does not retrigger the action.
void RespondToCommand(int32_t current_time_ms, int category, float score,
                      bool is_new_command);

#endif  // COMMAND_RESPONDER_H_
