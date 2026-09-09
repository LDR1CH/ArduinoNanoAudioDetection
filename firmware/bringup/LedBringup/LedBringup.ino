// Bring-up: all three LEDs, no model, no mic.
//
// Run this before the main firmware. It answers three questions:
//   1. Are the external LEDs wired correctly (anode to the pin through the
//      resistor, cathode to GND)? If one never lights, it is probably
//      backwards -- the long leg goes to the pin.
//   2. Which physical LED is on which pin? They alternate rather than blink
//      together, so you can tell them apart.
//   3. Does the onboard RGB LED behave active-LOW as expected? If the colors
//      look inverted, the convention in command_responder.cpp is wrong for
//      your board revision.
//
// Expected, once per second: the D2 LED lights for half a second, then the D3
// LED lights for half a second, while the onboard RGB LED steps through red,
// green, blue, amber, off.
//
// The pin numbers duplicate hardware_config.h and AudioCapture.ino. This
// sketch has to compile standalone, outside those sketch folders, so it cannot
// include from them -- and it is two numbers, used only during bring-up.

constexpr int kActionLedPin = 2;  // must match hardware_config.h
constexpr int kRecordLedPin = 3;  // must match AudioCapture.ino

void writeStatusLed(bool red, bool green, bool blue) {
  // Active LOW: LOW lights the channel.
  digitalWrite(LEDR, red ? LOW : HIGH);
  digitalWrite(LEDG, green ? LOW : HIGH);
  digitalWrite(LEDB, blue ? LOW : HIGH);
}

// Alternate the two external LEDs for a second while holding a status color,
// so all three can be checked -- and told apart -- in a single pass.
void holdFor(const char* name, bool red, bool green, bool blue) {
  Serial.println(name);
  writeStatusLed(red, green, blue);

  digitalWrite(kActionLedPin, HIGH);
  delay(500);
  digitalWrite(kActionLedPin, LOW);

  digitalWrite(kRecordLedPin, HIGH);
  delay(500);
  digitalWrite(kRecordLedPin, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(kActionLedPin, OUTPUT);
  pinMode(kRecordLedPin, OUTPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  Serial.println("led bring-up: D2 then D3 alternate, onboard RGB cycles");
}

void loop() {
  holdFor("red", true, false, false);
  holdFor("green", false, true, false);
  holdFor("blue", false, false, true);
  holdFor("amber (settling color)", true, true, false);
  holdFor("off", false, false, false);
}
