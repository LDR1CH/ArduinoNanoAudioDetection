# Hardware

A confident keyword lights an LED on D2; a second LED on D3 lights while the
microphone is capturing training clips; the board's onboard RGB LED shows state.
USB power, breadboard build, no soldering.

## Wiring

Two identical branches, one per LED:

```
 Nano D2 ----[330R]----|>|---- Nano GND      action LED
 Nano D3 ----[330R]----|>|---- Nano GND      record LED
                        ^
              long leg (anode) toward the pin
```

Plus the onboard RGB LED on pins 22/23/24, which needs no wiring at all.

| Nano pin | To | Notes |
|---|---|---|
| D2 | action LED anode, through 330 R to GND | Active HIGH. Driven by `KeywordSpotter`: on for "play", off for "pause". |
| D3 | record LED anode, through 330 R to GND | Active HIGH. Driven by `AudioCapture` only, lit during the 1 s capture window. `KeywordSpotter` leaves it alone. |
| GND | cathode side of both resistors | |
| LEDR/LEDG/LEDB (22/23/24) | onboard | No wiring. **Active LOW** -- writing LOW lights the channel. |

If an LED never lights, it is almost certainly backwards: the long leg goes to
the pin. `firmware/bringup/LedBringup/` alternates D2 and D3 rather than
blinking them together, so you can also confirm which physical LED is which.

## Bill of materials

Nano 33 BLE Sense, breadboard, jumpers, two red/yellow/amber LEDs, two 330 R
resistors, one USB cable. Nothing to buy if a starter kit is on hand.

Power is just the USB cable — from a PC during development (so the serial
monitor stays available) or from any phone charger once it runs standalone.
There is no external supply, no bulk capacitor, and no common-ground work,
because nothing here draws meaningful current.

## Why 330 R, and why not a blue LED

Two constraints on the Nano 33 BLE Sense, both worth knowing before changing
the resistor or swapping the LED:

| Constraint | Consequence |
|---|---|
| nRF52840 GPIO: ~2 mA at default "standard drive", **15 mA max per pin and ~25 mA total** across all pins plus VDD | The budget covers both external LEDs and the onboard RGB LED together. 330 R against a red LED's ~2.0 V drop draws about 4 mA from the 3.3 V rail: clearly visible, comfortably inside budget. Going below ~220 R starts pushing the pin harder than it wants to be pushed. Note that D2 and D3 are never lit by the same sketch, so in practice only one external LED draws at a time. |
| **No pin is 5V tolerant** (3.3 V I/O only) | Nothing at 5V may reach a GPIO. Not an issue for this circuit, but it rules out driving anything from a 5V rail later without a level shifter. |

**Use a red, yellow, or amber LED.** Blue and white LEDs drop 3.0–3.2 V, leaving
almost nothing across the resistor from a 3.3 V rail. They come out dim and vary
noticeably from part to part.

If the LED needs to be brighter than ~4 mA allows, drive a small-signal MOSFET
(2N7000, BSS138) from D2 rather than raising the current on the pin directly.

## Placement

The mic is the sensor, so give it a clear path: don't route the jumpers over the
top of the board, and don't lay it face-down on foam. Nothing in this build makes
noise, so there are no other acoustic constraints — the LED is silent, which is
why the firmware needs no hold-off between a detection and the next inference.

## Bring-up order

Both steps run before a model exists, which matters because `model_data.cc` is a
placeholder that deliberately refuses to compile until `scripts/quantize.py` has run.

1. **LEDs alone** — flash `firmware/bringup/LedBringup/`. Confirms the D2 and D3
   wiring and LED polarity, tells you which physical LED sits on which pin, and
   checks that the onboard RGB LED's active-LOW convention matches your board.
2. **Mic settle window** — verify no samples are returned for the first
   `kMicSettleMs` after reset, and that they look sane afterward.
3. **Integrated** — say each keyword ~20 times at ~1 m, recording accept/reject
   counts per keyword.

## What each LED means

| LED | Sketch | Meaning |
|---|---|---|
| D2 | KeywordSpotter | The decision: on after "play", off after "pause" |
| D3 | AudioCapture | Lit while the mic is capturing a training clip |
| Onboard RGB | KeywordSpotter | What the spotter is doing (below) |

The onboard RGB LED reports what the spotter is doing; D2 reports what it decided.

| Color | Meaning |
|---|---|
| Amber | Mic still settling after reset — not yet listening |
| Blue | Listening |
| Green | Keyword accepted (reverts to blue after `kStatusFlashMs`) |
| Red | Heard something, scored below threshold |

## Known risk

**PDM settling time.** The onboard mic returns unreliable data for roughly the
first 3.5 s after power-up or reset. `kMicSettleMs` (4 s, rounded up) covers it;
without it the board fires phantom detections the instant it is plugged in.

## References

- [Nano 33 BLE Sense datasheet (ABX00031)](https://docs.arduino.cc/resources/datasheets/ABX00031-datasheet.pdf)
- [Control the RGB LED on Nano 33 BLE boards](https://support.arduino.cc/hc/en-us/articles/360016724140-Control-the-RGB-LED-on-Nano-33-BLE-boards)
- [nRF52840 GPIO drive capability](https://devzone.nordicsemi.com/f/nordic-q-a/82106/nrf52840-gpio-drive-capability)
- [PDM mic settling time after reset](https://github.com/arduino/ArduinoCore-mbed/issues/674)
