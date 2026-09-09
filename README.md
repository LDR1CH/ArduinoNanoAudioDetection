# KeywordSpotter

TinyML keyword spotter for the Arduino Nano 33 BLE Sense. Records audio on the
onboard PDM mic (MP34DT05), classifies a small keyword set with a quantized
DS-CNN running on-device under TensorFlow Lite for Microcontrollers, and
triggers a physical action on a confident match.

Two independent halves: a Python training pipeline (`scripts/`) and Arduino
firmware (`firmware/`). Neither needs the other to be present to build or run.
The only bridge is the generated model, committed into `firmware/`.

## Layout

```
data/raw/                     recorded WAV clips, one folder per class
data/processed/               MFCC arrays (.npy), train/val/test splits
scripts/                      training pipeline (Python)
models/checkpoints/           Keras .h5 checkpoints
models/tflite/                quantized .tflite files
firmware/KeywordSpotter/      Arduino firmware
results/                      training curves, confusion matrix
firmware/bringup/             standalone hardware bring-up sketch
docs/                         hardware design notes
```

## Training pipeline

From the repo root, in order:

```
pip install -r requirements.txt
python scripts/record_samples.py      # capture clips (needs AudioCapture flashed)
python scripts/extract_features.py    # MFCC extraction, build splits
python scripts/train.py               # train the DS-CNN
python scripts/evaluate.py            # confusion matrix, per-class accuracy
python scripts/quantize.py            # INT8 TFLite export + C sources
```

`scripts/config.py` holds every path, label and hyperparameter. Change things
there, not in the individual scripts.

Recording runs **through the Nano**, not the host mic: flash
`firmware/AudioCapture/` first, then run `record_samples.py`, which drives the
board over USB serial and writes the clips it returns. This keeps the training
set on the same microphone that will run inference.

## Firmware

Open `firmware/KeywordSpotter/KeywordSpotter.ino` in the Arduino IDE, set the
board to Arduino Nano 33 BLE, and flash. Requires the Arduino Mbed OS Nano
board package and the **Chirale_TensorFlowLite** library; no drivers or elevated
permissions.

> Arduino no longer publishes `Arduino_TensorFlowLite`. `Chirale_TensorFlowLite`
> is generated from the same TFLite Micro source and keeps the same
> `tensorflow/lite/...` header paths; only the umbrella header differs
> (`<Chirale_TensorFlowLite.h>` instead of `<TensorFlowLite.h>`).

`model_data.cc` ships as a placeholder that refuses to compile until you run
`scripts/quantize.py` -- a stub model would otherwise build fine and then fail
on the board at `AllocateTensors()`.

### Building from the command line

`arduino-cli` shares its data directory with the Arduino IDE, so a core or
library installed either way is visible to both.

```
arduino-cli core install arduino:mbed_nano
arduino-cli lib install Chirale_TensorFLowLite

arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/bringup/LedBringup
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/AudioCapture
arduino-cli upload  --fqbn arduino:mbed_nano:nano33ble -p COM3 firmware/AudioCapture
```

The FQBN `arduino:mbed_nano:nano33ble` covers the Sense as well -- the core
lists a single Nano 33 BLE board.

## Hardware

A confident keyword lights an LED on D2; a second LED on D3 lights while the
mic is capturing training clips. The board's onboard RGB LED shows state.
Wiring and the reasoning are in [docs/hardware.md](docs/hardware.md).

```
 Nano D2 ----[330R]----|>|---- Nano GND      action LED
 Nano D3 ----[330R]----|>|---- Nano GND      record LED
```

Two things to know:

- **Use a red, yellow, or amber LED, and keep the resistor at ~330 R.** Blue and
  white LEDs drop 3.0-3.2 V, which leaves nothing across the resistor on a 3.3 V
  rail. The nRF52840 also allows only ~25 mA across all pins combined, a budget
  shared with the onboard RGB LED.
- **Check the hardware before the model exists.**
  `firmware/bringup/LedBringup/` runs standalone and verifies all three LEDs.

## Classes

Defined once in `scripts/config.py` and generated into
`firmware/KeywordSpotter/model_settings.h`, so the two lists cannot drift:

```
_background_noise, music, play, pause, skip
```

`_background_noise` is a real class at index 0, giving the model a rejection
option instead of forcing every input into a keyword.
