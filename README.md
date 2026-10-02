# KeywordSpotter

A spoken-keyword detector that runs entirely on an Arduino Nano 33 BLE Sense —
no network, no phone, no cloud. It listens through the board's built-in
microphone, recognises four command words, and lights an LED when it hears one.

```
music    play    pause    skip    (+ background noise)
```

Background noise is a real category, not a leftover. Without it the model would
be forced to label every sound as one of the four words; with it, it can decide
that what it heard wasn't a command at all.

## How it works

```
record clips  ->  convert to features  ->  train the model  ->  flash to board
    (PC)                 (PC)                    (PC)             (board)
```

Each one-second clip is summarised as **MFCCs** — a compact description of which
frequencies are present over time, much smaller than raw audio and easier to
learn from. The model is a small convolutional network, compressed to 8-bit
whole numbers so it fits in the chip's limited memory, and runs through
TensorFlow Lite for Microcontrollers at roughly two predictions per second.

Clips are recorded **through the Nano itself**, not a laptop microphone, so the
model learns from the same hardware it will run on.

## Layout

```
data/raw/                     recorded clips, one folder per word
data/processed/               extracted features, train/val/test splits
scripts/                      training pipeline (Python)
models/                       trained checkpoints and the compressed model
firmware/KeywordSpotter/      the sketch that runs on the board
firmware/AudioCapture/        recording sketch, used to collect clips
firmware/bringup/             standalone LED test sketch
docs/                         wiring and recording notes
results/                      training curves, confusion matrix
```

## Running the pipeline

From the repo root, in order:

```
pip install -r requirements.txt
python scripts/record_samples.py      # capture clips (needs AudioCapture flashed)
python scripts/extract_features.py    # convert audio to features, build splits
python scripts/train.py               # train the model
python scripts/evaluate.py            # accuracy and confusion matrix
python scripts/quantize.py            # compress, and generate the firmware files
```

`scripts/config.py` holds every path, label and setting — change things there
rather than in the individual scripts.

**`train.py` and `quantize.py` need TensorFlow, which has no release for Python
3.14 yet.** Use Python 3.11 or 3.12 for those two steps; a separate conda
environment works well. The other scripts run fine on 3.14.

## Flashing the board

Open `firmware/KeywordSpotter/KeywordSpotter.ino` in the Arduino IDE, set the
board to Arduino Nano 33 BLE, and upload. You need the Arduino Mbed OS Nano
board package and the **Chirale_TensorFlowLite** library — Arduino stopped
publishing its own `Arduino_TensorFlowLite`, and Chirale's is built from the
same source. No drivers or admin rights needed.

Run `scripts/quantize.py` first. Until you do, `model_data.cc` is a placeholder
that deliberately refuses to compile, rather than building fine and then failing
on the board.

## Hardware

One LED on D2 lights on a recognised word; a second on D3 lights while clips are
being recorded. The onboard RGB LED shows status. Full wiring and reasoning:
[docs/hardware.md](docs/hardware.md).

```
 Nano D2 ----[330R]----|>|---- Nano GND      detection LED
 Nano D3 ----[330R]----|>|---- Nano GND      recording LED
```

Use a red, yellow or amber LED. Blue and white ones need 3.0–3.2 V, which leaves
almost nothing across the resistor on a 3.3 V board, so they barely light.

## Things that tripped me up

- **Trained to 100%, predicted nothing useful.** Keras's default
  batch-normalisation momentum assumes far more training steps than this small
  dataset gives it, so its internal averages never settled. Lowering it took
  validation accuracy from 53% to 98%. See `BN_MOMENTUM` in
  [scripts/config.py](scripts/config.py).

- **The board hung silently at startup.** TensorFlow Lite Micro needs a list of
  exactly which operations the model uses. A Keras `Flatten()` layer looks like
  one but expands into four, so the board couldn't find one and stopped dead
  with no output. See
  [KeywordSpotter.ino](firmware/KeywordSpotter/KeywordSpotter.ino).

- **98% accuracy, and the device never reacted.** Validation clips hold one
  neatly-positioned word; live, the board sees a sliding window that usually
  catches half a word and scores low. The confidence threshold had to be
  calibrated against the hardware, not the validation set. Good validation
  numbers are not the same thing as a working device.

## Results

`scripts/evaluate.py` writes a confusion matrix and per-class accuracy into
`results/`. The model reached roughly 98% validation accuracy on ~500 clips —
a small dataset, so a promising figure rather than a robust one.
