# data/raw/

One folder per class, named exactly as in `scripts/config.py` LABELS:

```
data/raw/_background_noise/
data/raw/music/
data/raw/play/
data/raw/pause/
data/raw/skip/
```

Clips are mono 16 kHz WAV, 1 second long, written by
`scripts/record_samples.py` and captured **on the Nano itself** via
`firmware/AudioCapture` -- the same microphone that runs inference. Record
`_background_noise` from the room the device will actually live in: silence,
fans, and speech that is not a keyword.

Recordings are tracked in git for now. If the dataset grows large, uncomment
the `data/raw/**/*.wav` rule in `.gitignore`; the set stays reproducible
through `record_samples.py`.

`_background_noise` clips come from two recording modes and look identical on
disk (both are plain 1-second WAVs, same naming): `--label _background_noise`
for ambient room noise, and `--sentences` for natural speech that is not a
keyword, read from [docs/test_sentences.md](../docs/test_sentences.md) and
sliced into 1-second windows. See that file for the sentence list and the
reasoning.
