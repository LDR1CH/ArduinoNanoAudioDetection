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

Recordings are not tracked in git (`data/raw/**/*.wav` is gitignored) -- the
set is reproducible through `record_samples.py` plus
[docs/test_sentences.md](../docs/test_sentences.md), so there is nothing here
that can't be regenerated.

`_background_noise` clips come from two recording modes and look identical on
disk (both are plain 1-second WAVs, same naming): `--label _background_noise`
for ambient room noise, and `--sentences` for natural speech that is not a
keyword, read from [docs/test_sentences.md](../docs/test_sentences.md) and
sliced into five overlapping 1-second windows per sentence. See that file for
the sentence list and the reasoning.

`--sentences` also writes `_background_noise/_groups.json`, a sidecar mapping
each sliced clip's filename to the sentence reading it came from. The five
windows from one reading overlap by 50% and share most of their audio, so
`extract_features.py` uses this file to keep an entire group on one side of
the train/val/test split -- splitting them apart would let near-duplicate
audio leak from train into val/test and quietly inflate those scores. Also
gitignored, for the same reason as the WAVs themselves: it's regenerated
alongside them, not hand-maintained.
