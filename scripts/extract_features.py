"""MFCC extraction and train/val/test split construction.

Reads every WAV under data/raw/<label>/, computes a
(NUM_FRAMES, NUM_MFCC) MFCC array per clip, and writes stratified splits to
data/processed/ as .npy arrays:

    x_train.npy y_train.npy  x_val.npy y_val.npy  x_test.npy y_test.npy

The MFCC parameters here are the contract with the firmware feature_provider:
if they change, feature_provider must change with them (see the framing note
in config.py, next to WINDOW_SIZE_SAMPLES).

Splitting is group-aware: windows sliced from the same sentence reading (see
record_samples.py --sentences) share most of their audio, so they are kept
together on one side of train/val/test rather than split with plain
per-clip stratification -- otherwise val/test scores on the background class
would be inflated by near-duplicates the model already saw in training.

Deliberately NOT done here: per-feature normalization. That is train.py's job,
computed from the train split only -- doing it here would let val/test
statistics leak into what "normal" looks like during training.
"""

import json

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

import config


def load_clip(path):
    """Load a WAV as float32 mono in [-1, 1], padded/trimmed to CLIP_SAMPLES.

    Every clip this project records is already exactly CLIP_SAMPLES at
    config.SAMPLE_RATE (see firmware/AudioCapture and record_samples.py), so
    padding/trimming here is a defensive fallback, not the expected path. A
    wrong sample rate is not silently resampled -- resampling a training clip
    that was never actually heard at that rate is exactly the kind of
    silent train/deploy mismatch this project has been careful to avoid.
    """
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # collapse to mono defensively

    if sample_rate != config.SAMPLE_RATE:
        raise ValueError(
            f"{path}: sample rate {sample_rate} != config.SAMPLE_RATE "
            f"({config.SAMPLE_RATE}). This file was not recorded by this "
            f"project's pipeline -- resampling it would risk training on "
            f"audio that doesn't match what the mic actually captured.")

    if len(samples) < config.CLIP_SAMPLES:
        samples = np.pad(samples, (0, config.CLIP_SAMPLES - len(samples)))
    elif len(samples) > config.CLIP_SAMPLES:
        samples = samples[:config.CLIP_SAMPLES]

    return samples


def compute_mfcc(samples):
    """Return a (NUM_FRAMES, NUM_MFCC) MFCC array for one clip.

    center=False and n_fft == win_length keep framing exact (see the note by
    WINDOW_SIZE_SAMPLES in config.py): frame i is samples
    [i*stride : i*stride+win_length], no reflect-padding at the edges, no
    zero-padding inside the FFT.
    """
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.NUM_MFCC,
        n_mels=config.N_MELS,
        n_fft=config.WINDOW_SIZE_SAMPLES,
        win_length=config.WINDOW_SIZE_SAMPLES,
        hop_length=config.WINDOW_STRIDE_SAMPLES,
        window="hann",
        center=False,
    )
    mfcc = mfcc.T.astype(np.float32)  # librosa returns (n_mfcc, frames)

    if mfcc.shape != config.FEATURE_SHAPE:
        raise RuntimeError(
            f"MFCC shape {mfcc.shape} != config.FEATURE_SHAPE "
            f"{config.FEATURE_SHAPE} -- a config.py constant changed "
            f"without NUM_FRAMES/FEATURE_SHAPE being recomputed to match.")

    return mfcc


def load_clip_groups(label_dir):
    """Load the sentence-group sidecar for one class dir, or {} if absent.

    Only _background_noise has one (written by record_samples.py
    --sentences). Every other class has no sidecar, so every clip in it is
    its own singleton group -- which is the correct default anyway, since
    those clips were each recorded independently.
    """
    path = label_dir / config.BACKGROUND_GROUPS_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset():
    """Walk data/raw/, return (features, labels, groups) over all classes.

    Label order is config.LABELS -- the same order the firmware emits and
    evaluate.py's confusion matrix uses -- so the class index a clip gets
    here is the index the model will learn to predict for it.

    groups[i] identifies which clips must stay on the same side of the
    train/val/test split as clip i. Almost every clip is its own singleton
    group (its full path, guaranteed unique); the exception is clips sliced
    from the same sentence reading (see load_clip_groups), which share a
    group id so overlapping windows can't leak across splits.
    """
    features = []
    labels = []
    groups = []
    filenames = []

    print("Loading clips:")
    for class_index, label in enumerate(config.LABELS):
        label_dir = config.RAW_DIR / label
        clips = sorted(label_dir.glob(f"{label}_*.wav"))
        if not clips:
            raise RuntimeError(
                f"No clips found for class {label!r} in {label_dir}. Every "
                f"class in config.LABELS needs at least some recordings "
                f"before features can be extracted.")

        clip_groups = load_clip_groups(label_dir)
        n_grouped = 0
        for clip_path in clips:
            samples = load_clip(clip_path)
            features.append(compute_mfcc(samples))
            labels.append(class_index)
            group = clip_groups.get(clip_path.name)
            if group is not None:
                n_grouped += 1
            groups.append(group if group is not None else str(clip_path))
            filenames.append(f"{label}/{clip_path.name}")

        note = f", {n_grouped} in sentence groups" if n_grouped else ""
        print(f"  {label:20} {len(clips):4d} clips{note}")

    total = len(labels)
    print(f"  {'total':20} {total:4d} clips\n")

    return (np.stack(features), np.array(labels, dtype=np.int64),
            np.array(groups, dtype=object), np.array(filenames, dtype=object))


def _grouped_split(features, labels, groups, fraction, seed):
    """One stratified, group-respecting split, holding out ~fraction of rows.

    sklearn has no direct "StratifiedGroupShuffleSplit" for an exact
    percentage, so this approximates it with StratifiedGroupKFold: build
    round(1/fraction) folds and hold out one. With imbalanced group sizes
    (34 five-clip sentence groups vs hundreds of singletons here) the held-out
    fraction lands close to but not exactly on target -- fine for a dataset
    this size, and far better than leaking sentence windows across splits.
    Returns (keep_idx, held_out_idx).
    """
    n_folds = max(2, round(1 / fraction))
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    keep_idx, held_out_idx = next(splitter.split(features, labels, groups))
    return keep_idx, held_out_idx


def split_and_save(features, labels, groups, filenames):
    """Group-respecting stratified split by config.VAL_SPLIT / TEST_SPLIT.

    Both fractions target the FULL dataset; the val cut is rescaled below so
    it still lands near VAL_SPLIT of the original after test is removed. See
    _grouped_split for why the result is approximate rather than exact.

    Alongside x_/y_<split>.npy, writes <split>_files.txt -- the source
    data/raw/<label>/<clip>.wav for every row in that split, in order. Useful
    for tracing a misclassified row back to the actual clip later (see
    evaluate.py), and it's what makes the leakage check below meaningful from
    outside this function, not just self-reported.
    """
    train_val_idx, test_idx = _grouped_split(
        features, labels, groups, config.TEST_SPLIT, config.RANDOM_SEED)

    val_fraction_of_remainder = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    train_sub_idx, val_sub_idx = _grouped_split(
        features[train_val_idx], labels[train_val_idx], groups[train_val_idx],
        val_fraction_of_remainder, config.RANDOM_SEED)
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    # A group split apart would mean two rows sharing a group id in
    # different splits -- verify that never happened, rather than trust the
    # library silently.
    split_groups = {name: set(groups[idx]) for name, idx in split_indices.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[a] & split_groups[b]
        if overlap:
            raise RuntimeError(
                f"Group leakage between {a} and {b}: {overlap!r} appears in "
                f"both. This should be impossible with StratifiedGroupKFold "
                f"-- something upstream broke the group assignment.")

    print("Splits:")
    for name, idx in split_indices.items():
        x, y = features[idx], labels[idx]
        np.save(config.PROCESSED_DIR / f"x_{name}.npy", x)
        np.save(config.PROCESSED_DIR / f"y_{name}.npy", y)
        (config.PROCESSED_DIR / f"{name}_files.txt").write_text(
            "\n".join(filenames[idx]) + "\n", encoding="utf-8")
        counts = np.bincount(y, minlength=config.NUM_CLASSES)
        breakdown = ", ".join(
            f"{label}={n}" for label, n in zip(config.LABELS, counts))
        print(f"  {name:6} {len(y):4d} clips  ({breakdown})")


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features, labels, groups, filenames = build_dataset()
    split_and_save(features, labels, groups, filenames)


if __name__ == "__main__":
    main()
