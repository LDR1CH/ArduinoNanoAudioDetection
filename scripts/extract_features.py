"""MFCC extraction and train/val/test split construction.

Reads every WAV under data/raw/<label>/, computes a
(NUM_FRAMES, NUM_MFCC) MFCC array per clip, and writes stratified splits to
data/processed/ as .npy arrays:

    x_train.npy y_train.npy  x_val.npy y_val.npy  x_test.npy y_test.npy

The MFCC parameters here are the contract with the firmware feature_provider:
if they change, feature_provider must change with them.
"""

import config


def load_clip(path):
    """Load a WAV as float32 mono, pad or trim to config.CLIP_SAMPLES."""
    raise NotImplementedError  # TODO: soundfile.read + pad/trim


def compute_mfcc(samples):
    """Return a (NUM_FRAMES, NUM_MFCC) MFCC array for one clip."""
    raise NotImplementedError  # TODO: librosa.feature.mfcc with config windows


def build_dataset():
    """Walk data/raw/, return (features, labels) arrays over all classes."""
    raise NotImplementedError  # TODO: iterate config.LABELS, index == class id


def split_and_save(features, labels):
    """Stratified split by config.VAL_SPLIT / TEST_SPLIT, save to processed/."""
    raise NotImplementedError  # TODO: train_test_split twice, np.save each


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features, labels = build_dataset()
    split_and_save(features, labels)


if __name__ == "__main__":
    main()
