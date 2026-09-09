"""Train the DS-CNN keyword spotter on the extracted MFCC splits.

Reads data/processed/, writes the best checkpoint to
models/checkpoints/ds_cnn_best.h5 and training curves to results/.

All paths and hyperparameters come from config.py -- none are defined here.
"""

import config


def load_splits():
    """Load x/y train and val arrays from data/processed/."""
    raise NotImplementedError  # TODO: np.load each split


def build_model():
    """DS-CNN: conv stem + DS_CNN_BLOCKS depthwise-separable blocks + GAP head.

    Input shape is (*config.FEATURE_SHAPE, 1); output is NUM_CLASSES softmax.
    Keep the ops TFLM-friendly -- no layers the Micro interpreter lacks a
    kernel for, since this graph is what quantize.py converts.
    """
    raise NotImplementedError  # TODO: keras model


def train(model, splits):
    """Fit with checkpointing on val accuracy; return the History."""
    raise NotImplementedError  # TODO: ModelCheckpoint -> config.CHECKPOINT_PATH


def plot_curves(history):
    """Save accuracy/loss curves to results/training_curves.png."""
    raise NotImplementedError  # TODO: matplotlib


def main():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    model = build_model()
    history = train(model, splits)
    plot_curves(history)


if __name__ == "__main__":
    main()
