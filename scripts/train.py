"""Train the DS-CNN keyword spotter on the extracted MFCC splits.

Reads data/processed/, writes the best checkpoint to
models/checkpoints/ds_cnn_best.h5, per-feature normalization stats to
models/checkpoints/normalization.npz, and training curves to results/.

All paths and hyperparameters come from config.py -- none are defined here.

Requires TensorFlow. As of this writing there is no TensorFlow wheel for
Python 3.14 -- run this under a Python 3.11/3.12 environment instead (this
project was set up with a dedicated `keywordspotter` conda env: Python 3.11 +
this repo's requirements.txt).
"""

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


def load_splits():
    """Load x/y train and val arrays from data/processed/, add a channel dim.

    Conv2D expects (N, H, W, C); extract_features.py saves (N, H, W). Does
    NOT normalize -- that happens in normalize_splits() below, fit on train
    only, so val statistics never leak into what "normal" looks like.
    """
    def load(name):
        x = np.load(config.PROCESSED_DIR / f"x_{name}.npy")
        y = np.load(config.PROCESSED_DIR / f"y_{name}.npy")
        return x[..., np.newaxis], y

    x_train, y_train = load("train")
    x_val, y_val = load("val")
    return (x_train, y_train), (x_val, y_val)


def normalize_splits(train, val):
    """Per-MFCC-coefficient standardization, fit on train, saved for reuse.

    Stats are per coefficient (last axis, NUM_MFCC values), not a single
    global mean/std: MFCC coefficient 0 (roughly log-energy) sits on a very
    different scale than the higher coefficients, and a single scalar would
    let the low-index coefficients dominate.

    Saved to config.NORMALIZATION_PATH -- quantize.py and, eventually,
    feature_provider.cpp both need these exact numbers. Computing this fresh
    at inference time (e.g. per-clip normalization) would silently diverge
    from what the model was trained on.
    """
    x_train, y_train = train
    x_val, y_val = val

    mean = x_train.mean(axis=(0, 1), keepdims=True)  # shape (1,1,NUM_MFCC,1)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)  # guard a coefficient that happens to be constant

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.NORMALIZATION_PATH,
             mean=mean.squeeze(), std=std.squeeze())
    print(f"Saved normalization stats -> {config.NORMALIZATION_PATH}")

    return (x_train, y_train), (x_val, y_val)


def _reg():
    """Fresh L2 regularizer instance -- Keras regularizer objects are not
    meant to be shared across layers, so this is called at each use site."""
    return keras.regularizers.l2(config.WEIGHT_DECAY)


def _ds_block(x, filters):
    """One depthwise-separable block: depthwise conv + pointwise conv.

    Both convs carry activation='relu' directly rather than a following
    Activation layer -- the TFLite converter fuses a trailing ReLU into the
    conv it follows either way, but writing it inline makes the intended
    fused op count explicit at the model-definition level.

    SpatialDropout2D vanishes entirely from the converted inference graph
    (dropout is a training-only op TFLite drops on export), so it costs
    nothing on-device -- it's here purely to fight the overfitting a ~50
    clip/class dataset invites in a model this size.
    """
    x = keras.layers.DepthwiseConv2D(
        config.DS_CNN_BLOCK_KERNEL, strides=(1, 1), padding="same",
        use_bias=False, depthwise_regularizer=_reg())(x)
    x = keras.layers.BatchNormalization(momentum=config.BN_MOMENTUM)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(
        filters, (1, 1), strides=(1, 1), padding="same", use_bias=False,
        kernel_regularizer=_reg())(x)
    x = keras.layers.BatchNormalization(momentum=config.BN_MOMENTUM)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.SpatialDropout2D(config.DROPOUT)(x)
    return x


def build_model():
    """DS-CNN: conv stem + DS_CNN_BLOCKS depthwise-separable blocks + GAP head.

    Input shape is (*config.FEATURE_SHAPE, 1); output is NUM_CLASSES softmax.
    See the DS-CNN comment block in config.py for the op-level reasoning --
    every layer here maps to one of the six ops KeywordSpotter.ino's resolver
    TODO already commits to (conv2d, depthwise_conv2d, fully_connected,
    softmax, reshape, average_pool_2d).
    """
    inputs = keras.Input(shape=(*config.FEATURE_SHAPE, 1))

    x = keras.layers.Conv2D(
        config.DS_CNN_FILTERS, config.DS_CNN_STEM_KERNEL,
        strides=config.DS_CNN_STEM_STRIDE, padding="same",
        use_bias=False, kernel_regularizer=_reg())(inputs)
    x = keras.layers.BatchNormalization(momentum=config.BN_MOMENTUM)(x)
    x = keras.layers.ReLU()(x)

    for _ in range(config.DS_CNN_BLOCKS):
        x = _ds_block(x, config.DS_CNN_FILTERS)

    # Full-extent average pool, not GlobalAveragePooling2D -- see the note in
    # config.py on why that distinction matters for the on-device op list.
    _, h, w, _ = x.shape
    x = keras.layers.AveragePooling2D(pool_size=(h, w))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(config.DROPOUT)(x)
    outputs = keras.layers.Dense(
        config.NUM_CLASSES, activation="softmax",
        kernel_regularizer=_reg())(x)

    return keras.Model(inputs, outputs, name="ds_cnn")


def _augment_one(x):
    """Randomly perturb one already-normalized (H, W, 1) MFCC clip.

    Pure numpy so it's trivial to reason about and unit-test in isolation;
    wrapped for tf.data via tf.numpy_function in make_train_dataset. Applied
    fresh every epoch (not once, up front) so it acts as a real regularizer
    rather than just enlarging the dataset by a fixed multiple.
    """
    x = x.copy()
    h, w, _ = x.shape

    # Time shift: the word can land anywhere in the 1s capture window: the
    # countdown-triggered capture has no notion of speech onset. Zero-padding
    # the vacated edge approximates silence in normalized space (~= mean).
    shift = np.random.randint(-config.AUGMENT_TIME_SHIFT_FRAMES,
                               config.AUGMENT_TIME_SHIFT_FRAMES + 1)
    if shift > 0:
        x = np.concatenate([np.zeros((shift, w, 1), x.dtype), x[:-shift]], axis=0)
    elif shift < 0:
        x = np.concatenate([x[-shift:], np.zeros((-shift, w, 1), x.dtype)], axis=0)

    # SpecAugment-style time/frequency masking.
    mask_len = np.random.randint(0, config.AUGMENT_TIME_MASK_FRAMES + 1)
    if mask_len > 0:
        start = np.random.randint(0, h - mask_len + 1)
        x[start:start + mask_len, :, :] = 0.0

    mask_len = np.random.randint(0, config.AUGMENT_FREQ_MASK_BINS + 1)
    if mask_len > 0:
        start = np.random.randint(0, w - mask_len + 1)
        x[:, start:start + mask_len, :] = 0.0

    x = x + np.random.normal(0, config.AUGMENT_NOISE_STD, size=x.shape)
    return x.astype(np.float32)


def make_train_dataset(x_train, y_train):
    """tf.data pipeline applying _augment_one per-example, reshuffled and
    re-augmented every epoch."""
    def tf_augment(x, y):
        x_aug = tf.numpy_function(_augment_one, [x], tf.float32)
        x_aug.set_shape(x.shape)
        return x_aug, y

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds = ds.shuffle(len(x_train), seed=config.RANDOM_SEED,
                    reshuffle_each_iteration=True)
    ds = ds.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train(model, train_split, val_split):
    """Fit with checkpointing on val accuracy; return the History.

    Training data goes through make_train_dataset (augmented, reshuffled
    every epoch); val is passed as plain arrays -- it must never be
    augmented, or "val accuracy" stops measuring real generalization.
    """
    x_train, y_train = train_split
    x_val, y_val = val_split

    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.CHECKPOINT_PATH),
            monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True),
    ]

    train_ds = make_train_dataset(x_train, y_train)
    return model.fit(
        train_ds,
        validation_data=(x_val, y_val),
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )


def plot_curves(history):
    """Save accuracy/loss curves to results/training_curves.png."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))

    ax_acc.plot(history.history["accuracy"], label="train")
    ax_acc.plot(history.history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()

    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    fig.tight_layout()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves -> {out_path}")


def main():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_split, val_split = load_splits()
    train_split, val_split = normalize_splits(train_split, val_split)

    model = build_model()
    history = train(model, train_split, val_split)
    plot_curves(history)

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest val_accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved -> {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
