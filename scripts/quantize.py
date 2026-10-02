"""INT8 TFLite export, plus the generated C sources the firmware compiles.

Produces:
    models/tflite/ds_cnn_int8.tflite
    firmware/KeywordSpotter/model_data.cc   (the .tflite as a C byte array)
    firmware/KeywordSpotter/model_data.h
    firmware/KeywordSpotter/model_settings.h/.cc

model_settings.h/.cc is generated rather than hand-written so the class
labels, input shape, sample rate and normalization stats cannot drift from
config.py / train.py. All generated files are committed, so firmware/ builds
standalone without Python. When you regenerate them, note the source
checkpoint and the date in the commit message.

INT8 rather than float32: the Cortex-M4F has an FPU, but int8 ops are still
smaller and faster, and the model needs to fit comfortably alongside the rest
of the sketch. If accuracy drops unacceptably after
quantization -- this script checks and prints exactly that -- try
quantization-aware training before reverting to float32.

Requires TensorFlow; see the environment note in train.py / README.md.
"""

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config

NUM_REPRESENTATIVE_SAMPLES = 100  # enough to calibrate int8 ranges without
                                    # taking long; doesn't need to be all of
                                    # train


def load_normalization():
    """Load the mean/std train.py fit on the train split.

    Reused here rather than recomputed: the representative dataset below and
    the constants baked into model_settings.h must be the exact numbers the
    checkpoint was trained against, or calibration (and on-device inference)
    would target a different input distribution than the model actually
    learned.
    """
    if not config.NORMALIZATION_PATH.exists():
        raise RuntimeError(
            f"{config.NORMALIZATION_PATH} not found. Run train.py first -- "
            f"it computes and saves the normalization stats this script "
            f"reuses for calibration and bakes into model_settings.h.")
    stats = np.load(config.NORMALIZATION_PATH)
    return stats["mean"], stats["std"]


def _load_normalized_split(name, mean, std):
    x = np.load(config.PROCESSED_DIR / f"x_{name}.npy")
    y = np.load(config.PROCESSED_DIR / f"y_{name}.npy")
    x = (x - mean) / std
    return x[..., np.newaxis].astype(np.float32), y


def representative_dataset():
    """Yield ~100 real, normalized training samples for int8 calibration.

    Drawn from train (never val/test -- the converter only uses this to
    measure activation ranges, but reaching into held-out data for anything,
    even calibration, blurs the same line extract_features.py's group-aware
    split exists to keep clean).
    """
    mean, std = load_normalization()
    x_train, _ = _load_normalized_split("train", mean, std)

    rng = np.random.default_rng(config.RANDOM_SEED)
    n = min(NUM_REPRESENTATIVE_SAMPLES, len(x_train))
    for i in rng.choice(len(x_train), size=n, replace=False):
        yield [x_train[i:i + 1]]


def convert_to_int8():
    """Convert the Keras checkpoint to a fully-INT8 .tflite, return its bytes.

    Full-integer quantization -- int8 activations throughout, not just
    weights, and int8 tensors at the graph boundary -- matching
    kFeatureElements-sized int8 input the firmware's feature_provider
    produces and the int8 output KeywordSpotter.ino dequantizes.
    """
    model = keras.models.load_model(config.CHECKPOINT_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def check_accuracy(tflite_bytes):
    """Compare float-checkpoint vs int8-.tflite accuracy on val.

    Not a substitute for evaluate.py's full confusion matrix -- just an
    immediate, impossible-to-miss signal if quantization tanked accuracy,
    per the int8-vs-float32 tradeoff described at the top of this file.
    """
    mean, std = load_normalization()
    x_val, y_val = _load_normalized_split("val", mean, std)

    keras_model = keras.models.load_model(config.CHECKPOINT_PATH)
    keras_preds = keras_model.predict(x_val, verbose=0).argmax(axis=1)
    keras_acc = (keras_preds == y_val).mean()

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = in_detail["quantization"]
    out_scale, out_zero = out_detail["quantization"]

    correct = 0
    for i in range(len(y_val)):
        x_q = np.round(x_val[i:i + 1] / in_scale + in_zero).astype(np.int8)
        interpreter.set_tensor(in_detail["index"], x_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(out_detail["index"])[0]
        out = (out_q.astype(np.float32) - out_zero) * out_scale
        correct += int(np.argmax(out) == y_val[i])
    tflite_acc = correct / len(y_val)

    print(f"val accuracy: keras (float) = {keras_acc:.4f}   "
          f"tflite (int8) = {tflite_acc:.4f}")
    drop = keras_acc - tflite_acc
    if drop > 0.05:
        print(f"WARNING: int8 quantization cost {drop:.4f} accuracy "
              f"(> 0.05). Consider quantization-aware training before "
              f"flashing this model.")
    return tflite_acc


def write_model_data(tflite_bytes):
    """Emit model_data.cc/.h -- the model as an aligned const unsigned char[]."""
    rows = []
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        rows.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    body = "\n".join(rows)

    tflite_rel = config.TFLITE_PATH.relative_to(config.REPO_ROOT).as_posix()
    cc = f"""\
// GENERATED by scripts/quantize.py from {tflite_rel}.
// Do not hand-edit -- re-run scripts/quantize.py to regenerate. Note the
// source checkpoint and the date in the commit message when you do.

#include "model_data.h"

alignas(8) const unsigned char g_model_data[] = {{
{body}
}};
const int g_model_data_len = {len(tflite_bytes)};
"""
    path = config.FIRMWARE_DIR / "model_data.cc"
    path.write_text(cc, encoding="utf-8")
    print(f"Wrote {path} ({len(tflite_bytes)} bytes)")


def write_model_settings():
    """Emit model_settings.h/.cc from config.py + train.py's normalization
    stats: labels, shape, arena, threshold, and the per-MFCC mean/std
    feature_provider must apply on-device before quantizing its features.
    """
    mean, std = load_normalization()
    background_index = config.LABELS.index(config.BACKGROUND_LABEL)

    header = f"""\
// model_settings.h -- single source of truth for the firmware side.
//
// Class labels, input shape, sample rate, the tensor arena size and the
// per-MFCC normalization stats all live here. Nothing else under firmware/
// may hardcode a label string or an array size; include this instead.
//
// GENERATED by scripts/quantize.py from scripts/config.py and
// models/checkpoints/normalization.npz. Edit those and re-run the pipeline
// rather than editing this file, so the Python and C++ sides cannot drift.

#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

// --- Audio capture (audio_provider) --------------------------------------
constexpr int kAudioSampleRate = {config.SAMPLE_RATE};
constexpr int kClipDurationMs = {config.CLIP_DURATION_MS};
constexpr int kClipSamples = kAudioSampleRate * kClipDurationMs / 1000;

// --- MFCC features (feature_provider) ------------------------------------
// Recipe is documented in full in scripts/config.py next to N_FFT -- window
// convention, why n_fft != window size, and why top_db-based dynamic range
// clipping is deliberately NOT part of this (it isn't reproducible frame by
// frame). Read that before changing anything here.
constexpr int kWindowSizeMs = {config.WINDOW_SIZE_MS};
constexpr int kWindowStrideMs = {config.WINDOW_STRIDE_MS};
constexpr int kWindowSizeSamples = {config.WINDOW_SIZE_SAMPLES};
constexpr int kWindowStrideSamples = {config.WINDOW_STRIDE_SAMPLES};
constexpr int kFftSize = {config.N_FFT};             // zero-padded FFT length
constexpr int kSpectrumBins = kFftSize / 2 + 1;  // real FFT output length
constexpr int kFeatureFrames = {config.NUM_FRAMES};
constexpr int kNumMels = {config.N_MELS};
constexpr int kFeatureCount = {config.NUM_MFCC};
constexpr int kFeatureElements = kFeatureFrames * kFeatureCount;

// Slaney-normalized mel filterbank (librosa.filters.mel), kNumMels x
// kSpectrumBins, row-major. feature_provider does a plain matrix-vector
// multiply (this matrix times the frame's power spectrum) rather than
// reconstructing triangular filter shapes/normalization on-device -- the one
// piece of this recipe with real per-filter arithmetic subtlety is exactly
// these librosa-computed weights, not a hand reimplementation of them.
// ~{"{:.0f}".format(config.N_MELS * (config.N_FFT // 2 + 1) * 4 / 1024)} KB in flash, dense
// (only ~5% of entries are nonzero -- a sparse encoding would shrink this a
// lot if flash ever gets tight, but there's no need for it yet).
extern const float kMelFilterbank[kNumMels * kSpectrumBins];

// Per-MFCC-coefficient normalization, fit on the training split by
// train.py and applied identically here: feature_provider must subtract
// kFeatureMean[bin] and divide by kFeatureStd[bin] for each MFCC
// coefficient BEFORE quantizing to int8. The model was trained on
// normalized input, not raw MFCC magnitudes -- skipping this on-device
// would feed the model data far outside the range it ever saw in training.
extern const float kFeatureMean[kFeatureCount];
extern const float kFeatureStd[kFeatureCount];

// --- Model ----------------------------------------------------------------
constexpr int kCategoryCount = {config.NUM_CLASSES};
constexpr int kBackgroundIndex = {background_index};  // the rejection class
extern const char* kCategoryLabels[kCategoryCount];

// Softmax score (0..1, after dequantizing the int8 output with its own
// scale/zero-point -- read those off the output tensor at init time, don't
// hardcode them here) required before command_responder fires the action.
constexpr float kDetectionThreshold = {config.DETECTION_THRESHOLD}f;

// Scratch memory for the TFLM interpreter. Bump if AllocateTensors() fails;
// shrink if the sketch runs out of RAM.
constexpr int kTensorArenaSize = {config.TENSOR_ARENA_SIZE};

#endif  // MODEL_SETTINGS_H_
"""

    mel_filterbank = librosa.filters.mel(
        sr=config.SAMPLE_RATE, n_fft=config.N_FFT, n_mels=config.N_MELS,
        norm="slaney").astype(np.float32)  # (N_MELS, N_FFT//2+1), row-major

    labels_body = ",\n".join(f'    "{label}"' for label in config.LABELS)
    mean_body = ", ".join(f"{v:.8f}f" for v in mean)
    std_body = ", ".join(f"{v:.8f}f" for v in std)

    # One row per line: readable diffs if the recipe ever changes, and each
    # row is exactly one mel filter, which is a natural grouping.
    filterbank_rows = []
    for row in mel_filterbank:
        row_body = ", ".join(f"{v:.8f}f" for v in row)
        filterbank_rows.append(f"    {row_body},")
    filterbank_body = "\n".join(filterbank_rows)

    cc = f"""\
// GENERATED by scripts/quantize.py -- see model_settings.h.
//
// Label order matches scripts/config.py LABELS exactly; index
// kBackgroundIndex is the background/rejection class. kFeatureMean/
// kFeatureStd come from models/checkpoints/normalization.npz. kMelFilterbank
// is librosa.filters.mel(sr=kAudioSampleRate, n_fft=kFftSize,
// n_mels=kNumMels, norm="slaney"), flattened row-major (row i = mel filter
// i's weight for each of the kSpectrumBins power-spectrum bins).

#include "model_settings.h"

const char* kCategoryLabels[kCategoryCount] = {{
{labels_body},
}};

const float kFeatureMean[kFeatureCount] = {{{mean_body}}};
const float kFeatureStd[kFeatureCount] = {{{std_body}}};

const float kMelFilterbank[kNumMels * kSpectrumBins] = {{
{filterbank_body}
}};
"""

    h_path = config.FIRMWARE_DIR / "model_settings.h"
    cc_path = config.FIRMWARE_DIR / "model_settings.cc"
    h_path.write_text(header, encoding="utf-8")
    cc_path.write_text(cc, encoding="utf-8")
    print(f"Wrote {h_path}")
    print(f"Wrote {cc_path}")


def main():
    config.TFLITE_DIR.mkdir(parents=True, exist_ok=True)

    tflite_bytes = convert_to_int8()
    config.TFLITE_PATH.write_bytes(tflite_bytes)
    print(f"Wrote {config.TFLITE_PATH} ({len(tflite_bytes)} bytes)")

    check_accuracy(tflite_bytes)

    write_model_data(tflite_bytes)
    write_model_settings()


if __name__ == "__main__":
    main()
