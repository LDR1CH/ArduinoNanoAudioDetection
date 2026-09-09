"""Evaluate the trained checkpoint on the held-out test split.

Writes results/confusion_matrix.png and results/per_class_accuracy.csv, and
prints overall accuracy. Class label order comes from config.LABELS -- the
same order the firmware prints -- so the matrix axes mean what they say.

    python scripts/evaluate.py                  # the .h5 checkpoint
    python scripts/evaluate.py --tflite         # the quantized model instead
"""

import argparse

import config


def load_test_split():
    """Load x_test/y_test from data/processed/."""
    raise NotImplementedError  # TODO: np.load


def predict_keras(x):
    """Predicted class indices from the .h5 checkpoint."""
    raise NotImplementedError  # TODO: keras.models.load_model + argmax


def predict_tflite(x):
    """Predicted class indices from the INT8 .tflite via the TFLite interpreter.

    Quantize inputs and dequantize outputs with the tensor scale/zero-point so
    this measures what the Nano will actually compute.
    """
    raise NotImplementedError  # TODO: tf.lite.Interpreter, per-tensor quant params


def report(y_true, y_pred):
    """Confusion matrix plot + per-class accuracy CSV into results/."""
    raise NotImplementedError  # TODO: sklearn confusion_matrix, matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", action="store_true",
                        help="evaluate the quantized model instead of the .h5")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_test, y_true = load_test_split()
    y_pred = predict_tflite(x_test) if args.tflite else predict_keras(x_test)
    report(y_true, y_pred)


if __name__ == "__main__":
    main()
