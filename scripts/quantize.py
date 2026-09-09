"""INT8 TFLite export, plus the generated C sources the firmware compiles.

Produces:
    models/tflite/ds_cnn_int8.tflite
    firmware/KeywordSpotter/model_data.cc   (the .tflite as a C byte array)
    firmware/KeywordSpotter/model_data.h
    firmware/KeywordSpotter/model_settings.h

model_settings.h is generated rather than hand-written so the class labels,
input shape and sample rate cannot drift from config.py. All three generated
files are committed, so firmware/ builds standalone without Python. When you
regenerate them, note the source .tflite and the date in the commit message.

INT8 rather than float32: the Cortex-M4F has an FPU, but int8 is smaller and
faster and has to share the RAM/flash on the Nano with the rest of the sketch.
If accuracy drops too far, try quantization-aware training before reverting.
"""

import config


def representative_dataset():
    """Yield ~100 real training samples so the converter can calibrate ranges."""
    raise NotImplementedError  # TODO: yield from x_train, shaped (1, *FEATURE_SHAPE, 1)


def convert_to_int8():
    """Convert the Keras checkpoint to a fully-INT8 .tflite, return its bytes."""
    raise NotImplementedError  # TODO: TFLiteConverter, int8 in/out, INT8 ops only


def write_model_data(tflite_bytes):
    """Emit model_data.cc/.h -- the model as an aligned const unsigned char[]."""
    raise NotImplementedError  # TODO: hex rows, alignas(8), extern decls in .h


def write_model_settings():
    """Emit model_settings.h from config.py: labels, shape, arena, threshold."""
    raise NotImplementedError  # TODO: render the header, labels in config order


def main():
    config.TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    tflite_bytes = convert_to_int8()
    config.TFLITE_PATH.write_bytes(tflite_bytes)
    write_model_data(tflite_bytes)
    write_model_settings()


if __name__ == "__main__":
    main()
