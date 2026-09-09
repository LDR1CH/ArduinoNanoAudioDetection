"""Single source of truth for the Python half of the pipeline.

Every path, label and hyperparameter used by record_samples.py,
extract_features.py, train.py, evaluate.py and quantize.py lives here.
Nothing in scripts/ should hardcode a path or a magic number elsewhere.

The firmware mirror of this file is firmware/KeywordSpotter/model_settings.h,
which quantize.py regenerates so the two cannot drift apart.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
TFLITE_DIR = REPO_ROOT / "models" / "tflite"
RESULTS_DIR = REPO_ROOT / "results"
FIRMWARE_DIR = REPO_ROOT / "firmware" / "KeywordSpotter"

CHECKPOINT_PATH = CHECKPOINT_DIR / "ds_cnn_best.h5"
TFLITE_PATH = TFLITE_DIR / "ds_cnn_int8.tflite"

# --- Classes -------------------------------------------------------------
# Order matters: the index of a label here is the index the model emits and
# the index firmware/model_settings.h must agree on. Background noise is a
# real class, always index 0, so the model has a rejection option.
BACKGROUND_LABEL = "_background_noise"

# Sidecar written by record_samples.py --sentences (data/raw/_background_noise/
# _groups.json): maps clip filename -> a group id shared by every window
# sliced from the same sentence reading. extract_features.py keeps a group
# together on one side of the train/val/test split -- overlapping windows
# from one reading share most of their audio, so splitting them apart would
# leak train-set content into val/test and inflate those scores. A clip
# absent from the sidecar (every ambient clip, and every keyword clip) is
# its own singleton group.
BACKGROUND_GROUPS_FILENAME = "_groups.json"
KEYWORDS = ["music", "play", "pause", "skip"]
LABELS = [BACKGROUND_LABEL] + KEYWORDS
NUM_CLASSES = len(LABELS)

# --- Audio capture -------------------------------------------------------
SAMPLE_RATE = 16000          # Hz; must match the PDM config in audio_provider
CLIP_DURATION_MS = 1000
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION_MS // 1000
SAMPLES_PER_CLASS = 60       # target recordings per class in record_samples.py

# Longer captures for _background_noise negative speech (record_samples.py
# --sentences): read a whole sentence, then slide a CLIP_SAMPLES-long window
# across it at SLICE_STRIDE_MS steps, saving each window as its own training
# clip. Every saved clip is still exactly CLIP_SAMPLES long -- that is what
# the model is trained and run on -- this just gets more than one out of a
# single natural utterance. Must match kSentenceSamples in AudioCapture.ino.
SENTENCE_DURATION_MS = 3000
SENTENCE_SAMPLES = SAMPLE_RATE * SENTENCE_DURATION_MS // 1000
SLICE_STRIDE_MS = 500
SLICE_STRIDE_SAMPLES = SAMPLE_RATE * SLICE_STRIDE_MS // 1000
# 1 + (3000 - 1000) / 500 = 5 windows per sentence reading
SENTENCES_PATH = REPO_ROOT / "docs" / "test_sentences.md"

# --- MFCC features -------------------------------------------------------
# The (NUM_FRAMES, NUM_MFCC) array here is the model's input shape and is
# what feature_provider must produce on-device.
#
# Framing is non-centered (no reflect-padding at the clip edges): frame i
# starts at sample i * WINDOW_STRIDE_SAMPLES, and the window is exactly
# WINDOW_SIZE_SAMPLES long with no zero-padding inside the FFT. That is what
# makes the NUM_FRAMES formula below exact: with CLIP_SAMPLES=16000,
# WINDOW_SIZE_SAMPLES=640, WINDOW_STRIDE_SAMPLES=320, the 49th frame's last
# sample is exactly the clip's last sample, with nothing dropped or padded.
# feature_provider.cpp must frame the same way, or the two feature sets
# disagree about what "frame 0" even is.
WINDOW_SIZE_MS = 40
WINDOW_STRIDE_MS = 20
WINDOW_SIZE_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_MS // 1000
WINDOW_STRIDE_SAMPLES = SAMPLE_RATE * WINDOW_STRIDE_MS // 1000
N_MELS = 40                  # mel filterbank channels before DCT truncation
NUM_MFCC = 10                # coefficients kept after the DCT
NUM_FRAMES = 1 + (CLIP_DURATION_MS - WINDOW_SIZE_MS) // WINDOW_STRIDE_MS  # 49
FEATURE_SHAPE = (NUM_FRAMES, NUM_MFCC)

# --- Training ------------------------------------------------------------
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 1337

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3

# DS-CNN topology: one standard conv, then N depthwise-separable blocks.
DS_CNN_FILTERS = 64
DS_CNN_BLOCKS = 4
DROPOUT = 0.2

# --- Inference / firmware ------------------------------------------------
DETECTION_THRESHOLD = 0.80   # softmax score needed to fire the action
TENSOR_ARENA_SIZE = 30 * 1024  # bytes reserved for TFLM on the Nano
