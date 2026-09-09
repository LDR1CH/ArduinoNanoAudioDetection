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
# Per-MFCC-coefficient mean/std from the train split, computed by train.py.
# quantize.py applies the same normalization to its representative dataset,
# and must also bake these constants into model_settings.h -- feature_provider
# has to apply identical normalization on-device, or the model sees
# differently-scaled input at inference than it was trained on.
NORMALIZATION_PATH = CHECKPOINT_DIR / "normalization.npz"

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
WINDOW_SIZE_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_MS // 1000    # 640
WINDOW_STRIDE_SAMPLES = SAMPLE_RATE * WINDOW_STRIDE_MS // 1000  # 320
# FFT length: a power of two, NOT WINDOW_SIZE_SAMPLES (640 -- not a power of
# two, and a direct/mixed-radix transform at that size is both slow and hard
# to hand-verify on the Cortex-M4F). The windowed 640-sample frame is
# zero-padded to 1024 before the FFT, on both the Python and firmware side --
# this only changes frequency-bin resolution, not the frame timing that
# NUM_FRAMES depends on.
N_FFT = 1024
N_MELS = 40                  # mel filterbank channels before DCT truncation
NUM_MFCC = 10                # coefficients kept after the DCT
NUM_FRAMES = 1 + (CLIP_DURATION_MS - WINDOW_SIZE_MS) // WINDOW_STRIDE_MS  # 49
FEATURE_SHAPE = (NUM_FRAMES, NUM_MFCC)

# The exact MFCC recipe, verified against librosa's actual source (not just
# its docs) because feature_provider.cpp has to reproduce every step of it on
# the Nano bit-for-bit-close, and librosa.feature.mfcc()'s convenience
# wrapper hides two conventions that are NOT safe to reproduce as-is:
#
# 1. Window: PERIODIC Hann (scipy sym=False), not the symmetric variant --
#    w[n] = 0.5 - 0.5*cos(2*pi*n / WINDOW_SIZE_SAMPLES), n = 0..639.
#    Verified: matches librosa.filters.get_window('hann', 640) to float
#    precision (~1e-16). The symmetric variant differs by up to 0.0038 --
#    small, but a needless, avoidable mismatch.
#
# 2. power_to_db's top_db=80.0 default clips every value to within top_db of
#    log_spec.max() -- the max over the WHOLE ARRAY PASSED IN. Called once
#    per whole clip (as librosa.feature.mfcc does), that max spans all 49
#    frames at once: a fundamentally non-causal, whole-clip-dependent
#    operation. feature_provider computes frames incrementally as audio
#    streams in and cannot know a "whole clip max" -- worse, under streaming
#    there often isn't a well-defined clip boundary at all. This project sets
#    top_db=None (see extract_features.py's compute_mfcc, which calls
#    melspectrogram + power_to_db + DCT directly rather than through
#    feature.mfcc(), specifically to disable this). Every frame's log-power
#    is then a pure function of that frame alone, which is what actually
#    makes on-device reproduction possible.
#
# Everything else matches librosa.feature.mfcc's defaults directly: mel
# filterbank Slaney-normalized (mel_norm='slaney', htk=False), power
# spectrogram (power=2.0), ref=1.0/amin=1e-10 in power_to_db, DCT-II with
# norm='ortho' (verified against a from-scratch formula to ~1e-14).
#
# The mel filterbank itself (librosa.filters.mel(...)) is exported by
# quantize.py as a constant matrix into model_settings.h/.cc --
# feature_provider does a matrix-multiply against those weights rather than
# reconstructing Slaney-normalized triangular filters on-device, so the one
# part of this recipe with real per-filter arithmetic subtlety is exactly the
# librosa-computed values, not a hand reimplementation of them.

# --- Training ------------------------------------------------------------
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 1337

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15  # epochs of no val_accuracy gain before stopping

# DS-CNN topology: one standard conv, then N depthwise-separable blocks, then
# average-pool + dense + softmax. This is the "Hello Edge" DS-CNN shape (Zhang
# et al.) sized for exactly this FEATURE_SHAPE input -- a stem kernel wide in
# the time axis (frames carry more discriminative structure than individual
# MFCC bins), stride-2 stem to cut the sequence down once, then stride-1
# depthwise-separable blocks that don't shrink it further.
#
# Op mapping, registered in KeywordSpotter.ino's resolver: Conv2D/
# DepthwiseConv2D -> CONV_2D/DEPTHWISE_CONV_2D, BatchNorm+ReLU fuse into the
# preceding conv at TFLite-conversion time (no runtime op of their own),
# AveragePooling2D over the full remaining spatial extent -> AVERAGE_POOL_2D
# (deliberately not GlobalAveragePooling2D, which can lower to a MEAN op
# instead), Dense -> FULLY_CONNECTED, softmax -> SOFTMAX.
#
# Flatten does NOT map to a single RESHAPE as its name suggests -- confirmed
# the hard way, on real hardware. With a symbolic batch dimension, Keras's
# Flatten() lowered to SHAPE + STRIDED_SLICE + PACK + RESHAPE (computing the
# flattened size at graph-run time), not the plain static RESHAPE a "one op
# per layer" assumption predicts. Registering a resolver against only the
# "obviously needed" 6 ops compiled fine and then silently hung on the board
# at AllocateTensors() ("Didn't find op for builtin opcode 'SHAPE'"), with no
# other symptom -- exactly the kind of assumption this project verifies
# instead of trusting elsewhere. KeywordSpotter.ino's resolver comment has
# the full 9-op list and how to re-derive it if this model changes.
DS_CNN_FILTERS = 64
DS_CNN_BLOCKS = 4
DS_CNN_STEM_KERNEL = (10, 4)
DS_CNN_STEM_STRIDE = (2, 2)
DS_CNN_BLOCK_KERNEL = (3, 3)
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4  # L2 on conv/dense kernels; pure training-time cost, no
                      # runtime op added to the converted graph

# BatchNormalization's moving-average momentum. Keras's default (0.99) is
# tuned for large-batch, many-step training; at BATCH_SIZE=64 over ~400-500
# training clips there are only ~7 optimizer steps per epoch, nowhere near
# enough for a momentum-0.99 moving average to converge within EPOCHS. The
# symptom, verified empirically on this dataset: train accuracy climbs to
# ~100% (real batch statistics, train-mode) while val accuracy sits frozen at
# exactly the background-class fraction with near-uniform softmax output --
# predict() uses the still-unconverged moving average and produces near-noise
# on every input. 0.7 fixed it completely (0.53 -> 0.98 val accuracy on the
# same run, matching a plain LogisticRegression baseline on these features).
# Revisit upward only if the training set grows enough to give many more
# steps per epoch.
BN_MOMENTUM = 0.7

# --- Data augmentation -----------------------------------------------------
# Applied to TRAINING clips only, freshly randomized every epoch via a
# tf.data pipeline -- never to val/test. At ~50-60 clips per keyword,
# recorded in one or two sittings each, the model has almost no genuine
# variation to learn invariance from; without this, train.py reaches 100%
# train accuracy while val accuracy sits at chance (verified empirically:
# it did, before these were added). All are applied to already-normalized
# features, so magnitudes are in standard-deviation units.
AUGMENT_TIME_SHIFT_FRAMES = 4    # max frames to shift the clip earlier/later,
                                  # zero-padding the gap -- robustness to
                                  # where in the 1s capture window the word
                                  # actually landed
AUGMENT_TIME_MASK_FRAMES = 8     # SpecAugment-style: zero a random
                                  # contiguous block of frames (of this max
                                  # length) each epoch
AUGMENT_FREQ_MASK_BINS = 2       # same idea, across MFCC coefficients
AUGMENT_NOISE_STD = 0.1          # additive Gaussian noise, in normalized units

# --- Inference / firmware ------------------------------------------------
DETECTION_THRESHOLD = 0.80   # softmax score needed to fire the action
TENSOR_ARENA_SIZE = 30 * 1024  # bytes reserved for TFLM on the Nano
