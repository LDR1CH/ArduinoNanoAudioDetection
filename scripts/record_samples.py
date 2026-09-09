"""Capture WAV clips per class into data/raw/<label>/, recorded on the Nano.

Requires firmware/AudioCapture flashed to the board. Clips come from the same
MP34DT05 microphone that will run inference, which avoids the train/deploy
mismatch you get by recording on a laptop mic.

Usage:
    python scripts/record_samples.py --list-ports
    python scripts/record_samples.py --label play
    python scripts/record_samples.py --label _background_noise --count 200
    python scripts/record_samples.py --sentences          # see below
    python scripts/record_samples.py                      # every label in turn

--sentences reads whole sentences aloud for _background_noise instead of
single words, using docs/test_sentences.md. Each ~3s reading is sliced into
several 1-second training clips (see slice_windows), so it covers natural
speech -- including words that sound like your keywords -- far faster than
recording single negative words one at a time.

Existing clips are counted first, so re-running the same command tops a class
up rather than overwriting it. Ctrl-C stops cleanly.
"""

import argparse
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports
import soundfile as sf

import config

MAGIC = b"KS16"
CLIP_BYTES = config.CLIP_SAMPLES * 2
SENTENCE_MAGIC = b"KS3S"
SENTENCE_BYTES = config.SENTENCE_SAMPLES * 2
SERIAL_TIMEOUT_S = 15

# Bytes per block in the paced transfer. Must match kBlockBytes in
# firmware/AudioCapture. The host requests one block at a time rather than
# letting the board push the whole clip: a single large write stalls at 28672
# bytes on this core and only resumes when the host sends another byte.
BLOCK_BYTES = 1000


# USB vendor IDs that identify an Arduino board. Matching on the VID rather
# than on name strings matters on Windows: the generic USB CDC driver reports
# the manufacturer as "Microsoft" and the description as "USB Serial Device
# (COMn)", so the word "arduino" appears nowhere at all.
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1B4F, 0x239A}


def is_arduino(port):
    return port.vid in ARDUINO_VIDS


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        vid_pid = f"{port.vid:04X}:{port.pid:04X}" if port.vid else "----:----"
        mark = "  <- Arduino" if is_arduino(port) else ""
        print(f"  {port.device:8} {vid_pid}  {port.description}{mark}")


def autodetect_port():
    """Pick the Arduino port, or None if there is not exactly one."""
    candidates = [p for p in serial.tools.list_ports.comports() if is_arduino(p)]
    if len(candidates) == 1:
        return candidates[0].device
    return None


def open_board(port):
    """Open the serial port and wait for the board's READY line."""
    board = serial.Serial(port, 115200, timeout=SERIAL_TIMEOUT_S)

    # Opening the port resets the board, so the mic settle window restarts.
    print(f"Waiting for the board on {port} (mic settling)...", flush=True)
    deadline = time.time() + SERIAL_TIMEOUT_S
    while time.time() < deadline:
        line = board.readline().decode("ascii", "replace").strip()
        if line == "READY":
            print("Board ready.\n")
            return board
        if line.startswith("ERROR"):
            raise RuntimeError(f"Board reported: {line}")

    # Already-running board that printed READY before we connected.
    board.reset_input_buffer()
    board.write(b"P")
    if board.readline().decode("ascii", "replace").strip() == "PONG":
        print("Board ready.\n")
        return board
    raise RuntimeError("No response from the board. Is AudioCapture flashed?")


def await_magic(board, magic):
    """Scan the stream for a clip header.

    Scanning rather than reading a fixed number of bytes means a previous
    aborted transfer cannot desynchronise every clip that follows it.
    """
    window = bytearray()
    deadline = time.time() + SERIAL_TIMEOUT_S
    while time.time() < deadline:
        byte = board.read(1)
        if not byte:
            continue
        window += byte
        if len(window) > len(magic):
            window.pop(0)
        if bytes(window) == magic:
            return
    raise RuntimeError(f"Timed out waiting for header {magic!r}")


def _read_paced(board, command, magic, total_bytes):
    """Shared body of record_clip/record_long_clip: request, then pull the
    clip one block at a time so the board never has more than one block
    in flight (see AudioCapture.ino for why that matters)."""
    board.reset_input_buffer()
    board.write(command)
    await_magic(board, magic)

    raw = bytearray()
    while len(raw) < total_bytes:
        want = min(BLOCK_BYTES, total_bytes - len(raw))
        board.write(b"G")
        block = board.read(want)
        if len(block) != want:
            raise RuntimeError(
                f"Short block at offset {len(raw)}: "
                f"got {len(block)} of {want} bytes")
        raw += block

    return np.frombuffer(bytes(raw), dtype="<i2")


def record_clip(board):
    """Ask the board for one 1-second clip; return it as an int16 array."""
    return _read_paced(board, b"R", MAGIC, CLIP_BYTES)


def record_long_clip(board):
    """Ask the board for one SENTENCE_SAMPLES-long clip; return an int16 array.

    Used only for background/negative speech -- a whole sentence read aloud,
    later sliced into CLIP_SAMPLES-long windows by slice_windows().
    """
    return _read_paced(board, b"L", SENTENCE_MAGIC, SENTENCE_BYTES)


def slice_windows(samples):
    """Slide a CLIP_SAMPLES window across a longer clip at SLICE_STRIDE_MS
    steps. Every slice is exactly CLIP_SAMPLES long, matching what inference
    sees on-device -- this only gets more than one training clip out of a
    single natural sentence."""
    windows = []
    start = 0
    while start + config.CLIP_SAMPLES <= len(samples):
        windows.append(samples[start:start + config.CLIP_SAMPLES])
        start += config.SLICE_STRIDE_SAMPLES
    return windows


def describe(samples, label=None):
    """A one-line health check, so bad clips are obvious as they land."""
    peak = int(np.abs(samples).max())
    pct = peak / 32767 * 100
    if peak >= 32767:
        return f"peak {pct:5.1f}%  CLIPPING -- lower kPdmGain or back off"
    # Background clips are supposed to be quiet; warning on them would cry wolf
    # through the longest recording session of the set.
    if pct < 5 and label != config.BACKGROUND_LABEL:
        return f"peak {pct:5.1f}%  very quiet -- raise kPdmGain or move closer"
    return f"peak {pct:5.1f}%"


def next_index(label_dir, label):
    """Continue numbering after whatever is already recorded."""
    existing = sorted(label_dir.glob(f"{label}_*.wav"))
    if not existing:
        return 0
    last = existing[-1].stem.rsplit("_", 1)[-1]
    return int(last) + 1


def record_label(board, label, count):
    label_dir = config.RAW_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    start = next_index(label_dir, label)

    print(f"--- {label}: recording {count} clips (starting at {start:03d}) ---")
    if label == config.BACKGROUND_LABEL:
        print("Background: stay quiet, or make ordinary room noise. Include")
        print("unrelated speech -- this class is what stops false triggers.\n")
    else:
        print(f'Say "{label}" once per clip. Vary distance, volume, and pace.\n')

    for i in range(count):
        index = start + i
        for countdown in (3, 2, 1):
            print(f"  {index:03d}  {countdown}...", end="\r", flush=True)
            time.sleep(0.6)
        print(f"  {index:03d}  NOW    ", end="\r", flush=True)

        samples = record_clip(board)
        path = label_dir / f"{label}_{index:03d}.wav"
        sf.write(path, samples, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"  {index:03d}  saved  {describe(samples, label)}")

    print()


def load_sentences(path=None):
    """Read sentences to speak aloud from a markdown bullet list.

    Any top-level "- " line in the file counts, regardless of which section
    it is under -- the headers in docs/test_sentences.md are for a human
    reader, not for this parser.
    """
    path = path or config.SENTENCES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No sentence file at {path}. See docs/test_sentences.md.")

    sentences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            sentences.append(stripped[2:].strip())
    if not sentences:
        raise ValueError(f"No '- ' bullet lines found in {path}")
    return sentences


def record_background_sentences(board, sentences_path=None):
    """Read whole sentences aloud for the _background_noise class.

    Each SENTENCE_DURATION_MS reading is sliced into several CLIP_SAMPLES
    windows (see slice_windows), so this yields multiple training clips per
    sentence spoken.
    """
    label = config.BACKGROUND_LABEL
    label_dir = config.RAW_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(sentences_path)
    windows_per_sentence = 1 + (
        (config.SENTENCE_SAMPLES - config.CLIP_SAMPLES)
        // config.SLICE_STRIDE_SAMPLES)
    index = next_index(label_dir, label)

    print(f"--- {label}: {len(sentences)} sentences, "
          f"~{windows_per_sentence} clips each "
          f"(~{len(sentences) * windows_per_sentence} total), "
          f"starting at {index:03d} ---")
    print("Read each sentence aloud at a normal pace. Do not say your")
    print('keywords ("music", "play", "pause", "skip") -- the point of')
    print("this class is everything that is NOT a keyword.\n")

    for i, sentence in enumerate(sentences):
        print(f'  [{i + 1}/{len(sentences)}] "{sentence}"')
        for countdown in (3, 2, 1):
            print(f"      {countdown}...", end="\r", flush=True)
            time.sleep(0.6)
        print("      reading now, keep talking     ", end="\r", flush=True)

        samples = record_long_clip(board)
        windows = slice_windows(samples)
        for window in windows:
            path = label_dir / f"{label}_{index:03d}.wav"
            sf.write(path, window, config.SAMPLE_RATE, subtype="PCM_16")
            index += 1
        print(f"      saved {len(windows)} clips  {describe(samples, label)}"
              + " " * 10)

    print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=config.LABELS, default=None,
                        help="record one class (default: every class in turn)")
    parser.add_argument("--count", type=int, default=config.SAMPLES_PER_CLASS,
                        help="clips to record for each class")
    parser.add_argument("--port", default=None,
                        help="serial port, e.g. COM3 (default: autodetect)")
    parser.add_argument("--list-ports", action="store_true",
                        help="list serial ports and exit")
    parser.add_argument("--sentences", action="store_true",
                        help="record _background_noise from sentences in "
                             "docs/test_sentences.md instead of single words "
                             "(see that file). Ignores --label/--count.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_ports:
        list_ports()
        return
    if args.sentences and args.label and args.label != config.BACKGROUND_LABEL:
        print("--sentences only records _background_noise; drop --label "
              f"{args.label!r} or omit --sentences.")
        sys.exit(1)

    port = args.port or autodetect_port()
    if port is None:
        print("Could not identify the board. Ports visible right now:")
        list_ports()
        print("")
        print("Pass the right one with --port, e.g. --port COM7.")
        print("If none are marked Arduino, check the cable is a data cable")
        print("and that AudioCapture is flashed.")
        sys.exit(1)

    board = open_board(port)
    try:
        if args.sentences:
            record_background_sentences(board)
        else:
            labels = [args.label] if args.label else config.LABELS
            for label in labels:
                record_label(board, label, args.count)
    except KeyboardInterrupt:
        print("\nStopped. Clips recorded so far are saved.")
    finally:
        board.close()


if __name__ == "__main__":
    main()
