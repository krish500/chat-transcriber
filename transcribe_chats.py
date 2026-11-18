#!/usr/bin/env python3
"""
chat transcriber - turns my recorded calls into markdown transcripts

I record my coffee chats / networking calls (mostly .mkv from OBS) and kept
forgetting stuff people told me, so this transcribes them so I can go back
and actually find things.

setup:
    pip install faster-whisper "av<17"
    (need the av<17 pin, av 19 breaks faster-whisper's audio loading)

if you have an nvidia gpu (way faster):
    pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

how i run it:
    python transcribe_chats.py "C:\\Users\\me\\Videos"   # whole folder
    python transcribe_chats.py "some call.mkv"         # one file
    python transcribe_chats.py Videos --model base.en  # faster but less accurate
    python transcribe_chats.py Videos --accurate       # slower, best quality

it saves "<name>.transcript.md" in the same folder as the video and skips
anything already done, so i just rerun it on my videos folder after a call
"""
import argparse
import datetime as dt
import os
import sys
import time
import warnings
from pathlib import Path

MEDIA_EXTS = {".mkv", ".mp4", ".mov", ".m4a", ".mp3", ".wav", ".webm", ".flv"}


def find_recordings(paths):
    files = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in MEDIA_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Not found: {p}", file=sys.stderr)
    return files


def stamp(seconds):
    # 83 -> "01:23", only shows hours if the call went that long
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def register_nvidia_dlls():
    # windows can't find the cuda dlls from the nvidia pip packages on its own,
    # so point it at their bin folders (this was the cublas64_12.dll error)
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def load_model(name, device):
    from faster_whisper import WhisperModel
    import numpy as np

    if device in ("auto", "cuda"):
        register_nvidia_dlls()
        try:
            model = WhisperModel(name, device="cuda", compute_type="float16")
            # cuda doesn't actually load until the first transcribe, so test it on
            # 1 sec of silence here instead of crashing halfway through a real file
            list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
            return model, "GPU"
        except Exception as e:
            if device == "cuda":
                raise
            print(f"  GPU not usable ({str(e).splitlines()[0]}), using CPU")
    # cpu fallback - int8 + all cores is a lot faster than the defaults
    threads = os.cpu_count() or 4
    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads), f"CPU ({threads} threads)"


def transcribe(model, path, language, accurate, batch_size):
    from faster_whisper import BatchedInferencePipeline

    # beam_size 1 is way faster and barely worse for a clear 1-on-1 call
    opts = dict(language=language, beam_size=5 if accurate else 1,
                condition_on_previous_text=False)
    if batch_size > 1:
        # batched mode does chunks in parallel and skips silence automatically
        segments, info = BatchedInferencePipeline(model).transcribe(
            str(path), batch_size=batch_size, **opts)
    else:
        segments, info = model.transcribe(str(path), vad_filter=True, **opts)

    lines, paragraph, para_start, last_end = [], [], None, 0.0
    for seg in segments:
        # new paragraph after a 2s+ pause, usually means the other person is talking
        if paragraph and seg.start - last_end > 2.0:
            lines.append(f"**[{stamp(para_start)}]** " + " ".join(paragraph))
            paragraph = []
        if not paragraph:
            para_start = seg.start
        paragraph.append(seg.text.strip())
        last_end = seg.end
        print(f"\r  {stamp(seg.end)} / {stamp(info.duration)}", end="", flush=True)
    if paragraph:
        lines.append(f"**[{stamp(para_start)}]** " + " ".join(paragraph))
    print()
    return lines, info.duration


def main():
    ap = argparse.ArgumentParser(description="Transcribe chat recordings to Markdown.")
    ap.add_argument("paths", nargs="*", default=["."], help="files or folders (default: current folder)")
    ap.add_argument("--model", default="small.en",
                    help="tiny.en / base.en / small.en (default) / medium.en / large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="auto", help="auto (default: GPU if it works, else CPU), cuda, or cpu")
    ap.add_argument("--accurate", action="store_true", help="beam search: slower, slightly more accurate")
    ap.add_argument("--batch-size", type=int, default=8, help="chunks processed at once (1 = no batching)")
    ap.add_argument("--force", action="store_true", help="redo files that already have a transcript")
    args = ap.parse_args()

    todo = []
    for f in find_recordings(args.paths):
        out = f.with_suffix(".transcript.md")
        if out.exists() and not args.force:
            print(f"Skip (already done): {f.name}")
        else:
            todo.append((f, out))
    if not todo:
        print("Nothing to transcribe.")
        return

    # hide the huggingface symlink warning spam on windows
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"Loading model {args.model} ...")
    model, where = load_model(args.model, args.device)
    print(f"  Running on {where}")

    for f, out in todo:
        print(f"Transcribing {f.name}")
        t0 = time.time()
        lines, duration = transcribe(model, f, args.language, args.accurate, args.batch_size)
        # file modified time is close enough to when the call happened
        recorded = dt.datetime.fromtimestamp(f.stat().st_mtime)
        header = [
            f"# Transcript: {f.stem}",
            "",
            f"- **Recorded:** {recorded:%a %b %d, %Y %I:%M %p}",
            f"- **Length:** {stamp(duration)}",
            f"- **Source file:** {f.name}",
            "",
            "---",
            "",
        ]
        out.write_text("\n".join(header) + "\n\n".join(lines) + "\n", encoding="utf-8")
        took = time.time() - t0
        print(f"  Saved {out.name} ({took:.0f}s, {duration / max(took, 1):.1f}x realtime)")


if __name__ == "__main__":
    main()
