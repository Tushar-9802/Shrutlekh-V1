"""Single benchmark run: one model x one clip x one device. Appends one JSONL row."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--clip", required=True)
parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
parser.add_argument("--threads", type=int, default=4)  # CPU config only; the Rs45k-laptop proxy
parser.add_argument("--results", default="results/asr_bench.jsonl")
parser.add_argument("--transcripts", default="results/transcripts")
args = parser.parse_args()

run_key = (args.model, Path(args.clip).stem, args.device)
results_path = Path(args.results)
results_path.parent.mkdir(parents=True, exist_ok=True)

if results_path.exists():  # resumable: skip runs already recorded
    done = {(r["model"], r["clip"], r["device"]) for r in map(json.loads, results_path.open(encoding="utf-8"))}
    if run_key in done:
        print(f"skip (already done): {run_key}")
        raise SystemExit(0)

import torch  # imported after the skip check so a skip costs nothing

if args.device == "cpu":
    torch.set_num_threads(args.threads)

import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

DTYPE = "float16" if args.device == "cuda" else "float32"  # CPUs have no native fp16 math
PARAMS = {
    "language": "hi",
    "task": "transcribe",
    "temperature": 0.0,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
}

console = Console()
console.print(f"[bold]{args.model}[/] | {args.clip} | {args.device}/{DTYPE}")

t0 = time.perf_counter()
asr = pipeline(
    "automatic-speech-recognition",
    model=args.model,
    dtype=DTYPE,
    device=args.device,
    chunk_length_s=30,
)
load_s = time.perf_counter() - t0

duration = sf.info(args.clip).duration

with console.status("warm-up run (untimed)..."):
    asr(args.clip, generate_kwargs=PARAMS)

with console.status("timed run..."):
    t0 = time.perf_counter()
    out = asr(args.clip, generate_kwargs=PARAMS)
    wall = time.perf_counter() - t0

transcript_dir = Path(args.transcripts)
transcript_dir.mkdir(parents=True, exist_ok=True)
transcript_path = transcript_dir / f"{args.model.split('/')[-1]}__{Path(args.clip).stem}__{args.device}.txt"
transcript_path.write_text(out["text"], encoding="utf-8")

row = {
    "model": args.model,
    "clip": Path(args.clip).stem,
    "device": args.device,
    "dtype": DTYPE,
    "threads": args.threads if args.device == "cpu" else None,
    "load_s": round(load_s, 2),
    "audio_s": round(duration, 2),
    "wall_s": round(wall, 2),
    "rtf": round(wall / duration, 4),
    "transcript": str(transcript_path),
    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
with results_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()

console.print(f"RTF [bold]{row['rtf']}[/]  (load {row['load_s']}s, wall {row['wall_s']}s / audio {row['audio_s']}s)")
console.print(Panel(out["text"].strip(), title=str(transcript_path)))
