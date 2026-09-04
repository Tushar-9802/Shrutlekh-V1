"""int8-CT2 CPU bench via faster-whisper: RTF + WER for converted Vaani models.

chunk_length=15 is load-bearing: Whisper caps decoding at 448 tokens per window and
Devanagari costs ~2 tokens/char, so default 30s windows silently truncate dense Hindi.
"""
import argparse
import json
import re
import time
from pathlib import Path

import jiwer
import soundfile as sf
from faster_whisper import WhisperModel
from hindi_normalize import normalize_transcript

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["models/vaani-small-int8", "models/vaani-medium-int8"])
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--chunk-length", type=int, default=15)
parser.add_argument("--results", default="results/asr_bench_int8.jsonl")
args = parser.parse_args()

DECODE = dict(
    language="hi",
    temperature=0.0,
    beam_size=1,
    condition_on_previous_text=False,
    without_timestamps=True,
    chunk_length=args.chunk_length,
)


def norm(s):
    s = normalize_transcript(s)
    s = re.sub(r"[।॥.,!?;:\"'()\-]", " ", s)  # danda + latin punctuation out of WER
    return re.sub(r"\s+", " ", s).strip()


refs = {p.stem.replace(".ref", ""): norm(p.read_text(encoding="utf-8")) for p in Path("samples").glob("*.ref.txt")}
out = Path("results/transcripts")
out.mkdir(parents=True, exist_ok=True)

rows = []
for model_dir in args.models:
    name = Path(model_dir).name
    m = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=args.threads)
    warm = True
    for clip in sorted(Path("samples").glob("*.wav")):
        dur = sf.info(clip).duration
        if warm:  # first call pays memory/plan init, never measured
            list(m.transcribe(str(clip), **DECODE)[0])
            warm = False
        t0 = time.perf_counter()
        text = "".join(s.text for s in m.transcribe(str(clip), **DECODE)[0])
        wall = time.perf_counter() - t0
        (out / f"{name}__{clip.stem}__cpu.txt").write_text(text, encoding="utf-8")
        row = {
            "model": f"{name}-ct2",
            "clip": clip.stem,
            "device": "cpu",
            "threads": args.threads,
            "rtf": round(wall / dur, 3),
            "wer": round(jiwer.wer(refs[clip.stem], norm(text)), 3) if clip.stem in refs else None,
            "chars": len(text),
            "decode": f"chunk{args.chunk_length}-nots",
        }
        rows.append(row)
        print(row)

with open(args.results, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
