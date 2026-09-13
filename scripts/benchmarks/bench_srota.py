"""Srota (moorlee/qwen3-asr-0.6b-hinglish) bench. Runs under envs/srota (qwen-asr pins transformers 4.57.6):
    envs/srota/Scripts/python.exe scripts/benchmarks/bench_srota.py
"""
import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="moorlee/qwen3-asr-0.6b-hinglish")
parser.add_argument("--window", type=float, default=30.0)  # card: keep segments <= 30 s
parser.add_argument("--results", default="results/asr_bench_hinglish.jsonl")
args = parser.parse_args()

SR = 16000
name = args.repo.split("/")[-1]
model = Qwen3ASRModel.from_pretrained(args.repo, dtype=torch.bfloat16, device_map="cuda:0")
_PREFIX = re.compile(r"^\s*language\s+\S+\s*<asr_text>\s*")  # raw decode may carry the training prefix


def transcribe(audio, tmp):
    parts = []
    step = int(args.window * SR)
    for i, start in enumerate(range(0, len(audio), step)):
        path = tmp / f"w{i}.wav"
        sf.write(path, audio[start:start + step], SR)
        text = model.transcribe(audio=str(path), language=None)[0].text
        parts.append(_PREFIX.sub("", text).strip())
    return " ".join(p for p in parts if p)


out_dir = Path("results/transcripts")
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
clips = sorted(Path("samples").glob("*.wav"))
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    transcribe(sf.read(clips[0])[0][:SR * 5], tmp)  # warm-up
    for clip in clips:
        audio, sr = sf.read(clip)
        assert sr == SR, (clip, sr)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        t0 = time.perf_counter()
        text = transcribe(audio, tmp)
        wall = time.perf_counter() - t0
        (out_dir / f"{name}__{clip.stem}__cuda.txt").write_text(text, encoding="utf-8")
        row = {"model": name, "clip": clip.stem, "device": "cuda", "rtf": round(wall / (len(audio) / SR), 3),
               "chars": len(text), "decode": f"win{int(args.window)}-langNone"}
        rows.append(row)
        print(row)
        print("  ", text[:160])

with open(args.results, "a", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
