"""Trelis/whisper-hinglish-preview bench: hand-built decoder prefix, with/without <|mixedcode|>.

The mixed-script switch is a real added token (id 51866) placed between <|hi|> and
<|transcribe|>; pipeline()/generate(language=) cannot emit it, so the prefix is built by hand.
Audio is cut into fixed windows because the 448-token/window cap truncates dense Devanagari
(~2 tok/char) well before 30 s.
"""
import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="Trelis/whisper-hinglish-preview")
parser.add_argument("--window", type=float, default=15.0)
parser.add_argument("--no-mixedcode", action="store_true")
parser.add_argument("--results", default="results/asr_bench_hinglish.jsonl")
args = parser.parse_args()

SR = 16000
name = args.repo.split("/")[-1] + ("-hi-only" if args.no_mixedcode else "")

proc = WhisperProcessor.from_pretrained(args.repo)
model = WhisperForConditionalGeneration.from_pretrained(args.repo, dtype=torch.bfloat16).to("cuda").eval()

tok = proc.tokenizer
ids = tok.convert_tokens_to_ids
prefix = [ids("<|startoftranscript|>"), ids("<|hi|>")]
if not args.no_mixedcode:
    prefix += tok("<|mixedcode|>", add_special_tokens=False).input_ids
prefix += [ids("<|transcribe|>"), ids("<|notimestamps|>")]
assert all(i is not None and i >= 0 for i in prefix), prefix
prefix_t = torch.tensor([prefix], device="cuda")


def transcribe(audio):
    parts = []
    step = int(args.window * SR)
    for start in range(0, len(audio), step):
        feat = proc.feature_extractor(audio[start:start + step], sampling_rate=SR, return_tensors="pt")
        feat = feat.input_features.to("cuda", torch.bfloat16)
        with torch.inference_mode():
            out = model.generate(input_features=feat, decoder_input_ids=prefix_t, max_new_tokens=440, num_beams=1, do_sample=False)
        parts.append(tok.decode(out[0], skip_special_tokens=True).strip())
    return " ".join(p for p in parts if p)


out_dir = Path("results/transcripts")
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
clips = sorted(Path("samples").glob("*.wav"))
transcribe(sf.read(clips[0])[0][:SR * 5])  # warm-up
for clip in clips:
    audio, sr = sf.read(clip)
    assert sr == SR, (clip, sr)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t0 = time.perf_counter()
    text = transcribe(audio)
    wall = time.perf_counter() - t0
    (out_dir / f"{name}__{clip.stem}__cuda.txt").write_text(text, encoding="utf-8")
    row = {"model": name, "clip": clip.stem, "device": "cuda", "rtf": round(wall / (len(audio) / SR), 3),
           "chars": len(text), "decode": f"win{int(args.window)}-prefix{'-mc' if not args.no_mixedcode else ''}"}
    rows.append(row)
    print(row)
    print("  ", text[:160])

with open(args.results, "a", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
