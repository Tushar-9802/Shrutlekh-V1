"""Score every results/transcripts/{model}__{clip}__{device}.txt against samples/{clip}.ref.txt."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import jiwer
from hindi_normalize import normalize_transcript
from rich.console import Console
from rich.table import Table

parser = argparse.ArgumentParser()
parser.add_argument("--refs", default="samples")
parser.add_argument("--transcripts", default="results/transcripts")
parser.add_argument("--out", default="results/wer.jsonl")
parser.add_argument("--device", default="cpu", help="cpu|cuda|all — GPU/CPU outputs are identical, so default to one")
args = parser.parse_args()

PUNCT = re.compile(r"[।॥.,!?;:\"'()\-]")


def norm(s):
    s = PUNCT.sub(" ", normalize_transcript(s)).lower()  # Latin case is not an ASR error
    return re.sub(r"\s+", " ", s).strip()


refs = {p.name.removesuffix(".ref.txt"): norm(p.read_text(encoding="utf-8")) for p in Path(args.refs).glob("*.ref.txt")}
rows = []
for t in sorted(Path(args.transcripts).glob("*__*__*.txt")):
    model, clip, device = t.stem.rsplit("__", 2)
    if clip not in refs or (args.device != "all" and device != args.device):
        continue
    hyp = norm(t.read_text(encoding="utf-8"))
    rows.append({"model": model, "clip": clip, "device": device,
                 "wer": round(jiwer.wer(refs[clip], hyp), 3), "cer": round(jiwer.cer(refs[clip], hyp), 3)})

with open(args.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

by_model = defaultdict(dict)
for r in rows:
    by_model[r["model"]][r["clip"]] = r
clips = sorted(refs)
table = Table(title=f"WER / CER vs references ({args.device})")
table.add_column("model")
for c in clips:
    table.add_column(c.replace("_", "\n"), justify="right")
for model, per_clip in sorted(by_model.items(), key=lambda kv: sum(v["wer"] for v in kv[1].values())):
    table.add_row(model, *[f"{per_clip[c]['wer']:.3f} / {per_clip[c]['cer']:.3f}" if c in per_clip else "—" for c in clips])
Console().print(table)
