"""Grid runner: fresh subprocess per (model, clip, device) so no state leaks between runs."""
import itertools
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

MODELS = [
    "ARTPARK-IISc/whisper-small-vaani-hindi",
    "ARTPARK-IISc/whisper-medium-vaani-hindi",
    "ARTPARK-IISc/whisper-large-v3-vaani-hindi",
    "openai/whisper-large-v3",
    # "ai4bharat/indic-conformer-600m-multilingual",  # different API — needs its own loader branch first
]
CLIPS = sorted(Path("samples").glob("*.wav"))
DEVICES = ["cuda", "cpu"]
RESULTS = "results/asr_bench.jsonl"

console = Console()
failures = []

grid = list(itertools.product(MODELS, CLIPS, DEVICES))
for i, (model, clip, device) in enumerate(grid, 1):
    console.rule(f"[{i}/{len(grid)}] {model.split('/')[-1]} | {clip.name} | {device}")
    cmd = [sys.executable, "scripts/benchmarks/bench_one.py",
           "--model", model, "--clip", str(clip), "--device", device, "--results", RESULTS]
    if subprocess.run(cmd).returncode != 0:
        failures.append((model, clip.name, device))  # keep going; a broken combo shouldn't kill the grid

rows = [json.loads(line) for line in open(RESULTS, encoding="utf-8")]
table = Table(title="Shrutlekh ASR bake-off — all runs")
for col in ("model", "clip", "config", "load s", "RTF"):
    table.add_column(col, justify="right" if col in ("load s", "RTF") else "left")
for r in sorted(rows, key=lambda r: (r["clip"], r["device"], r["rtf"])):
    table.add_row(r["model"].split("/")[-1], r["clip"], f"{r['device']}/{r['dtype']}", f"{r['load_s']:.1f}", f"{r['rtf']:.3f}")
console.print(table)

if failures:
    console.print(f"[red]{len(failures)} failed:[/] {failures}")
