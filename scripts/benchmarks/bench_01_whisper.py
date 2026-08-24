import time

import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # warnings acknowledged once, silenced for benchmark runs

MODEL = "openai/whisper-large-v3"
CLIP = "samples/fleurs_hi_1842.wav"
DEVICE = "cuda"
DTYPE = "float16" if DEVICE == "cuda" else "float32"  # CPUs have no native fp16 math

PARAMS = {
    "language": "hi",
    "task": "transcribe",
    "temperature": 0.0,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
}

console = Console()

t0 = time.perf_counter()
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL,
    dtype=DTYPE,
    device=DEVICE,
    chunk_length_s=30,
)
load_s = time.perf_counter() - t0

duration = sf.info(CLIP).duration

with console.status("warm-up run (untimed)..."):
    asr(CLIP, generate_kwargs=PARAMS)

with console.status("timed run..."):
    t0 = time.perf_counter()
    out = asr(CLIP, generate_kwargs=PARAMS)
    wall = time.perf_counter() - t0

table = Table(title="Shrutlekh ASR bake-off")
for col in ("model", "clip", "config", "load s", "audio s", "wall s", "RTF"):
    table.add_column(col, justify="right" if col[-1] == "s" or col == "RTF" else "left")
table.add_row(
    MODEL.split("/")[-1],
    CLIP.split("/")[-1],
    f"{DEVICE}/{DTYPE}",
    f"{load_s:.1f}",
    f"{duration:.1f}",
    f"{wall:.1f}",
    f"{wall / duration:.3f}",
)
console.print(table)
console.print(Panel(out["text"].strip(), title=CLIP))

with open("transcript_step1.txt", "w", encoding="utf-8") as f:
    f.write(out["text"])
