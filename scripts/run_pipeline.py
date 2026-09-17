import argparse
import json
from pathlib import Path

from shrutlekh import config, pipeline

p = argparse.ArgumentParser()
p.add_argument("audio", type=Path)
p.add_argument("--mode", choices=("hindi", "hinglish"), default="hindi")
p.add_argument("--template", choices=tuple(pipeline.summarize.TEMPLATES), default="meeting")
p.add_argument("--tier", choices=("cpu", "gpu"))
args = p.parse_args()

notes = pipeline.run(args.audio, config.load(args.tier), args.mode, args.template)

for s in notes.segments:
    print(f"[{s.start:6.1f}-{s.end:6.1f}] {s.text}")
print("\n--- summary ---\n" + notes.summary)
print("\n--- action items ---\n" + json.dumps(notes.action_items, ensure_ascii=False, indent=1))
print("\n--- timings ---", notes.timings)
