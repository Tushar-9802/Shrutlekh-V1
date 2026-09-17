"""audio -> ASR -> normalize -> summary + action items. HTTP-free: the CLI and the API both call run()."""
import time
from dataclasses import dataclass, field
from pathlib import Path

from shrutlekh import asr, normalize, summarize
from shrutlekh.config import Settings


@dataclass
class Notes:
    segments: list[asr.Segment]
    summary: str
    action_items: list[dict]
    timings: dict[str, float] = field(default_factory=dict)


def run(path: Path, settings: Settings, mode: str = "hindi", template: str = "meeting") -> Notes:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    segments = asr.transcribe(path, settings, mode)
    t["asr_s"] = round(time.perf_counter() - t0, 2)
    if not segments:
        raise ValueError(f"no speech recognized in {path}")

    segments = normalize.normalize(segments)

    t0 = time.perf_counter()
    summary = summarize.summarize(segments, settings, template)
    items = summarize.action_items(segments, settings)
    t["llm_s"] = round(time.perf_counter() - t0, 2)
    return Notes(segments, summary, items, t)
