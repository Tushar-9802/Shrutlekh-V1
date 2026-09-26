"""The job: pending -> processing -> ready | failed. Called by the CLI now, by a background task later."""
from __future__ import annotations

import threading
from pathlib import Path

from shrutlekh import asr, config, pipeline, store

_LOCK = threading.Lock()  # one recording at a time: two models would fight over VRAM


def process(rid: str, tier: str | None = None) -> None:
    with _LOCK:
        _process(rid, tier)


def _process(rid: str, tier: str | None) -> None:
    rec = store.get_recording(rid)
    if rec is None:
        raise ValueError(f"no recording {rid}")
    settings = config.load(tier)
    raw = Path(rec["raw_path"])
    store.set_status(rid, "processing")
    try:
        notes = pipeline.run(raw, settings, rec["mode"], rec["template"])
        store.save_transcript(rid, notes.segments, settings, asr.audio_duration(raw))
        store.save_notes(rid, notes, rec["template"], settings.ollama_model)
        store.set_status(rid, "ready")
    except Exception as exc:  # the row carries the reason; the caller still sees the traceback
        store.set_status(rid, "failed", f"{type(exc).__name__}: {exc}")
        raise
