"""HTTP surface. All work happens in store/worker; these only validate and map errors."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from shrutlekh import config, store, worker
from shrutlekh.schemas import (
    ConfigOut,
    NoteOut,
    RecordingCreated,
    RecordingOut,
    SearchHit,
    SegmentOut,
)
from shrutlekh.summarize import TEMPLATES

MODES = ("hindi", "hinglish")
router = APIRouter(prefix="/api")


def _require(rid: str) -> dict:
    rec = store.get_recording(rid)
    if rec is None:
        raise HTTPException(404, "Recording not found.")
    return rec


@router.post("/recordings", response_model=RecordingCreated, status_code=202)
def create_recording(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("hindi"),
    template: str = Form("meeting"),
    title: str | None = Form(None),
    tier: str | None = Form(None),
) -> RecordingCreated:
    """Accept an upload and queue transcription; returns at once with an id to poll."""
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if template not in TEMPLATES:
        raise HTTPException(400, f"template must be one of {tuple(TEMPLATES)}")
    if tier not in (None, "cpu", "gpu"):
        raise HTTPException(400, "tier must be cpu or gpu")
    try:
        rid = store.ingest_stream(file.filename or "audio", file.file, mode, template, title)
    except store.IngestError as exc:
        raise HTTPException(400, str(exc)) from exc
    background.add_task(worker.process, rid, tier)
    return RecordingCreated(id=rid, status="pending")


@router.get("/recordings", response_model=list[RecordingOut])
def list_recordings(limit: int = 50) -> list[RecordingOut]:
    return [RecordingOut(**r) for r in store.list_recordings(limit)]


@router.get("/recordings/{rid}", response_model=RecordingOut)
def get_recording(rid: str) -> RecordingOut:
    rec = _require(rid)
    rec["n_segments"] = len(store.get_segments(rid))
    return RecordingOut(**rec)


@router.get("/recordings/{rid}/segments", response_model=list[SegmentOut])
def get_segments(rid: str) -> list[SegmentOut]:
    _require(rid)
    return [SegmentOut(start=s.start, end=s.end, text=s.text) for s in store.get_segments(rid)]


@router.get("/recordings/{rid}/notes", response_model=list[NoteOut])
def get_notes(rid: str) -> list[NoteOut]:
    _require(rid)
    return [NoteOut(**n) for n in store.get_notes(rid)]


@router.delete("/recordings/{rid}", status_code=204)
def delete_recording(rid: str) -> None:
    rec = _require(rid)
    if rec["status"] == "processing":
        raise HTTPException(409, "This recording is being processed; try again in a moment.")
    store.delete_recording(rid)


@router.get("/search", response_model=list[SearchHit])
def search(q: str, limit: int = 20) -> list[SearchHit]:
    return [SearchHit(**h) for h in store.search(q, limit)]


@router.get("/config", response_model=ConfigOut)
def get_config() -> ConfigOut:
    s = config.load()
    return ConfigOut(
        tier=s.tier, asr_model=s.asr_model, ollama_model=s.ollama_model,
        modes=list(MODES), templates=list(TEMPLATES),
        max_upload_mb=store.MAX_UPLOAD_BYTES // (1024 * 1024),
        audio_extensions=sorted(store.AUDIO_EXTS),
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
