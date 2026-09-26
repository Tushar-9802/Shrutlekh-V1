"""Request/response models. Explicit fields only — no dict[str, Any] on the wire."""
from __future__ import annotations

from pydantic import BaseModel


class RecordingCreated(BaseModel):
    id: str
    status: str


class RecordingOut(BaseModel):
    id: str
    title: str
    duration_s: float | None = None
    mode: str
    template: str
    tier: str | None = None
    asr_model: str | None = None
    created_at: int
    processed_at: int | None = None
    status: str
    error: str | None = None
    n_segments: int = 0


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class ActionItem(BaseModel):
    what: str
    who: str | None = None
    when: str | None = None


class NoteOut(BaseModel):
    id: str
    template: str
    summary: str
    action_items: list[ActionItem]
    model: str | None = None
    timings: dict[str, float]
    created_at: int


class SearchHit(BaseModel):
    recording_id: str
    title: str
    t_start: float
    snippet: str
    rank: float


class ConfigOut(BaseModel):
    tier: str
    asr_model: str
    ollama_model: str
    modes: list[str]
    templates: list[str]
    max_upload_mb: int
    audio_extensions: list[str]
