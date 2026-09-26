"""Every SQL statement in the app. Callers (CLI, worker, API) work in dicts and Segments."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ulid import ULID

from shrutlekh.asr import Segment
from shrutlekh.config import Settings
from shrutlekh.db import RAW_DIR, get_connection
from shrutlekh.pipeline import Notes


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma", ".mkv", ".mov", ".3gp"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


class IngestError(ValueError):
    """Bad input from the user: unsupported type, empty, or too large."""


def _register(rid: str, title: str, source_path: str | None, raw: Path, mode: str, template: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO recordings (id, title, source_path, raw_path, mode, template, created_at, status)"
            " VALUES (?,?,?,?,?,?,?,'pending')",
            (rid, title, source_path, str(raw), mode, template, int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


def ingest(source: Path, mode: str, template: str, title: str | None = None) -> str:
    """Copy a local file into data/raw and register a pending recording."""
    ext = source.suffix.lower()
    if ext not in AUDIO_EXTS:
        raise IngestError(f"unsupported file type '{ext}' - audio or video only")
    rid = str(ULID())
    raw = RAW_DIR / f"{rid}{ext}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw)  # the copy is what we process; the user's file is never touched again
    _register(rid, title or source.stem, str(source), raw, mode, template)
    return rid


def ingest_stream(filename: str, fileobj, mode: str, template: str, title: str | None = None) -> str:
    """Stream an upload into data/raw. Never buffers the whole file: meeting audio runs to hundreds of MB."""
    ext = Path(filename).suffix.lower()
    if ext not in AUDIO_EXTS:
        raise IngestError(f"unsupported file type '{ext}' - audio or video only")
    rid = str(ULID())
    raw = RAW_DIR / f"{rid}{ext}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with open(raw, "wb") as out:
            while chunk := fileobj.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise IngestError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                out.write(chunk)
        if size == 0:
            raise IngestError("file is empty")
    except Exception:
        raw.unlink(missing_ok=True)  # no half-written file left behind for a failed upload
        raise
    _register(rid, title or Path(filename).stem, None, raw, mode, template)
    return rid


def set_status(rid: str, status: str, error: str | None = None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE recordings SET status = ?, error = ?,"
            " processed_at = CASE WHEN ? IN ('ready','failed') THEN ? ELSE processed_at END"
            " WHERE id = ?",
            (status, error, status, int(time.time()), rid),
        )
        conn.commit()
    finally:
        conn.close()


def save_transcript(rid: str, segments: list[Segment], settings: Settings, duration_s: float) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM segments WHERE recording_id = ?", (rid,))  # re-transcribe replaces
        conn.executemany(
            "INSERT INTO segments (recording_id, seq, t_start, t_end, text) VALUES (?,?,?,?,?)",
            [(rid, i, s.start, s.end, s.text) for i, s in enumerate(segments)],
        )
        conn.execute(
            "UPDATE recordings SET duration_s = ?, tier = ?, asr_model = ? WHERE id = ?",
            (duration_s, settings.tier, settings.asr_model, rid),
        )
        conn.commit()
    finally:
        conn.close()


def save_notes(rid: str, notes: Notes, template: str, model: str) -> str:
    nid = str(ULID())
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO notes (id, recording_id, template, summary, action_items, model, timings, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (nid, rid, template, notes.summary, json.dumps(notes.action_items, ensure_ascii=False),
             model, json.dumps(notes.timings), int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()
    return nid


def get_recording(rid: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rid,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_segments(rid: str) -> list[Segment]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT t_start, t_end, text FROM segments WHERE recording_id = ? ORDER BY seq", (rid,)
        ).fetchall()
        return [Segment(r["t_start"], r["t_end"], r["text"]) for r in rows]
    finally:
        conn.close()


def get_notes(rid: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM notes WHERE recording_id = ? ORDER BY created_at DESC", (rid,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["action_items"] = json.loads(d["action_items"])
            d["timings"] = json.loads(d["timings"]) if d["timings"] else {}
            out.append(d)
        return out
    finally:
        conn.close()


def list_recordings(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT r.*, (SELECT count(*) FROM segments s WHERE s.recording_id = r.id) AS n_segments"
            " FROM recordings r ORDER BY r.created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _snippet(text: str, q: str, width: int = 70) -> str:
    """Window the text around the first hit; trigram phrase matches are literal substrings."""
    i = text.casefold().find(q.casefold())
    if i < 0:
        return text[: width * 2]
    start = max(0, i - width)
    end = min(len(text), i + len(q) + width)
    return ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")


def search(query: str, limit: int = 20) -> list[dict]:
    """Keyword search over transcripts. Quoted as one phrase so user punctuation can't be FTS syntax."""
    q = query.strip()
    if not q:
        return []
    conn = get_connection()
    try:
        if len(q) < 3:  # trigram indexes 3-char windows; shorter queries would silently match nothing
            rows = conn.execute(
                "SELECT s.recording_id, r.title, s.t_start, s.text, 0.0 AS rank"
                " FROM segments s JOIN recordings r ON r.id = s.recording_id"
                " WHERE s.text LIKE ? ORDER BY r.created_at DESC LIMIT ?", (f"%{q}%", limit)
            ).fetchall()
        else:
            phrase = '"' + q.replace('"', '""') + '"'
            rows = conn.execute(
                "SELECT s.recording_id, r.title, s.t_start, s.text, bm25(segments_fts) AS rank"
                " FROM segments_fts JOIN segments s ON s.id = segments_fts.rowid"
                " JOIN recordings r ON r.id = s.recording_id"
                " WHERE segments_fts MATCH ? ORDER BY rank LIMIT ?", (phrase, limit)
            ).fetchall()
        return [dict(r) | {"snippet": _snippet(r["text"], q)} for r in rows]
    finally:
        conn.close()


def delete_recording(rid: str) -> bool:
    rec = get_recording(rid)
    if rec is None:
        return False
    conn = get_connection()
    try:
        conn.execute("DELETE FROM recordings WHERE id = ?", (rid,))  # segments + notes cascade
        conn.commit()
    finally:
        conn.close()
    if rec["raw_path"]:
        Path(rec["raw_path"]).unlink(missing_ok=True)
    return True
