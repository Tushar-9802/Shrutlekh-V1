"""SQLite store: one file under data/, plain sqlite3, no ORM."""
from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "shrutlekh.db"

# CREATE ... IF NOT EXISTS throughout so init_db() is idempotent.
SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
  id TEXT PRIMARY KEY,               -- ULID: sorts by creation time
  title TEXT NOT NULL,
  source_path TEXT,                  -- where the user's file came from
  raw_path TEXT,                     -- untouched copy; re-transcribe reads this
  duration_s REAL,
  mode TEXT NOT NULL,                -- hindi | hinglish
  template TEXT NOT NULL,
  tier TEXT,                         -- cpu | gpu, whichever engine ran
  asr_model TEXT,                    -- exact model id, so a later upgrade is detectable
  created_at INTEGER NOT NULL,
  processed_at INTEGER,
  status TEXT NOT NULL,              -- pending | processing | ready | failed
  error TEXT
);

CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY,            -- rowid; doubles as the FTS docid
  recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,              -- position within the recording
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  text TEXT NOT NULL
);

-- Several notes per recording: another template, or a re-summary with a better model.
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY,
  recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
  template TEXT NOT NULL,
  summary TEXT NOT NULL,
  action_items TEXT NOT NULL,        -- JSON array
  model TEXT,
  timings TEXT,                      -- JSON
  created_at INTEGER NOT NULL
);

-- trigram, not unicode61: unicode61 splits Devanagari at every matra (अच्छी -> अच + छ + ई),
-- so अच्छा matches अच्छी. Trigram indexes 3-char windows and keeps the vowel signs.
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
  text, content='segments', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
  INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
  INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
  INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE INDEX IF NOT EXISTS idx_segments_recording ON segments(recording_id);
CREATE INDEX IF NOT EXISTS idx_notes_recording ON notes(recording_id);
CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)  # timeout: wait out the worker's write instead of raising
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite defaults this OFF, per connection
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # readers never block the worker's writes
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


# Columns added after a table first shipped; CREATE IF NOT EXISTS won't add them.
_ADDED_COLUMNS: list[tuple[str, str, str]] = []


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, decl in _ADDED_COLUMNS:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
