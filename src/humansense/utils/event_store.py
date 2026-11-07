"""Persists events to a local SQLite file (events.db in the project root)."""

import os
import sqlite3
import threading

DB_PATH = os.environ.get("HUMANSENSE_EVENTS_DB", os.path.join(os.getcwd(), "events.db"))
_lock = threading.Lock()


def _connect(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS events ("
                 "id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, "
                 "type TEXT NOT NULL, detail TEXT NOT NULL)")
    return conn


def save_event(event_type: str, detail: str, path: str = DB_PATH) -> None:
    """Best-effort: a storage problem must never break video processing."""
    try:
        with _lock, _connect(path) as conn:
            conn.execute("INSERT INTO events (type, detail) VALUES (?, ?)", (event_type, detail))
    except Exception:
        pass


def recent_events(limit: int = 100, path: str = DB_PATH):
    with _lock, _connect(path) as conn:
        return conn.execute("SELECT ts, type, detail FROM events ORDER BY id DESC LIMIT ?",
                            (limit,)).fetchall()
