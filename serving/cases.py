"""SQLite `cases` table: one row per scored case.

A graph-only case has NULL voice fields, a voice-only case has a NULL node_id, and the combined
fields are filled only when both inputs exist -- nothing is ever back-filled with a made-up value.
"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id           INTEGER,
    voice_clip        TEXT,
    graph_score       REAL,
    graph_verdict     TEXT,
    voice_score       REAL,
    voice_verdict     TEXT,
    voice_model       TEXT,
    combined_score    REAL,
    combined_verdict  TEXT,
    w_graph           REAL,
    w_voice           REAL,
    created_at        TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_only ON cases(node_id)
    WHERE voice_clip IS NULL AND node_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_only ON cases(voice_clip)
    WHERE node_id IS NULL AND voice_clip IS NOT NULL;
DROP VIEW IF EXISTS flagged_cases;
CREATE VIEW flagged_cases AS
SELECT *,
       COALESCE(combined_score, graph_score, voice_score) AS risk_score,
       COALESCE(combined_verdict, graph_verdict, voice_verdict) AS risk_verdict
FROM cases
WHERE COALESCE(combined_verdict, graph_verdict, voice_verdict) IN ('medium', 'high')
ORDER BY risk_score DESC;
"""


class CaseStore:
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._tx() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_graph(self, node_id, score, verdict):
        """Graph-only case: one row per node, refreshed in place if the node is scored again."""
        with self._tx() as c:
            c.execute(
                """INSERT INTO cases (node_id, graph_score, graph_verdict, created_at) VALUES (?,?,?,?)
                   ON CONFLICT(node_id) WHERE voice_clip IS NULL AND node_id IS NOT NULL
                   DO UPDATE SET graph_score=excluded.graph_score, graph_verdict=excluded.graph_verdict,
                                 created_at=excluded.created_at""",
                (node_id, score, verdict, self._now()))

    def record_voice(self, clip, score, verdict, model):
        """Voice-only case: one row per clip name."""
        with self._tx() as c:
            c.execute(
                """INSERT INTO cases (voice_clip, voice_score, voice_verdict, voice_model, created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(voice_clip) WHERE node_id IS NULL AND voice_clip IS NOT NULL
                   DO UPDATE SET voice_score=excluded.voice_score, voice_verdict=excluded.voice_verdict,
                                 voice_model=excluded.voice_model, created_at=excluded.created_at""",
                (clip, score, verdict, model, self._now()))

    def record_combined(self, node_id, clip, graph, voice, voice_model, fused):
        """Combined case (always a new row: the same node can be paired with different clips)."""
        with self._tx() as c:
            cur = c.execute(
                """INSERT INTO cases (node_id, voice_clip, graph_score, graph_verdict, voice_score,
                                      voice_verdict, voice_model, combined_score, combined_verdict,
                                      w_graph, w_voice, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (node_id, clip, graph["score"], graph["verdict"], voice["score"], voice["verdict"],
                 voice_model, fused["combined_score"], fused["combined_verdict"],
                 fused["weights"]["graph"], fused["weights"]["voice"], self._now()))
            return cur.lastrowid

    def flagged(self, limit=50):
        with self._tx() as c:
            return [dict(r) for r in c.execute("SELECT * FROM flagged_cases LIMIT ?", (limit,))]

    def all_cases(self, limit=200):
        with self._tx() as c:
            return [dict(r) for r in c.execute("SELECT * FROM cases ORDER BY case_id DESC LIMIT ?", (limit,))]

    def count(self):
        with self._tx() as c:
            return c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
