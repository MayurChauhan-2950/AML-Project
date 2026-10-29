"""Local persistence: SQLite (standard library) plus image files on disk.

Layout (under ``data_dir``):
  analyses.db            one row per analysis
  uploads/<id>.<ext>     the uploaded MRI (validated PNG/JPEG)
  heatmaps/<id>.png      the Grad-CAM overlay (optional)

A new connection is opened per operation, which is safe for FastAPI's
threadpool. Search strings are bound as parameters (no SQL injection).
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import config

_ID_RE = re.compile(r"^[0-9a-f]{8}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    model_version TEXT NOT NULL,
    image_file TEXT NOT NULL,
    heatmap_file TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
"""


def is_valid_id(analysis_id: str) -> bool:
    return bool(_ID_RE.match(analysis_id))


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Storage:
    def __init__(self, data_dir: Path | None = None) -> None:
        base = Path(data_dir) if data_dir else config.DATA_DIR
        self.data_dir = base
        self.db_path = base / "analyses.db"
        self.upload_dir = base / "uploads"
        self.heatmap_dir = base / "heatmaps"
        self.init()

    # -- setup ----------------------------------------------------------
    def init(self) -> None:
        """Create folders and tables (idempotent)."""
        for folder in (self.data_dir, self.upload_dir, self.heatmap_dir):
            folder.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writes ---------------------------------------------------------
    def create(
        self,
        *,
        filename: str,
        predicted_class: str,
        confidence: float,
        probabilities: dict[str, float],
        model_version: str,
        image_bytes: bytes,
        image_ext: str,
        heatmap_png: bytes | None,
    ) -> dict[str, Any]:
        """Persist files and the DB row; returns the stored record."""
        analysis_id = self._new_id()
        image_file = f"{analysis_id}{image_ext}"
        heatmap_file = f"{analysis_id}.png" if heatmap_png else None
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (self.upload_dir / image_file).write_bytes(image_bytes)
        if heatmap_png and heatmap_file:
            (self.heatmap_dir / heatmap_file).write_bytes(heatmap_png)
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO analyses (id, created_at, filename, predicted_class, confidence,"
                    " probabilities, model_version, image_file, heatmap_file)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (analysis_id, created_at, filename, predicted_class, confidence,
                     json.dumps(probabilities), model_version, image_file, heatmap_file),
                )
        except Exception:
            self._remove_files(image_file, heatmap_file)  # do not leave orphans
            raise
        record = self.get(analysis_id)
        assert record is not None
        return record

    def delete(self, analysis_id: str) -> bool:
        """Delete the row and its files. Returns False if the id does not exist."""
        record = self.get(analysis_id)
        if record is None:
            return False
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        self._remove_files(record["image_file"], record["heatmap_file"])
        return True

    def _remove_files(self, image_file: str | None, heatmap_file: str | None) -> None:
        for folder, name in ((self.upload_dir, image_file), (self.heatmap_dir, heatmap_file)):
            if name:
                (folder / name).unlink(missing_ok=True)

    def _new_id(self) -> str:
        while True:
            candidate = uuid.uuid4().hex[:8]
            if self.get(candidate) is None:
                return candidate

    # -- reads ----------------------------------------------------------
    def get(self, analysis_id: str) -> dict[str, Any] | None:
        if not is_valid_id(analysis_id):
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_analyses(
        self, *, search: str | None, label: str | None, page: int, page_size: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Newest first. ``search`` matches id, filename or class name (case-insensitive)."""
        where: list[str] = []
        params: list[Any] = []
        if label:
            where.append("predicted_class = ?")
            params.append(label)
        if search and search.strip():
            like = f"%{_escape_like(search.strip())}%"
            where.append(
                "(id LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\'"
                " OR predicted_class LIKE ? ESCAPE '\\')"
            )
            params.extend([like, like, like])
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with closing(self._connect()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM analyses{clause}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM analyses{clause} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
        return [self._row_to_dict(r) for r in rows], total

    def stats(self) -> dict[str, Any]:
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with closing(self._connect()) as conn:
            total, avg = conn.execute("SELECT COUNT(*), AVG(confidence) FROM analyses").fetchone()
            recent = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE created_at >= ?", (week_ago,)
            ).fetchone()[0]
            counts = dict(conn.execute(
                "SELECT predicted_class, COUNT(*) FROM analyses GROUP BY predicted_class"
            ).fetchall())
        by_class = {c: int(counts.get(c, 0)) for c in config.CLASSES}
        return {
            "total_scans": int(total),
            "last_7_days": int(recent),
            "by_class": by_class,
            "average_confidence": round(float(avg), 4) if avg is not None else 0.0,
        }

    # -- file access ----------------------------------------------------
    def image_path(self, record: dict[str, Any]) -> Path:
        return self.upload_dir / record["image_file"]

    def heatmap_path(self, record: dict[str, Any]) -> Path | None:
        name = record.get("heatmap_file")
        if not name:
            return None
        path = self.heatmap_dir / name
        return path if path.is_file() else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["probabilities"] = json.loads(data["probabilities"])
        return data
