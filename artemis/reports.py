"""Disk persistence for completed /v1/responses payloads.

Every completed response is written to ``settings.reports_dir`` as
``{id}.json`` so admins can inspect past reports on disk for debugging.
Files are purged after ``settings.reports_ttl_hours`` by a periodic
cleanup task started in the FastAPI lifespan.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _job_path(reports_dir: str, report_id: str) -> Path | None:
    if "/" in report_id or "\\" in report_id or report_id in {"", ".", ".."}:
        return None
    return Path(reports_dir) / f"{report_id}.json"


def save(reports_dir: str, report_id: str, payload: dict[str, Any]) -> None:
    """Persist a completed response payload keyed by ``report_id``.

    Failures are logged and swallowed — losing a debug copy must never
    break the live API response.
    """
    path = _job_path(reports_dir, report_id)
    if path is None:
        logger.warning("Refusing to save report with unsafe id: %r", report_id)
        return
    try:
        record = {"saved_at": int(time.time()), "payload": payload}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to persist report %s: %s", report_id, exc)


def cleanup_expired(reports_dir: str, ttl_hours: int) -> int:
    """Delete report files older than ``ttl_hours``. Returns count deleted."""
    p = Path(reports_dir)
    if not p.exists():
        return 0

    cutoff = time.time() - (ttl_hours * 3600)
    count = 0
    for f in p.glob("*.json"):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
            saved_at = record.get("saved_at") or f.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            saved_at = f.stat().st_mtime if f.exists() else cutoff + 1
        if saved_at < cutoff:
            try:
                f.unlink()
                count += 1
            except OSError as exc:
                logger.warning("Failed to delete expired report %s: %s", f, exc)
    if count:
        logger.info("Purged %d expired report(s) older than %dh.", count, ttl_hours)
    return count
