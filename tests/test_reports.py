"""Unit tests for artemis.reports persistence module."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from artemis import reports


class ReportsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="artemis-reports-")

    def tearDown(self) -> None:
        for f in Path(self.tmp).glob("*"):
            f.unlink()
        os.rmdir(self.tmp)

    def test_save_writes_json_file(self) -> None:
        reports.save(self.tmp, "abc123", {"hello": "world"})
        path = Path(self.tmp) / "abc123.json"
        self.assertTrue(path.exists())
        record = json.loads(path.read_text())
        self.assertEqual(record["payload"], {"hello": "world"})
        self.assertIn("saved_at", record)

    def test_save_rejects_unsafe_id(self) -> None:
        reports.save(self.tmp, "../escape", {"x": 1})
        reports.save(self.tmp, "a/b", {"x": 1})
        self.assertEqual(list(Path(self.tmp).glob("*.json")), [])

    def test_cleanup_deletes_expired_files(self) -> None:
        reports.save(self.tmp, "old", {"x": 1})
        reports.save(self.tmp, "fresh", {"x": 2})

        old_path = Path(self.tmp) / "old.json"
        record = json.loads(old_path.read_text())
        record["saved_at"] = int(time.time()) - 48 * 3600
        old_path.write_text(json.dumps(record))

        deleted = reports.cleanup_expired(self.tmp, ttl_hours=24)
        self.assertEqual(deleted, 1)
        self.assertFalse(old_path.exists())
        self.assertTrue((Path(self.tmp) / "fresh.json").exists())

    def test_cleanup_handles_missing_dir(self) -> None:
        self.assertEqual(reports.cleanup_expired("/nonexistent/path", 24), 0)


if __name__ == "__main__":
    unittest.main()
