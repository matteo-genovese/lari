"""Monthly usage report: persistent ledger and the /usage route."""
from lari.config import get_settings
import asyncio
import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from lari import server
from lari import usage


class UsageLedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "usage.json"

    def test_records_persist_across_instances(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        ledger.record("2026-09-29", turns=2, realtime_s=30.0, local_turns=1)
        fresh = usage.UsageLedger(self.path, settings=get_settings())
        summary = fresh.month_summary("2026-09")
        self.assertEqual(summary["turns"], 2)
        self.assertEqual(summary["local_turns"], 1)
        self.assertEqual(summary["realtime_s"], 30.0)
        self.assertEqual(summary["realtime_min"], 0.5)

    def test_month_summary_aggregates_only_that_month(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        ledger.record("2026-09-01", turns=1, realtime_s=60.0)
        ledger.record("2026-08-31", turns=5, realtime_s=600.0)
        summary = ledger.month_summary("2026-09")
        self.assertEqual(summary["turns"], 1)
        self.assertEqual(list(summary["days"]), ["2026-09-01"])
        self.assertIsNone(summary["est_eur"])

    def test_estimate_uses_the_configured_rate_only(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        ledger.record("2026-09-29", realtime_s=120.0)
        summary = ledger.month_summary("2026-09", eur_per_min=0.5)
        self.assertEqual(summary["est_eur"], 1.0)

    def test_truncated_history_is_preserved_when_recording(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        ledger.record("2026-09-01", turns=2, realtime_s=60.0)
        ledger.record("2026-09-02", turns=3, local_turns=1)
        content = self.path.read_bytes()
        truncated = content[:len(content) // 2]
        self.path.write_bytes(truncated)

        with self.assertLogs("lari.usage", level="WARNING"):
            with self.assertRaises(usage.UsageLedgerCorrupt):
                ledger.record("2026-09-03", turns=1)

        copies = list(self.path.parent.glob("usage.json.corrupt-*"))
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].read_bytes(), truncated)
        self.assertEqual(self.path.read_bytes(), truncated)
        self.assertFalse(self.path.with_name("usage.json.tmp").exists())

    def test_missing_ledger_loads_empty_and_records_normally(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        self.assertEqual(ledger._load(), {})
        ledger.record("2026-09-01", turns=1)
        self.assertEqual(ledger._load()["2026-09-01"]["turns"], 1)
        self.assertEqual(list(self.path.parent.glob("usage.json.corrupt-*")), [])

    def test_record_raises_for_corrupt_ledger(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        self.path.write_bytes(b'{"2026-09-01":')
        with self.assertLogs("lari.usage", level="WARNING"):
            with self.assertRaises(usage.UsageLedgerCorrupt) as caught:
                ledger.record("2026-09-02", turns=1)
        self.assertTrue(caught.exception.corrupt_copy.is_file())

    def test_corrupt_month_summary_reports_unreadable_data(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        content = b'{"2026-09-01":'
        self.path.write_bytes(content)
        with self.assertLogs("lari.usage", level="WARNING"):
            summary = ledger.month_summary("2026-09", eur_per_min=0.5)
        self.assertEqual(summary, {
            "month": "2026-09", "turns": 0, "local_turns": 0,
            "realtime_s": 0.0, "realtime_min": 0.0, "est_eur": 0.0,
            "days": {}, "corrupt": True, "corrupt_copy": summary["corrupt_copy"],
        })
        self.assertEqual(Path(summary["corrupt_copy"]).read_bytes(), content)
        self.assertEqual(self.path.read_bytes(), content)

    def test_identical_corruption_creates_only_one_copy(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        self.path.write_bytes(b'{"2026-09-01":')
        with self.assertLogs("lari.usage", level="WARNING"):
            first = ledger.month_summary("2026-09")
            copy = Path(first["corrupt_copy"])
            original_stat = copy.stat()
            second = ledger.month_summary("2026-09")
        self.assertEqual(first["corrupt_copy"], second["corrupt_copy"])
        self.assertEqual(list(self.path.parent.glob("usage.json.corrupt-*")), [copy])
        self.assertEqual(copy.stat().st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual(copy.stat().st_ino, original_stat.st_ino)

    def test_non_dict_and_invalid_utf8_are_preserved(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        for content in (b'[]', b'null', b'"text"', b'\xff'):
            with self.subTest(content=content):
                self.path.write_bytes(content)
                with self.assertLogs("lari.usage", level="WARNING"):
                    with self.assertRaises(usage.UsageLedgerCorrupt) as caught:
                        ledger._load()
                self.assertEqual(caught.exception.corrupt_copy.read_bytes(), content)
                self.assertEqual(self.path.read_bytes(), content)

    def test_read_error_preserves_bytes_when_retry_succeeds(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        self.path.write_bytes(b'{}')
        with patch.object(Path, "read_bytes", side_effect=[OSError("read failed"), b'{}']):
            with self.assertLogs("lari.usage", level="WARNING"):
                with self.assertRaises(usage.UsageLedgerCorrupt) as caught:
                    ledger.record("2026-09-01", turns=1)
        self.assertEqual(caught.exception.corrupt_copy.read_bytes(), b'{}')
        self.assertEqual(self.path.read_bytes(), b'{}')

    def test_unreadable_ledger_still_reports_corruption_when_copy_fails(self):
        ledger = usage.UsageLedger(self.path, settings=get_settings())
        self.path.write_bytes(b'{}')
        with patch.object(Path, "read_bytes", side_effect=PermissionError("read denied")):
            with self.assertLogs("lari.usage", level="WARNING"):
                with self.assertRaises(usage.UsageLedgerCorrupt):
                    ledger.record("2026-09-01", turns=1)
                summary = ledger.month_summary("2026-09")
        self.assertTrue(summary["corrupt"])
        self.assertIsNone(summary["corrupt_copy"])
        self.assertEqual(summary["turns"], 0)
        self.assertEqual(self.path.read_bytes(), b'{}')


class UsageRouteTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        ledger = usage.UsageLedger(Path(self.dir.name) / "usage.json", settings=get_settings())
        ledger.record(datetime.date.today().isoformat(),
                      turns=3, realtime_s=45.0, local_turns=2)
        for name, target in (("TOKEN", "unit-test-token"), ("USAGE_LEDGER", ledger)):
            mock = patch.object(server, name, target)
            mock.start()
            self.addCleanup(mock.stop)

    def get(self, path):
        async def request():
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(path)

        return asyncio.run(request())

    def test_usage_summary_is_served_with_the_installation_token(self):
        response = self.get("/unit-test-token/usage")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["turns"], 3)
        self.assertEqual(data["local_turns"], 2)
        self.assertEqual(data["realtime_s"], 45.0)
        self.assertIn("days", data)
        self.assertIn(datetime.date.today().isoformat(), data["days"])

    def test_usage_rejects_an_invalid_token(self):
        self.assertEqual(self.get("/wrong-token/usage").status_code, 403)


if __name__ == "__main__":
    unittest.main()
