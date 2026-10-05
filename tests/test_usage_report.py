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
