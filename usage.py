"""Persistent per-day usage ledger: turns, paid realtime audio, local turns.

Separate from ``stt_backends.DailyAudioBudget`` (which only enforces the local
realtime daily cap and rolls over every day): this ledger keeps history for the
monthly report served at ``/<token>/usage``.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class UsageLedger:
    def __init__(self, path: str | Path | None = None):
        default = Path(__file__).resolve().parent / "usage.json"
        self.path = Path(path or os.environ.get("BUDDY_USAGE_LEDGER", default))
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def record(self, day: str, *, turns: int = 0, realtime_s: float = 0.0,
               local_turns: int = 0) -> None:
        with self._lock:
            days = self._load()
            entry = days.setdefault(
                day, {"turns": 0, "realtime_s": 0.0, "local_turns": 0})
            entry["turns"] += turns
            entry["realtime_s"] = round(entry["realtime_s"] + realtime_s, 3)
            entry["local_turns"] += local_turns
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(days, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def month_summary(self, month: str, eur_per_min: float | None = None) -> dict:
        """Aggregate the ledger's days for ``month`` (``YYYY-MM``).

        ``est_eur`` stays ``None`` unless a rate is configured: prices must
        come from the operator's own provider dashboard, never be invented.
        """
        days = {d: v for d, v in sorted(self._load().items()) if d.startswith(month)}
        totals = {
            "month": month,
            "turns": sum(v.get("turns", 0) for v in days.values()),
            "local_turns": sum(v.get("local_turns", 0) for v in days.values()),
            "realtime_s": round(sum(v.get("realtime_s", 0.0) for v in days.values()), 3),
            "days": days,
        }
        totals["realtime_min"] = round(totals["realtime_s"] / 60.0, 2)
        totals["est_eur"] = (
            round(totals["realtime_min"] * eur_per_min, 2)
            if eur_per_min is not None else None
        )
        return totals
