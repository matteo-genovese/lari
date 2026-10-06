"""Persistent per-day usage ledger: turns, paid realtime audio, local turns.

Separate from ``stt_backends.DailyAudioBudget`` (which only enforces the local
realtime daily cap and rolls over every day): this ledger keeps history for the
monthly report served at ``/<token>/usage``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from .config import Settings


log = logging.getLogger(__name__)


class UsageLedgerCorrupt(Exception):
    """The ledger is unreadable; corrupt_copy identifies any preserved bytes."""

    def __init__(self, path: Path, corrupt_copy: Path | None):
        super().__init__(f"Usage ledger non leggibile: {path}; copia: {corrupt_copy}")
        self.corrupt_copy = corrupt_copy


class UsageLedger:
    def __init__(self, path: str | Path | None = None, *, settings: Settings):
        self.path = Path(path or settings.usage_ledger)
        self._lock = threading.Lock()

    def _load(self) -> dict:
        content = None
        try:
            content = self.path.read_bytes()
        except FileNotFoundError:
            return {}
        except OSError as error:
            self._raise_corrupt(error)
        try:
            data = json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self._raise_corrupt(error, content)
        if not isinstance(data, dict):
            self._raise_corrupt(ValueError("il ledger deve essere un dizionario"), content)
        return data

    def _raise_corrupt(self, error: Exception, content: bytes | None = None) -> None:
        corrupt_copy = None
        try:
            # Retry a failed read to preserve bytes after a transient I/O error.
            if content is None:
                content = self.path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            copy_path = self.path.with_name(f"{self.path.name}.corrupt-{digest}")
            try:
                with copy_path.open("xb") as copy:
                    copy.write(content)
            except FileExistsError:
                pass
            corrupt_copy = copy_path
        except OSError:
            log.warning("Impossibile conservare una copia del ledger %s", self.path,
                        exc_info=True)
        log.warning("Usage ledger non leggibile: %s; copia: %s; errore: %s",
                    self.path, corrupt_copy, error)
        raise UsageLedgerCorrupt(self.path, corrupt_copy) from error

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
        corruption = {}
        try:
            loaded = self._load()
        except UsageLedgerCorrupt as error:
            loaded = {}
            corruption = {
                "corrupt": True,
                "corrupt_copy": str(error.corrupt_copy) if error.corrupt_copy else None,
            }
        days = {d: v for d, v in sorted(loaded.items()) if d.startswith(month)}
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
        totals.update(corruption)
        return totals
