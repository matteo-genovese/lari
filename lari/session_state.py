"""Best-effort transcript ids, indexed by a digest of the device's session key."""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
import threading

from .config import Settings

log = logging.getLogger("lari")
_LOCK = threading.Lock()
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


def valid_session_id(value) -> bool:
    """Bounded, printable identifiers safe to send as an HTTP header."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


class SessionState:
    def __init__(self, settings: Settings):
        self.path = settings.session_state
        self.enabled = settings.session_resume
        self.key = hashlib.sha256(settings.session_key.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items()
                        if re.fullmatch(r"[0-9a-f]{64}", k) and valid_session_id(v)}
        except (OSError, ValueError, UnicodeError):
            pass
        return {}

    def get(self) -> str | None:
        # Atomic replacement makes an unlocked reader see either complete file.
        return self._load().get(self.key) if self.enabled else None

    def save(self, session_id: str) -> None:
        if self.enabled and valid_session_id(session_id):
            self._update(session_id)

    def forget(self, rejected_id: str) -> None:
        if self.enabled:
            self._update(None, rejected_id)

    def _update(self, value: str | None, rejected_id: str | None = None) -> None:
        temporary = None
        try:
            with _LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # A stable sidecar locks read/modify/replace across processes too.
                with os.fdopen(os.open(str(self.path) + ".lock",
                                       os.O_CREAT | os.O_RDWR, 0o600), "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    data = self._load()
                    current = data.get(self.key)
                    if current == value or (rejected_id is not None and current != rejected_id):
                        return
                    if value is None:
                        data.pop(self.key, None)
                    else:
                        data[self.key] = value
                    fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp",
                                                      dir=self.path.parent)
                    with os.fdopen(fd, "w", encoding="utf-8") as output:
                        os.fchmod(output.fileno(), 0o600)
                        json.dump(data, output, sort_keys=True)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, self.path)
                    temporary = None
        except (OSError, ValueError, UnicodeError):
            # Never expose ids, keys, credentials or filesystem exception text.
            log.warning("stato sessioni Hermes non aggiornabile")
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
