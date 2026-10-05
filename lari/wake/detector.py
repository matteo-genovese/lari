"""Local wake detection and command resolution."""
from __future__ import annotations
import logging
import numpy as np
from ..config import Settings
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
import json
import importlib.util
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from . import config as wake_config
_vosk_models = {}
_vosk_lock = threading.Lock()
_hermes_import_lock = threading.RLock()


@contextmanager
def _hermes_import_path(root):
    """Scope Hermes' absolute sibling imports to optional engine construction."""
    with _hermes_import_lock:
        previous = sys.path[:]
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            sys.path[:] = previous

def _wake_engine_builder(hermes_root):
    """Lazy loader for the optional openwakeword engine from the Hermes source.

    The default wake provider (energy VAD + Vosk) never needs it; importing it
    lazily keeps the bridge importable without a Hermes checkout (CI, fresh
    installs).
    """
    path = Path(hermes_root) / "tools" / "wake_word.py"
    if not path.is_file():
        raise RuntimeError(
            f"optional wake engine: missing Hermes module {path}; "
            "set LARI_HERMES_ROOT to a Hermes checkout containing tools/wake_word.py"
        )

    def build(cfg):
        # A private name avoids reusing tools.wake_word from another checkout.
        name = f"_lari_hermes_wake_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"optional wake engine: cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        with _hermes_import_path(hermes_root):
            # dataclasses and other module introspection require this during exec.
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
                builder = getattr(module, "_build_engine", None)
                if not callable(builder):
                    raise RuntimeError(f"optional wake engine: missing callable _build_engine in {path}")
                return builder(cfg)
            except ImportError as exc:
                raise RuntimeError(
                    f"optional wake engine: missing dependency {exc.name or str(exc)!r} "
                    f"while loading {path}; check LARI_HERMES_ROOT and install "
                    "the dependencies for the configured Hermes wake provider"
                ) from exc
            finally:
                sys.modules.pop(name, None)

    return build


def wake_command(text: str, cfg: wake_config.WakeConfig) -> str | None:
    """Return the request after the wake; None when not addressed to us."""
    return cfg.command(text)


def resolve_command(text: str, conversation_until: float, now: float,
                    cfg: wake_config.WakeConfig) -> str | None:
    """Wake on the first turn, then free dialog only inside the follow-up window."""
    command = wake_command(text, cfg)
    if command is not None:
        return command
    if now < conversation_until and text.strip():
        return text.strip()
    return None


def get_vosk(model_dir):
    """Share heavy models by path across sessions; recognizers remain per call."""
    if model_dir is None:
        raise ValueError("inject the Vosk model path through Settings.wake_config")
    model_dir = str(model_dir)
    with _vosk_lock:
        if model_dir not in _vosk_models:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)
            _vosk_models[model_dir] = Model(str(model_dir))
        return _vosk_models[model_dir]


def vosk_wake(pcm: np.ndarray, cfg: wake_config.WakeConfig) -> bool:
    """Detect the wake in the first 2.5 s without forcing the rest through the grammar."""
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(get_vosk(cfg.model_dir), SAMPLE_RATE, json.dumps(list(cfg.grammar)))
    rec.AcceptWaveform(pcm[:int(2.5 * SAMPLE_RATE)].astype(np.int16, copy=False).tobytes())
    heard = json.loads(rec.FinalResult()).get("text", "")
    log.info("Vosk wake: %r", heard)
    return bool(cfg.loose_re.search(heard))


def resolve_vosk_command(text: str, wake: bool, followup: bool,
                         cfg: wake_config.WakeConfig) -> str | None:
    if not wake and not followup:
        return None
    if wake:
        command = wake_command(text, cfg)
        if command is not None:
            return command
        # Free-form ASR sometimes renders the wake as one spurious word; the
        # separate grammar already confirmed it, so strip up to two junk words.
        text = cfg.strip_junk(text)
    return text.strip()


def make_engine(settings: Settings):
    cfg = {
        "provider": settings.wake_provider,
        "phrase": settings.wake_config.phrase,
        "sensitivity": settings.wake_sensitivity,
        "confirmation_frames": settings.confirm_frames,
        "profile_routing": False,   # only the bridge's phrase: no Hermes profile routing
        "openwakeword": {"model": "hey_hermes"},
    }
    return _wake_engine_builder(settings.hermes_root)(cfg)


def __getattr__(name):
    if name == "transcribe_vosk":
        from ..stt.vosk import transcribe_vosk
        return transcribe_vosk
    raise AttributeError(name)
