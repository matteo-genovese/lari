"""Satellite identity validation and device context for bridge diagnostics."""
from contextvars import ContextVar
from functools import wraps
import inspect
import logging
import re


_DEVICE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
_log_device = ContextVar("lari_log_device", default=None)


def validate_device(device: str) -> str:
    if not isinstance(device, str) or _DEVICE_NAME.fullmatch(device) is None:
        raise ValueError("invalid device")
    return device


class _DeviceFilter(logging.Filter):
    def filter(self, record):
        device = getattr(record, "device", None) or _log_device.get()
        if device is not None:
            record.msg = "[device=%s] %s" % (device, record.msg)
        return True


logging.getLogger("lari").addFilter(_DeviceFilter())


def device_logs(method):
    """Propagate identity to backend logs, child tasks and executor calls."""
    if inspect.iscoroutinefunction(method):
        @wraps(method)
        async def wrapped(self, *args, **kwargs):
            token = _log_device.set(self.device)
            try:
                return await method(self, *args, **kwargs)
            finally:
                _log_device.reset(token)
    else:
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            token = _log_device.set(self.device)
            try:
                return method(self, *args, **kwargs)
            finally:
                _log_device.reset(token)
    return wrapped
