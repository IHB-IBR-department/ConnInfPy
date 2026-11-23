"""Enhancement backend detection, dispatch tracking, and once-per-run diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("conninfpy.enhance")

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Fallback decorator when numba is not installed."""

        def decorator(func):
            """No-op stand-in for numba's njit when numba is absent."""
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


_ENHANCE_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_conninfpy_enhance_call_ctx", default=None
)
_SUPPRESS_WARNINGS: ContextVar[bool] = ContextVar(
    "_conninfpy_enhance_suppress_warnings", default=False
)


@contextmanager
def _enhance_call_scope() -> Iterator[Dict[str, Any]]:
    """Scope diagnostic deduplication and executed-backend tracking to one user call."""
    existing = _ENHANCE_CTX.get()
    if existing is not None:
        yield existing
        return
    ctx: Dict[str, Any] = {"emitted": set(), "actual_backend": None}
    token = _ENHANCE_CTX.set(ctx)
    try:
        yield ctx
    finally:
        _ENHANCE_CTX.reset(token)


@contextmanager
def _suppress_fallback_warnings() -> Iterator[None]:
    """Suppress fallback warnings inside permutation workers."""
    token = _SUPPRESS_WARNINGS.set(True)
    try:
        yield
    finally:
        _SUPPRESS_WARNINGS.reset(token)


def _record_executed_backend(
    actual_backend: str,
    *,
    method_label: str,
    emit_fallback_warning: bool = True,
    fallback_reason: Optional[str] = None,
) -> None:
    """Record the backend executed by dispatch and log fallback once per user call."""
    ctx = _ENHANCE_CTX.get()
    if ctx is not None:
        ctx["actual_backend"] = actual_backend

    if not emit_fallback_warning or _SUPPRESS_WARNINGS.get() or actual_backend == "numba":
        return

    key = (method_label, actual_backend, fallback_reason)
    if ctx is not None:
        if key in ctx["emitted"]:
            return
        ctx["emitted"].add(key)

    if fallback_reason == "no_numba":
        logger.warning(
            "Numba is not installed; %s is running on the '%s' fallback backend.",
            method_label,
            actual_backend,
        )
    elif fallback_reason == "disabled":
        logger.warning(
            "Numba backend disabled; %s is running on the '%s' fallback backend.",
            method_label,
            actual_backend,
        )
