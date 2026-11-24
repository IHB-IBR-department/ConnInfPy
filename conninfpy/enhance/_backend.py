"""Enhancement backend detection, dispatch tracking, and once-per-run diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
import threading
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("conninfpy.enhance")

try:
    from numba import njit as _numba_njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    _numba_njit = None


# NETSTAT-27 R1-5: numba's ``@njit(cache=True)`` creates the configured cache
# directory at decoration time, i.e. while ``import conninfpy`` runs — even
# for pure diagnostics such as ``conninfpy info``. Kernels are therefore
# decorated without caching here, and caching is enabled lazily on the first
# real enhancement call (``_enhance_call_scope``). The user's cache settings
# and numerical behaviour are unchanged; only the moment of directory
# creation moves from package import to first actual kernel use.
_CACHED_KERNELS: list = []
_CACHING_LOCK = threading.Lock()
_ENABLED_KERNELS: set = set()


if HAS_NUMBA:

    def njit(*args, **kwargs):
        """``numba.njit`` wrapper deferring ``cache=True`` to first real use.

        The returned object is the ordinary numba dispatcher (same type,
        pickling and introspection as before); only the cache activation is
        postponed to :func:`_enable_kernel_caching`.
        """
        cache = bool(kwargs.pop("cache", False))

        def _register(disp):
            if cache:
                _CACHED_KERNELS.append(disp)
            return disp

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _register(_numba_njit(args[0]))

        def decorator(func):
            return _register(_numba_njit(**kwargs)(func))

        return decorator

else:

    def njit(*args, **kwargs):
        """Fallback decorator when numba is not installed."""

        def decorator(func):
            """No-op stand-in for numba's njit when numba is absent."""
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def _enable_kernel_caching() -> None:
    """Enable numba caching for registered kernels (idempotent, thread-safe).

    Called when a real enhancement dispatch starts. Failures are logged, not
    hidden, and never change numerical results (caching is an optimization).
    """
    if not HAS_NUMBA:
        return
    with _CACHING_LOCK:
        for disp in _CACHED_KERNELS:
            if id(disp) in _ENABLED_KERNELS:
                continue
            try:
                disp.enable_caching()
            except Exception as exc:
                logger.warning("numba kernel caching could not be enabled: %s", exc)
            _ENABLED_KERNELS.add(id(disp))


_ENHANCE_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_conninfpy_enhance_call_ctx", default=None
)
_SUPPRESS_WARNINGS: ContextVar[bool] = ContextVar(
    "_conninfpy_enhance_suppress_warnings", default=False
)


@contextmanager
def _enhance_call_scope() -> Iterator[Dict[str, Any]]:
    """Scope diagnostic deduplication and executed-backend tracking to one user call."""
    # First real enhancement use activates numba kernel caching (NETSTAT-27
    # R1-5); package import and diagnostics never create the cache directory.
    _enable_kernel_caching()
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
