"""Compatibility import; implementation lives in :mod:`conninfpy.stats._workers`."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("conninfpy.stats._workers")
