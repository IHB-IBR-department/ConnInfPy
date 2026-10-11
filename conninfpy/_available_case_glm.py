"""Compatibility import; implementation lives in :mod:`conninfpy.stats._missing_glm`."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("conninfpy.stats._missing_glm")
