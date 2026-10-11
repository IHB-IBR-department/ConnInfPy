"""Run the existing edge-model feature-stability workflow (separate from atlas contrasts)."""
from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from pathlib import Path
import sys
from examples.open_close.feature_stability.experiment import main

if __name__ == "__main__":
    main()
