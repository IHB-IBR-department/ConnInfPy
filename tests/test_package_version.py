"""Keep distribution metadata, runtime version and citation synchronized."""
import importlib
import tomllib
from pathlib import Path

import conninfpy
import yaml


def test_version_single_source():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    assert "version" in config["project"]["dynamic"]
    attr = config["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    module, name = attr.rsplit(".", 1)
    source_version = getattr(importlib.import_module(module), name)
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    assert conninfpy.__version__ == source_version == str(citation["version"])
