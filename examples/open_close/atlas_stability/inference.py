"""Paired edge inference with validated reuse of saved maps."""
from pathlib import Path
import hashlib
import json

import numpy as np
import scipy
from conninfpy import analyze
from conninfpy.results import InferenceResult
from examples._shared.paths import ROOT, sha256

TFNBS = dict(method="tfnbs", e=0.4, h=3.0, n=10, acceleration=None, rng=42, use_mp=True)


def array_digest(array) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def cache_identity(branch, eo_z, ec_z, n_perm, settings) -> dict:
    return {
        "schema": 1, "cohort": branch.cohort, "branch": branch.name,
        "subjects": list(branch.subjects),
        "parcels": branch.parcels[["id", "name"]].astype(str).to_dict("list"),
        "eo_sha256": array_digest(eo_z), "ec_sha256": array_digest(ec_z),
        "n_permutations": n_perm, "test_type": "paired", "fisher_z": False,
        "settings": settings,
        "cache_source": sha256(Path(__file__)),
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "library_sources": {str(p.relative_to(ROOT)): sha256(p)
                            for p in sorted((ROOT / "conninfpy").rglob("*.py"))},
    }


def tfnbs(branch, eo_z, ec_z, n_perm, reuse, output: Path, settings=None):
    settings = TFNBS if settings is None else settings
    path = output / "tfnbs" / f"{branch.cohort}__{branch.name}.npz"
    identity = cache_identity(branch, eo_z, ec_z, n_perm, settings)
    keys = ("positive", "negative", "stat_positive", "stat_negative")
    if reuse and path.exists():
        with np.load(path, allow_pickle=False) as data:
            if "cache_identity" not in data or json.loads(str(data["cache_identity"].item())) != identity:
                raise ValueError(f"Unverified or stale cache: {path}. Recompute without --reuse or choose a new --output.")
            maps = {key: data[key] for key in keys}
        shape = (eo_z.shape[1], eo_z.shape[1])
        if any(a.shape != shape or not np.isfinite(a).all() for a in maps.values()):
            raise ValueError(f"Invalid cached map shape or values: {path}")
        if any(np.any((maps[k] < 0) | (maps[k] > 1)) for k in ("positive", "negative")):
            raise ValueError(f"Invalid cached p-values: {path}")
        return InferenceResult(maps["positive"], maps["negative"], method="tfnbs",
                               stat_positive=maps["stat_positive"], stat_negative=maps["stat_negative"])
    inference = analyze(group1=eo_z, group2=ec_z, test_type="paired", fisher_z=False,
                        n_permutations=n_perm, **settings).inference
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: getattr(inference, key) for key in keys},
                        cache_identity=json.dumps(identity, sort_keys=True))
    return inference
