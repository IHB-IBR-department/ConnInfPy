"""Bundled atlas snapshots, configured external originals and derived caches.

Materialized package paths remain alive for lazy image reads until process exit.
Optional originals are never located relative to the caller's working directory.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import urllib.error
import urllib.request
import warnings
from contextlib import ExitStack
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

_RESOURCE_CONTEXTS = ExitStack()
atexit.register(_RESOURCE_CONTEXTS.close)


@lru_cache(maxsize=None)
def bundled_path(filename: str) -> Path:
    """Resolve an immutable package asset with a process-long lifetime."""
    return Path(
        _RESOURCE_CONTEXTS.enter_context(
            resources.as_file(resources.files("conninfpy.atlas_data") / filename)
        )
    )


def read_bna_hierarchy(*, index: str = "roi_id"):
    """Read a fresh hierarchy table in the requested key order."""
    import pandas as pd

    return pd.read_csv(bundled_path("bna246_hierarchy.csv")).set_index(index)


_BUILTIN_ALIASES = {
    "bna_246_2mm": "BN_Atlas_246_2mm.nii.gz",
    "bna_246_1mm": "BN_Atlas_246_1mm.nii.gz",
    "bna246.csv": "bna246.csv",
    "schaefer100": "schaefer100_yeo7.csv",
    "schaefer200": "schaefer200_yeo7.csv",
    "schaefer400": "schaefer400_yeo7.csv",
}


def builtin_atlas_path(token: str) -> Optional[Path]:
    """Resolve a manifest's builtin token through the package resource locator."""
    filename = _BUILTIN_ALIASES.get(token, token)
    try:
        candidate = bundled_path(filename)
    except FileNotFoundError:
        return None
    return candidate if candidate.is_file() else None


_BNA_BD_URL = "https://atlas.brainnetome.org/js/bnatlasviewer/BDf_FDR05.json"
_BNA_PC_URL = "https://atlas.brainnetome.org/js/bnatlasviewer/PCf_FDR05.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_cache() -> Path:
    base = os.environ.get("CONNINFPY_CACHE", str(Path.home() / ".cache" / "conninfpy"))
    return Path(base) / "bna246"


def brainnetome_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Directory for optional Brainnetome originals; does not create it.

    Precedence: explicit ``data_dir``, legacy ``CONNINFPY_BNA_DIR``, then
    ``CONNINFPY_DATA/brainnetome`` (default ``~/conninfpy_data/brainnetome``).
    An explicit directory is authoritative, even if it does not exist.
    Derived caches use ``CONNINFPY_CACHE`` separately.
    """
    if data_dir is not None:
        path = Path(data_dir)
    elif os.environ.get("CONNINFPY_BNA_DIR"):
        path = Path(os.environ["CONNINFPY_BNA_DIR"])
    else:
        base = Path(os.environ.get("CONNINFPY_DATA", str(Path.home() / "conninfpy_data")))
        path = base / "brainnetome"
    return path.expanduser().resolve()


def _find_data_dir(data_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """Compatibility helper for optional references without local originals."""
    path = brainnetome_data_dir(data_dir)
    return path if path.is_dir() else None


@lru_cache(maxsize=32)
def _original_sha256(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """Reuse integrity checks while the original's filesystem identity is stable."""
    return _sha256(Path(path))


def brainnetome_resource_path(
    filename: str,
    *,
    data_dir: Optional[Union[str, Path]] = None,
    required: bool = False,
) -> Optional[Path]:
    """Locate and verify a versioned optional original from the source manifest.

    Missing optional files return ``None``. ``required=True`` gives manual
    acquisition instructions. A different/corrupt source raises ``ValueError``;
    no implicit download or fallback to another directory occurs.
    """
    manifest = json.loads(bundled_path("brainnetome_external.json").read_text())
    if filename not in manifest["resources"]:
        raise ValueError(f"Unknown Brainnetome original: {filename!r}")
    path = brainnetome_data_dir(data_dir) / filename
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Missing Brainnetome original: {path}. Obtain {Path(filename).name} "
                f"from {manifest['source_url']} under the provider's terms and place "
                "it here, or set data_dir, CONNINFPY_BNA_DIR or CONNINFPY_DATA."
            )
        return None
    return verify_brainnetome_resource(path, filename)


def verify_brainnetome_resource(path: Union[str, Path], filename: str) -> Path:
    """Verify an explicitly located original against its manifest resource ID.

    This also supports rebuilding from source tables in an arbitrary directory.
    ``filename`` is the manifest key, such as ``sources/BNA_subregions.xlsx``.
    """
    manifest = json.loads(bundled_path("brainnetome_external.json").read_text())
    if filename not in manifest["resources"]:
        raise ValueError(f"Unknown Brainnetome original: {filename!r}")
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    digest = _original_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    expected = manifest["resources"][filename]["sha256"]
    if digest != expected:
        raise ValueError(
            f"{path}: SHA256 {digest} does not match the versioned Brainnetome "
            f"original ({expected}). Check the source/version before using it."
        )
    return path


def get_bna_246_nifti_path(resolution_mm: int = 2) -> Path:
    """Get the file system path of the bundled Brainnetome 246 NIfTI atlas image."""
    filename = f"BN_Atlas_246_{resolution_mm}mm.nii.gz"
    return bundled_path(filename)


def bna_profile_paths() -> Tuple[Path, Path]:
    """Paths of the bundled BrainMap profiles of BNA-246 (BDf_FDR05.json, PCf_FDR05.json)."""
    return bundled_path("BDf_FDR05.json"), bundled_path("PCf_FDR05.json")


def fetch_bna_behavioral_decoding(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    as_dataframe: bool = False,
) -> Union[Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]], Any]:
    """Brainnetome Atlas (BNA-246) BrainMap behavioral decoding data.

    The forward-inference Behavioral Domain (BD) and Paradigm Class (PC)
    association scores (thresholded at FDR < 0.05) published with the
    Brainnetome Atlas (`atlas.brainnetome.org`) are bundled with the package
    (``conninfpy/atlas_data``, under the Brainnetome terms in
    ``NOTICE_BRAINNETOME.txt``) and read without network access by default.
    With ``cache_dir`` or ``force_download=True`` they are downloaded from the
    official repository instead and cached.

    Data attribution & licensing
    ----------------------------
    The Behavioral Domain and Paradigm Class taxonomy and coordinate data are
    derived from the BrainMap database (Laird et al., 2005, 2011)
    and computed for the Brainnetome Atlas by Fan et al. (2016). When publishing
    analyses that use these decodings, please cite:

    1. Fan L, Li H, Zhuo J, et al. (2016). The Human Brainnetome Atlas: A New Brain
       Atlas Based on Connectional Architecture. Cerebral Cortex, 26(8):3508–3526.
    2. Laird AR, Lancaster JL, Fox PT (2005). BrainMap: The social evolution of a human
       brain mapping database. Neuroinformatics, 3(1):65–78.
    3. Laird AR, Eickhoff SB, Fox PM, et al. (2011). The BrainMap strategy for
       standardization, sharing, and meta-analysis of neuroimaging data. BMC Research
       Notes, 4:349.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Download into (and read from) this directory instead of using the
        bundled files.
    force_download : bool, default=False
        Download a fresh copy even when bundled or cached files exist (into
        ``cache_dir``, default ``$CONNINFPY_CACHE/bna246`` or
        ``~/.cache/conninfpy/bna246``).
    as_dataframe : bool, default=False
        If True, returns a consolidated ``pandas.DataFrame`` merged with
        BNA-246 parcel names and ROI IDs.

    Returns
    -------
    tuple of (dict, dict) or pandas.DataFrame
        If ``as_dataframe=False``, returns ``(behavioral_domains, paradigm_classes)``
        where each is a dict mapping string ROI ID (``"1"`` to ``"246"``) to
        ``{term: score}``.
        If ``as_dataframe=True``, returns a DataFrame indexed by ``roi_id`` (1..246).
    """
    if cache_dir is None and not force_download:
        bd_file, pc_file = bna_profile_paths()
        with open(bd_file, "r", encoding="utf-8") as f:
            bd_data = json.load(f)
        with open(pc_file, "r", encoding="utf-8") as f:
            pc_data = json.load(f)
        return _bna_decoding_output(bd_data, pc_data, as_dataframe)

    if cache_dir is None:
        base_cache = os.environ.get(
            "CONNINFPY_CACHE",
            str(Path.home() / ".cache" / "conninfpy"),
        )
        cache_path = Path(base_cache) / "bna246"
    else:
        cache_path = Path(cache_dir)

    cache_path.mkdir(parents=True, exist_ok=True)
    bd_file = cache_path / "BDf_FDR05.json"
    pc_file = cache_path / "PCf_FDR05.json"

    def _download_if_needed(url: str, target: Path) -> None:
        if force_download or not target.exists() or target.stat().st_size == 0:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "conninfpy/2.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read()
                tmp_target = target.with_suffix(".tmp")
                with open(tmp_target, "wb") as f:
                    f.write(content)
                tmp_target.replace(target)
            except Exception as exc:
                if target.exists() and target.stat().st_size > 0:
                    warnings.warn(
                        f"Failed to refresh {target.name} from {url}: {exc}. "
                        "Using existing cached copy.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise IOError(
                        f"Failed to download Brainnetome decoding data from {url}: {exc}"
                    ) from exc

    _download_if_needed(_BNA_BD_URL, bd_file)
    _download_if_needed(_BNA_PC_URL, pc_file)

    with open(bd_file, "r", encoding="utf-8") as f:
        bd_data = json.load(f)
    with open(pc_file, "r", encoding="utf-8") as f:
        pc_data = json.load(f)
    return _bna_decoding_output(bd_data, pc_data, as_dataframe)


def _bna_decoding_output(bd_data, pc_data, as_dataframe):
    if not as_dataframe:
        return bd_data, pc_data

    import pandas as pd

    from conninfpy.atlas.metadata import AtlasInfo

    atlas = AtlasInfo.bna_246()
    rows = []
    for roi_id in range(1, len(atlas) + 1):
        s_id = str(roi_id)
        name = atlas.labels[roi_id - 1]
        network = atlas.networks[roi_id - 1]
        roi_bd = bd_data.get(s_id, {})
        roi_pc = pc_data.get(s_id, {})

        top_bd = max(roi_bd.items(), key=lambda x: x[1])[0] if roi_bd else None
        top_pc = max(roi_pc.items(), key=lambda x: x[1])[0] if roi_pc else None

        rows.append(
            {
                "roi_id": roi_id,
                "name": name,
                "network": network,
                "behavioral_domains": roi_bd,
                "paradigm_classes": roi_pc,
                "top_behavioral_domain": top_bd,
                "top_paradigm_class": top_pc,
            }
        )
    return pd.DataFrame(rows).set_index("roi_id")
