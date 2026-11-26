"""Bundled MIT atlas snapshots, on-demand third-party atlas cache, and external originals.

Materialized package paths remain alive for lazy image reads until process exit.
Third-party Brainnetome and SUIT atlases are not bundled in the MIT package; they
are resolved from ``CONNINFPY_ATLAS_DIR`` (default ``~/.cache/conninfpy/atlases``)
and verified against SHA-256 manifests. Optional originals are never located
relative to the caller's working directory.
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


def atlas_cache_dir(atlas_dir: Optional[Union[str, Path]] = None) -> Path:
    """Directory for on-demand third-party Brainnetome/SUIT atlases; does not create it.

    Precedence: explicit ``atlas_dir``, ``CONNINFPY_ATLAS_DIR``, then
    ``~/.cache/conninfpy/atlases``.
    """
    if atlas_dir is not None:
        path = Path(atlas_dir)
    elif os.environ.get("CONNINFPY_ATLAS_DIR"):
        path = Path(os.environ["CONNINFPY_ATLAS_DIR"])
    else:
        path = Path.home() / ".cache" / "conninfpy" / "atlases"
    return path.expanduser().resolve()


def _atlas_manifest_entries() -> Dict[str, Dict[str, str]]:
    """Expected SHA-256 and provenance metadata for all on-demand atlas files."""
    entries: Dict[str, Dict[str, str]] = {}

    bna_manifest = json.loads(bundled_path("brainnetome_external.json").read_text(encoding="utf-8"))
    for fname, meta in bna_manifest.get("atlas_files", {}).items():
        entries[fname] = {
            "sha256": meta["sha256"],
            "suite": "Brainnetome BNA-246",
            "source_url": meta.get("url") or bna_manifest.get("source_url", "https://atlas.brainnetome.org/download.html"),
            "license": meta.get("license", "Brainnetome non-commercial research use, attribution, and share-alike (NOTICE_BRAINNETOME.txt)"),
        }

    suit_manifest = json.loads(bundled_path("cerebellar_atlases_provenance.json").read_text(encoding="utf-8"))
    suit_url = suit_manifest.get("atlas_set", {}).get("repository", "https://github.com/DiedrichsenLab/cerebellar_atlases")
    for fname, meta in suit_manifest.get("atlas_set", {}).get("files", {}).items():
        entries[fname] = {
            "sha256": meta["packaged_sha256"],
            "suite": "DiedrichsenLab SUIT v1.2",
            "source_url": meta.get("upstream_url", suit_url),
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        }

    return entries


def _verify_atlas_file(path: Path, expected_sha256: str, suite: str) -> Path:
    path = path.expanduser().resolve()
    stat = path.stat()
    digest = _original_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if digest != expected_sha256:
        raise ValueError(
            f"{path}: SHA256 {digest} does not match the expected {suite} checksum "
            f"({expected_sha256}). Re-run `python scripts/download_atlases.py`."
        )
    return path


def atlas_resource_path(
    filename: str,
    *,
    atlas_dir: Optional[Union[str, Path]] = None,
    verify: bool = True,
) -> Path:
    """Locate and SHA-256-verify an on-demand Brainnetome or SUIT atlas file.

    Raises ``FileNotFoundError`` with download instructions and upstream terms
    when the file is absent from ``atlas_cache_dir(atlas_dir)``; never downloads
    implicitly.
    """
    entries = _atlas_manifest_entries()
    if filename not in entries:
        raise ValueError(f"Unknown on-demand atlas resource: {filename!r}")
    meta = entries[filename]
    cache_root = atlas_cache_dir(atlas_dir)
    path = cache_root / filename
    if path.is_file():
        return _verify_atlas_file(path, meta["sha256"], meta["suite"]) if verify else path

    # Support explicit test mocks/overrides of conninfpy.atlas_data when no cache dir is forced.
    if atlas_dir is None:
        try:
            bundled = bundled_path(filename)
            if bundled.is_file():
                return _verify_atlas_file(bundled, meta["sha256"], meta["suite"]) if verify else bundled
        except (FileNotFoundError, KeyError, OSError):
            pass

    raise FileNotFoundError(
        f"Missing {meta['suite']} atlas file {filename!r} (looked in {cache_root}). "
        f"Third-party atlas files are not bundled in the MIT package. "
        f"Run `python scripts/download_atlases.py` to populate the local cache, or set "
        f"CONNINFPY_ATLAS_DIR to a directory containing the verified files. "
        f"Upstream source: {meta['source_url']} | Terms: {meta['license']}."
    )


def atlas_resource_available(
    filename: str,
    *,
    atlas_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Return True if an on-demand atlas file exists in the local cache and passes SHA-256."""
    try:
        atlas_resource_path(filename, atlas_dir=atlas_dir, verify=True)
        return True
    except (FileNotFoundError, ValueError):
        return False


def read_bna_hierarchy(*, index: str = "roi_id"):
    """Read a fresh hierarchy table in the requested key order."""
    import pandas as pd

    return pd.read_csv(atlas_resource_path("bna246_hierarchy.csv")).set_index(index)


_BUILTIN_ALIASES = {
    "bna_246_2mm": "BN_Atlas_246_2mm.nii.gz",
    "bna_246_1mm": "BN_Atlas_246_1mm.nii.gz",
    "bna246.csv": "bna246.csv",
    "schaefer100": "schaefer100_yeo7.csv",
    "schaefer200": "schaefer200_yeo7.csv",
    "schaefer400": "schaefer400_yeo7.csv",
}


def builtin_atlas_path(token: str) -> Optional[Path]:
    """Resolve a manifest's builtin token through bundled MIT assets or the atlas cache."""
    filename = _BUILTIN_ALIASES.get(token, token)
    if filename in _atlas_manifest_entries():
        try:
            return atlas_resource_path(filename)
        except FileNotFoundError:
            return None
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
    """Get the file system path of the cached Brainnetome 246 NIfTI atlas image."""
    if resolution_mm not in (1, 2):
        raise ValueError(f"resolution_mm must be 1 or 2; got {resolution_mm!r}")
    filename = f"BN_Atlas_246_{resolution_mm}mm.nii.gz"
    return atlas_resource_path(filename)


def bna_profile_paths() -> Tuple[Path, Path]:
    """Paths of the cached BrainMap profiles of BNA-246 (BDf_FDR05.json, PCf_FDR05.json)."""
    return atlas_resource_path("BDf_FDR05.json"), atlas_resource_path("PCf_FDR05.json")


def fetch_bna_behavioral_decoding(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    as_dataframe: bool = False,
) -> Union[Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]], Any]:
    """Brainnetome Atlas (BNA-246) BrainMap behavioral decoding data.

    The forward-inference Behavioral Domain (BD) and Paradigm Class (PC)
    association scores (thresholded at FDR < 0.05) published with the
    Brainnetome Atlas (`atlas.brainnetome.org`) are read from the local
    on-demand atlas cache (``CONNINFPY_ATLAS_DIR`` or
    ``~/.cache/conninfpy/atlases``, under the Brainnetome terms in
    ``NOTICE_BRAINNETOME.txt``) without network access by default.
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
        on-demand atlas cache.
    force_download : bool, default=False
        Download a fresh copy even when cached files exist (into
        ``cache_dir``, default ``atlas_cache_dir()``).
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

    cache_path = atlas_cache_dir() if cache_dir is None else Path(cache_dir)

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
