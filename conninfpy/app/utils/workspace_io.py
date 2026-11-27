"""Safe, versioned JSON+NPZ workspace save and restore (no pickle).

All arrays live in ``workspace/arrays/arrays.npz`` (written and read strictly
with ``allow_pickle=False``); structured metadata, DataFrames, atlases,
results, annotations, and allowlisted UI state live in ``workspace/state.json``;
integrity checksums and schema metadata live in ``workspace/manifest.json``.

A workspace can be saved either as a standalone workspace archive
(``workspace/*`` + root ``manifest.json``) or embedded as the additive
``workspace/`` layer inside a full analysis archive
(:mod:`conninfpy.app.utils.archive_export`). Inspection and restoration never import
``pickle`` and never read ``bundle.pkl``.

``state.json`` carries the explicit contract sections ``source_refs``,
``results``, ``annotation`` (``RoiAnnotation`` + ``evidence_packet``),
``ui_state`` (allowlisted, secret-free) and optional ``data``. Schema version 2
is current; version 1 archives written before the LLM narrative report was
removed still load, but their narrative/LLM fields, keys, and widgets are
skipped and reported as legacy instead of being restored.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd


WORKSPACE_ARCHIVE_KIND = "conninfpy-workspace"
# Version 2 drops the retired LLM narrative surface (NETSTAT-39) from the
# contract. Version 1 archives stay readable: their narrative/LLM fields are
# skipped and reported, never restored.
WORKSPACE_SCHEMA_VERSION = 2
LEGACY_WORKSPACE_SCHEMA_VERSIONS = (1,)
SUPPORTED_WORKSPACE_SCHEMA_VERSIONS = (*LEGACY_WORKSPACE_SCHEMA_VERSIONS, WORKSPACE_SCHEMA_VERSION)
WORKSPACE_EXTENSION = ".conninfpy.zip"
WORKSPACE_PREFIX = "workspace/"
WORKSPACE_MANIFEST_PATH = "workspace/manifest.json"
WORKSPACE_STATE_PATH = "workspace/state.json"
WORKSPACE_ARRAYS_PATH = "workspace/arrays/arrays.npz"

# Top-level contract sections of workspace/state.json (schema version 2+).
_REQUIRED_STATE_SECTIONS = ("source_refs", "results", "annotation", "ui_state", "data")

_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_TYPE_TAG = "__conninf_type__"

# Sidebar and session keys explicitly allowlisted for UI restoration.
_SIDEBAR_WIDGET_KEYS = (
    "atlas_mode",
    "atlas_choice",
    "custom_atlas_source",
    "custom_atlas_path",
)

# Scopes whose persistent widgets belong to workspace UI state.
_ALLOWLISTED_WIDGET_SCOPES = frozenset({
    "atlas",
    "data",
    "design",
    "results",
    "annotation",
    "companion",
})

# Tabs that no longer exist are redirected to their closest surviving surface
# instead of reviving a removed view.
LEGACY_TAB_REDIRECTS = {
    "5. Narrative Report": "4. Brainnetome Annotation",
}

# Fields of the retired LLM narrative report (NETSTAT-39). They are never
# exported and never restored; values found in an old archive are skipped and
# reported as legacy, so no narrative state, token, or widget can be revived.
LEGACY_NARRATIVE_ANNOTATION_KEYS = (
    "narrative_results",
    "narrative_text",
    "llm_provider",
    "llm_model",
)
LEGACY_NARRATIVE_SESSION_KEYS = (*LEGACY_NARRATIVE_ANNOTATION_KEYS, "llm_api_key")
LEGACY_NARRATIVE_WIDGET_PREFIX = "narrative:"
LEGACY_LLM_WIDGET_KEYS = frozenset({"llm_api_key"})

# Keys or prefixes that must NEVER be exported/imported (secrets, uploaders, transient triggers).
# Retired narrative/LLM widget keys are covered by _is_excluded_widget_key().
_EXCLUDED_WIDGET_KEYS = frozenset({
    "custom_atlas_upload",
    "custom_atlas_nifti_upload",
    "custom_atlas_lut_upload",
    "custom_atlas_template_download",
    "workspace_restore_upload",
    "workspace_restore_file",
    "active_tab",
    "next_tab",
    "nav_docs_btn",
    "run_inference_btn",
    "cancel_inference_btn",
    "load_dataset_btn",
    "generate_dataset_btn",
    "archive_export_download",
    "workspace_save_download",
    "workspace_apply_restore_btn",
})

_EXCLUDED_WIDGET_PREFIXES = (
    "_",
    "FormSubmitter:",
    "nav_step_",
    "__persisted__",
    "__widget_",
)

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|credential|auth|bearer)",
    re.IGNORECASE,
)

# Keys managed directly by workspace restoration (cleared before applying restored state).
_MANAGED_SESSION_KEYS = (
    "connectivity_data",
    "connectivity_data_kind",
    "pheno_df",
    "dataset_info",
    "dataset_load_notes",
    "dataset_preprocessing",
    "dataset_fisher_history",
    "last_fisher_clip_note",
    "dataset_atlas",
    "sub_atlas",
    "loaded_reference_atlas",
    "roi_indices",
    "loaded_atlas_signature",
    "loaded_settings_hash",
    "current_settings_hash",
    "inference_result",
    "edges_df",
    "run_plan",
    "companion_inference_result",
    "companion_edges_df",
    "companion_method",
    "annotation",
    "evidence_packet",
    "_synthetic_effect_mask",
    "_synthetic_scenario_name",
    "_synthetic_confound_mask",
    "_synthetic_confound_name",
    "_empirical_corr",
    "_annotation_result_id",
    "_annotation_run_alpha",
    "_workspace_source_verification",
    "_workspace_legacy_unknowns",
    "_workspace_restored_meta",
    # Tested-family mask retained across a restore (registered in the widget
    # registry under the design scope, restored with its values when saved).
    "design:edge_mask_array",
    "design:edge_mask_source",
)


class WorkspaceArchiveError(ValueError):
    """Raised when a workspace archive is malformed, unsupported, or unsafe."""


@dataclass
class WorkspaceInspection:
    """Validated non-mutating inspection summary of a workspace archive."""

    manifest: Dict[str, Any]
    workspace_schema_version: int
    kind: str
    include_data: bool
    saved_at: Optional[str]
    active_tab: Optional[str]
    has_results: bool
    has_companion: bool
    has_annotation: bool
    has_evidence_packet: bool
    has_data: bool
    dataset_summary: Dict[str, Any]
    source_verification: Dict[str, Any]
    legacy_unknowns: List[str] = field(default_factory=list)
    legacy_fields_skipped: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _dumps_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _npy_bytes(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    if arr.dtype == object or arr.dtype.kind == "O":
        raise WorkspaceArchiveError("Object-dtype arrays cannot be stored with allow_pickle=False.")
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            member_name = f"{key}.npy" if not key.endswith(".npy") else key
            info = zipfile.ZipInfo(member_name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, _npy_bytes(arrays[key]))
    return buf.getvalue()


def _validate_zip_members(zf: zipfile.ZipFile) -> None:
    seen = set()
    for info in zf.infolist():
        name = info.filename
        if not name or "\\" in name or name.startswith("/"):
            raise WorkspaceArchiveError(f"Unsafe archive member path: {name!r}")
        parts = name.split("/")
        if any(part in ("", ".", "..") for part in parts if not name.endswith("/")):
            raise WorkspaceArchiveError(f"Unsafe archive member path: {name!r}")
        if ".." in parts:
            raise WorkspaceArchiveError(f"Path traversal not permitted in archive member: {name!r}")
        if name in seen:
            raise WorkspaceArchiveError(f"Duplicate archive member: {name!r}")
        seen.add(name)


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_RE.search(str(key)))


def _is_excluded_widget_key(key: str) -> bool:
    """True for keys that must never be exported or restored (secrets, retired UI)."""
    key_str = str(key)
    return (
        key_str in _EXCLUDED_WIDGET_KEYS
        or key_str in LEGACY_LLM_WIDGET_KEYS
        or key_str.startswith(LEGACY_NARRATIVE_WIDGET_PREFIX)
        or _is_secret_key(key_str)
        or any(key_str.startswith(prefix) for prefix in _EXCLUDED_WIDGET_PREFIXES)
    )


def _normalise_active_tab(active_tab: Optional[str]) -> Optional[str]:
    """Point a retired tab label at the surviving surface it maps to."""
    if active_tab is None:
        return None
    label = str(active_tab)
    return LEGACY_TAB_REDIRECTS.get(label, label)


def _unwrap_result(res: Any) -> Any:
    return getattr(res, "inference", res)


def _is_inference_like(result: Any) -> bool:
    """True only for a real directional/omnibus inference payload, never a stub."""
    inner = _unwrap_result(result)
    if not isinstance(inner, Mapping):
        return False
    return "positive" in inner or "omnibus" in inner


# ---------------------------------------------------------------------------
# Source file capture and verification
# ---------------------------------------------------------------------------

def _classify_path(raw: str, root: Path) -> Tuple[Path, str]:
    p = Path(raw)
    resolved = p if p.is_absolute() else (root / p)
    try:
        resolved_norm = resolved.resolve()
        root_norm = root.resolve()
    except OSError:
        resolved_norm = resolved
        root_norm = root
    if not p.is_absolute():
        location = "project"
    elif str(resolved_norm).startswith(str(root_norm)):
        location = "inside-project"
    else:
        location = "external"
    return resolved, location


def _collect_provenance_paths(
    dataset_info: Optional[Mapping[str, Any]],
    *,
    manifest_file: Optional[str] = None,
    resolved_paths: Optional[Mapping[str, Any]] = None,
    extra_paths: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    candidates: Dict[str, str] = {}
    info = dataset_info or {}
    source = str(info.get("source") or "")
    manifest_like = manifest_file or info.get("manifest_file") or (
        source if re.search(r"\.ya?ml$", source, re.IGNORECASE) else None
    )
    if manifest_like:
        candidates["manifest"] = str(manifest_like)

    if isinstance(resolved_paths, Mapping):
        for key, value in resolved_paths.items():
            if isinstance(value, (str, Path)) and str(value):
                candidates[f"manifest.{key}"] = str(value)

    provenance = info.get("loader_provenance") or {}
    if isinstance(provenance, Mapping):
        for key in (
            "data_path",
            "pheno_csv_path",
            "pheno_path",
            "atlas_metadata",
            "atlas",
            "open_path",
            "close_path",
            "subject_list_path",
            "prepared_npz",
            "set84_bnt_ids",
            "manifest_path",
        ):
            value = provenance.get(key)
            if isinstance(value, (str, Path)) and str(value):
                candidates.setdefault(key, str(value))
        sites = provenance.get("sites")
        if isinstance(sites, Mapping):
            for site_name, site_cfg in sites.items():
                if isinstance(site_cfg, Mapping):
                    for key, value in site_cfg.items():
                        if isinstance(value, (str, Path)) and str(value):
                            candidates.setdefault(f"sites.{site_name}.{key}", str(value))

    if isinstance(extra_paths, Mapping):
        for key, value in extra_paths.items():
            if isinstance(value, (str, Path)) and str(value):
                candidates.setdefault(str(key), str(value))

    return candidates


def capture_source_hashes(
    *,
    dataset_info: Optional[Mapping[str, Any]] = None,
    manifest_file: Optional[str] = None,
    resolved_paths: Optional[Mapping[str, Any]] = None,
    extra_paths: Optional[Mapping[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Hash actual readable source files at dataset load time."""
    from conninfpy.app.utils.helpers import is_public_mode
    if is_public_mode():
        return {}
    root = Path(project_root) if project_root is not None else Path.cwd()
    candidates = _collect_provenance_paths(
        dataset_info,
        manifest_file=manifest_file,
        resolved_paths=resolved_paths,
        extra_paths=extra_paths,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for name, raw in sorted(candidates.items()):
        resolved, location = _classify_path(raw, root)
        exists = resolved.exists()
        is_file = resolved.is_file()
        sha = _sha256_file(resolved) if is_file else None
        size = int(resolved.stat().st_size) if is_file else None
        out[name] = {
            "path": str(raw),
            "resolved": str(resolved),
            "exists": bool(exists),
            "is_file": bool(is_file),
            "location": location,
            "captured_sha256": sha,
            "captured_bytes": size,
        }
    return out


def _source_entries_for_save(
    dataset_info: Optional[Mapping[str, Any]],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Preserve captured source hashes from load time; never fabricate hashes for unhashed paths."""
    from conninfpy.app.utils.helpers import is_public_mode
    if is_public_mode():
        return {}
    info = dataset_info or {}
    root = Path(project_root) if project_root is not None else Path.cwd()
    recorded = info.get("source_files")
    entries: Dict[str, Dict[str, Any]] = {}
    if isinstance(recorded, Mapping):
        for name, item in recorded.items():
            if isinstance(item, Mapping):
                raw = str(item.get("path") or item.get("resolved") or "")
                if not raw:
                    continue
                resolved, location = _classify_path(raw, root)
                entries[str(name)] = {
                    "path": raw,
                    "resolved": str(item.get("resolved") or resolved),
                    "exists": bool(item.get("exists", False)),
                    "is_file": bool(item.get("is_file", False)),
                    "location": str(item.get("location") or location),
                    "captured_sha256": item.get("captured_sha256"),
                    "captured_bytes": item.get("captured_bytes"),
                }

    # Any legacy path present only in loader_provenance/source without a load-time
    # hash MUST stay unhashed (captured_sha256=None -> "unknown", never "verified").
    for name, raw in _collect_provenance_paths(info).items():
        if name not in entries:
            resolved, location = _classify_path(raw, root)
            entries[name] = {
                "path": raw,
                "resolved": str(resolved),
                "exists": bool(resolved.exists()),
                "is_file": bool(resolved.is_file()),
                "location": location,
                "captured_sha256": None,
                "captured_bytes": None,
            }
    return entries


def verify_source_references(
    source_files: Optional[Mapping[str, Any]],
    *,
    dataset_info: Optional[Mapping[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify current disk files against captured source references without lying about unhashed paths.

    Possible per-file statuses:
    - ``verified``: file exists on disk and current SHA-256 matches ``captured_sha256``
    - ``stale``: file exists on disk and current SHA-256 differs from ``captured_sha256``
    - ``missing``: referenced file is not present on disk
    - ``unknown``: file exists on disk (or source is unhashed/directory), no ``captured_sha256``
    - ``unknown`` + ``malformed``: the recorded reference is not a JSON object, so
      nothing about it can be verified. It is counted and reported explicitly
      (never skipped) and it can never yield an ``overall`` of ``verified``.
    """
    from conninfpy.app.utils.helpers import is_public_mode
    if is_public_mode():
        return {"overall": "unknown", "files": {},
                "counts": {"verified": 0, "stale": 0, "missing": 0, "unknown": 0, "malformed": 0},
                "message": "Source path verification is disabled in public mode."}
    info = dataset_info or {}
    root = Path(project_root) if project_root is not None else Path.cwd()
    entries: Mapping[str, Any] = source_files if isinstance(source_files, Mapping) else {}
    if not entries and info:
        entries = _source_entries_for_save(info, project_root=root)

    is_synthetic = (
        info.get("mode") == "Generate Synthetic Dataset"
        or str(info.get("source") or "").strip().lower() == "synthetic simulation"
    )
    if not entries:
        if is_synthetic:
            return {
                "overall": "synthetic",
                "files": {},
                "counts": {"verified": 0, "stale": 0, "missing": 0, "unknown": 0, "malformed": 0},
                "message": "Synthetic simulation (no external source files).",
            }
        return {
            "overall": "unknown",
            "files": {},
            "counts": {"verified": 0, "stale": 0, "missing": 0, "unknown": 0, "malformed": 0},
            "message": "No external source hashes were recorded for this workspace.",
        }

    counts = {"verified": 0, "stale": 0, "missing": 0, "unknown": 0, "malformed": 0}
    file_statuses: Dict[str, Dict[str, Any]] = {}
    for name, item in sorted(entries.items()):
        if not isinstance(item, Mapping):
            # A malformed record is reported, not silently dropped: it is
            # unverifiable, so it degrades the overall status and stays visible.
            counts["unknown"] += 1
            counts["malformed"] += 1
            file_statuses[str(name)] = {
                "path": None,
                "resolved": None,
                "location": "unknown",
                "exists": False,
                "captured_sha256": None,
                "current_sha256": None,
                "status": "unknown",
                "malformed": True,
                "message": ("Recorded source reference is not a JSON object; "
                            "it cannot be verified."),
            }
            continue
        raw = str(item.get("path") or item.get("resolved") or "")
        captured_sha = item.get("captured_sha256")
        resolved, location = _classify_path(raw, root)
        if not resolved.exists() and item.get("resolved"):
            alt = Path(str(item["resolved"]))
            if alt.exists():
                resolved = alt
        exists = resolved.exists()
        is_file = resolved.is_file()
        current_sha = _sha256_file(resolved) if is_file else None
        if not exists:
            status = "missing"
        elif captured_sha is None:
            status = "unknown"
        elif current_sha is None:
            status = "unknown"
        elif current_sha == captured_sha:
            status = "verified"
        else:
            status = "stale"
        counts[status] += 1
        file_statuses[str(name)] = {
            "path": raw,
            "resolved": str(resolved),
            "location": str(item.get("location") or location),
            "exists": bool(exists),
            "captured_sha256": captured_sha,
            "current_sha256": current_sha,
            "status": status,
        }

    if counts["missing"] > 0:
        overall = "missing"
    elif counts["stale"] > 0:
        overall = "stale"
    elif counts["unknown"] > 0:
        overall = "unknown"
    elif counts["verified"] > 0:
        overall = "verified"
    else:
        overall = "unknown"

    messages = {
        "verified": f"All {counts['verified']} referenced source file(s) match their captured SHA-256 hashes.",
        "stale": f"{counts['stale']} source file(s) changed on disk since capture.",
        "missing": f"{counts['missing']} referenced source file(s) are missing on disk.",
        "unknown": "Source freshness is unknown (unhashed source reference(s) or directory source).",
    }
    if overall == "unknown" and counts["malformed"]:
        unhashed = counts["unknown"] - counts["malformed"]
        details = [f"{counts['malformed']} malformed source reference(s) (not a JSON "
                   "object) could not be verified"]
        if unhashed > 0:
            details.append(f"{unhashed} reference(s) are unhashed or directories")
        messages["unknown"] = "Source freshness is unknown: " + "; ".join(details) + "."
    return {
        "overall": overall,
        "files": file_statuses,
        "counts": counts,
        "message": messages[overall],
    }


# ---------------------------------------------------------------------------
# Safe JSON + NPZ Codec (zero pickle)
# ---------------------------------------------------------------------------

class _Codec:
    """Recursive lossless encoder/decoder with NPZ array storage (allow_pickle=False)."""

    def __init__(self) -> None:
        self._arrays: Dict[str, np.ndarray] = {}
        self._counter = 0

    def _next_array_key(self, hint: str = "arr") -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", hint).strip("_") or "arr"
        key = f"a{self._counter:04d}_{safe}"
        self._counter += 1
        return key

    def store_array(self, arr: np.ndarray, hint: str = "arr") -> Dict[str, Any]:
        arr = np.asarray(arr)
        if arr.dtype == object or arr.dtype.kind == "O":
            # Only allow pure primitive string/None object arrays as JSON, NEVER pickled NPZ
            flat = arr.tolist()
            return {
                _TYPE_TAG: "string_array",
                "shape": list(arr.shape),
                "items": self.encode(flat),
            }
        key = self._next_array_key(hint)
        self._arrays[key] = np.ascontiguousarray(arr)
        return {
            _TYPE_TAG: "npz_array",
            "key": key,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }

    def encode(self, value: Any, hint: str = "val") -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return {_TYPE_TAG: "float", "value": "nan"}
            if math.isinf(value):
                return {_TYPE_TAG: "float", "value": "inf" if value > 0 else "-inf"}
            return value
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return self.encode(float(value), hint=hint)
        if isinstance(value, Path):
            return {_TYPE_TAG: "path", "value": str(value)}
        if isinstance(value, tuple):
            return {_TYPE_TAG: "tuple", "items": [self.encode(v, hint=hint) for v in value]}
        if isinstance(value, set):
            items = [self.encode(v, hint=hint) for v in sorted(value, key=lambda x: str(x))]
            return {_TYPE_TAG: "set", "items": items}
        if isinstance(value, list):
            return [self.encode(v, hint=hint) for v in value]
        if isinstance(value, np.ndarray):
            return self.store_array(value, hint=hint)
        if isinstance(value, pd.DataFrame):
            return self._encode_dataframe(value, hint=hint)
        if isinstance(value, pd.Series):
            return self._encode_series(value, hint=hint)
        if isinstance(value, pd.Index):
            return self._encode_index(value, hint=hint)

        # ConnInfPy domain types
        cls_name = type(value).__name__
        mod_name = type(value).__module__

        if cls_name == "AnalyzeResult" and mod_name.startswith("conninfpy"):
            return {
                _TYPE_TAG: "AnalyzeResult",
                "inference": self.encode(value.inference, hint=f"{hint}_inference"),
                "combat_diagnostics": self.encode(value.combat_diagnostics, hint=f"{hint}_combat"),
                "design_diagnostics": self.encode(value.design_diagnostics, hint=f"{hint}_design"),
                "flags": self.encode(list(value.flags), hint=f"{hint}_flags"),
                "resolved_strategy": self.encode(value.resolved_strategy, hint=f"{hint}_strategy"),
                # NETSTAT-28: additive opt-in descriptive side-car. Absent in
                # older states (decoded as None), never reconstructed.
                "motion_qc": self.encode(getattr(value, "motion_qc", None), hint=f"{hint}_motion_qc"),
            }
        if cls_name == "InferenceResult" and mod_name.startswith("conninfpy"):
            return self._encode_inference_result(value, hint=hint)
        if cls_name == "OmnibusInferenceResult" and mod_name.startswith("conninfpy"):
            return self._encode_omnibus_result(value, hint=hint)
        if cls_name == "AtlasInfo" and mod_name.startswith("conninfpy"):
            return {
                _TYPE_TAG: "AtlasInfo",
                "labels": list(value.labels),
                "networks": list(value.networks),
                "coords": self.encode(value.coords, hint=f"{hint}_coords"),
                "hemisphere": list(value.hemisphere) if value.hemisphere is not None else None,
                "source": value.source,
                "image": self.encode(value.image, hint=f"{hint}_image"),
                "label_ids": self.encode(value.label_ids, hint=f"{hint}_label_ids"),
                "assignment": self.encode(value.assignment, hint=f"{hint}_assignment"),
            }
        if cls_name == "BrainnetomeReference" and mod_name.startswith("conninfpy"):
            prov = getattr(value, "provenance", None) or {}
            if (
                isinstance(prov, Mapping)
                and prov.get("atlas") == "Brainnetome BNA-246 (Fan et al. 2016)"
                and prov.get("label_image_sha256")
            ):
                res_mm = 2 if "2 mm" in str(prov.get("label_image", "")) else 1
                return {
                    _TYPE_TAG: "BrainnetomeReferenceBundled",
                    "resolution_mm": res_mm,
                    "profiles": bool(getattr(value, "profiles", None) is not None),
                    "data_dir": self.encode(value.data_dir, hint=f"{hint}_data_dir"),
                    "cache_dir": self.encode(value.cache_dir, hint=f"{hint}_cache_dir"),
                    "provenance": self.encode(dict(prov), hint=f"{hint}_provenance"),
                }
            return {
                _TYPE_TAG: "BrainnetomeReference",
                "regions": self.encode(value.regions, hint=f"{hint}_regions"),
                "label_image": self.encode(value.label_image, hint=f"{hint}_label_image"),
                "data_dir": self.encode(value.data_dir, hint=f"{hint}_data_dir"),
                "cache_dir": self.encode(value.cache_dir, hint=f"{hint}_cache_dir"),
                "profiles": self.encode(value.profiles, hint=f"{hint}_profiles"),
                "provenance": self.encode(value.provenance, hint=f"{hint}_provenance"),
            }
        if cls_name == "RoiAnnotation" and mod_name.startswith("conninfpy"):
            return {
                _TYPE_TAG: "RoiAnnotation",
                "table": self.encode(value.table, hint=f"{hint}_table"),
                "weights": self.encode(value.weights, hint=f"{hint}_weights"),
                "membership": self.encode(value.membership, hint=f"{hint}_membership"),
                "reference": self.encode(value.reference, hint=f"{hint}_reference"),
                "settings": self.encode(value.settings, hint=f"{hint}_settings"),
            }
        if mod_name.startswith("nibabel") and hasattr(value, "dataobj") and hasattr(value, "affine"):
            return {
                _TYPE_TAG: "nifti_image",
                "data": self.store_array(np.asarray(value.dataobj), hint=f"{hint}_nii_data"),
                "affine": self.store_array(np.asarray(value.affine, dtype=np.float64), hint=f"{hint}_nii_affine"),
            }

        if isinstance(value, Mapping):
            # Preserve non-string dict keys (e.g. int keys in BrainnetomeReference.profiles)
            if all(isinstance(k, str) and k != _TYPE_TAG for k in value.keys()):
                return {k: self.encode(v, hint=f"{hint}_{k}") for k, v in value.items()}
            return {
                _TYPE_TAG: "dict",
                "entries": [
                    [self.encode(k, hint=f"{hint}_k"), self.encode(v, hint=f"{hint}_v")]
                    for k, v in value.items()
                ],
            }

        if is_dataclass(value):
            return {
                _TYPE_TAG: "dataclass_dict",
                "fields": {
                    k: self.encode(getattr(value, k), hint=f"{hint}_{k}")
                    for k in value.__dataclass_fields__
                },
            }

        raise WorkspaceArchiveError(
            f"Unsupported non-JSON type {mod_name}.{cls_name} encountered during safe serialization."
        )

    def _encode_index(self, idx: pd.Index, hint: str) -> Dict[str, Any]:
        if isinstance(idx, pd.MultiIndex):
            return {
                _TYPE_TAG: "pd_multiindex",
                "names": self.encode(list(idx.names), hint=f"{hint}_names"),
                "tuples": self.encode(list(idx.tolist()), hint=f"{hint}_tuples"),
            }
        arr = idx.to_numpy()
        values_enc = (
            self.encode(list(idx.tolist()), hint=f"{hint}_idx")
            if arr.dtype == object or arr.dtype.kind in ("O", "U", "S")
            else self.store_array(arr, hint=f"{hint}_idx")
        )
        return {
            _TYPE_TAG: "pd_index",
            "name": self.encode(idx.name, hint=f"{hint}_name"),
            "dtype": str(idx.dtype),
            "values": values_enc,
        }

    def _encode_dataframe(self, df: pd.DataFrame, hint: str) -> Dict[str, Any]:
        columns = []
        for pos, col_name in enumerate(df.columns):
            series = df.iloc[:, pos]
            arr = series.to_numpy()
            if arr.dtype == object or arr.dtype.kind == "O":
                col_payload: Dict[str, Any] = {
                    "name": self.encode(col_name, hint=f"{hint}_col"),
                    "storage": "json_list",
                    "dtype": str(series.dtype),
                    "values": self.encode(series.tolist(), hint=f"{hint}_c{pos}"),
                }
            elif isinstance(series.dtype, pd.CategoricalDtype):
                col_payload = {
                    "name": self.encode(col_name, hint=f"{hint}_col"),
                    "storage": "categorical",
                    "categories": self.encode(list(series.cat.categories.tolist()), hint=f"{hint}_cat"),
                    "ordered": bool(series.cat.ordered),
                    "codes": self.store_array(series.cat.codes.to_numpy(), hint=f"{hint}_codes{pos}"),
                }
            else:
                col_payload = {
                    "name": self.encode(col_name, hint=f"{hint}_col"),
                    "storage": "npz",
                    "dtype": str(series.dtype),
                    "values": self.store_array(arr, hint=f"{hint}_c{pos}"),
                }
            columns.append(col_payload)
        return {
            _TYPE_TAG: "pd_dataframe",
            "index": self._encode_index(df.index, hint=f"{hint}_index"),
            "columns": columns,
            "attrs": self.encode(dict(df.attrs), hint=f"{hint}_attrs"),
        }

    def _encode_series(self, ser: pd.Series, hint: str) -> Dict[str, Any]:
        arr = ser.to_numpy()
        storage = "json_list" if (arr.dtype == object or arr.dtype.kind == "O") else "npz"
        return {
            _TYPE_TAG: "pd_series",
            "name": self.encode(ser.name, hint=f"{hint}_name"),
            "dtype": str(ser.dtype),
            "storage": storage,
            "index": self._encode_index(ser.index, hint=f"{hint}_index"),
            "values": (
                self.encode(ser.tolist(), hint=f"{hint}_vals")
                if storage == "json_list"
                else self.store_array(arr, hint=f"{hint}_vals")
            ),
            "attrs": self.encode(dict(ser.attrs), hint=f"{hint}_attrs"),
        }

    def _encode_inference_result(self, res: Any, hint: str) -> Dict[str, Any]:
        known = (
            "method", "n_permutations", "acceleration", "wall_time_s",
            "null_max_dist", "stat_positive", "stat_negative", "stat_type",
            "harmonized", "preserve_provided", "strata_provided",
            "combat_diagnostics", "e_grid", "h_grid", "missing_report",
            "block_report", "component_report", "two_tailed", "tested_tail",
            "tail_selection", "beta", "design", "analysis_provenance",
            # NETSTAT-32 Monte-Carlo precision contract. Listed explicitly (not
            # left to the `extra` catch-all) so a restore cannot silently drop
            # counts/kind/RNG/space; absent keys in an older archive decode as
            # None = "unavailable", never as an inferred value.
            "exceedance_counts", "count_kind", "rng_provenance",
            "permutation_space",
        )
        attrs = {k: self.encode(getattr(res, k, None), hint=f"{hint}_{k}") for k in known}
        extra = {
            k: self.encode(v, hint=f"{hint}_{k}")
            for k, v in vars(res).items()
            if k not in known
        }
        return {
            _TYPE_TAG: "InferenceResult",
            "positive": self.store_array(np.asarray(res["positive"]), hint=f"{hint}_pos"),
            "negative": self.store_array(np.asarray(res["negative"]), hint=f"{hint}_neg"),
            "attrs": attrs,
            "extra": extra,
        }

    def _encode_omnibus_result(self, res: Any, hint: str) -> Dict[str, Any]:
        known = (
            "method", "n_permutations", "acceleration", "wall_time_s",
            "null_max_dist", "stat_omnibus", "stat_type", "harmonized",
            "preserve_provided", "strata_provided", "combat_diagnostics",
            "missing_report", "block_report", "component_report", "beta",
            "design", "analysis_provenance",
            # See _encode_inference_result: explicit NETSTAT-32 contract.
            "exceedance_counts", "count_kind", "rng_provenance",
            "permutation_space",
        )
        attrs = {k: self.encode(getattr(res, k, None), hint=f"{hint}_{k}") for k in known}
        extra = {
            k: self.encode(v, hint=f"{hint}_{k}")
            for k, v in vars(res).items()
            if k not in known
        }
        return {
            _TYPE_TAG: "OmnibusInferenceResult",
            "omnibus": self.store_array(np.asarray(res["omnibus"]), hint=f"{hint}_omni"),
            "attrs": attrs,
            "extra": extra,
        }

    @property
    def arrays(self) -> Dict[str, np.ndarray]:
        return self._arrays


class _Decoder:
    """Strict non-pickle decoder reading arrays from a pre-loaded NPZ dict."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._arrays = arrays

    def decode(self, node: Any) -> Any:
        if node is None or isinstance(node, (bool, int, float, str)):
            return node
        if isinstance(node, list):
            return [self.decode(item) for item in node]
        if not isinstance(node, dict):
            raise WorkspaceArchiveError(f"Unexpected JSON value type: {type(node).__name__}")

        tag = node.get(_TYPE_TAG)
        if tag is None:
            return {k: self.decode(v) for k, v in node.items()}

        if tag == "float":
            raw: Any = node.get("value")
            if raw == "nan":
                return float("nan")
            if raw == "inf":
                return float("inf")
            if raw == "-inf":
                return float("-inf")
            return float(raw)
        if tag == "path":
            return Path(str(node["value"]))
        if tag == "tuple":
            return tuple(self.decode(v) for v in node["items"])
        if tag == "set":
            return set(self.decode(v) for v in node["items"])
        if tag == "dict":
            return {self.decode(k): self.decode(v) for k, v in node["entries"]}
        if tag == "dataclass_dict":
            return {k: self.decode(v) for k, v in node["fields"].items()}
        if tag == "object_ndarray":
            raise WorkspaceArchiveError("object_ndarray entries are forbidden (no pickle).")
        if tag == "string_array":
            arr = np.array(self.decode(node["items"]), dtype=object)
            return arr.reshape(tuple(node["shape"]))
        if tag == "npz_array":
            key = str(node["key"])
            if key not in self._arrays:
                raise WorkspaceArchiveError(f"Missing array {key!r} in workspace/arrays/arrays.npz")
            arr = np.array(self._arrays[key], copy=True)
            if arr.dtype == object or arr.dtype.kind == "O":
                raise WorkspaceArchiveError(f"Array {key!r} has forbidden object dtype.")
            expected_shape = tuple(node.get("shape") or arr.shape)
            if arr.shape != expected_shape:
                raise WorkspaceArchiveError(
                    f"Array {key!r} shape mismatch: expected {expected_shape}, got {arr.shape}"
                )
            return arr
        if tag == "nifti_image":
            import nibabel as nib
            data = self.decode(node["data"])
            affine = self.decode(node["affine"])
            return nib.Nifti1Image(data, affine)
        if tag == "pd_index":
            vals = self.decode(node["values"])
            name = self.decode(node["name"])
            dtype = node.get("dtype")
            idx = pd.Index(vals, name=name)
            if dtype and str(idx.dtype) != dtype:
                try:
                    idx = idx.astype(dtype)
                except Exception:
                    pass
            return idx
        if tag == "pd_multiindex":
            names = self.decode(node["names"])
            tuples = [tuple(t) for t in self.decode(node["tuples"])]
            return pd.MultiIndex.from_tuples(tuples, names=names)
        if tag == "pd_dataframe":
            index = self.decode(node["index"])
            cols_data: List[Tuple[Any, pd.Series]] = []
            for col in node["columns"]:
                cname = self.decode(col["name"])
                storage = col["storage"]
                if storage == "json_list":
                    vals = self.decode(col["values"])
                    ser = pd.Series(vals, index=index)
                elif storage == "categorical":
                    cats = self.decode(col["categories"])
                    codes = self.decode(col["codes"])
                    cat = pd.Categorical.from_codes(codes, categories=cats, ordered=bool(col.get("ordered", False)))
                    ser = pd.Series(cat, index=index)
                else:
                    arr = self.decode(col["values"])
                    ser = pd.Series(arr, index=index)
                    dtype = col.get("dtype")
                    if dtype and str(ser.dtype) != dtype:
                        try:
                            ser = ser.astype(dtype)
                        except Exception:
                            pass
                cols_data.append((cname, ser))
            if cols_data:
                df = pd.DataFrame({f"__col_{i}": s for i, (_, s) in enumerate(cols_data)}, index=index)
                df.columns = pd.Index([cname for cname, _ in cols_data])
            else:
                df = pd.DataFrame(index=index)
            attrs = self.decode(node.get("attrs") or {})
            if isinstance(attrs, dict):
                df.attrs.update(attrs)
            return df
        if tag == "pd_series":
            index = self.decode(node["index"])
            vals = self.decode(node["values"])
            name = self.decode(node.get("name"))
            ser = pd.Series(vals, index=index, name=name)
            attrs = self.decode(node.get("attrs") or {})
            if isinstance(attrs, dict):
                ser.attrs.update(attrs)
            return ser
        if tag == "AtlasInfo":
            from conninfpy.atlas.metadata import AtlasInfo
            return AtlasInfo(
                labels=list(node["labels"]),
                networks=list(node["networks"]),
                coords=self.decode(node["coords"]),
                hemisphere=list(node["hemisphere"]) if node.get("hemisphere") is not None else None,
                source=node.get("source"),
                image=self.decode(node.get("image")),
                label_ids=self.decode(node.get("label_ids")),
                assignment=self.decode(node.get("assignment")),
            )
        if tag == "BrainnetomeReferenceBundled":
            from conninfpy.atlas.brainnetome import BrainnetomeReference
            data_dir = self.decode(node.get("data_dir"))
            cache_dir = self.decode(node.get("cache_dir"))
            if data_dir is not None and not Path(data_dir).exists():
                data_dir = None
            if cache_dir is not None and not Path(cache_dir).parent.exists():
                cache_dir = None
            return BrainnetomeReference.load(
                data_dir=data_dir,
                resolution_mm=int(node.get("resolution_mm", 1)),
                profiles=bool(node.get("profiles", True)),
                cache_dir=cache_dir,
            )
        if tag == "BrainnetomeReference":
            from conninfpy.atlas.brainnetome import BrainnetomeReference
            ref = BrainnetomeReference(
                regions=self.decode(node["regions"]),
                label_image=self.decode(node["label_image"]),
                data_dir=self.decode(node.get("data_dir")),
                cache_dir=self.decode(node.get("cache_dir")),
                profiles=self.decode(node.get("profiles")),
                provenance=self.decode(node.get("provenance") or {}),
            )
            return ref
        if tag == "RoiAnnotation":
            from conninfpy.interpret.annotate import RoiAnnotation
            return RoiAnnotation(
                table=self.decode(node["table"]),
                weights=self.decode(node["weights"]),
                membership=self.decode(node.get("membership")),
                reference=self.decode(node["reference"]),
                settings=self.decode(node.get("settings") or {}),
            )
        if tag == "InferenceResult":
            from conninfpy.results import InferenceResult
            pos = self.decode(node["positive"])
            neg = self.decode(node["negative"])
            attrs = {k: self.decode(v) for k, v in (node.get("attrs") or {}).items()}
            # InferenceResult.__new__ uses keyword arguments; component_report handled cleanly
            kwargs = {k: v for k, v in attrs.items() if k in {
                "method", "n_permutations", "acceleration", "wall_time_s",
                "null_max_dist", "stat_positive", "stat_negative", "stat_type",
                "harmonized", "preserve_provided", "strata_provided",
                "combat_diagnostics", "e_grid", "h_grid", "missing_report",
                "block_report", "two_tailed", "tested_tail", "tail_selection",
                "beta", "design", "analysis_provenance", "component_report",
                # NETSTAT-32: legacy archives simply omit these keys.
                "exceedance_counts", "count_kind", "rng_provenance",
                "permutation_space",
            }}
            obj: Union[InferenceResult, OmnibusInferenceResult] = InferenceResult(pos, neg, **kwargs)
            for k, v in (node.get("extra") or {}).items():
                setattr(obj, k, self.decode(v))
            return obj
        if tag == "OmnibusInferenceResult":
            from conninfpy.results import OmnibusInferenceResult
            omni = self.decode(node["omnibus"])
            attrs = {k: self.decode(v) for k, v in (node.get("attrs") or {}).items()}
            kwargs = {k: v for k, v in attrs.items() if k in {
                "method", "n_permutations", "acceleration", "wall_time_s",
                "null_max_dist", "stat_omnibus", "stat_type", "harmonized",
                "preserve_provided", "strata_provided", "combat_diagnostics",
                "missing_report", "block_report", "beta", "design",
                "analysis_provenance", "component_report",
                # NETSTAT-32: legacy archives simply omit these keys.
                "exceedance_counts", "count_kind", "rng_provenance",
                "permutation_space",
            }}
            obj = OmnibusInferenceResult(omni, **kwargs)
            for k, v in (node.get("extra") or {}).items():
                setattr(obj, k, self.decode(v))
            return obj
        if tag == "AnalyzeResult":
            from conninfpy.analysis import AnalyzeResult
            return AnalyzeResult(
                inference=self.decode(node["inference"]),
                combat_diagnostics=self.decode(node.get("combat_diagnostics")),
                design_diagnostics=self.decode(node.get("design_diagnostics")),
                flags=list(self.decode(node.get("flags") or [])),
                resolved_strategy=self.decode(node.get("resolved_strategy")),
                motion_qc=self.decode(node.get("motion_qc")),
            )

        raise WorkspaceArchiveError(f"Unsupported tagged object type in workspace state: {tag!r}")


# ---------------------------------------------------------------------------
# UI state extraction & allowlisting (zero secret leakage)
# ---------------------------------------------------------------------------

def _extract_ui_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    registry_raw = state.get("_workspace_widget_keys") or {}
    registry: Dict[str, str] = {}
    widgets: Dict[str, Any] = {}

    if isinstance(registry_raw, Mapping):
        for wkey, entry in registry_raw.items():
            key_str = str(wkey)
            if _is_excluded_widget_key(key_str):
                continue
            scope = entry.get("scope") if isinstance(entry, Mapping) else (str(entry) if entry is not None else None)
            if scope not in _ALLOWLISTED_WIDGET_SCOPES:
                continue
            registry[key_str] = scope
            if key_str in state:
                val = state[key_str]
                if not hasattr(val, "read"):  # never serialize UploadedFile handles
                    widgets[key_str] = val

    sidebar: Dict[str, Any] = {}
    for skey in _SIDEBAR_WIDGET_KEYS:
        if skey in state and not _is_secret_key(skey):
            sidebar[skey] = state[skey]

    # Also capture known un-registered comparison/view keys if present
    for extra_key in (
        "results:select_result_view_mode",
        "results:grid_idx:primary",
        "results:grid_idx:companion",
    ):
        if extra_key in state and extra_key not in widgets and not _is_secret_key(extra_key):
            widgets[extra_key] = state[extra_key]

    return {
        "active_tab": _normalise_active_tab(state.get("active_tab")),
        "sidebar": sidebar,
        "widget_registry": registry,
        "widgets": widgets,
    }


def _validate_result_bundle_if_captured(result: Any) -> bool:
    """Validate result via ResultBundle.from_result when analysis_provenance is captured."""
    if result is None:
        return False
    inner = _unwrap_result(result)
    provenance = getattr(inner, "analysis_provenance", None)
    if not provenance:
        return False
    from conninfpy.post_analysis import ResultBundle
    ResultBundle.from_result(result)
    return True


def _redact_mapping(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _is_secret_key(str(k)):
                continue
            out[k] = _redact_mapping(v)
        return out
    if isinstance(obj, list):
        return [_redact_mapping(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Public Workspace Serialization API
# ---------------------------------------------------------------------------

def build_workspace_layer_files(
    state: Mapping[str, Any],
    *,
    include_data: bool = False,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    """Build the ``workspace/*`` files dict and its ``workspace/manifest.json`` dict."""
    from conninfpy import __version__

    raw_info = state.get("dataset_info")
    dataset_info = _redact_mapping(dict(raw_info)) if isinstance(raw_info, Mapping) else None
    source_files = _source_entries_for_save(dataset_info, project_root=project_root)

    run_plan_raw = state.get("run_plan")
    run_plan = _redact_mapping(dict(run_plan_raw)) if isinstance(run_plan_raw, Mapping) else None

    primary_res = state.get("inference_result")
    companion_res = state.get("companion_inference_result")

    primary_validated = _validate_result_bundle_if_captured(primary_res)
    companion_validated = _validate_result_bundle_if_captured(companion_res)

    has_synth = any(
        state.get(k) is not None
        for k in ("_synthetic_effect_mask", "_synthetic_scenario_name", "_synthetic_confound_mask", "_synthetic_confound_name")
    ) or (isinstance(dataset_info, Mapping) and dataset_info.get("mode") == "Generate Synthetic Dataset")

    synthetic_payload = None
    if has_synth:
        synthetic_payload = {
            "effect_mask": state.get("_synthetic_effect_mask"),
            "scenario_name": state.get("_synthetic_scenario_name"),
            "confound_mask": state.get("_synthetic_confound_mask"),
            "confound_name": state.get("_synthetic_confound_name"),
        }

    ui_raw = _extract_ui_state(state)

    codec = _Codec()
    state_doc: Dict[str, Any] = {
        "kind": WORKSPACE_ARCHIVE_KIND,
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "include_data": bool(include_data),
        "meta": {
            "conninfpy_version": __version__,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "active_tab": ui_raw["active_tab"],
        },
        "source_refs": codec.encode({
            "dataset_info": dataset_info,
            "source_files": source_files,
            "connectivity_data_kind": state.get("connectivity_data_kind"),
            "dataset_preprocessing": state.get("dataset_preprocessing"),
            "dataset_fisher_history": state.get("dataset_fisher_history"),
            "last_fisher_clip_note": state.get("last_fisher_clip_note"),
            "dataset_load_notes": state.get("dataset_load_notes"),
            "dataset_atlas": state.get("dataset_atlas"),
            "sub_atlas": state.get("sub_atlas"),
            "loaded_reference_atlas": state.get("loaded_reference_atlas"),
            "roi_indices": state.get("roi_indices"),
            "active_atlas_signature": state.get("active_atlas_signature"),
            "loaded_atlas_signature": state.get("loaded_atlas_signature"),
            "loaded_settings_hash": state.get("loaded_settings_hash"),
            "current_settings_hash": state.get("current_settings_hash"),
            "synthetic": synthetic_payload,
        }, hint="src"),
        "results": {
            "primary": codec.encode({
                "inference_result": primary_res,
                "edges_df": state.get("edges_df"),
                "run_plan": run_plan,
                "bundle_validated": primary_validated,
            }, hint="res") if primary_res is not None else None,
            "companion": codec.encode({
                "inference_result": companion_res,
                "edges_df": state.get("companion_edges_df"),
                "method": state.get("companion_method"),
                "bundle_validated": companion_validated,
            }, hint="comp") if companion_res is not None else None,
        },
        "annotation": codec.encode({
            "annotation": state.get("annotation"),
            "evidence_packet": state.get("evidence_packet"),
        }, hint="ann"),
        "ui_state": codec.encode(ui_raw, hint="ui"),
        "data": codec.encode({
            "connectivity_data": state.get("connectivity_data"),
            "pheno_df": state.get("pheno_df"),
            "_empirical_corr": state.get("_empirical_corr"),
        }, hint="data") if include_data else None,
    }

    state_bytes = _dumps_json(state_doc)
    npz_bytes = _npz_bytes(codec.arrays)

    layer_files: Dict[str, bytes] = {
        WORKSPACE_STATE_PATH: state_bytes,
        WORKSPACE_ARRAYS_PATH: npz_bytes,
    }

    ws_manifest: Dict[str, Any] = {
        "kind": WORKSPACE_ARCHIVE_KIND,
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "include_data": bool(include_data),
        "saved_at": state_doc["meta"]["saved_at"],
        "conninfpy_version": __version__,
        "encoding": "json+npz",
        "allow_pickle": False,
        "contents": {
            "has_results": primary_res is not None,
            "has_companion": companion_res is not None,
            "has_annotation": state.get("annotation") is not None,
            "has_evidence_packet": state.get("evidence_packet") is not None,
            "has_data": bool(include_data and state.get("connectivity_data") is not None),
            "array_count": len(codec.arrays),
        },
        "files": {
            name: {"sha256": _sha256_bytes(blob), "bytes": len(blob)}
            for name, blob in sorted(layer_files.items())
        },
    }
    layer_files[WORKSPACE_MANIFEST_PATH] = _dumps_json(ws_manifest)
    return layer_files, ws_manifest


def build_workspace_archive(
    state: Mapping[str, Any],
    *,
    include_data: bool = False,
    project_root: Optional[Path] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Build a deterministic safe workspace ZIP archive (no pickle).

    When ``inference_result`` and ``run_plan`` are present, returns the unified
    analysis-archive ZIP containing both the standard ``NETSTAT-24`` files and
    the additive ``workspace/`` layer covered in root ``manifest.json``.
    Otherwise returns a clean workspace ZIP with ``workspace/*`` + root
    ``manifest.json``.

    This is the default workspace save path (sidebar *Save workspace*, Cloud
    save): it omits the separate ``result_bundle.zip`` member (and never writes
    Python pickle files). The explicit ``NETSTAT-24`` *Download Full Analysis
    Archive* button keeps calling ``build_analysis_archive`` directly
    (``include_bundle=True``) to include the pickle-free ``result_bundle.zip``.
    """
    from conninfpy.app.utils.archive_export import (
        ARCHIVE_KIND,
        ARCHIVE_SCHEMA_VERSION,
        archive_zip_bytes,
        build_analysis_archive,
    )
    from conninfpy import __version__

    if (
        state.get("inference_result") is not None
        and isinstance(state.get("run_plan"), Mapping)
        and _is_inference_like(state["inference_result"])
    ):
        inner = _unwrap_result(state["inference_result"])
        is_omni = "omnibus" in inner
        p_arr = inner["omnibus"] if is_omni else inner["positive"]
        has_stats = (
            getattr(inner, "stat_omnibus", None) is not None
            if is_omni
            else (
                getattr(inner, "stat_positive", None) is not None
                and getattr(inner, "stat_negative", None) is not None
            )
        )
        if has_stats:
            n_nodes = int(np.asarray(p_arr).shape[0])
            atlas = (
                state.get("sub_atlas")
                or state.get("dataset_atlas")
                or state.get("loaded_reference_atlas")
            )
            if atlas is not None and len(atlas) != n_nodes:
                atlas = None
            return build_analysis_archive(
                state["inference_result"],
                state["run_plan"],
                state.get("dataset_info") or {},
                atlas=atlas,
                effect_mask=state.get("_synthetic_effect_mask"),
                grid_cell=state.get("results:grid_idx:primary"),
                project_root=project_root,
                workspace_state=state,
                include_workspace_data=include_data,
                include_bundle=False,
                edge_family_mask=state.get("design:edge_mask_array"),
            )

    ws_files, ws_manifest = build_workspace_layer_files(
        state, include_data=include_data, project_root=project_root
    )
    files = dict(ws_files)
    root_manifest: Dict[str, Any] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "kind": ARCHIVE_KIND,
        "export": {
            "exported_utc": ws_manifest["saved_at"],
            "exporter": "conninfpy-app-workspace",
            "conninfpy_version": __version__,
        },
        "workspace": {
            "kind": ws_manifest["kind"],
            "workspace_schema_version": ws_manifest["workspace_schema_version"],
            "include_data": ws_manifest["include_data"],
        },
        "files": {
            name: {"sha256": _sha256_bytes(blob), "bytes": len(blob)}
            for name, blob in sorted(files.items())
        },
    }
    files["manifest.json"] = _dumps_json(root_manifest)
    return archive_zip_bytes(files), ws_manifest


def workspace_filename(state: Mapping[str, Any]) -> str:
    """Deterministic download filename for a workspace archive."""
    from conninfpy.app.utils.helpers import safe_filename_part

    info = state.get("dataset_info") or {}
    name = safe_filename_part(str(info.get("name") or "workspace"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"conninf_workspace_{name}_{stamp}{WORKSPACE_EXTENSION}"


# ---------------------------------------------------------------------------
# Workspace Inspection & Restoration (no pickle, strict validation)
# ---------------------------------------------------------------------------

def _load_npz_safe(raw_npz: bytes) -> Dict[str, np.ndarray]:
    if not raw_npz:
        return {}
    arrays: Dict[str, np.ndarray] = {}
    try:
        with np.load(io.BytesIO(raw_npz), allow_pickle=False) as npz:
            for name in npz.files:
                arr = npz[name]
                if arr.dtype == object or arr.dtype.kind == "O":
                    raise WorkspaceArchiveError(f"Array {name!r} has forbidden object dtype.")
                arrays[name] = np.array(arr, copy=True)
    except WorkspaceArchiveError:
        raise
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        raise WorkspaceArchiveError(f"Unsafe or malformed workspace array archive: {exc}") from exc
    return arrays


def _parse_and_validate_zip(
    zip_bytes: bytes,
    *,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    if not isinstance(zip_bytes, (bytes, bytearray)) or not zip_bytes:
        raise WorkspaceArchiveError("Workspace archive is empty.")
    try:
        zf_ctx = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    except zipfile.BadZipFile as exc:
        raise WorkspaceArchiveError(f"Invalid ZIP workspace archive: {exc}") from exc

    with zf_ctx as zf:
        _validate_zip_members(zf)
        names = set(zf.namelist())
        if WORKSPACE_MANIFEST_PATH not in names:
            raise WorkspaceArchiveError(
                "Archive does not contain a 'workspace/manifest.json' layer "
                "(legacy analysis archives without a workspace layer cannot be auto-restored; "
                "bundle.pkl is never loaded)."
            )

        try:
            ws_manifest = json.loads(zf.read(WORKSPACE_MANIFEST_PATH).decode("utf-8"))
        except Exception as exc:
            raise WorkspaceArchiveError(f"Corrupt workspace/manifest.json: {exc}") from exc

        if not isinstance(ws_manifest, dict):
            raise WorkspaceArchiveError("workspace/manifest.json must be a JSON object.")
        if ws_manifest.get("kind") != WORKSPACE_ARCHIVE_KIND:
            raise WorkspaceArchiveError(
                f"Unsupported workspace archive kind: {ws_manifest.get('kind')!r}"
            )
        version = ws_manifest.get("workspace_schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise WorkspaceArchiveError(f"Invalid workspace_schema_version: {version!r}")
        if version not in SUPPORTED_WORKSPACE_SCHEMA_VERSIONS:
            raise WorkspaceArchiveError(
                f"Unsupported workspace_schema_version {version}; this app supports "
                f"versions {list(SUPPORTED_WORKSPACE_SCHEMA_VERSIONS)}."
            )
        if ws_manifest.get("allow_pickle") is True:
            raise WorkspaceArchiveError("Pickle-enabled workspaces are rejected.")

        # Verify workspace/manifest.json file inventory & SHA-256 digests
        ws_files_meta = ws_manifest.get("files")
        if not isinstance(ws_files_meta, dict) or WORKSPACE_STATE_PATH not in ws_files_meta:
            raise WorkspaceArchiveError("workspace/manifest.json is missing required file inventory.")
        for member, meta in ws_files_meta.items():
            if not isinstance(meta, Mapping):
                raise WorkspaceArchiveError(
                    "workspace/manifest.json file inventory entry "
                    f"{str(member)!r} is not a JSON object."
                )

        ws_members = {n for n in names if n.startswith(WORKSPACE_PREFIX) and not n.endswith("/")}
        expected_ws_members = set(ws_files_meta.keys()) | {WORKSPACE_MANIFEST_PATH}
        if ws_members != expected_ws_members:
            raise WorkspaceArchiveError(
                "Workspace member inventory does not match workspace/manifest.json."
            )

        for member, meta in ws_files_meta.items():
            if member not in names:
                raise WorkspaceArchiveError(f"Missing workspace member: {member}")
            raw_member = zf.read(member)
            if len(raw_member) != meta.get("bytes") or _sha256_bytes(raw_member) != meta.get("sha256"):
                raise WorkspaceArchiveError(f"Checksum mismatch for workspace member: {member}")

        # If root manifest.json exists, cross-check all workspace/* members against it too
        if "manifest.json" in names:
            try:
                root_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            except Exception as exc:
                raise WorkspaceArchiveError(f"Corrupt root manifest.json: {exc}") from exc
            if not isinstance(root_manifest, dict):
                raise WorkspaceArchiveError("Root manifest.json must be a JSON object.")
            root_files = root_manifest.get("files")
            if not isinstance(root_files, dict):
                raise WorkspaceArchiveError("Root manifest.json is missing 'files' checksums.")
            for member in ws_members:
                if member not in root_files:
                    raise WorkspaceArchiveError(
                        f"Workspace member {member!r} is missing from root manifest.json."
                    )
                raw_member = zf.read(member)
                entry = root_files[member]
                if not isinstance(entry, Mapping):
                    raise WorkspaceArchiveError(
                        f"Root manifest.json checksum entry for {member!r} is not a JSON object."
                    )
                if len(raw_member) != entry.get("bytes") or _sha256_bytes(raw_member) != entry.get("sha256"):
                    raise WorkspaceArchiveError(
                        f"Root manifest checksum mismatch for workspace member: {member}"
                    )

        try:
            state_doc = json.loads(zf.read(WORKSPACE_STATE_PATH).decode("utf-8"))
        except Exception as exc:
            raise WorkspaceArchiveError(f"Corrupt workspace/state.json: {exc}") from exc

        if not isinstance(state_doc, dict):
            raise WorkspaceArchiveError("workspace/state.json must be a JSON object.")
        if state_doc.get("kind") != WORKSPACE_ARCHIVE_KIND:
            raise WorkspaceArchiveError(f"Unsupported workspace state kind: {state_doc.get('kind')!r}")
        state_version = state_doc.get("workspace_schema_version")
        if isinstance(state_version, bool) or not isinstance(state_version, int):
            # 2.0 (float) must never satisfy the manifest's integer 2.
            raise WorkspaceArchiveError(
                "Invalid workspace/state.json workspace_schema_version: "
                f"{state_version!r} (expected an integer)."
            )
        if state_version != version:
            raise WorkspaceArchiveError("workspace/state.json schema version does not match manifest.")
        _validate_state_contract(state_doc)

        arrays: Dict[str, np.ndarray] = {}
        if WORKSPACE_ARRAYS_PATH in names:
            try:
                arrays = _load_npz_safe(zf.read(WORKSPACE_ARRAYS_PATH))
            except WorkspaceArchiveError:
                raise
            except Exception as exc:
                raise WorkspaceArchiveError(f"Failed to load workspace arrays safely: {exc}") from exc

    return ws_manifest, state_doc, arrays


def _validate_state_contract(state_doc: Mapping[str, Any]) -> None:
    """Check the explicit data/results/annotation/ui_state contract of state.json."""
    missing = [section for section in _REQUIRED_STATE_SECTIONS if section not in state_doc]
    if missing:
        raise WorkspaceArchiveError(
            "workspace/state.json is missing required section(s): " + ", ".join(missing)
        )
    for section in _REQUIRED_STATE_SECTIONS:
        value = state_doc.get(section)
        if section == "data":
            if value is not None and not isinstance(value, Mapping):
                raise WorkspaceArchiveError(
                    "workspace/state.json section 'data' must be an object or null."
                )
        elif not isinstance(value, Mapping):
            raise WorkspaceArchiveError(
                f"workspace/state.json section {section!r} must be a JSON object."
            )


def _legacy_narrative_skipped(
    annotation_section: Optional[Mapping[str, Any]],
    ui_section: Optional[Mapping[str, Any]],
) -> List[str]:
    """Names of retired narrative/LLM fields present in an archive (skipped, never restored)."""
    skipped: List[str] = []
    if isinstance(annotation_section, Mapping):
        for key in LEGACY_NARRATIVE_ANNOTATION_KEYS:
            if annotation_section.get(key) is not None:
                skipped.append(f"annotation.{key}")
    if isinstance(ui_section, Mapping):
        for container in ("widget_registry", "widgets"):
            block = ui_section.get(container)
            if not isinstance(block, Mapping):
                continue
            for key in block:
                key_str = str(key)
                if key_str.startswith(LEGACY_NARRATIVE_WIDGET_PREFIX) or key_str in LEGACY_LLM_WIDGET_KEYS:
                    skipped.append(f"ui_state.{container}.{key_str}")
    return sorted(set(skipped))


def _legacy_unknowns_from_decoded(
    source_refs: Mapping[str, Any],
    primary_block: Optional[Mapping[str, Any]],
    source_verification: Mapping[str, Any],
    *,
    annotation_section: Optional[Mapping[str, Any]] = None,
    ui_section: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    unknowns: List[str] = []
    if primary_block and not primary_block.get("bundle_validated"):
        unknowns.append(
            "Saved primary result lacks captured ResultBundle analysis_provenance "
            "(masks/ROI IDs stay unverified rather than fabricated)."
        )
    if source_verification.get("overall") == "unknown":
        unknowns.append("Source file freshness is unknown (missing load-time SHA-256 hash).")
    prep = source_refs.get("dataset_preprocessing")
    if prep is None and source_refs.get("dataset_info"):
        unknowns.append("Dataset preprocessing provenance was not recorded.")
    fisher_hist = source_refs.get("dataset_fisher_history")
    if fisher_hist is None and source_refs.get("dataset_info"):
        unknowns.append("Upstream Fisher transform history was not recorded.")
    skipped = _legacy_narrative_skipped(annotation_section, ui_section)
    if skipped:
        unknowns.append(
            "Retired LLM narrative fields from an earlier workspace were skipped "
            "and are not restored: " + ", ".join(skipped) + "."
        )
    return unknowns


def _require_decoded_mapping(value: Any, label: str) -> Dict[str, Any]:
    """Insist a decoded section is a JSON object, else raise ``WorkspaceArchiveError``."""
    if not isinstance(value, Mapping):
        raise WorkspaceArchiveError(
            f"workspace/state.json section {label!r} must decode to a JSON object "
            f"(got {type(value).__name__})."
        )
    return dict(value)


def _decode_workspace_payload_impl(
    zip_bytes: bytes,
    *,
    project_root: Optional[Path] = None,
) -> Tuple[WorkspaceInspection, Dict[str, Any]]:
    """Parse, validate, decode and summarize an archive in one pass."""
    ws_manifest, state_doc, arrays = _parse_and_validate_zip(zip_bytes, project_root=project_root)
    decoder = _Decoder(arrays)
    source_refs = _require_decoded_mapping(decoder.decode(state_doc.get("source_refs") or {}), "source_refs")
    results_section = state_doc.get("results") or {}
    primary_raw = results_section.get("primary")
    companion_raw = results_section.get("companion")
    annotation_section = _require_decoded_mapping(decoder.decode(state_doc.get("annotation") or {}), "annotation")
    data_section = decoder.decode(state_doc.get("data")) if state_doc.get("data") is not None else None
    if data_section is not None and not isinstance(data_section, Mapping):
        raise WorkspaceArchiveError("workspace/state.json section 'data' must decode to a JSON object or null.")
    ui_section = _require_decoded_mapping(decoder.decode(state_doc.get("ui_state") or {}), "ui_state")
    primary_decoded = (
        _require_decoded_mapping(decoder.decode(primary_raw), "results.primary")
        if primary_raw is not None else None
    )
    companion_decoded = (
        _require_decoded_mapping(decoder.decode(companion_raw), "results.companion")
        if companion_raw is not None else None
    )
    run_plan = (primary_decoded or {}).get("run_plan")
    if run_plan is not None and not isinstance(run_plan, Mapping):
        raise WorkspaceArchiveError("workspace/state.json results.primary.run_plan must be a JSON object.")

    dataset_info = source_refs.get("dataset_info")
    if dataset_info is None:
        dataset_info = {}
    if not isinstance(dataset_info, Mapping):
        raise WorkspaceArchiveError("workspace/state.json source_refs.dataset_info must be a JSON object.")
    source_files = source_refs.get("source_files")
    if source_files is None:
        source_files = {}
    if not isinstance(source_files, Mapping):
        raise WorkspaceArchiveError("workspace/state.json source_refs.source_files must be a JSON object.")
    verification = verify_source_references(
        source_files,
        dataset_info=dataset_info,
        project_root=project_root,
    )

    # Validate result structure during decode: a bad bundle must fail before apply
    if primary_decoded is not None and primary_decoded.get("inference_result") is not None:
        try:
            _validate_result_bundle_if_captured(primary_decoded["inference_result"])
        except Exception as exc:
            raise WorkspaceArchiveError(f"Primary result failed bundle validation: {exc}") from exc
    if companion_decoded is not None and companion_decoded.get("inference_result") is not None:
        try:
            _validate_result_bundle_if_captured(companion_decoded["inference_result"])
        except Exception as exc:
            raise WorkspaceArchiveError(f"Companion result failed bundle validation: {exc}") from exc

    legacy_unknowns = _legacy_unknowns_from_decoded(
        source_refs,
        primary_decoded,
        verification,
        annotation_section=annotation_section,
        ui_section=ui_section,
    )
    legacy_fields_skipped = _legacy_narrative_skipped(annotation_section, ui_section)
    contents = ws_manifest.get("contents") or {}
    if not isinstance(contents, Mapping):
        raise WorkspaceArchiveError("workspace/manifest.json 'contents' must be a JSON object.")
    summary = {
        "name": dataset_info.get("name") or "Workspace",
        "mode": dataset_info.get("mode"),
        "source": dataset_info.get("source"),
        "observations": dataset_info.get("observations"),
        "n_rois": dataset_info.get("n_rois"),
        "atlas": dataset_info.get("atlas"),
        "method": ((primary_decoded or {}).get("run_plan") or {}).get("method"),
        "alpha": ((primary_decoded or {}).get("run_plan") or {}).get("alpha"),
        "source_status": verification["overall"],
    }
    inspection = WorkspaceInspection(
        manifest=ws_manifest,
        workspace_schema_version=int(ws_manifest["workspace_schema_version"]),
        kind=str(ws_manifest["kind"]),
        include_data=bool(ws_manifest.get("include_data", False)),
        saved_at=ws_manifest.get("saved_at"),
        active_tab=_normalise_active_tab(ui_section.get("active_tab")),
        has_results=bool(primary_decoded and primary_decoded.get("inference_result") is not None),
        has_companion=bool(companion_decoded and companion_decoded.get("inference_result") is not None),
        has_annotation=annotation_section.get("annotation") is not None,
        has_evidence_packet=annotation_section.get("evidence_packet") is not None,
        has_data=bool(data_section is not None and contents.get("has_data", False)),
        dataset_summary=summary,
        source_verification=verification,
        legacy_unknowns=legacy_unknowns,
        legacy_fields_skipped=legacy_fields_skipped,
        summary=summary,
    )
    payload: Dict[str, Any] = {
        "manifest": ws_manifest,
        "source_refs": source_refs,
        "primary_decoded": primary_decoded,
        "companion_decoded": companion_decoded,
        "annotation_section": annotation_section,
        "ui_section": ui_section,
        "data_section": data_section,
        "verification": verification,
        "legacy_unknowns": legacy_unknowns,
        "legacy_fields_skipped": legacy_fields_skipped,
    }
    return inspection, payload


def _decode_workspace_payload(
    zip_bytes: bytes,
    *,
    project_root: Optional[Path] = None,
) -> Tuple[WorkspaceInspection, Dict[str, Any]]:
    """Single decode/validate entry point for inspection and restore.

    A malformed JSON document (wrong type, missing key, `[]` where an object
    belongs, ...) always surfaces as ``WorkspaceArchiveError`` with a readable
    message instead of leaking ``AttributeError``/``TypeError``/``KeyError``
    into the UI, and never after any target state was touched.

    The same holds for ZIP-level damage found while reading a member (typical
    download corruption: one flipped byte in a stored body, central directory
    intact): a raw ``zipfile.BadZipFile`` from ``zf.read(...)`` of any member is
    normalised here, on the single external boundary, and never reaches the UI.
    """
    try:
        return _decode_workspace_payload_impl(zip_bytes, project_root=project_root)
    except WorkspaceArchiveError:
        raise
    except zipfile.BadZipFile as exc:
        raise WorkspaceArchiveError(
            f"Corrupt workspace archive (BadZipFile): {exc}"
        ) from exc
    except (AttributeError, TypeError, KeyError, IndexError) as exc:
        raise WorkspaceArchiveError(
            f"Malformed workspace archive ({type(exc).__name__}): {exc}"
        ) from exc


def inspect_workspace_archive(
    zip_bytes: bytes,
    *,
    project_root: Optional[Path] = None,
) -> WorkspaceInspection:
    """Validate and summarize a workspace ZIP without mutating session state."""
    return _decode_workspace_payload(zip_bytes, project_root=project_root)[0]


def _build_restore_patch(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any], Dict[str, Any], Optional[str]]:
    """Build the fully validated session patch from an already-decoded archive.

    Pure: it never touches ``target_state``. Every shape assumption fails here,
    before the first write, so a malformed archive cannot leave half a restore
    behind.
    """
    ws_manifest = payload["manifest"]
    source_refs = payload["source_refs"]
    primary_decoded = payload["primary_decoded"]
    companion_decoded = payload["companion_decoded"]
    annotation_section = payload["annotation_section"]
    ui_section = payload["ui_section"]
    data_section = payload["data_section"]
    verification = payload["verification"]
    legacy_unknowns = payload["legacy_unknowns"]
    legacy_fields_skipped = payload["legacy_fields_skipped"]

    primary_res = (primary_decoded or {}).get("inference_result")
    companion_res = (companion_decoded or {}).get("inference_result")
    run_plan = (primary_decoded or {}).get("run_plan")

    dataset_info = source_refs.get("dataset_info")
    if isinstance(dataset_info, Mapping):
        dataset_info = _redact_mapping(dict(dataset_info))

    # Build full validated patch dict BEFORE mutating target_state (atomicity)
    patch: Dict[str, Any] = {k: None for k in _MANAGED_SESSION_KEYS}
    patch["dataset_info"] = dataset_info
    patch["connectivity_data_kind"] = source_refs.get("connectivity_data_kind")
    patch["dataset_preprocessing"] = source_refs.get("dataset_preprocessing")
    patch["dataset_fisher_history"] = source_refs.get("dataset_fisher_history")
    patch["last_fisher_clip_note"] = source_refs.get("last_fisher_clip_note")
    patch["dataset_load_notes"] = source_refs.get("dataset_load_notes") or []
    patch["dataset_atlas"] = source_refs.get("dataset_atlas")
    patch["sub_atlas"] = source_refs.get("sub_atlas")
    patch["loaded_reference_atlas"] = source_refs.get("loaded_reference_atlas")
    patch["roi_indices"] = source_refs.get("roi_indices")
    patch["active_atlas_signature"] = source_refs.get("active_atlas_signature")
    patch["loaded_atlas_signature"] = source_refs.get("loaded_atlas_signature")
    patch["loaded_settings_hash"] = source_refs.get("loaded_settings_hash")
    patch["current_settings_hash"] = source_refs.get("current_settings_hash")

    synth = source_refs.get("synthetic")
    if isinstance(synth, Mapping):
        patch["_synthetic_effect_mask"] = synth.get("effect_mask")
        patch["_synthetic_scenario_name"] = synth.get("scenario_name")
        patch["_synthetic_confound_mask"] = synth.get("confound_mask")
        patch["_synthetic_confound_name"] = synth.get("confound_name")

    if data_section is not None:
        patch["connectivity_data"] = data_section.get("connectivity_data")
        patch["pheno_df"] = data_section.get("pheno_df")
        patch["_empirical_corr"] = data_section.get("_empirical_corr")

    if primary_decoded is not None:
        patch["inference_result"] = primary_res
        patch["edges_df"] = primary_decoded.get("edges_df")
        patch["run_plan"] = _redact_mapping(run_plan) if isinstance(run_plan, Mapping) else run_plan

    if companion_decoded is not None:
        patch["companion_inference_result"] = companion_res
        patch["companion_edges_df"] = companion_decoded.get("edges_df")
        patch["companion_method"] = companion_decoded.get("method")

    patch["annotation"] = annotation_section.get("annotation")
    patch["evidence_packet"] = annotation_section.get("evidence_packet")

    # Widget-guard keys so Results & Annotation keep restored selections without reset
    run_alpha = 0.05
    try:
        run_alpha = float((run_plan or {}).get("alpha") or 0.05)
    except (TypeError, ValueError) as exc:
        raise WorkspaceArchiveError(
            f"workspace/state.json records a non-numeric run alpha: {exc}"
        ) from exc
    if primary_res is not None:
        inner_primary = _unwrap_result(primary_res)
        primary_method = str((run_plan or {}).get("method") or getattr(inner_primary, "method", "Primary")).upper()
        patch["_annotation_result_id"] = id(inner_primary)
        patch["_annotation_run_alpha"] = run_alpha
        patch[f"_{primary_method}_graph_result_id"] = id(inner_primary)
        patch[f"_{primary_method}_graph_run_alpha"] = run_alpha

    if companion_res is not None:
        inner_comp = _unwrap_result(companion_res)
        comp_method = str(
            (companion_decoded or {}).get("method") or getattr(inner_comp, "method", "Companion")
        ).upper()
        patch[f"_{comp_method}_graph_result_id"] = id(inner_comp)
        patch[f"_{comp_method}_graph_run_alpha"] = run_alpha

    patch["_workspace_source_verification"] = verification
    patch["_workspace_legacy_unknowns"] = legacy_unknowns
    patch["_workspace_legacy_fields_skipped"] = legacy_fields_skipped
    patch["_workspace_restored_meta"] = {
        "saved_at": ws_manifest.get("saved_at"),
        "include_data": bool(ws_manifest.get("include_data", False)),
        "has_data": bool(patch.get("connectivity_data") is not None),
        "workspace_schema_version": int(ws_manifest["workspace_schema_version"]),
    }

    # Build sanitized UI widget patch (ui_state.py stores _workspace_widget_keys[key] = scope)
    widget_registry_patch: Dict[str, str] = {}
    widget_values_patch: Dict[str, Any] = {}

    raw_reg = ui_section.get("widget_registry") or {}
    if isinstance(raw_reg, Mapping):
        for wkey, entry in raw_reg.items():
            key_str = str(wkey)
            if _is_excluded_widget_key(key_str):
                continue
            scope = entry.get("scope") if isinstance(entry, Mapping) else (str(entry) if entry is not None else None)
            if scope not in _ALLOWLISTED_WIDGET_SCOPES:
                continue
            widget_registry_patch[key_str] = scope

    raw_widgets = ui_section.get("widgets") or {}
    if isinstance(raw_widgets, Mapping):
        for wkey, val in raw_widgets.items():
            key_str = str(wkey)
            if _is_excluded_widget_key(key_str):
                continue
            widget_values_patch[key_str] = val

    sidebar_patch: Dict[str, Any] = {}
    raw_sidebar = ui_section.get("sidebar") or {}
    if isinstance(raw_sidebar, Mapping):
        for skey in _SIDEBAR_WIDGET_KEYS:
            if skey in raw_sidebar and not _is_secret_key(skey):
                sidebar_patch[skey] = raw_sidebar[skey]

    target_active_tab = _normalise_active_tab(ui_section.get("active_tab"))

    return patch, widget_registry_patch, widget_values_patch, sidebar_patch, target_active_tab


def restore_workspace_archive(
    zip_bytes: bytes,
    target_state: Any,
    *,
    project_root: Optional[Path] = None,
) -> WorkspaceInspection:
    """Atomically decode, validate, and apply a saved workspace onto ``target_state``.

    Parsing, validation and the returned inspection all complete BEFORE the
    first write to ``target_state``; the archive is never re-parsed after the
    mutation, and any failure leaves ``target_state`` completely unchanged.
    """
    inspection, payload = _decode_workspace_payload(zip_bytes, project_root=project_root)
    try:
        patch, widget_registry_patch, widget_values_patch, sidebar_patch, target_active_tab = (
            _build_restore_patch(payload)
        )
    except WorkspaceArchiveError:
        raise
    except (AttributeError, TypeError, KeyError, ValueError) as exc:
        raise WorkspaceArchiveError(
            f"Malformed workspace archive ({type(exc).__name__}): {exc}"
        ) from exc

    # Atomic apply phase (cannot raise type/validation errors from payload above)
    # Retired narrative/LLM session keys stay absent: an old archive's values are
    # never revived and no narrative widget is recreated.
    for legacy_key in LEGACY_NARRATIVE_SESSION_KEYS:
        target_state.pop(legacy_key, None)
    old_reg = target_state.get("_workspace_widget_keys")
    if isinstance(old_reg, Mapping):
        for old_key in list(old_reg.keys()):
            target_state.pop(old_key, None)
            target_state.pop(f"__persisted__{old_key}", None)

    for old_guard_key in [k for k in list(target_state.keys()) if str(k).endswith(("_graph_result_id", "_graph_run_alpha"))]:
        target_state.pop(old_guard_key, None)

    # Clear previous archive memo buffers / transient task handles
    for old_k in [k for k in list(target_state.keys()) if isinstance(k, tuple) and k[:1] == ("_archive_zip",)]:
        target_state.pop(old_k, None)
    target_state.pop("inference_task", None)

    for k, v in patch.items():
        if k == "loaded_reference_atlas":
            target_state[k] = v
        elif v is None and k in ("_synthetic_effect_mask", "_synthetic_scenario_name", "_synthetic_confound_mask", "_synthetic_confound_name", "_empirical_corr"):
            target_state[k] = None
        else:
            target_state[k] = v

    target_state["_workspace_widget_keys"] = widget_registry_patch
    for k, v in widget_values_patch.items():
        target_state[k] = v
    for k, v in sidebar_patch.items():
        target_state[k] = v

    if target_active_tab:
        target_state["active_tab"] = target_active_tab

    return inspection


def apply_pending_workspace_restore(state: Any, *, project_root: Optional[Path] = None) -> Optional[WorkspaceInspection]:
    """Apply ``_pending_workspace_restore`` bytes before any Streamlit widgets are instantiated."""
    pending = state.pop("_pending_workspace_restore", None)
    if pending is None:
        return None
    inspection = restore_workspace_archive(pending, state, project_root=project_root)
    state["_workspace_restore_banner"] = (
        f"Restored workspace '{inspection.summary.get('name', 'Workspace')}' "
        f"({inspection.source_verification['overall']})."
    )
    return inspection
