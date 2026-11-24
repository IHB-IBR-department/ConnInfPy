"""Self-contained analysis archive exported from a finished GUI run.

One ZIP download stores the trusted post-analysis bundle (NETSTAT-3
``ResultBundle``), the corrected evidence and coefficient arrays, the
edge / network / node / component / block tables, figures built by the
shared ``result_views`` builders, a deterministic ``methods.md`` written
only from captured facts, and — when the data source is honestly
reproducible — a CLI ``analysis.yaml`` with an explicit replay status.

The export is pure: it reads the finished result and the captured
``run_plan`` / ``dataset_info`` records and never reruns preprocessing,
inference, ComBat, community detection or simulation. Original analysis
facts stay separate from the display choices of the moment (grid cell,
current widgets); unknown facts are reported as unknown rather than
reconstructed from today's defaults. Only an explicit allowlist of
result/run/source artifacts is archived — no session state, phenotype
rows, subject time series or credentials.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from apps.utils.result_views import (
    build_cnbs_block_figure,
    build_connectivity_maps_figure,
    build_network_summary_figure,
    build_null_max_figure,
    describe_result_scale_and_stat,
    get_2d_result,
    prepare_cnbs_blocks_table,
    prepare_network_summary,
    prepare_node_summary,
    unwrap_result,
)

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_KIND = "conninfpy-analysis-archive"

# Fixed ZIP member timestamp keeps byte-identical archives for identical
# captured facts (the content itself carries the real run times).
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# Honest wording for an export that deliberately omits the trusted pickle
# (default workspace saves). It must never read as "this result lacked".
_PICKLE_FREE_ARCHIVE_NOTE = (
    "bundle.pkl omitted: this export is pickle-free by design (workspace save); "
    "no trusted bundle was requested."
)
_PICKLE_FREE_METHODS_REASON = (
    "No bundle.pkl is included: this export is pickle-free by design (workspace "
    "save). Arrays, tables and the workspace/ layer remain a faithful record of "
    "the run, and workspace restore never needs a pickle."
)

_BUILTIN_ATLAS_NAMES = {
    "Schaefer-100 Yeo7": "schaefer_100_yeo7",
    "Schaefer-200 Yeo-7": "schaefer_200_yeo7",
    "Schaefer-400 Yeo-7": "schaefer_400_yeo7",
    "BNA-246": "bna_246",
}

_SECRET_KEY_RE = re.compile(r"(password|passwd|secret|token|api[_-]?key|credential)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._\-]{16,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
)


def _redact(value: Any) -> Tuple[Any, int]:
    """Drop obviously secret-looking keys/values; return (value, n_redactions).

    A conservative guard against credentials riding along in captured
    settings; everything else is preserved verbatim.
    """
    if isinstance(value, Mapping):
        out = {}
        n = 0
        for key, item in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                n += 1
                continue
            cleaned, sub = _redact(item)
            n += sub
            out[key] = cleaned
        return out, n
    if isinstance(value, (list, tuple)):
        out = []
        n = 0
        for item in value:
            cleaned, sub = _redact(item)
            n += sub
            out.append(cleaned)
        return (out if isinstance(value, list) else tuple(out)), n
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return "<redacted possible credential>", 1
    return value, 0


def _jsonable(value: Any) -> Any:
    """Convert captured values to JSON-safe primitives without inventing data."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        if value.size > 500:
            return {"shape": list(value.shape), "dtype": str(value.dtype), "omitted": "array"}
        return [_jsonable(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None:
        return None
    if isinstance(value, (datetime, Path)):
        return str(value)
    return str(value)


def _dumps_json(payload: Any) -> bytes:
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _dumps_yaml(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        _jsonable(payload), sort_keys=False, default_flow_style=False, allow_unicode=True
    ).encode("utf-8")


def _fig_bytes(fig: Any) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    return buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Captured facts
# ---------------------------------------------------------------------------

def _inner(result: Any) -> Any:
    return unwrap_result(result)


def _resolve_display_cell(result: Any, grid_cell: Optional[int]) -> Tuple[Any, Optional[int], Optional[Dict[str, Any]]]:
    """The 2-D projection used for figures, plus which grid cell it is."""
    inner = _inner(result)
    if not getattr(inner, "is_grid", False):
        return inner, None, None
    idx = 0 if grid_cell is None else int(grid_cell)
    e_grid, h_grid = getattr(inner, "e_grid", None), getattr(inner, "h_grid", None)
    projected, _ = get_2d_result(inner, idx)
    cell = {
        "grid_cell": idx,
        "e": None if e_grid is None or idx >= len(e_grid) else float(e_grid[idx]),
        "h": None if h_grid is None or idx >= len(h_grid) else float(h_grid[idx]),
        "n_cells": 0 if e_grid is None else int(len(e_grid)),
    }
    return projected, idx, cell


def _fisher_summary(record: Any) -> Dict[str, Any]:
    """Human- and YAML-usable summary of the captured Fisher record."""
    from conninfpy._prepare import normalize_fisher_record

    try:
        normalized = normalize_fisher_record(record)
    except Exception:
        normalized = None
    if normalized is None:
        return {"available": False, "applied": [], "upstream": "unknown"}
    applied = []
    for entry in normalized.get("applied") or []:
        params = entry.get("parameters") or {}
        applied.append({
            "stage": entry.get("stage"),
            "scope": entry.get("scope"),
            "source": entry.get("source"),
            "policy": entry.get("policy"),
            "parameters": dict(params),
        })
    upstream = normalized.get("upstream")
    if upstream == "unknown" or upstream is None:
        upstream_summary: Any = "unknown" if upstream == "unknown" else None
    else:
        upstream_summary = [
            {"stage": e.get("stage"), "scope": e.get("scope"), "source": e.get("source"),
             "policy": e.get("policy"), "parameters": dict(e.get("parameters") or {})}
            for e in upstream
        ]
    return {"available": True, "applied": applied, "upstream": upstream_summary}


# Legacy run_plan summaries stored human-readable labels, not raw kwargs.
_LEGACY_OPERATOR_LABELS = {
    "E": "e", "H": "h", "start threshold": "start_thres",
    "minimum block size": "min_cluster_size", "normalization": "normalization",
    "tau": "threshold", "component statistic": "nbs_stat",
}


def _operator_parameters(run_plan: Mapping[str, Any]) -> Dict[str, Any]:
    """The actual method/operator arguments of the run (new captures first)."""
    if "operator_kwargs" in run_plan:
        captured = run_plan.get("operator_kwargs")
        if isinstance(captured, Mapping):
            return dict(captured)   # possibly {}: the run passed no operator arguments
    legacy = run_plan.get("method_parameters")
    if isinstance(legacy, Mapping):
        out: Dict[str, Any] = {}
        for key, value in legacy.items():
            if key in _LEGACY_OPERATOR_LABELS:
                out[_LEGACY_OPERATOR_LABELS[key]] = value
            elif key == "integration" and value == "exact":
                out["n"] = None   # 'exact' means n was None
        return out
    return {}


def _dataset_files(dataset_info: Mapping[str, Any], project_root: Optional[Path]) -> Dict[str, Any]:
    """Classify the captured source paths as available / external / missing."""
    provenance = dataset_info.get("loader_provenance") or {}
    candidates: Dict[str, str] = {}
    source = str(dataset_info.get("source") or "")
    manifest_like = dataset_info.get("manifest_file") or (
        source if re.search(r"\.ya?ml$", source) else None
    )
    if manifest_like:
        candidates["manifest"] = str(manifest_like)
    for key in ("data_path", "pheno_csv_path", "atlas_metadata", "atlas"):
        value = provenance.get(key)
        if isinstance(value, (str, Path)) and value:
            candidates[key] = str(value)
    root = Path(project_root) if project_root is not None else Path.cwd()
    files: Dict[str, Dict[str, Any]] = {}
    for name, raw in candidates.items():
        path = Path(raw)
        resolved = path if path.is_absolute() else (root / path)
        exists = resolved.exists()
        files[name] = {
            "path": raw,
            "resolved": str(resolved),
            "exists": bool(exists),
            "location": "project" if not path.is_absolute() else (
                "inside-project" if str(resolved).startswith(str(root)) else "external"
            ),
        }
    return files


# ---------------------------------------------------------------------------
# Replay configuration
# ---------------------------------------------------------------------------

def _manifest_reference(dataset_info: Mapping[str, Any]) -> Optional[str]:
    """The manifest path the CLI can load, when it was actually captured."""
    source = str(dataset_info.get("source") or "")
    if dataset_info.get("manifest_file"):
        return str(dataset_info["manifest_file"])
    if re.search(r"\.ya?ml$", source):
        return source
    choice = str(dataset_info.get("dataset_choice") or "")
    if choice == "ABIDE-mini":
        # The application's built-in loader applies the same loader and
        # missing-data policy as this bundled manifest.
        return "datasets/abide_mini.yaml"
    if choice == "OpenClose":
        return {"OpenClose": "datasets/open_close.yaml",
                "China-CloseClose": "datasets/china_close_close.yaml"}.get(
            str(dataset_info.get("cohort") or ""), None)
    return None


def _result_sourced_operator_args(result: Any, method: Any) -> Dict[str, Any]:
    """Operator values the finished result genuinely records on itself.

    An explicit, narrow contract: an NBS result carries its threshold and
    component statistic; a parameter-grid result carries its (E, H)
    coordinates. Nothing else is guessed from a result.
    """
    if result is None:
        return {}
    inner = _inner(result)
    out: Dict[str, Any] = {}
    if str(method) == "nbs":
        report = getattr(inner, "component_report", None)
        if isinstance(report, Mapping):
            if report.get("threshold") is not None:
                out["threshold"] = float(report["threshold"])
            if report.get("stat_type") is not None:
                out["nbs_stat"] = str(report["stat_type"])
    e_grid, h_grid = getattr(inner, "e_grid", None), getattr(inner, "h_grid", None)
    if e_grid is not None and h_grid is not None:
        out["e"] = [float(x) for x in np.asarray(e_grid).tolist()]
        out["h"] = [float(x) for x in np.asarray(h_grid).tolist()]
    return out


def _manifest_dependency_state(manifest_path: Path) -> Dict[str, Any]:
    """Existence of the manifest's declared path dependencies (text only).

    Reads the manifest YAML itself — never the datasets — so an export-time
    check stays cheap. Returns per-path state and an ``error`` when the
    manifest cannot be parsed.
    """
    out: Dict[str, Any] = {"paths": {}, "error": None}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as exc:
        out["error"] = f"manifest is not readable YAML ({exc})"
        return out
    if not isinstance(doc, Mapping):
        out["error"] = "manifest is not a YAML mapping"
        return out
    paths = doc.get("paths")
    base = manifest_path.parent
    if isinstance(paths, Mapping):
        for key, raw in paths.items():
            if not isinstance(raw, (str, Path)) or not str(raw).strip():
                continue
            p = Path(str(raw))
            resolved = p if p.is_absolute() else base / p
            out["paths"][key] = {"path": str(resolved), "exists": bool(resolved.exists())}
    return out


def build_replay_config(
    run_plan: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
    *,
    project_root: Optional[Path] = None,
    result: Any = None,
) -> Dict[str, Any]:
    """Build the CLI replay config and its honest status.

    Status is ``replayable`` only when every ingredient of the finished run
    (data source, design coding, preprocessing policy, operator arguments,
    RNG and directional family) is either expressed in the YAML or already
    embodied in the referenced manifest. Otherwise the status names what is
    missing; nothing is silently defaulted. ``result`` (the finished run)
    is an explicit, documented fallback source: values a result genuinely
    records on itself (an NBS threshold, the (E, H) grid coordinates, cNBS
    block labels, the missing-data policy) may be sourced from it; values
    it does not carry stay blockers, never guesses.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    reasons: List[str] = []
    prereqs: List[Dict[str, str]] = []
    sourced_from_result: List[str] = []

    if not isinstance(run_plan, Mapping) or not run_plan.get("method"):
        return {"status": "not-replayable",
                "reasons": ["the finished run did not record its launch settings (run_plan)"],
                "config": None}

    mode = str(dataset_info.get("mode") or "")
    if mode == "Generate Synthetic Dataset":
        return {
            "status": "not-replayable",
            "reasons": [
                "synthetic dataset: the generated observations are not part of the archive "
                "(subject data are never archived); the CLI has no dataset generator. "
                "Regenerate in the application from the captured generation parameters, "
                "then replay.",
            ],
            "config": None,
            "generation_parameters": dataset_info.get("generation_parameters", {}),
        }

    manifest = _manifest_reference(dataset_info)
    if manifest is None:
        return {"status": "requires-input",
                "reasons": ["the data source of the finished run was not captured "
                            "(no manifest path in the dataset record)"],
                "config": None}
    resolved = (Path(manifest) if Path(manifest).is_absolute() else root / manifest).resolve()
    # The YAML keeps the resolved absolute path: the CLI resolves `data:`
    # relative to the analysis file, so a relative captured reference would
    # only replay from one particular extraction directory.
    manifest_ref = str(resolved)
    if not resolved.exists():
        prereqs.append({"kind": "missing-file", "path": manifest_ref,
                        "detail": "referenced data manifest is not available at export time"})
        reasons.append(f"data manifest {manifest_ref!r} is missing at export time; "
                       "provide it at this path before replaying")
    else:
        dep_state = _manifest_dependency_state(resolved)
        if dep_state["error"] is not None:
            reasons.append(f"the data manifest cannot be read as a dataset manifest: "
                           f"{dep_state['error']}")
            return {"status": "not-replayable", "reasons": reasons, "config": None,
                    "prerequisites": prereqs}
        for key, state in sorted(dep_state["paths"].items()):
            if not state["exists"]:
                prereqs.append({"kind": "missing-file", "path": state["path"],
                                "detail": f"manifest dependency paths.{key} is not available "
                                          f"at export time"})
                reasons.append(f"manifest dependency paths.{key} ({state['path']!r}) is "
                               "missing at export time")
    # Captured loader paths are part of the same dependency set even when the
    # manifest does not declare them (e.g. built-in flows that record the
    # absolute files they actually read).
    provenance = dataset_info.get("loader_provenance") or {}
    for key in ("data_path", "pheno_csv_path", "atlas_metadata", "atlas"):
        raw = provenance.get(key)
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            continue
        p = Path(str(raw))
        candidate = (p if p.is_absolute() else root / p).resolve()
        if not candidate.exists():
            prereqs.append({"kind": "missing-file", "path": str(candidate),
                            "detail": f"captured source file ({key}) is not available "
                                      f"at export time"})
            reasons.append(f"captured source file {key} ({str(candidate)!r}) is missing "
                           "at export time")

    # ---- connectivity / Fisher policy -------------------------------------
    input_kind = run_plan.get("input")
    scale_map = {"correlation": "r", "fisher_z": "z"}
    if input_kind not in ("r", "z"):
        reasons.append("the input scale of the finished run was not recorded")
        return {"status": "not-replayable", "reasons": reasons, "config": None, "prerequisites": prereqs}
    connectivity_block: Dict[str, Any] = {}
    fisher_summary = _fisher_summary(run_plan.get("fisher"))
    # Transforms this library applied BEFORE analyze (the application's
    # ingestion option / design button) are recorded as *upstream* entries on
    # the finished result; analyze's own applied entries are reproduced by
    # analyze itself on replay and need no adapter.
    upstream_entries = fisher_summary.get("upstream") or []
    upstream_entries = [e for e in upstream_entries if isinstance(e, Mapping)]
    if input_kind == "z":
        if not upstream_entries:
            provenance_kind = (dataset_info.get("data_kind")
                               or (dataset_info.get("loader_provenance") or {}).get("connectivity", {}).get("kind"))
            if provenance_kind in ("fisher_z", "z"):
                pass  # the loaded matrices were already on the z scale; nothing to replay
            elif fisher_summary.get("upstream") == "unknown":
                reasons.append(
                    "the run analysed Fisher-z input whose upstream transform history is "
                    "unknown; the z scale cannot be reproduced from the source data")
                return {"status": "not-replayable", "reasons": reasons, "config": None,
                        "prerequisites": prereqs}
            else:
                reasons.append(
                    "the run analysed Fisher-z input but no applied transform was captured; "
                    "how the z scale was obtained is unknown")
                return {"status": "not-replayable", "reasons": reasons, "config": None,
                        "prerequisites": prereqs}
        else:
            if len(upstream_entries) != 1:
                reasons.append("more than one upstream Fisher transform was captured; "
                               "the replay adapter supports exactly one")
                return {"status": "not-replayable", "reasons": reasons, "config": None,
                        "prerequisites": prereqs}
            applied_entry = upstream_entries[0]
            policy = applied_entry.get("policy")
            params = applied_entry.get("parameters") or {}
            if policy == "clip_r":
                if "clip_r" not in params:
                    reasons.append("the captured clip_r Fisher policy does not record its "
                                   "actual clip_r value; it is not invented at export time")
                    return {"status": "not-replayable", "reasons": reasons, "config": None,
                            "prerequisites": prereqs}
                connectivity_block["fisher"] = {"policy": "clip_r",
                                                "clip_r": float(params["clip_r"])}
            elif policy == "clip_z":
                if "max_z" not in params:
                    reasons.append("the captured clip_z Fisher policy does not record its "
                                   "actual max_z value; it is not invented at export time")
                    return {"status": "not-replayable", "reasons": reasons, "config": None,
                            "prerequisites": prereqs}
                connectivity_block["fisher"] = {"policy": "clip_z",
                                                "max_z": float(params["max_z"])}
            elif policy == "unbounded":
                connectivity_block["fisher"] = {"policy": "unbounded"}
            else:
                reasons.append(f"unknown Fisher policy {policy!r} in the captured history")
                return {"status": "not-replayable", "reasons": reasons, "config": None,
                        "prerequisites": prereqs}
    elif upstream_entries:
        reasons.append("an upstream Fisher transform was captured but the run used raw "
                       "correlations; this combination is inconsistent and cannot be replayed")
        return {"status": "not-replayable", "reasons": reasons, "config": None, "prerequisites": prereqs}

    provenance = dataset_info.get("loader_provenance") or {}
    conn_provenance = provenance.get("connectivity") or {}
    if conn_provenance.get("kind") == "spearman":
        reasons.append("the connectivity matrices were Spearman rank correlations computed "
                       "in the application at load time; the CLI has no rank pipeline")
        return {"status": "not-replayable", "reasons": reasons, "config": None, "prerequisites": prereqs}
    # The construction block is emitted from the captured loader record
    # whenever it exists (only timeseries-constructed sources carry it) —
    # independently of the Fisher adapter above: partial correlations with
    # an upstream transform replay as partial + transform, not as Pearson.
    if conn_provenance.get("kind") in ("correlation", "partial_correlation"):
        kind = "correlation" if conn_provenance.get("kind") == "correlation" else "partial_correlation"
        connectivity_block["kind"] = kind
        shrinkage = conn_provenance.get("shrinkage")
        if shrinkage is not None:
            connectivity_block["shrinkage"] = shrinkage
        if conn_provenance.get("standardize") is not None:
            connectivity_block["standardize"] = bool(conn_provenance.get("standardize"))

    # ---- design ------------------------------------------------------------
    family = run_plan.get("design_family")
    question = run_plan.get("question_type")
    design: Dict[str, Any] = {}
    if family == "two-sample":
        design["type"] = "two-sample"
        design["group"] = run_plan.get("group_col")
        design["levels"] = [run_plan.get("reference_group"), run_plan.get("target_group")]
    elif family == "glm":
        design["type"] = "glm"
        if question == "Group Difference":
            design["interest"] = run_plan.get("group_col")
            if run_plan.get("reference_group") is not None:
                design["interest_reference"] = run_plan.get("reference_group")
        else:
            design["interest"] = run_plan.get("interest_var")
        if run_plan.get("confound_vars"):
            design["confounds"] = list(run_plan["confound_vars"])
    elif family == "paired":
        design["type"] = "paired"
        design["condition"] = run_plan.get("condition_col")
        design["subject"] = run_plan.get("subject_col") or "subject_id"
        design["levels"] = [run_plan.get("baseline_condition"), run_plan.get("target_condition")]
        # Condition-specific confounds (values that may differ between the two
        # conditions). The CLI reads exactly this key for a paired design and
        # subtracts B - A itself, so replay keeps the same partialling-out.
        # Strict subject-ID bijection makes the alignment deterministic:
        # condition A row order, condition B matched by ID.
        condition_confounds = run_plan.get("condition_confound_cols")
        if condition_confounds:
            design["confounds"] = list(condition_confounds)
    else:
        reasons.append(f"the design family {family!r} of the finished run has no CLI mapping")
        return {"status": "not-replayable", "reasons": reasons, "config": None, "prerequisites": prereqs}

    if any(v is None for key, v in design.items() if key != "confounds") or (
            design.get("type") in ("two-sample", "paired") and None in (design.get("levels") or [])):
        reasons.append("design columns/levels of the finished run were not fully recorded")
        return {"status": "not-replayable", "reasons": reasons, "config": None, "prerequisites": prereqs}
    if run_plan.get("site_col"):
        design["sites"] = run_plan.get("site_col")
    if "harmonize" in run_plan:
        design["harmonize"] = run_plan.get("harmonize")

    # ---- inference ---------------------------------------------------------
    # Required launch facts: an unknown RNG, directional family or variance
    # choice is a blocker, never an invitation to replay with today's
    # defaults.
    if run_plan.get("seed") is None:
        reasons.append("the inference RNG seed was not recorded for this finished run; "
                       "an identical replay cannot be promised")
        return {"status": "not-replayable", "reasons": reasons, "config": None,
                "prerequisites": prereqs}
    if run_plan.get("tested_tail") not in (None, "positive", "negative") or (
            run_plan.get("tested_tail") is None and not isinstance(run_plan.get("two_tailed"), bool)):
        reasons.append("the directional family (two_tailed / tested_tail) of the run was "
                       "not recorded")
        return {"status": "not-replayable", "reasons": reasons, "config": None,
                "prerequisites": prereqs}
    if not isinstance(run_plan.get("n_permutations"), (int, np.integer)) or run_plan["n_permutations"] < 0:
        reasons.append("the permutation count of the run was not recorded")
        return {"status": "not-replayable", "reasons": reasons, "config": None,
                "prerequisites": prereqs}
    if family == "two-sample" and run_plan.get("t_statistic") not in ("student", "welch"):
        reasons.append("the two-sample variance choice (Student/Welch) of the run was "
                       "not recorded")
        return {"status": "not-replayable", "reasons": reasons, "config": None,
                "prerequisites": prereqs}
    inference: Dict[str, Any] = {
        "method": run_plan.get("method"),
        "n_permutations": int(run_plan["n_permutations"]),
        "rng": run_plan.get("seed"),
        "use_mp": bool(run_plan.get("use_mp", False)),
    }
    if run_plan.get("tested_tail"):
        inference["tested_tail"] = run_plan.get("tested_tail")
    else:
        inference["two_tailed"] = bool(run_plan.get("two_tailed"))
    if run_plan.get("acceleration") is not None:
        inference["acceleration"] = run_plan.get("acceleration")
    if family == "two-sample":
        inference["var_equal"] = run_plan.get("t_statistic") == "student"
    if input_kind == "r":
        # The application analyses raw correlations as loaded (r_to_z=False);
        # the CLI default would Fisher-transform them once.
        inference["r_to_z"] = False

    # Operator arguments: captured kwargs first; legacy run_plan summaries
    # (label keys) are normalized back; otherwise values the finished result
    # genuinely records on itself may be sourced from it, once each, under an
    # explicit contract. Anything else stays a blocker.
    operator = _operator_parameters(run_plan)
    sourced = _result_sourced_operator_args(result, run_plan.get("method"))
    for key in ("e", "h", "n", "start_thres", "threshold", "nbs_stat",
                "min_cluster_size", "normalization", "stat_type"):
        if key in operator and key not in ("_source",):
            inference[key] = operator[key]
        elif key in sourced and key not in inference:
            inference[key] = sourced[key]
            sourced_from_result.append(f"inference.{key}")
    if str(run_plan.get("method")) == "nbs" and "threshold" not in inference:
        reasons.append("the NBS threshold of the run was not captured and is not "
                       "recorded on the result")
        return {"status": "not-replayable", "reasons": reasons, "config": None,
                "prerequisites": prereqs}

    # Network labels: replay always uses the captured vector — the actual
    # partition in original ROI order — regardless of whether it came from an
    # SC-prior Louvain run or from atlas network labels; today's atlas
    # selection is never substituted for it.
    net_labels = run_plan.get("net_labels") if isinstance(run_plan.get("net_labels"), Mapping) else None
    labels = net_labels.get("labels") if net_labels else None
    label_source = net_labels.get("source") if net_labels else None
    if not (isinstance(labels, (list, tuple)) and labels):
        labels = None
        if result is not None and str(run_plan.get("method")) == "cnbs":
            block_report = getattr(_inner(result), "block_report", None)
            node_labels = (block_report or {}).get("node_labels")
            if isinstance(node_labels, (list, tuple)) and node_labels:
                labels = [int(x) for x in node_labels]
                label_source = "result_block_report"
                sourced_from_result.append("inference.net_labels")
    method_needs_labels = str(run_plan.get("method")) in ("ni_tfnbs", "fbc_tfnbs", "cnbs")
    if method_needs_labels:
        if labels:
            inference["net_labels"] = [int(x) for x in labels]
        elif net_labels is not None and label_source not in ("sc_prior", "atlas_networks"):
            reasons.append("network labels were used but their source was not recorded")
            return {"status": "not-replayable", "reasons": reasons, "config": None,
                    "prerequisites": prereqs}
        else:
            reasons.append("the run used network labels (SC-prior Louvain or atlas networks) "
                           "but the actual label vector was not captured")
            return {"status": "not-replayable", "reasons": reasons, "config": None,
                    "prerequisites": prereqs}

    # Missing-data policy: captured fact, or the result's own record (a
    # missing_report means available_case ran; its absence means the run
    # raised on missing values — analyze's documented default).
    if "missing_policy" in run_plan:
        policy = run_plan.get("missing_policy")
    elif result is not None:
        policy = "available_case" if getattr(_inner(result), "missing_report", None) is not None else "raise"
        sourced_from_result.append("inference.missing (policy)")
    else:
        policy = None
    if policy == "available_case":
        captured_coverage = run_plan.get("min_coverage")
        if not isinstance(captured_coverage, (int, float, np.floating)):
            # Legacy plan: the GUI did not capture the coverage floor, so the
            # per-edge tested family cannot be reproduced.
            reasons.append("the run used available-case missing handling but its "
                           "min_coverage was not captured")
            return {"status": "not-replayable", "reasons": reasons, "config": None,
                    "prerequisites": prereqs}
        inference["missing"] = "available_case"
        inference["min_coverage"] = float(captured_coverage)
    elif policy is not None and policy != "raise":
        reasons.append(f"unknown missing-data policy {policy!r}")
        return {"status": "not-replayable", "reasons": reasons, "config": None,
                "prerequisites": prereqs}

    # Declared primary (E, H) cell of a sensitivity grid: a reporting fact fixed
    # before the run, never re-derived from the archived p-value arrays. The
    # operator arguments above keep the full grid so the replay reproduces every
    # cell; the confirmatory cell is named separately.
    declared_primary = None
    primary_cell = run_plan.get("primary_cell")
    if isinstance(primary_cell, Mapping):
        declared_primary = {
            "e": primary_cell.get("e"),
            "h": primary_cell.get("h"),
            "param_idx": primary_cell.get("param_idx"),
            "selection": primary_cell.get("selection"),
            "note": ("cell declared before the run; every other grid cell is a "
                     "descriptive sensitivity analysis"),
        }

    # Tested-family mask (edge family): the exact array is bundled with the
    # archive, so a replay can reproduce the tested family instead of silently
    # testing all edges.
    edge_mask_fact = run_plan.get("edge_mask")
    if isinstance(edge_mask_fact, Mapping):
        mask_path = "arrays/edge_family_mask.npy"
        prereqs.append({
            "kind": "edge-family-mask",
            "path": mask_path,
            "detail": ("the tested edge family of the finished run is the boolean "
                       f"{edge_mask_fact.get('shape')} mask bundled as {mask_path} "
                       f"({edge_mask_fact.get('eligible_edges')} eligible edges, sha256 "
                       f"{edge_mask_fact.get('sha256')}, source "
                       f"{edge_mask_fact.get('source_filename') or 'not recorded'}); "
                       "pass it as the method's edge_mask. It is not an ROI subset "
                       "and not an SC community label vector"),
        })
        reasons.append("the run restricted the tested edge family with an edge mask; "
                       "apply the bundled arrays/edge_family_mask.npy as edge_mask "
                       "before replaying")

    roi_indices = run_plan.get("roi_indices")
    if roi_indices:
        prereqs.append({
            "kind": "roi-subset",
            "path": manifest_ref,
            "detail": "the run analysed an explicit ROI subset "
                      f"({len(roi_indices)} of the dataset ROIs); express it via the "
                      "manifest's keep_rois before replaying",
        })
        reasons.append("the run analysed an ROI subset selected in the application; "
                       "apply the same keep_rois in the data manifest before replaying")

    alpha = run_plan.get("alpha")
    cfg: Dict[str, Any] = {"data": manifest_ref, "design": design, "inference": inference}
    if connectivity_block:
        cfg["connectivity"] = connectivity_block
    if alpha is not None:
        cfg["alpha"] = float(alpha)

    status = "replayable" if not reasons else "requires-input"
    out: Dict[str, Any] = {
        "status": status,
        "reasons": reasons,
        "prerequisites": prereqs,
        "config": cfg,
        "command": "conninfpy run analysis.yaml",
    }
    if sourced_from_result:
        out["sourced_from_result"] = sorted(set(sourced_from_result))
    if declared_primary is not None:
        out["declared_primary_cell"] = declared_primary
    return out


# ---------------------------------------------------------------------------
# methods.md
# ---------------------------------------------------------------------------

def _fmt(value: Any, unknown: str = "unknown / not recorded") -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return unknown
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fmt_list(values: Optional[Sequence[Any]]) -> str:
    if not values:
        return "none"
    return ", ".join(_fmt(v, "unknown") for v in values)


def build_methods_markdown(
    result: Any,
    run_plan: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
    scale_meta: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    alpha: Optional[float],
    display_cell: Optional[Mapping[str, Any]],
    fisher_summary: Mapping[str, Any],
    dataset_files: Mapping[str, Any],
    effect_mask_available: bool,
    bundle_available: bool = True,
    bundle_omitted_reason: Optional[str] = None,
) -> str:
    """Deterministic methods text built only from captured facts."""
    inner = _inner(result)
    lines: List[str] = []
    plan = dict(run_plan or {})
    info = dict(dataset_info or {})
    add = lines.append

    add("# Analysis methods")
    add("")
    add(f"Archive schema: {ARCHIVE_KIND} v{ARCHIVE_SCHEMA_VERSION}. "
        "This file is generated deterministically from the facts captured at "
        "launch/load time; values that were not captured are stated as unknown.")
    add("")

    # --- run identity ---
    add("## Run")
    add("")
    add(f"- Started (captured at launch): {_fmt(plan.get('started_at'))}")
    add(f"- Wall time: {_fmt(plan.get('wall_time_s'))} s")
    add(f"- conninfpy version at run: {_fmt(plan.get('package_version'))}")
    # Backend facts, kept distinct: which enhancement kernels a method runs
    # at all, what actually executed (only a genuine runtime record may say
    # — none exists today), and the numba installation fact of the run
    # environment. A legacy ambiguous label is qualified, never relabelled.
    method_key = str(inner.method).lower()
    no_enhancement_methods = {"tstat", "bonferroni", "bh_fdr", "bh_fdr_perm", "nbs"}
    backend_capture = plan.get("enhancement_backend", "__absent__")
    legacy_label = backend_capture if backend_capture in ("numba", "scipy") else None
    if method_key in no_enhancement_methods:
        line = ("- Enhancement backend: not applicable — this method runs no "
                "enhancement-kernel computation (component labelling uses SciPy).")
        if legacy_label:
            line += (f" (An earlier capture recorded '{legacy_label}', but that label "
                     "measured module availability, not execution.)")
        add(line)
    elif legacy_label:
        add(f"- Enhancement backend (executed): unknown / not recorded. An earlier "
            f"capture recorded '{legacy_label}', but that label measured module "
            "availability in the run environment, not which backend executed.")
    else:
        add("- Enhancement backend (executed): unknown / not recorded (the "
            "enhancement kernels do not record which backend executed).")
    if "numba_module_found" in plan:
        add(f"- numba module present in the run environment (installation fact, "
            f"distinct from the executed backend): "
            f"{'yes' if plan['numba_module_found'] else 'no'}.")
    add(f"- Multiprocessing: {_fmt(plan.get('use_mp'))}")
    add(f"- Random seed (inference RNG): {_fmt(plan.get('seed'))}")
    add("")
    add(f"- Dataset: {_fmt(info.get('name'))}")
    add(f"- Design family: {_fmt(plan.get('design_family'))}; question: {_fmt(plan.get('question_type'))}")
    add(f"- Method (enhancement operator): {_fmt(plan.get('method'))}")
    add(f"- Run α: {_fmt(alpha)}" if alpha is not None else "- Run α: unknown / not recorded")
    add("")

    # --- data ---
    add("## Data and provenance")
    add("")
    add(f"- Observations loaded: {_fmt(info.get('observations'))}; ROIs: {_fmt(info.get('n_rois'))}; "
        f"subjects: {_fmt(info.get('subjects'))}")
    add(f"- Atlas (display metadata): {_fmt(info.get('atlas'))}")
    add(f"- Input data kind: {_fmt(plan.get('data_kind'))}; analysed scale: {_fmt(plan.get('analysis_scale'))}")
    if plan.get("roi_indices"):
        add(f"- ROI subset: the run analysed {len(plan['roi_indices'])} explicitly selected ROIs "
            "(captured in run_plan.json).")
    for name, entry in sorted(dataset_files.items()):
        state = ("available" if entry["exists"] else
                 ("external" if entry["location"] == "external" else "missing"))
        add(f"- Source file ({name}): `{entry['path']}` — {state} at export time")
    generation = info.get("generation_parameters") if isinstance(info.get("generation_parameters"), Mapping) else None
    if info.get("mode") == "Generate Synthetic Dataset":
        add("- Synthetic dataset: generation parameters below are the actual generator "
            "inputs captured when the dataset was created (seed included); they are "
            "kept separate from the inference RNG.")
        if generation:
            for key in sorted(generation):
                add(f"  - {key}: {_fmt(generation[key])}")
        else:
            add("  - generation parameters: unknown / not recorded")
    add("")
    add("### Preprocessing facts")
    add("")
    preprocessing = plan.get("preprocessing")
    if isinstance(preprocessing, Mapping) and preprocessing:
        from apps.utils.helpers import describe_per_site_facts, describe_preprocessing_facts

        for label, value, source in describe_preprocessing_facts(preprocessing):
            add(f"- {label}: {value} — {source}")
        per_site = describe_per_site_facts(preprocessing)
        if per_site:
            add("- Per-site facts (no common value is claimed across sites):")
            for site, value, source in per_site:
                add(f"  - Site `{site}` GSR: {value} — {source}")
    else:
        add("- No preprocessing facts were captured (each fact would carry its own "
            "source; unknown stays unknown).")
    add("")
    add("### Fisher transform history")
    add("")
    applied = list(fisher_summary.get("applied") or [])
    if applied:
        for entry in applied:
            params = entry.get("parameters") or {}
            param_text = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(params.items())) or "default parameters"
            add(f"- Applied transform: stage `{_fmt(entry.get('stage'))}`, scope "
                f"`{_fmt(entry.get('scope'))}`, policy `{_fmt(entry.get('policy'))}` ({param_text}); "
                f"applied by {_fmt(entry.get('source'))}.")
    else:
        add("- No Fisher r-to-z transform was applied inside the analyze() call "
            "itself.")
    upstream = fisher_summary.get("upstream")
    if upstream == "unknown":
        add("- Upstream transforms: unknown (the loaded data carry no history of "
            "transforms applied before loading).")
    elif upstream:
        for entry in upstream:
            params = entry.get("parameters") or {}
            param_text = (", ".join(f"{k}={_fmt(v)}" for k, v in sorted(params.items()))
                          or "no parameters recorded")
            add(f"- Upstream transform (reported by the data source): stage "
                f"`{_fmt(entry.get('stage'))}`, policy `{_fmt(entry.get('policy'))}` "
                f"({param_text}), source {_fmt(entry.get('source'))}.")
    add("")

    # --- design ---
    add("## Design and contrast")
    add("")
    question = plan.get("question_type")
    if question == "Group Difference":
        add(f"- Two groups from column `{_fmt(plan.get('group_col'))}`: reference "
            f"`{_fmt(plan.get('reference_group'))}`, target `{_fmt(plan.get('target_group'))}`. "
            "The contrast is target − reference; the positive statistical tail is the "
            "target group showing higher connectivity.")
    elif question == "Paired Condition":
        add(f"- Paired conditions from column `{_fmt(plan.get('condition_col'))}`: baseline "
            f"`{_fmt(plan.get('baseline_condition'))}`, target `{_fmt(plan.get('target_condition'))}` "
            f"(subject column `{_fmt(plan.get('subject_col'))}`). The contrast is target − baseline "
            "(B − A) within subject.")
    elif question == "Continuous Predictor":
        add(f"- Continuous predictor `{_fmt(plan.get('interest_var'))}` "
            "(positive tail: higher connectivity with higher predictor values).")
    if plan.get("confound_vars"):
        add(f"- Nuisance covariates: {_fmt_list(plan.get('confound_vars'))} "
            "(numeric columns as loaded; no re-coding was applied by the application).")
    if plan.get("site_col"):
        add(f"- Acquisition site column: `{_fmt(plan.get('site_col'))}`.")
        add(f"- Site recipe requested: {_fmt(plan.get('site_recipe'))}; "
            f"effective: {_fmt(plan.get('effective_site_recipe'))}.")
        harmonize = plan.get("harmonize", "__absent__")
        if harmonize == "__absent__":
            add("- Requested harmonize argument: unknown / not recorded.")
        elif harmonize is None:
            add("- Requested harmonize argument: None — no ComBat/site-dummy "
                "harmonization was requested (site labels, when given, still drive "
                "stratified permutation and the resolved strategy below).")
        else:
            add(f"- Requested harmonize argument: `{harmonize}`.")
        resolved = getattr(inner, "resolved_strategy", None)
        outer = getattr(result, "resolved_strategy", None)
        resolved = outer if outer is not None else resolved
        add(f"- Backend-resolved strategy: {_fmt(resolved)}.")
    else:
        add("- No site column was used; no site handling was requested.")
    if plan.get("t_statistic"):
        statistic_names = {"student": "Student's t (pooled variance)",
                           "welch": "Welch's t (separate variances)",
                           "glm": "GLM contrast with pooled residual variance"}
        add(f"- Test statistic: {statistic_names.get(plan['t_statistic'], _fmt(plan['t_statistic']))}.")
    missing_scope = getattr(inner, "missing_report", None)
    if missing_scope is not None:
        coverage = plan.get("min_coverage")
        coverage_text = (
            f"The captured coverage floor is {float(coverage):.0%}. "
            if isinstance(coverage, (int, float, np.floating)) else
            "The coverage floor was not captured for this run. "
        )
        add("- Missing data: analysed per-edge with available subjects (available_case); "
            "edges below the coverage threshold were not tested (their p-values are NaN). "
            f"{coverage_text}Counts are in the bundle's missing_report. The tested n "
            "therefore varies per edge and is never reported as one complete-case sample; "
            "the coverage rule adds no statistical correction.")
    else:
        add("- Missing data policy: the run stopped on missing connectivity values "
            "(missing='raise'); no missing values were present in the analysed arrays.")
    add("")

    # --- inference ---
    add("## Inference")
    add("")
    add(f"- Method: `{_fmt(inner.method)}`; inference unit: `{_fmt(getattr(inner, 'inference_unit', None))}`.")
    add(f"- Correction at the native unit: `{_fmt(getattr(inner, 'correction', None))}`.")
    family = getattr(inner, "tail_policy", None)
    add(f"- Directional family: `{_fmt(family)}`.")
    tested = getattr(inner, "tested_tail", None)
    selection = getattr(inner, "tail_selection", None)
    if tested:
        origin = {"prespecified": "fixed before the run",
                  "post-hoc": "selected after the run (post hoc)"}.get(selection, _fmt(selection))
        add(f"- Tested direction: `{tested}` ({origin}); the other direction was not tested.")
    if alpha is not None:
        add(f"- α (captured at launch, applies to the family above): {alpha:g}.")
    add(f"- Permutations: {_fmt(inner.n_permutations)}; acceleration: "
        f"{_fmt(inner.acceleration, 'none — exact empirical p-values')}.")
    operator = _operator_parameters(plan)
    operator_text = ("default method settings (no operator arguments beyond the "
                     "method itself)" if operator == {} else _fmt(operator))
    add(f"- Operator arguments (as passed to the run): {operator_text}")
    if getattr(inner, "is_grid", False):
        grid_axis = getattr(inner, "e_grid", None)
        cells = 0 if grid_axis is None else int(len(grid_axis))
        shown = "cell 0" if display_cell is None else f"cell {display_cell.get('grid_cell')}"
        add(f"- Parameter grid: the run evaluated {cells} (E, H) cells. The archived "
            f"p-value arrays keep the full grid; the figures show {shown} (a display "
            "fact of the export moment, not an analysis fact).")
        declared = plan.get("primary_cell")
        if isinstance(declared, Mapping) and declared.get("e") is not None:
            add(f"- Declared primary cell (fixed before the run): E={declared.get('e')}, "
                f"H={declared.get('h')} (cell {declared.get('param_idx')}). Only this cell "
                "carries the confirmatory claim; every other cell is a descriptive "
                "sensitivity analysis and the grid forms no corrected family-wise "
                "conclusion for a cell chosen after seeing the data.")
    edge_mask = plan.get("edge_mask")
    if isinstance(edge_mask, Mapping):
        add(f"- Tested edge family: the run analysed a boolean "
            f"{edge_mask.get('shape')} mask ({edge_mask.get('eligible_edges')} eligible "
            f"edges, sha256 {edge_mask.get('sha256')}, source "
            f"{edge_mask.get('source_filename') or 'not recorded'}). The mask restricts "
            "which edges enter the tested family; it is not an ROI exclusion subset and "
            "not an SC community label vector.")
    net_labels = plan.get("net_labels") if isinstance(plan.get("net_labels"), Mapping) else None
    if net_labels:
        source = net_labels.get("source")
        if source == "sc_prior":
            add("- Network labels: computed once from the supplied structural-connectivity "
                "prior by Louvain community detection (seed = the run seed); the actual "
                "label vector used is captured in run_plan.json and was not recomputed "
                "for this export.")
        elif source == "atlas_networks":
            add("- Network labels: integer-coded atlas network labels "
                "(first-appearance order of the atlas partition).")
        labels = net_labels.get("labels")
        if labels:
            add(f"- Label vector: {labels}")
    elif str(plan.get("method")) in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
        add("- Network labels: not recorded for this finished run (legacy run); the "
            "actual partition is not reconstructible from the result alone.")
    add("")

    # --- estimates / units ---
    add("## Estimates, units and uncertainty")
    add("")
    add(f"- Response scale: {_fmt(scale_meta.get('scale_label'))}.")
    add(f"- Reported effect: the native contrast effect c'β in recorded units "
        f"({_fmt(scale_meta.get('effect_unit_label'))}); it is not the t statistic and "
        "not a standardized effect size.")
    if scale_meta.get("scale_note"):
        add(f"- Scale note: {scale_meta['scale_note']}")
    design = getattr(inner, "design", None)
    has_beta = getattr(inner, "beta", None) is not None
    has_effect = has_beta and design is not None
    add(f"- Coefficient maps (β): {'saved' if has_beta else 'not saved on this result'}.")
    add(f"- Native effect map (c'β): {'saved' if has_effect else 'not saved on this result'}.")
    add("- Uncertainty: corrected p-values at the native inference unit. No confidence "
        "intervals, standard errors or standardized effects are computed or reconstructed "
        "from p-values or statistics; none are archived.")
    unit = getattr(inner, "inference_unit", "edge")
    if unit in ("component", "block"):
        add(f"- Because the inference unit is the {unit}, member edges of a detected "
            f"{unit} share its p-value; per-edge and per-node tables are descriptive "
            "projections, not independent edge findings.")
    if getattr(inner, "null_max_dist", None):
        add("- The permutation null max-distribution was saved on the result and is "
            "archived (figures/null_max.png).")
        if str(inner.method).lower() != "tstat":
            add("- The observed maximum on the enhanced score scale was not saved; the "
                "figure omits that reference line instead of substituting a raw statistic.")
    else:
        add("- The permutation null max-distribution was not saved on this result.")
    add("")

    # --- detections ---
    add("## Detections")
    add("")
    if alpha is not None:
        if unit in ("component", "block"):
            report = getattr(inner, "component_report" if unit == "component" else "block_report", None)
            if isinstance(report, Mapping):
                add(f"- Native {unit} detections at α = {alpha:g} are listed in the "
                    f"archived table; each member edge carries its {unit}'s corrected p-value.")
            else:
                add(f"- Native {unit} counts were not recorded on this result.")
        else:
            counts = inner.n_significant(alpha)
            if "omnibus" in counts:
                add(f"- Omnibus F detections at α = {alpha:g}: {counts['omnibus']} edges "
                    "(non-directional).")
            else:
                tested = getattr(inner, "tested_tail", None)
                if tested:
                    add(f"- One-sided ({tested}) detections at α = {alpha:g}: {counts[tested]} "
                        "edges; the other direction was not tested.")
                else:
                    add(f"- Positive-tail detections at α = {alpha:g}: {counts['positive']} edges; "
                        f"negative-tail: {counts['negative']} edges.")
    else:
        add("- The run's α was not captured; no detection counts are claimed.")
    if effect_mask_available:
        add("- Synthetic ground truth: the generator's planted effect mask is archived "
            "(arrays/synthetic_effect_mask.npy). Recovery statistics computed from it are "
            "descriptive edge-membership measures, not estimates of the "
            f"{unit}-level FWER.")
    add("")

    # --- figures ---
    add("## Figures")
    add("")
    if display_cell:
        add(f"- Grid result: figures show parameter-grid cell {display_cell.get('grid_cell')} "
            f"(E = {_fmt(display_cell.get('e'))}, H = {_fmt(display_cell.get('h'))} of "
            f"{_fmt(display_cell.get('n_cells'))} cells); the full grid evidence is in "
            "the archived arrays.")
    add("- figures/connectivity_maps.png: the test-statistic map, the signed −log10 p "
        "map (α contour at the run's α) and — when saved — the native effect map in its "
        "recorded units. ROI ordering follows the atlas network grouping when network "
        "labels are available (matrix order otherwise); the same ordering is used by the "
        "summaries.")
    add("- figures/network_summary.png: descriptive network-pair aggregation of the "
        "corrected detections; no network-level test is performed.")
    if fisher_summary.get("applied") or upstream:
        add("- Fisher-transform provenance is listed above; the exported arrays are on "
            "the analysed scale.")
    add("")

    # --- replay ---
    add("## Reproducing this analysis")
    add("")
    status = replay.get("status")
    if status == "replayable":
        add(f"- analysis.yaml in this archive is a CLI replay configuration: "
            f"`{replay.get('command')}` with the conninfpy version recorded above "
            "reproduces the statistical maps (identical seed). Data prerequisites below.")
        for key in replay.get("sourced_from_result") or []:
            add(f"  - `{key}` in the replay file was sourced from the finished result's "
                "own record (explicit contract), not from the captured launch plan.")
    else:
        add(f"- Replay status: **{status}**. The archived analysis.yaml is not "
            "self-sufficient:")
    for reason in replay.get("reasons") or []:
        add(f"  - {reason}")
    for prereq in replay.get("prerequisites") or []:
        add(f"  - Prerequisite ({prereq.get('kind')}): {prereq.get('detail')} "
            f"(`{prereq.get('path')}`).")
    if status != "replayable":
        add("- The tables, arrays and bundle in this archive remain a faithful record "
            "of the finished run regardless of replayability.")
    add("")

    # --- limits ---
    add("## Limits and unknowns")
    add("")
    add("- Values reported above as 'unknown / not recorded' were not captured at run "
        "time; they are never filled in from today's defaults or widgets.")
    add("- The archive contains no subject-level observations, phenotype rows or time "
        "series; group summaries cannot be recomputed from it, by design.")
    if bundle_available:
        add("- bundle.pkl is a Python pickle for trusted use only "
            "(ResultBundle.load); do not load archives from untrusted sources.")
    else:
        add("- " + (bundle_omitted_reason or (
            "No bundle.pkl is included: this finished result lacks the captured "
            "analysis metadata a trustworthy snapshot requires. Arrays and tables "
            "remain a faithful record of the run.")))
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Archive assembly
# ---------------------------------------------------------------------------

def _arrays_payload(inner: Any, cell_idx: Optional[int]) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    """Corrected evidence, statistic and coefficient arrays as .npy members.

    P-value maps keep the finished run's full parameter grid when there is
    one; ``cell_idx`` is only recorded as the display fact in the index
    (figures show that cell — the analysis facts are never projected).
    """
    files: Dict[str, bytes] = {}
    index: Dict[str, Dict[str, Any]] = {}
    tails = ["omnibus"] if "omnibus" in inner else ["positive", "negative"]
    is_grid = getattr(inner, "is_grid", False)
    for tail in tails:
        arr = np.asarray(inner[tail], dtype=np.float64)
        files[f"arrays/p_{tail}.npy"] = _npy_bytes(arr)
        entry = {"shape": list(arr.shape), "content": "corrected p-values"}
        if is_grid:
            entry["content"] += " (full parameter grid)"
            entry["display_cell"] = cell_idx
        index[f"p_{tail}"] = entry
    stat_map = getattr(inner, "stat_omnibus", None) if "omnibus" in inner else inner.stat_signed
    if stat_map is not None:
        name = "stat_omnibus" if "omnibus" in inner else "stat_signed"
        files[f"arrays/{name}.npy"] = _npy_bytes(np.asarray(stat_map, dtype=np.float64))
        index[name] = {"shape": list(np.asarray(stat_map).shape),
                       "content": "observed signed test statistic"}
    beta = getattr(inner, "beta", None)
    design = getattr(inner, "design", None)
    if beta is not None and design is not None:
        files["arrays/beta.npy"] = _npy_bytes(np.asarray(beta, dtype=np.float64))
        index["beta"] = {"shape": list(beta.shape), "content": "GLM coefficient maps (response scale)",
                         "columns": [str(c) for c in design.get("columns", ())]}
        effect = inner.effect_map() if hasattr(inner, "effect_map") else None
        if effect is not None:
            files["arrays/effect.npy"] = _npy_bytes(np.asarray(effect, dtype=np.float64))
            index["effect"] = {"shape": list(effect.shape),
                               "content": "native contrast effect c'β (recorded units)"}
    e_grid, h_grid = getattr(inner, "e_grid", None), getattr(inner, "h_grid", None)
    if e_grid is not None:
        files["arrays/e_grid.npy"] = _npy_bytes(np.asarray(e_grid, dtype=np.float64))
    if h_grid is not None:
        files["arrays/h_grid.npy"] = _npy_bytes(np.asarray(h_grid, dtype=np.float64))
    if e_grid is not None or h_grid is not None:
        index["e_grid"] = {"shape": list(np.asarray(e_grid).shape) if e_grid is not None else None,
                           "content": "parameter-grid E coordinates (full grid)"}
        index["h_grid"] = {"shape": list(np.asarray(h_grid).shape) if h_grid is not None else None,
                           "content": "parameter-grid H coordinates (full grid)"}
    null_dist = getattr(inner, "null_max_dist", None)
    if isinstance(null_dist, dict):
        for tail, arr in null_dist.items():
            if arr is not None:
                files[f"arrays/null_max_{tail}.npy"] = _npy_bytes(np.asarray(arr, dtype=np.float64))
                index[f"null_max_{tail}"] = {"shape": list(np.asarray(arr).shape),
                                             "content": "saved permutation null max-distribution"}
    return files, index


def _attrs_json(attrs: Mapping[str, Any]) -> bytes:
    return _dumps_json(dict(attrs))


def _shape_matches_recorded(recorded: Any, actual_shape: Sequence[int]) -> bool:
    """True when the recorded mask shape is exactly the array's shape (2-D)."""
    if not isinstance(recorded, (list, tuple)) or len(recorded) != 2:
        return False
    try:
        recorded_shape = tuple(int(v) for v in recorded)
    except (TypeError, ValueError):
        return False
    return recorded_shape == tuple(int(s) for s in actual_shape)


def build_analysis_archive(
    result: Any,
    run_plan: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
    *,
    atlas: Any = None,
    effect_mask: Optional[np.ndarray] = None,
    grid_cell: Optional[int] = None,
    project_root: Optional[Path] = None,
    exporter: str = "conninfpy-app",
    workspace_state: Optional[Mapping[str, Any]] = None,
    include_workspace_data: bool = False,
    include_bundle: bool = True,
    edge_family_mask: Optional[np.ndarray] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Build the deterministic analysis-archive ZIP bytes and its manifest.

    Parameters
    ----------
    result : AnalyzeResult or InferenceResult
        The finished run. Nothing is recomputed; only exporters are called.
    run_plan : mapping
        The captured launch settings (``task.run_plan``).
    dataset_info : mapping
        The captured dataset provenance record (``st.session_state.dataset_info``).
    atlas : AtlasInfo, optional
        Display/annotation atlas in matrix order (used for tables and figures).
    effect_mask : ndarray, optional
        Synthetic generator truth mask, archived as-is when present.
    edge_family_mask : ndarray, optional
        The boolean tested-edge-family mask of the finished run, archived as
        ``arrays/edge_family_mask.npy`` when the run recorded one; it is what a
        replay needs to reproduce the exact tested family.
    grid_cell : int, optional
        Parameter-grid cell shown in the application at export time (display
        fact; the archived arrays keep the full grid evidence).
    project_root : Path, optional
        Root for resolving relative source paths (defaults to cwd).
    include_bundle : bool, default True
        Whether to snapshot the trusted ``bundle.pkl`` pickle. ``True`` is the
        full analysis archive (NETSTAT-24) behaviour. ``False`` builds a
        pickle-free archive: no ``bundle.pkl`` member is produced and the
        manifest/methods state that omission honestly. Default workspace saves
        (``build_workspace_archive``) always pass ``False``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from conninfpy import __version__

    inner = _inner(result)
    projected, cell_idx, display_cell = _resolve_display_cell(result, grid_cell)
    plan, n_redacted_plan = _redact(dict(run_plan or {}))
    info, n_redacted_info = _redact(dict(dataset_info or {}))
    alpha = plan.get("alpha")
    alpha = float(alpha) if isinstance(alpha, (int, float, np.floating)) else None

    files: Dict[str, bytes] = {}
    notes: List[str] = []
    if n_redacted_plan or n_redacted_info:
        notes.append(f"{n_redacted_plan + n_redacted_info} credential-like value(s) redacted "
                     "from the captured records.")

    scale_meta = describe_result_scale_and_stat(result, plan)
    fisher_summary = _fisher_summary(plan.get("fisher"))
    dfiles = _dataset_files(info, project_root)
    replay = build_replay_config(plan, info, project_root=project_root, result=result)

    # ---- captured records ----
    files["run_plan.json"] = _dumps_json(plan)
    files["dataset_info.json"] = _dumps_json(info)

    # ---- trusted bundle (only when explicitly requested) ----
    bundle = None
    bundle_note = None
    if include_bundle:
        try:
            bundle = result.to_bundle() if hasattr(result, "to_bundle") else inner.to_bundle()
        except ValueError as exc:
            # A legacy result without captured analysis metadata cannot be
            # snapshotted; the archive records that honestly instead of failing
            # or reconstructing metadata from defaults.
            bundle_note = ("bundle.pkl omitted: this finished result lacks captured "
                           f"analysis metadata ({exc}).")
            notes.append(bundle_note)
        if bundle is not None:
            import pickle

            bundle_buffer = io.BytesIO()
            pickle.dump({"schema_version": bundle.schema_version, "bundle": bundle},
                        bundle_buffer, protocol=pickle.HIGHEST_PROTOCOL)
            files["bundle.pkl"] = bundle_buffer.getvalue()
    else:
        # A pickle-free export states the omission instead of implying that the
        # finished result lacked the metadata a snapshot needs.
        notes.append(_PICKLE_FREE_ARCHIVE_NOTE)

    # ---- arrays ----
    array_files, arrays_index = _arrays_payload(inner, cell_idx)
    files.update(array_files)
    files["arrays/index.json"] = _dumps_json(arrays_index)
    if effect_mask is not None:
        files["arrays/synthetic_effect_mask.npy"] = _npy_bytes(np.asarray(effect_mask))
        notes.append("synthetic_effect_mask.npy is the generator's planted truth "
                     "(generator fact, not an observation).")
        effect_mask_available = True
    else:
        effect_mask_available = False

    edge_mask_fact = plan.get("edge_mask")
    if isinstance(edge_mask_fact, Mapping):
        if edge_family_mask is None:
            notes.append("the run recorded a tested-family edge mask but the mask array is "
                         "not available here; arrays/edge_family_mask.npy is omitted and "
                         "the replay cannot reproduce the tested family")
        else:
            mask_array = np.asarray(edge_family_mask, dtype=bool)
            recorded_sha = edge_mask_fact.get("sha256")
            recorded_shape = edge_mask_fact.get("shape")
            # Bundle ONLY the mask of the finished run: a recorded sha256 is
            # required, the bytes must hash to it, and the recorded shape must
            # equal the array's shape (a reshape shares the bool bytes but is a
            # different ROI x ROI family).
            mismatch = None
            if not isinstance(recorded_sha, str) or not recorded_sha.strip():
                mismatch = ("the finished run did not record a sha256 for its edge mask, so "
                            "the available mask cannot be verified as the one that was "
                            "tested")
            elif np.asarray(edge_family_mask).ndim != 2:
                mismatch = ("the available edge mask is not a 2-D ROI x ROI array "
                            f"(shape {tuple(int(s) for s in np.asarray(edge_family_mask).shape)})")
            elif _sha256(np.ascontiguousarray(mask_array).tobytes()) != recorded_sha:
                mismatch = ("the available edge mask does not match the sha256 recorded "
                            "for the finished run (it was changed after the run)")
            elif not _shape_matches_recorded(recorded_shape, mask_array.shape):
                mismatch = (f"the available edge mask has shape {list(mask_array.shape)} but "
                            f"the finished run recorded {recorded_shape!r}; a reshaped mask "
                            "is not the tested family")
            if mismatch is not None:
                notes.append(f"{mismatch}; arrays/edge_family_mask.npy is omitted and the "
                             "replay cannot reproduce the tested family")
            else:
                files["arrays/edge_family_mask.npy"] = _npy_bytes(mask_array)
                notes.append("edge_family_mask.npy is the tested-edge-family mask of the "
                             "finished run (boolean ROI x ROI, eligible edges only), bundled so "
                             "the tested family can be reproduced; it is not an ROI subset and "
                             "not a community label vector.")

    # ---- tables (α-dependent; skipped honestly when the run α is unknown) ----
    if alpha is not None:
        edges = projected.significant_edges(atlas=atlas, alpha=alpha) if hasattr(
            projected, "significant_edges") else None
        if edges is not None:
            files["tables/edges.csv"] = _csv_bytes(edges)
        net_df, net_reason = prepare_network_summary(projected, atlas=atlas, alpha=alpha)
        if net_df is not None:
            files["tables/network_summary.csv"] = _csv_bytes(net_df)
        node_df, node_reason = prepare_node_summary(projected, atlas=atlas, alpha=alpha)
        if node_df is not None:
            files["tables/node_summary.csv"] = _csv_bytes(node_df)
        else:
            notes.append(f"node summary omitted: {node_reason}")
        if getattr(projected, "component_report", None) is not None:
            comps = projected.component_table(atlas=atlas, alpha=alpha)
            files["tables/nbs_components.csv"] = _csv_bytes(comps)
            files["tables/nbs_components.attrs.json"] = _attrs_json(comps.attrs)
        if getattr(projected, "block_report", None) is not None:
            blocks, _ = prepare_cnbs_blocks_table(projected, atlas=atlas, alpha=alpha)
            if blocks is not None:
                files["tables/cnbs_blocks.csv"] = _csv_bytes(blocks)
    else:
        notes.append("α-dependent tables and figures are omitted: the finished run did "
                     "not record its α (never re-chosen at export time).")

    # ---- methods ----
    methods = build_methods_markdown(
        result, plan, info, scale_meta, replay, alpha=alpha, display_cell=display_cell,
        fisher_summary=fisher_summary, dataset_files=dfiles,
        effect_mask_available=effect_mask_available,
        bundle_available="bundle.pkl" in files,
        bundle_omitted_reason=_PICKLE_FREE_METHODS_REASON if not include_bundle else None,
    )
    files["methods.md"] = methods.encode("utf-8")

    # ---- replay config ----
    if replay.get("config") is not None:
        files["analysis.yaml"] = _dumps_yaml(replay["config"])

    # ---- figures (α-dependent) ----
    if alpha is not None:
        method_name = str(plan.get("method") or inner.method or "analysis").upper()
        fig, fig_info = build_connectivity_maps_figure(
            projected, atlas=atlas, alpha=alpha, method_name=method_name,
            run_plan=plan, effect_mask=effect_mask if effect_mask_available else None)
        if fig is not None:
            files["figures/connectivity_maps.png"] = _fig_bytes(fig)
            plt.close(fig)
        net_fig = build_network_summary_figure(projected, atlas=atlas, alpha=alpha,
                                               method_name=method_name)
        if net_fig is not None:
            files["figures/network_summary.png"] = _fig_bytes(net_fig)
            plt.close(net_fig)
        null_fig, null_info = build_null_max_figure(projected, method_name=method_name)
        if null_fig is not None:
            files["figures/null_max.png"] = _fig_bytes(null_fig)
            plt.close(null_fig)
        if getattr(projected, "block_report", None) is not None:
            block_fig = build_cnbs_block_figure(projected, atlas=atlas, alpha=alpha,
                                                method_name=method_name)
            if block_fig is not None:
                files["figures/cnbs_blocks.png"] = _fig_bytes(block_fig)
                plt.close(block_fig)

    # ---- manifest ----
    manifest: Dict[str, Any] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "kind": ARCHIVE_KIND,
        "export": {
            "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "exporter": exporter,
            "conninfpy_version": __version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "display": {
                "grid_cell": cell_idx,
                "figure_alpha": alpha,
                "note": "display facts of the export moment; the captured analysis "
                        "facts live in run_plan.json and are never rewritten from these",
            },
        },
        "run": {
            "started_at": plan.get("started_at"),
            "wall_time_s": plan.get("wall_time_s"),
            "method": inner.method,
            "design_family": plan.get("design_family"),
            "alpha": alpha,
            "seed": plan.get("seed"),
            # Facts fixed before the run: the executed missing-connectivity
            # policy, the declared primary grid cell and the tested edge family.
            "missing_policy": plan.get("missing_policy"),
            "min_coverage": plan.get("min_coverage"),
            "primary_cell": plan.get("primary_cell"),
            "edge_mask": plan.get("edge_mask"),
        },
        "dataset": {
            "name": info.get("name"),
            "mode": info.get("mode"),
            "source": info.get("source"),
            "data_kind": plan.get("data_kind"),
            "synthetic": info.get("mode") == "Generate Synthetic Dataset",
            "source_files": dfiles,
        },
        "replay": {k: replay.get(k) for k in ("status", "reasons", "prerequisites",
                                              "command", "sourced_from_result")},
        "bundle_available": "bundle.pkl" in files,
        "trusted_only": ["bundle.pkl"] if "bundle.pkl" in files else [],
        "notes": notes,
    }
    if workspace_state is not None:
        from apps.utils.workspace_io import build_workspace_layer_files

        ws_files, ws_manifest = build_workspace_layer_files(
            workspace_state,
            include_data=include_workspace_data,
            project_root=project_root,
        )
        files.update(ws_files)
        manifest["workspace"] = {
            "kind": ws_manifest["kind"],
            "workspace_schema_version": ws_manifest["workspace_schema_version"],
            "include_data": ws_manifest["include_data"],
        }

    # Self-describing listing of every other member (a file cannot carry
    # its own checksum; manifest.json itself is covered by the ZIP listing
    # consumers build when extracting).
    manifest["files"] = {name: {"sha256": _sha256(data), "bytes": len(data)}
                         for name, data in sorted(files.items())}
    files["manifest.json"] = _dumps_json(manifest)

    zip_bytes = archive_zip_bytes(files)
    return zip_bytes, manifest


def archive_zip_bytes(files: Mapping[str, bytes]) -> bytes:
    """Deterministic ZIP: fixed member timestamps, sorted member order."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[name])
    return buffer.getvalue()


def archive_filename(contrast_name: str, method: str, started_at: Any = None) -> str:
    """Stable download filename for the archive."""
    from apps.utils.helpers import safe_filename_part

    stamp = ""
    if isinstance(started_at, str) and started_at:
        stamp = f"_{re.sub(r'[^0-9TZ]', '', started_at[:19])}"
    return (f"conninf_archive_{safe_filename_part(contrast_name)}"
            f"_{safe_filename_part(method)}{stamp}.zip")
