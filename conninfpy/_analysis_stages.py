"""Private stages of :func:`conninfpy.analysis.analyze`.

``analyze()`` is a thin orchestrator: it normalizes ``**method_kwargs``,
builds one :class:`_AnalysisState` and calls the stage functions below in a
fixed order. Every stage reads and writes named fields of that state, so the
orchestrator body stays a crisp list of phase calls while the exact
validation order, ``flags`` order and numerical path of the former
monolithic body are preserved:

1. :func:`_resolve_tail_policy` — ``tested_tail`` / ``two_tailed`` contract.
2. :func:`_align_paired_subjects` — subject-ID alignment of paired conditions.
3. :func:`_reduce_paired_interest` — paired GLM on the B − A difference;
   :func:`_expand_interactions` — centered product predictors.
4. :func:`_resolve_modes` — GLM / multi / t-test modes, method and net labels;
   :func:`_resolve_harmonization_strategy` — ComBat strategy and guards.
5. :func:`_prepare_inputs` — input scale, Fisher r→z exactly once, motion QC.
6. :func:`_run_combat` — ComBat plus the site-interest diagnostic.
7. :func:`_add_site_dummies` — site indicators as GLM nuisance.
8. :func:`_resolve_strata` — permutation strata from ``sites``.
9. :func:`_run_multi_inference` / :func:`_run_single_inference` — dispatch.
10. :func:`_finalize_single` — provenance threading onto the result.

The small helpers at the top of this module are imported by
:mod:`conninfpy.analysis` (which re-exports the ones used by the CLI and
tests) and by the stages.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from conninfpy._prepare import prepare_connectivity
from conninfpy._provenance import update_analysis_scale
from conninfpy._shared.alignment import (
    IdAlignmentError,
    alignment_record,
    check_subject_axis,
    paired_id_permutation,
    reorder_subject_axis,
)
from conninfpy._shared.rng import RngLike
from conninfpy.harmonize import combat_harmonize
from conninfpy.harmonize.diagnostics import analyze_design_diagnostics
from conninfpy.results import InferenceResult, OmnibusInferenceResult
from conninfpy.stats._design import build_design_matrix
from conninfpy.stats._types import StatMethod
from conninfpy.stats.glm import compute_p_val_glm
from conninfpy.stats.glm import compute_p_val_glm_multi
from conninfpy.stats.glm import compute_p_val_paired_glm
from conninfpy.stats.pairwise import compute_p_val

COMBAT_ONLY = "combat_only"
COMBAT_SITE_DUMMIES_GLM = "combat_site_dummies_glm"
SITE_DUMMIES_GLM = "site_dummies_glm"

InterestLike = Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]


# ---------------------------------------------------------------------------
# Small helpers shared by the stages (and by the CLI through conninfpy.analysis)
# ---------------------------------------------------------------------------
def _as_2d(a: Any) -> np.ndarray:
    """Promote a 1D regressor to a column matrix, leave 2D untouched."""
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def _build_multi_design(
    interest: Dict[str, Any],
    confounds: Optional[npt.NDArray[np.float64]],
) -> "tuple[np.ndarray, Dict[str, np.ndarray]]":
    """Assemble ``X = [intercept, confounds, *interest_columns]`` and one
    unit contrast per named predictor for the multi-contrast GLM.

    The intercept and every confound column are nuisance; each interest
    column gets a ``[0, ..., 0, 1, 0, ..., 0]`` contrast targeting it. The
    returned ``contrasts`` dict is keyed by the same names as ``interest``,
    ready for :func:`~conninfpy.compute_p_val_glm_multi`.
    """
    names = list(interest.keys())
    n = len(np.asarray(interest[names[0]]))
    blocks = [np.ones((n, 1), dtype=np.float64)]
    if confounds is not None:
        blocks.append(_as_2d(confounds).astype(np.float64))
    n_nuisance_cols = sum(b.shape[1] for b in blocks)

    interest_cols = [np.asarray(interest[nm], dtype=np.float64)[:, None]
                     for nm in names]
    X = np.hstack(blocks + interest_cols)
    p = X.shape[1]

    contrasts: Dict[str, np.ndarray] = {}
    for j, nm in enumerate(names):
        c = np.zeros(p, dtype=np.float64)
        c[n_nuisance_cols + j] = 1.0
        contrasts[nm] = c
    return X, contrasts


def _design_names(confounds: Any, confound_names: Optional[List[str]], interest_names: List[str]) -> List[str]:
    """Column names of [intercept, confounds, interest] as analyze() builds the design."""
    q = 0 if confounds is None else _as_2d(confounds).shape[1]
    names = list(confound_names) if confound_names is not None else [f"confound_{k + 1}" for k in range(q)]
    if len(names) != q:
        names = [f"confound_{k + 1}" for k in range(q)]
    return ["intercept"] + names + list(interest_names)


def _site_dummies(
    site_id: Any, drop_first: bool = True
) -> npt.NDArray[np.float64]:
    """One-hot encode site labels with optional drop-first reference coding.

    Returns ``(n, K - drop_first)`` float64 dummy matrix. The first
    unique site (sorted) is the reference category when
    ``drop_first=True`` (the standard one). Returns an ``(n, 0)`` array
    when there is only one site — the GLM design is unchanged.
    """
    sites = np.asarray(site_id)
    unique, inverse = np.unique(sites, return_inverse=True)
    n_sites = len(unique)
    if n_sites < 2:
        return np.zeros((len(sites), 0), dtype=np.float64)
    start = 1 if drop_first else 0
    out = np.zeros((len(sites), n_sites - start), dtype=np.float64)
    for j in range(start, n_sites):
        out[inverse == j, j - start] = 1.0
    return out


def _site_interest_association(
    sites: Any, interest: Any
) -> Optional[Dict[str, Any]]:
    """Multiple correlation ``R`` between site labels and the interest regressor(s).

    Each numeric interest column ``y`` is regressed by least squares on an
    intercept plus the drop-first site indicators, and ``R = sqrt(R^2)`` is
    returned, where ``R^2`` is the proportion of variance in ``y`` explained
    by site. ``R`` lies in ``[0, 1]``: it is the multiple correlation
    coefficient (eta) for ``K >= 2`` sites and reduces to ``|r|`` for two
    sites. With a dict of predictors the maximum over the columns is reported
    together with the per-column values.

    Returns ``None`` when the association is undefined: fewer than two sites,
    a length mismatch, a non-numeric / constant column, or a rank-deficient
    site design.
    """
    site_arr = np.asarray(sites)
    if site_arr.ndim != 1:
        return None
    n = int(site_arr.shape[0])
    dummies = _site_dummies(site_arr)
    if dummies.shape[1] == 0:
        return None
    try:
        if isinstance(interest, Mapping):
            columns = {
                str(name): np.asarray(values, dtype=np.float64).ravel()
                for name, values in interest.items()
            }
        else:
            columns = {"interest": np.asarray(interest, dtype=np.float64).ravel()}
    except (TypeError, ValueError):
        return None
    if not columns or any(col.shape[0] != n for col in columns.values()):
        return None
    design = np.column_stack([np.ones(n), dummies])
    per_column: Dict[str, Optional[float]] = {}
    for name, y in columns.items():
        finite = np.isfinite(y)
        y_f = y[finite]
        if y_f.size < design.shape[1] + 1 or np.ptp(y_f) == 0.0:
            per_column[name] = None
            continue
        x_f = design[finite]
        if np.linalg.matrix_rank(x_f) < design.shape[1]:
            per_column[name] = None
            continue
        beta, *_ = np.linalg.lstsq(x_f, y_f, rcond=None)
        residual = y_f - x_f @ beta
        total = float(np.sum((y_f - y_f.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 0.0
        per_column[name] = float(np.clip(r2, 0.0, 1.0))
    defined = {k: v for k, v in per_column.items() if v is not None}
    if not defined:
        return None
    best = max(defined, key=lambda k: defined[k])
    r2 = defined[best]
    return {
        "R": float(np.sqrt(r2)),
        "R2": r2,
        "column": best,
        "per_column_R2": per_column,
        "n_sites": int(dummies.shape[1] + 1),
    }


def _resolve_input_kind(
    input: Optional[str],
    fisher_z: Optional[bool],
    stacks: Sequence[Any],
) -> str:
    """Resolve ``input=`` (and the legacy ``fisher_z=``) to ``'r'`` or ``'z'``.

    An explicit value wins; otherwise the scale is inferred from the data.
    Ambiguous data raise instead of guessing, because guessing ``'r'`` on
    weak Fisher-z data would silently transform it twice.
    """
    from conninfpy._prepare import infer_input_kind

    if fisher_z is not None:
        legacy = "r" if fisher_z else "z"
        if input is not None and input != legacy:
            raise ValueError(
                f"input={input!r} contradicts fisher_z={fisher_z!r}; pass only "
                "input= ('r' for correlations, 'z' for Fisher-z data)."
            )
        input = legacy
    if input is not None:
        if input not in ("r", "z"):
            raise ValueError(f"input must be 'r', 'z' or None; got {input!r}.")
        return input
    kind = infer_input_kind(*stacks)
    if kind is None:
        raise ValueError(
            "Cannot tell whether the connectivity holds correlations or "
            "Fisher-z values: every value lies in [-1, 1] and the diagonal "
            "is not all ones. Pass input='r' (correlations; Fisher r→z is "
            "applied) or input='z' (already transformed)."
        )
    return kind


def _resolve_strategy(
    harmonize: Any,
    *,
    glm_mode: bool,
    ttest_mode: bool,
    sites_provided: bool,
    confounds_provided: bool,
) -> Optional[str]:
    """Resolve the user-facing ``harmonize=`` to a canonical strategy.

    Returns one of:

    * ``'combat_only'`` -- fit ComBat, then test on the harmonized matrix
      without adding site dummies to the GLM.
    * ``'combat_site_dummies_glm'`` -- fit ComBat, then add site dummies to the
      inferential GLM.
    * ``'site_dummies_glm'`` -- skip ComBat and model site directly in the
      inferential GLM.
    * ``None`` -- no ComBat, no site adjustment (no ``sites`` passed,
      or two-sample call where ComBat has no defensible recipe).
    """
    if harmonize in (COMBAT_SITE_DUMMIES_GLM, COMBAT_ONLY):
        return harmonize
    if harmonize in (SITE_DUMMIES_GLM, None):
        return SITE_DUMMIES_GLM if (glm_mode and sites_provided) else None
    if harmonize == "auto":
        if not sites_provided:
            return None
        if glm_mode:
            if confounds_provided:
                return COMBAT_SITE_DUMMIES_GLM
            return SITE_DUMMIES_GLM
        # two-sample + sites has no defensible ComBat recipe without
        # an interest column to preserve. Skip harmonization; the
        # caller is flagged in analyze() so the demotion isn't silent.
        return None
    raise ValueError(
        "harmonize= must be one of 'auto', 'combat_only', "
        "'combat_site_dummies_glm', 'site_dummies_glm', or None; "
        f"got {harmonize!r}."
    )


def _thread_preprocessing(result: Any, record: Dict[str, Any]) -> None:
    """Attach the normalized preprocessing record to a result's provenance."""
    provenance = getattr(result, "analysis_provenance", None)
    if provenance is not None:
        provenance["preprocessing"] = deepcopy(record)


def _thread_fisher(result: Any, record: Optional[Dict[str, Any]]) -> None:
    """Attach the Fisher-transform provenance record to a result."""
    provenance = getattr(result, "analysis_provenance", None)
    if provenance is not None:
        provenance["fisher"] = deepcopy(record)


def _estimator_aware_units(scale: str, record: Dict[str, Any]) -> Optional[str]:
    """Default response units that name the estimator when it is known."""
    method_fact = record.get("connectivity_method") if record else None
    method = method_fact.get("value") if isinstance(method_fact, dict) else None
    if scale == "fisher_z":
        if method == "partial":
            return "Fisher z (atanh of partial r)"
        if method == "spearman":
            return "Fisher z (atanh of Spearman ρ)"
    elif scale == "correlation":
        if method == "partial":
            return "partial correlation (r)"
        if method == "spearman":
            return "Spearman rank correlation (ρ)"
    return None


def _motion_qc_common(spec: Mapping[str, Any], kind: str, atlas: Any) -> Dict[str, Any]:
    """Shared motion_qc_fc kwargs resolved from one analyze(motion_qc=...) spec."""
    explicit_input = spec.get("input")
    if explicit_input is not None:
        scale = explicit_input
        scale_source = spec.get("input_source") or "explicit motion_qc input="
    else:
        scale = kind
        scale_source = spec.get("input_source") or "analyze resolved input scale"
    return dict(
        pheno=spec.get("pheno"),
        fd_units=spec.get("fd_units", "mm"),
        nuisance=spec.get("nuisance"),
        nuisance_names=spec.get("nuisance_names"),
        coords=spec.get("coords"),
        atlas=spec.get("atlas", atlas),
        space=spec.get("space"),
        roi_ids=spec.get("roi_ids"),
        scale=scale,
        scale_source=scale_source,
        pheno_subject_column=spec.get("pheno_subject_column"),
        fd_source_name=spec.get("fd_source_name"),
        unit_of_observation=spec.get("unit_of_observation", "subject"),
        input_stage="native input before Fisher/ComBat and before inference",
    )


def _motion_qc_single_diagnostics(
    spec: Mapping[str, Any], data: Any, kind: str, atlas: Any
) -> Dict[str, Any]:
    """One-set descriptive motion/QC-FC diagnostics (GLM, two-sample, one-sample)."""
    from conninfpy.interpret.motion_qc import motion_qc_fc

    return motion_qc_fc(
        data,
        fd=spec.get("fd"),
        subject_ids=spec.get("subject_ids"),
        condition_labels=spec.get("condition_labels"),
        condition=spec.get("condition"),
        index_values=spec.get("index_values"),
        **(_motion_qc_common(spec, kind, atlas)),
    )


def _motion_qc_paired_diagnostics(
    spec: Mapping[str, Any], group1: Any, group2: Any, kind: str, atlas: Any
) -> Dict[str, Any]:
    """Per-condition descriptive diagnostics for a paired design (A/B separate)."""
    from conninfpy.interpret.motion_qc import motion_qc_fc_paired

    return motion_qc_fc_paired(
        group1,
        group2,
        fd_group1=spec.get("fd_group1", spec.get("fd")),
        fd_group2=spec.get("fd_group2"),
        subject_ids=spec.get("subject_ids"),
        pool=spec.get("pool", True),
        condition_a=spec.get("condition_a", "A"),
        condition_b=spec.get("condition_b", "B"),
        id_alignment=spec.get("id_alignment_verified"),
        **(_motion_qc_common(spec, kind, atlas)),
    )


def _motion_qc_flag(diag: Mapping[str, Any]) -> str:
    """One plain-English flag line for the opted-in side-car."""
    status = str(diag.get("status"))
    if status == "available":
        where = "per condition" if diag.get("kind") == "motion_qc_fc_paired" else "one observation set"
        return (
            "motion/QC-FC side-car enabled (descriptive Spearman only, no p-values; "
            f"{where}; inference unaffected): status=available."
        )
    reason = diag.get("status_reason") or "see motion_qc.unavailable"
    return (
        "motion/QC-FC side-car enabled (descriptive only, inference unaffected) but "
        f"not fully available: status={status} - {reason}"
    )


def _encode_net_labels(networks: Sequence[Any]) -> tuple[npt.NDArray[np.int_], Dict[str, int]]:
    """Deterministically encode network labels to 0..M-1 in matrix row order."""
    seen: Dict[str, int] = {}
    codes = np.empty(len(networks), dtype=np.int_)
    for idx, net in enumerate(networks):
        key = str(net)
        if key not in seen:
            seen[key] = len(seen)
        codes[idx] = seen[key]
    return codes, seen


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
@dataclass
class _AnalysisState:
    """Everything the stages of one ``analyze()`` call thread between each other.

    The first block holds caller inputs that stages may rebind (``Y`` after
    ComBat, ``group2`` after the paired-ID permutation, ``interest`` after
    the interaction expansion, ...); the second holds the immutable call
    settings; the third the derived values produced by one stage and read by
    later ones.
    """

    # -- caller inputs that stages may rebind -----------------------------------
    Y: Optional[npt.NDArray[np.float64]]
    interest: Optional[InterestLike]
    confounds: Optional[npt.NDArray[np.float64]]
    group1: Optional[npt.NDArray[np.float64]]
    group2: Optional[npt.NDArray[np.float64]]
    confounds_group1: Optional[npt.NDArray[np.float64]]
    confounds_group2: Optional[npt.NDArray[np.float64]]
    sites: Optional[Sequence[Any]]
    preserve: Optional[npt.NDArray[np.float64]]
    input: Optional[str]
    fisher_z: Optional[bool]
    contrast_id: Optional[str]
    method_kwargs: Dict[str, Any]
    n_processes: Optional[int]
    progress: Optional[Any]
    cancel: Optional[Callable[[], bool]]
    #: ``dict(motion_qc)`` when the caller opted in; stages annotate it.
    motion_qc_spec: Optional[Dict[str, Any]]

    # -- immutable call settings ---------------------------------------------------
    test_type: str
    harmonize: Optional[str]
    r_to_z: bool
    var_equal: bool
    method: Union[str, StatMethod]
    two_tailed: Optional[bool]
    tested_tail: Optional[str]
    acceleration: Optional[str]
    n_permutations: int
    e: Union[float, Sequence[float]]
    h: Union[float, Sequence[float]]
    n: Optional[int]
    rng: RngLike
    verbose: bool
    use_mp: bool
    missing: str
    min_coverage: float
    atlas: Optional[Any]
    roi_ids: Optional[Sequence[Any]]
    groupings: Optional[Dict[str, Sequence[Any]]]
    units: Optional[str]
    #: Normalized once so every result and export carries the same record.
    preprocessing_record: Dict[str, Any]
    #: Upstream Fisher transforms, validated once.
    fisher_upstream: Optional[List[Dict[str, Any]]]

    # -- derived values threaded across stages -----------------------------------
    #: Plain-English warning / note lines, in emission order.
    flags: List[str] = field(default_factory=list)
    effective_two_tailed: bool = True
    #: ``design['paired_alignment']`` record of the subject-ID permutation.
    paired_alignment: Optional[Dict[str, Any]] = None
    #: ``interest`` given together with ``group1``/``group2`` (GLM on B − A).
    paired_interest: bool = False
    paired_details: Optional[Dict[str, Any]] = None
    derived_scale: Optional[str] = None
    derived_transform: Optional[str] = None
    centered_means: Optional[Dict[str, float]] = None
    motion_qc_diag: Optional[Dict[str, Any]] = None
    fisher_applied: List[Dict[str, Any]] = field(default_factory=list)
    fisher_upstream_state: Any = None  # list of entries | "unknown" | None
    fisher_upstream_decided: bool = False  # paired-interest resolves it on its own
    fisher_record: Dict[str, Any] = field(default_factory=dict)
    glm_mode: bool = False
    multi_mode: bool = False
    ttest_mode: bool = False
    paired_glm_mode: bool = False
    method_name: str = ""
    net_labels_mapping: Optional[Dict[str, int]] = None
    had_user_confounds: bool = False
    n_user_confounds: int = 0
    kind: str = "z"
    analysis_scale: str = "fisher_z"
    allow_missing: bool = False
    strategy: Optional[str] = None
    backend_resolved_strategy: str = "none"
    combat_diag: Optional[Dict[str, Any]] = None
    confound_names: Optional[List[str]] = None
    n_site_dummies: int = 0
    signflip_mode: bool = False
    auto_strata: Optional[npt.NDArray[np.float64]] = None


class _Inferred(NamedTuple):
    """One finished inference plus the diagnostics that wrap it."""

    result: Union[InferenceResult, OmnibusInferenceResult]
    design_diag: Dict[str, Any]
    flags: List[str]


# ---------------------------------------------------------------------------
# 1. Tail policy
# ---------------------------------------------------------------------------
def _resolve_tail_policy(state: _AnalysisState) -> None:
    """Two-sided by default; ``tested_tail`` opts into a prespecified direction."""
    tested_tail = state.tested_tail
    two_tailed = state.two_tailed
    if tested_tail is not None:
        if tested_tail not in ("positive", "negative"):
            raise ValueError(
                f"tested_tail must be 'positive' or 'negative'; got {tested_tail!r}."
            )
        if two_tailed:
            raise ValueError(
                "tested_tail=... declares a one-sided hypothesis and needs per-tail "
                "nulls; do not also pass two_tailed=True."
            )
        stat_type_arg = state.method_kwargs.get("stat_type", "tstat")
        stat_type_arg = getattr(stat_type_arg, "value", stat_type_arg)
        if str(stat_type_arg) == "fstat":
            raise ValueError(
                "tested_tail=... is undefined for an F-stat omnibus contrast: "
                "F is non-negative and has no direction."
            )
    state.effective_two_tailed = (
        False if tested_tail is not None else (True if two_tailed is None else bool(two_tailed))
    )
    if tested_tail is not None:
        state.flags.append(
            f"prespecified one-sided test ({tested_tail}): the direction was fixed "
            "before the run and the other direction was not tested."
        )
    elif two_tailed is False:
        state.flags.append(
            "two_tailed=False: positive and negative are corrected as separate "
            "one-sided families, each at the nominal alpha; the combined-direction "
            "FWER is not the nominal alpha (use alpha/2 per tail for that)."
        )


# ---------------------------------------------------------------------------
# 2. Paired subject-ID alignment
# ---------------------------------------------------------------------------
def _align_paired_subjects(
    state: _AnalysisState,
    ids_group1: Optional[Sequence[Any]],
    ids_group2: Optional[Sequence[Any]],
) -> None:
    """Move every condition-B subject-axis input into condition A's row order.

    Condition A (group1) is the reference row order. When the caller names
    the subjects, verify the A<->B bijection once and apply the permutation
    *before* the Fisher transform, the B-A differences and any inference.
    A-order inputs (interest, subject-level confounds, sites,
    confounds_group1) are validated against A and never permuted; edge-axis
    masks are invariant.
    """
    if ids_group1 is None and ids_group2 is None:
        return
    group1, group2, Y = state.group1, state.group2, state.Y
    if Y is not None or group1 is None or group2 is None or state.test_type != "paired":
        raise ValueError(
            "ids_group1/ids_group2 are only valid for the paired design: "
            "group1= and group2= with test_type='paired' and no Y. "
            "Two-sample, one-sample and Y-only (GLM) paths carry no "
            "within-subject pairs to align."
        )
    a_data, b_data = np.asarray(group1), np.asarray(group2)
    if a_data.ndim != 3 or b_data.ndim != 3 or a_data.shape[1:] != b_data.shape[1:]:
        raise ValueError(
            "paired conditions must have identical (subjects, regions, regions) "
            f"shapes; got {a_data.shape} and {b_data.shape}."
        )
    n_a = int(a_data.shape[0])
    try:
        # One-sided IDs are rejected inside ``paired_id_permutation``; the cast
        # only states that the validator owns the None check.
        perm = paired_id_permutation(
            cast(Sequence[Any], ids_group1), cast(Sequence[Any], ids_group2),
            n_group1=n_a, n_group2=int(b_data.shape[0]),
            name_group1="ids_group1", name_group2="ids_group2",
        )
        a_order = ["interest"] if state.interest is not None else []
        if isinstance(state.interest, dict):
            for name, values in state.interest.items():
                check_subject_axis(values, f"interest[{name!r}]", n_a, condition="group1")
        else:
            check_subject_axis(state.interest, "interest", n_a, condition="group1")
        for a_name, a_value in (("confounds", state.confounds), ("sites", state.sites),
                                ("confounds_group1", state.confounds_group1)):
            if a_value is not None:
                check_subject_axis(a_value, a_name, n_a, condition="group1")
                a_order.append(a_name)
        had_condition_confounds = state.confounds_group2 is not None
        state.confounds_group2 = reorder_subject_axis(
            state.confounds_group2, perm, "confounds_group2"
        )
    except IdAlignmentError as exc:
        raise ValueError(str(exc)) from exc
    state.group2 = b_data[perm]
    reordered = ["group2"] + (["confounds_group2"] if had_condition_confounds else [])
    if state.motion_qc_spec is not None and state.motion_qc_spec.get("fd_group2") is not None:
        # The side-car is descriptive, but its condition-B FD must follow the
        # exact same rows as condition B: reorder it with the one permutation.
        from conninfpy.interpret.motion_qc import _resolve_fd_vector

        fd_b = _resolve_fd_vector(
            state.motion_qc_spec["fd_group2"], state.motion_qc_spec.get("pheno"),
            int(b_data.shape[0]), "motion_qc['fd_group2']",
        )
        state.motion_qc_spec["fd_group2"] = fd_b[perm]
        state.motion_qc_spec["fd_group2_alignment"] = (
            "condition-B FD moved into condition A's row order with the same "
            "subject-ID permutation as condition B"
        )
    if state.motion_qc_spec is not None and ids_group1 is not None:
        state.motion_qc_spec["subject_ids"] = tuple(ids_group1)
        # analyze verified the A<->B bijection and moved condition B (and its
        # FD) into A order; the descriptive side-car records that fact.
        state.motion_qc_spec["id_alignment_verified"] = "exact_id_match"
    state.paired_alignment = alignment_record(
        perm,
        n_subjects=n_a,
        reference="group1 (condition A)",
        reordered_inputs=reordered,
        a_order_inputs=a_order,
    )
    if state.paired_alignment["identity_permutation"]:
        state.flags.append(
            f"paired subject IDs checked: {n_a} subjects form a one-to-one "
            "match between group1 and group2; condition B already followed "
            "condition A's row order (no permutation applied)."
        )
    else:
        state.flags.append(
            f"paired subject IDs aligned: {n_a} subjects matched one-to-one by "
            f"ID; {', '.join(reordered)} permuted into condition A's row order "
            "before the Fisher transform and the B - A difference. "
            "A-order inputs (interest, subject-level confounds, sites) were "
            "validated against A and not permuted."
        )


# ---------------------------------------------------------------------------
# 3. Paired-interest reduction and interactions
# ---------------------------------------------------------------------------
def _reduce_paired_interest(state: _AnalysisState) -> None:
    """Replace paired conditions + ``interest`` by the GLM response B − A.

    A repeated-measures regression has one observation per subject. Do the
    nonlinear transform before subtraction, never on the difference.
    """
    state.paired_interest = state.interest is not None and (state.group1 is not None or state.group2 is not None)
    if not state.paired_interest:
        return
    group1, group2 = state.group1, state.group2
    if state.Y is not None or group1 is None or group2 is None or state.test_type != "paired":
        raise ValueError("paired interest requires group1 and group2, test_type='paired', and no Y")
    a, b = np.asarray(group1), np.asarray(group2)
    if a.ndim != 3 or a.shape != b.shape:
        raise ValueError("paired conditions must have identical (subjects, regions, regions) shapes and subject order")
    original_kind = _resolve_input_kind(state.input, state.fisher_z, [a, b])
    if original_kind == "r" and not state.r_to_z:
        raise ValueError("paired interest with correlations requires Fisher transformation; r_to_z=False is not supported")
    if state.harmonize not in (None, "auto"):
        raise ValueError("paired interest supports harmonize=None or 'auto'; harmonization of condition differences is not automatic")
    if state.motion_qc_spec is not None:
        state.motion_qc_diag = _motion_qc_paired_diagnostics(
            state.motion_qc_spec, a, b, original_kind, state.atlas
        )
        state.flags.append(_motion_qc_flag(state.motion_qc_diag))
    if (state.confounds_group1 is None) != (state.confounds_group2 is None):
        raise ValueError("supply both condition-specific confounds or neither")
    n_subject_confounds = 0 if state.confounds is None else _as_2d(state.confounds).shape[1]
    n_condition_confounds = 0
    if state.confounds_group1 is not None:
        ca, cb = _as_2d(state.confounds_group1), _as_2d(state.confounds_group2)
        if ca.shape != cb.shape or ca.shape[0] != a.shape[0]:
            raise ValueError("condition-specific confounds must align with paired subjects and have equal shapes")
        delta_c = cb.astype(float) - ca.astype(float)
        n_condition_confounds = delta_c.shape[1]
        state.confounds = delta_c if state.confounds is None else np.column_stack((_as_2d(state.confounds), delta_c))
    state.paired_details = {"input": original_kind, "orientation": "B - A",
                            "subject_confound_columns": n_subject_confounds,
                            "condition_confound_columns": n_condition_confounds,
                            "condition_confound_transform": "B - A"}
    if state.paired_alignment is not None:
        state.paired_details["alignment"] = deepcopy(state.paired_alignment)
    allow_missing_pair = state.missing == "available_case"
    # Fisher is applied to each condition separately, never to the
    # difference; the captured entries record exactly that (stage
    # 'analysis_prepare_before_difference', scopes group1/group2).
    b_pre, fisher_entry_b = prepare_connectivity(
        b, original_kind, "group2", allow_missing=allow_missing_pair,
        stage="analysis_prepare_before_difference", return_report=True)
    a_pre, fisher_entry_a = prepare_connectivity(
        a, original_kind, "group1", allow_missing=allow_missing_pair,
        stage="analysis_prepare_before_difference", return_report=True)
    state.Y = b_pre - a_pre
    state.fisher_applied = [e for e in (fisher_entry_a, fisher_entry_b) if e is not None]
    if original_kind == "z":
        state.fisher_upstream_state = state.fisher_upstream if state.fisher_upstream is not None else "unknown"
    state.fisher_upstream_decided = True
    state.derived_scale = "difference_fisher_z"
    state.derived_transform = "Fisher r-to-z separately for each condition, then B minus A" if original_kind == "r" else "B minus A of Fisher z conditions"
    state.group1 = state.group2 = state.confounds_group1 = state.confounds_group2 = None
    state.input, state.fisher_z = "z", None


def _expand_interactions(state: _AnalysisState, interactions: Optional[Sequence[tuple[str, str]]]) -> None:
    """Center named predictors by their sample means and add a product column per pair."""
    interest = state.interest
    if interactions is None:
        return
    if not isinstance(interest, dict) or not interest:
        raise ValueError("interactions require a nonempty named interest dictionary of numeric predictors")
    centered = {}
    centered_means: Dict[str, float] = {}
    for name, values in interest.items():
        if not isinstance(name, str) or not name or ':' in name:
            raise ValueError("interaction predictor names must be nonempty strings without ':'")
        try:
            v = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"interaction predictor {name!r} must be numeric; encode categories explicitly") from exc
        if v.ndim != 1 or not np.isfinite(v).all():
            raise ValueError(f"interaction predictor {name!r} must be a finite 1D numeric vector")
        centered_means[name] = float(v.mean())
        centered[name] = v - centered_means[name]
    for pair in interactions:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or pair[0] == pair[1] or any(x not in centered_means for x in pair):
            raise ValueError("each interaction must name two distinct interest predictors")
        name = f"{pair[0]}:{pair[1]}"
        if name in centered or f"{pair[1]}:{pair[0]}" in centered:
            raise ValueError(f"duplicate interaction {name!r}")
        if centered[pair[0]].shape != centered[pair[1]].shape:
            raise ValueError("interaction predictors must have the same length")
        centered[name] = centered[pair[0]] * centered[pair[1]]
    state.centered_means = centered_means
    state.interest = centered


# ---------------------------------------------------------------------------
# 4. Modes, method and harmonization strategy
# ---------------------------------------------------------------------------
def _resolve_modes(state: _AnalysisState) -> None:
    """Decide GLM / multi-predictor / t-test mode and validate the inputs."""
    interest = state.interest
    state.glm_mode = interest is not None
    state.multi_mode = state.glm_mode and isinstance(interest, dict)
    state.ttest_mode = state.group1 is not None or state.group2 is not None
    if state.glm_mode and state.ttest_mode:
        raise ValueError(
            "analyze() takes either (interest, [confounds]) for the GLM path or "
            "(group1, group2) for the t-test path, not both."
        )
    if not state.glm_mode and not state.ttest_mode:
        raise ValueError(
            "analyze() requires either an `interest` regressor or `group1`/`group2`."
        )
    if state.glm_mode and state.Y is None:
        raise ValueError(
            "analyze() requires Y when using the GLM path (interest=, "
            "[confounds=]). Pass Y as the first positional argument."
        )
    state.method_name = str(getattr(state.method, "value", state.method)).lower()
    state.net_labels_mapping = None
    if state.method_name in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
        if (state.method_kwargs.get("net_labels") is None and state.atlas is not None
                and getattr(state.atlas, "networks", None) is not None):
            n_nodes_data = int(state.Y.shape[-1]) if state.Y is not None else int(np.asarray(state.group1).shape[-1])
            if len(state.atlas.networks) != n_nodes_data:
                raise ValueError(
                    f"atlas has {len(state.atlas.networks)} networks but data has {n_nodes_data} regions."
                )
            encoded_labels, state.net_labels_mapping = _encode_net_labels(state.atlas.networks)
            state.method_kwargs["net_labels"] = encoded_labels
        elif isinstance(state.method_kwargs.get("net_labels_mapping"), Mapping):
            state.net_labels_mapping = dict(state.method_kwargs.pop("net_labels_mapping"))
        else:
            state.method_kwargs.pop("net_labels_mapping", None)
    else:
        state.method_kwargs.pop("net_labels_mapping", None)

    if state.method_name == "cnbs":
        cnbs_corr = str(state.method_kwargs.get("cnbs_correction", "bonferroni")).lower()
        nl_arr = state.method_kwargs.get("net_labels")
        if cnbs_corr == "bonferroni" and nl_arr is not None:
            m_nets = int(len(np.unique(np.asarray(nl_arr))))
            k_blocks = m_nets * (m_nets + 1) / 2
            min_p = k_blocks / (int(state.n_permutations) + 1)
            if min_p > 0.05:
                cnbs_warn = (
                    f"cNBS with Bonferroni correction: M={m_nets} networks -> K={int(k_blocks)} blocks, "
                    f"minimum attainable p-value K/(n_permutations+1) = {min_p:.4g} > 0.05; "
                    "increase n_permutations or use cnbs_correction='bh'."
                )
                warnings.warn(cnbs_warn, RuntimeWarning, stacklevel=2)
                state.flags.append(cnbs_warn)
    if state.var_equal and not (state.ttest_mode and state.test_type == "two-sample"):
        raise ValueError(
            "var_equal applies to the two-sample t-test (group1=, group2=, "
            "test_type='two-sample'). The GLM t uses a pooled residual variance "
            "by construction; paired and one-sample tests have one group."
        )
    if state.multi_mode:
        # Several predictors → compute_p_val_glm_multi. Each value must be a
        # single 1D regressor; the dict key names the per-predictor result.
        assert isinstance(interest, dict)
        if len(interest) == 0:
            raise ValueError(
                "interest={} is empty. Pass a non-empty dict {name: vector}, "
                "or a single array for one predictor."
            )
        for nm, vec in interest.items():
            v = np.asarray(vec)
            if v.ndim != 1:
                raise ValueError(
                    f"interest[{nm!r}] must be a 1D predictor of shape "
                    f"(n_subjects,); got shape {v.shape}. Each dict entry is "
                    "one regressor of interest."
                )
    elif state.glm_mode:
        # `interest` is a SINGLE predictor. A (n, k>1) array or a Python list
        # of regressors would be tested by build_design_matrix's contrast on
        # the *last* column only (silently demoting the rest to nuisance), and
        # a list like [age, sex] is read as shape (k, n) and crashes
        # downstream. Reject both with a pointer to the dict (multi) API.
        assert interest is not None
        _interest_arr = np.asarray(interest)
        if _interest_arr.ndim >= 2 and not (
            _interest_arr.ndim == 2 and _interest_arr.shape[1] == 1
        ):
            raise ValueError(
                "interest= must be a single predictor of shape (n_subjects,) "
                f"or (n_subjects, 1); got array of shape {_interest_arr.shape}. "
                "To test several predictors, pass a dict "
                "interest={'name': vector, ...} (analyze() returns one result "
                "per predictor via compute_p_val_glm_multi), or build a "
                "multi-row contrast with compute_p_val_glm(..., "
                "stat_type='fstat') for a joint omnibus F-test."
            )

    # Repeated-measures GLM: paired conditions + condition-varying confounds.
    has_cg1 = state.confounds_group1 is not None
    has_cg2 = state.confounds_group2 is not None
    state.paired_glm_mode = has_cg1 or has_cg2
    if state.paired_glm_mode:
        if has_cg1 != has_cg2:
            raise ValueError(
                "Pass both confounds_group1 and confounds_group2, or neither — "
                "the repeated-measures GLM needs the confound in both conditions "
                "to form the per-subject difference."
            )
        if not state.ttest_mode:
            raise ValueError(
                "confounds_group1/confounds_group2 require the paired t-test "
                "inputs (group1=, group2=). For a between-subject continuous "
                "predictor with confounds use Y=, interest=, confounds=."
            )
        if state.test_type != "paired":
            raise ValueError(
                "confounds_group1/confounds_group2 are only valid with "
                f"test_type='paired'; got test_type={state.test_type!r}. Condition-"
                "varying confounds are differenced within subject, which only "
                "makes sense for a repeated-measures design."
            )
    if state.ttest_mode and state.confounds is not None:
        raise ValueError(
            "confounds= is the GLM-path nuisance design (used with interest=). "
            "For a paired design with condition-varying confounds use "
            "confounds_group1=/confounds_group2=; for a two-sample contrast with "
            "nuisance covariates promote to GLM with a binary interest= column."
        )

    state.had_user_confounds = state.confounds is not None
    state.n_user_confounds = 0 if state.confounds is None else _as_2d(state.confounds).shape[1]


def _resolve_harmonization_strategy(state: _AnalysisState) -> None:
    """Resolve ``harmonize=`` for this call shape and apply the explicit guards."""
    strategy = _resolve_strategy(
        state.harmonize,
        glm_mode=state.glm_mode,
        ttest_mode=state.ttest_mode,
        sites_provided=state.sites is not None,
        confounds_provided=state.confounds is not None,
    )
    if state.paired_interest:
        # Site labels constrain between-subject permutations, while
        # subject-constant additive site effects cancel in B - A.
        strategy = None
        if state.sites is not None:
            state.flags.append(
                "paired differences (GLM on B - A): ComBat skipped — "
                "subject-constant additive site effects cancel in B - A "
                "(arbitrary site × condition interactions are not automatically "
                "controlled unless modeled); sites= stratifies between-subject "
                "permutations."
            )

    # Explicit ComBat guards (only fire for explicit requests; 'auto'
    # resolves to combat_site_dummies_glm only when these guards are satisfied).
    if state.harmonize in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM):
        if state.sites is None:
            raise ValueError(
                f"{strategy} requires sites=...; otherwise "
                "there is no ComBat step to control."
            )
        if state.confounds is None:
            raise ValueError(
                f"{strategy} requires confounds=; otherwise "
                "the ComBat preserve design would be empty."
            )
        if not state.glm_mode:
            raise ValueError(
                f"{strategy} is GLM-only (interest=, confounds=). For "
                "two-sample / paired designs use "
                "harmonize='auto' or None."
            )
    # Two-sample + sites resolves to no-ComBat (no defensible preserve).
    # Flag so the demotion is visible and the caller can promote to GLM.
    if (
        state.ttest_mode
        and state.test_type != "paired"
        and state.sites is not None
        and state.harmonize in ("auto", None)
    ):
        state.flags.append(
            "two-sample + sites: ComBat skipped (no interest column to "
            "preserve). For combat_site_dummies_glm, promote to GLM "
            "with a binary interest indicator."
        )
    # Paired + sites: subject-constant additive site effects cancel in
    # group2 - group1 (not arbitrary site × condition effects), and
    # within-subject sign-flip permutations are stratum-invariant by construction.
    if (
        state.ttest_mode
        and state.test_type == "paired"
        and state.sites is not None
        and state.harmonize in ("auto", None)
    ):
        state.flags.append(
            "paired + sites: ComBat skipped — subject-constant additive site "
            "effects cancel in the within-subject difference (arbitrary site × "
            "condition interactions are not automatically controlled); "
            "within-subject sign-flip permutations are stratum-invariant by "
            "construction, so sites/strata do not alter the sign-flip null."
        )
    if strategy is None and state.preserve is not None:
        state.flags.append(
            "preserve= ignored because no ComBat harmonization strategy is "
            "active."
        )
        state.preserve = None
    state.strategy = strategy
    state.backend_resolved_strategy = strategy if strategy is not None else "none"


# ---------------------------------------------------------------------------
# 5. Input preparation: zero diagonal, Fisher r→z exactly once
# ---------------------------------------------------------------------------
def _prepare_inputs(state: _AnalysisState) -> None:
    """Resolve the input scale, prepare every stack and run the opt-in QC side-car."""
    glm_mode, ttest_mode = state.glm_mode, state.ttest_mode
    stacks = [state.Y] if glm_mode else [g for g in (state.group1, state.group2) if g is not None]
    kind = _resolve_input_kind(state.input, state.fisher_z, stacks)
    if state.motion_qc_spec is not None and state.motion_qc_diag is None:
        # Described on the caller's native input, before Fisher/ComBat/inference.
        if ttest_mode and state.test_type == "paired" and state.group1 is not None and state.group2 is not None:
            state.motion_qc_diag = _motion_qc_paired_diagnostics(
                state.motion_qc_spec, state.group1, state.group2, kind, state.atlas
            )
        else:
            if glm_mode:
                mq_data: Any = state.Y
            elif state.test_type == "two-sample" and state.group2 is not None:
                mq_data = np.concatenate([state.group1, state.group2], axis=0)
            else:
                mq_data = state.group1
            state.motion_qc_diag = _motion_qc_single_diagnostics(
                state.motion_qc_spec, mq_data, kind, state.atlas
            )
        state.flags.append(_motion_qc_flag(state.motion_qc_diag))
    # Already-transformed input: nothing is applied here. The upstream
    # transform is the caller's history when supplied, else explicitly
    # unknown — never an invented policy or zero-count clip. (The
    # paired-interest path decided this above on the original conditions.)
    if kind == "z" and not state.fisher_upstream_decided:
        state.fisher_upstream_state = state.fisher_upstream if state.fisher_upstream is not None else "unknown"
    analysis_scale = "correlation" if (kind == "r" and not state.r_to_z) else "fisher_z"
    if state.derived_scale:
        analysis_scale = state.derived_scale
    if analysis_scale == "correlation":
        # prepare_connectivity(..., 'z') checks and zeroes the diagonal
        # without transforming: the correlations are analysed as given.
        kind = "z"
    if state.missing not in ("raise", "available_case"):
        raise ValueError(f"missing must be 'raise' or 'available_case'; got {state.missing!r}.")
    state.allow_missing = state.missing == "available_case"
    allow_missing = state.allow_missing
    if glm_mode:
        assert state.Y is not None
        state.Y, _fisher_entry_y = prepare_connectivity(
            state.Y, kind, "Y", allow_missing=allow_missing, return_report=True)
        if _fisher_entry_y is not None:
            state.fisher_applied.append(_fisher_entry_y)
    else:
        if state.group1 is not None:
            state.group1, _fisher_entry_g1 = prepare_connectivity(
                state.group1, kind, "group1", allow_missing=allow_missing, return_report=True)
            if _fisher_entry_g1 is not None:
                state.fisher_applied.append(_fisher_entry_g1)
        if state.group2 is not None:
            state.group2, _fisher_entry_g2 = prepare_connectivity(
                state.group2, kind, "group2", allow_missing=allow_missing, return_report=True)
            if _fisher_entry_g2 is not None:
                state.fisher_applied.append(_fisher_entry_g2)
    state.kind = kind
    state.analysis_scale = analysis_scale
    state.fisher_record = {
        "schema_version": 1,
        "applied": state.fisher_applied,
        "upstream": state.fisher_upstream_state,
    }


# ---------------------------------------------------------------------------
# 6. ComBat (runs for combat_only and combat_site_dummies_glm)
# ---------------------------------------------------------------------------
def _run_combat(state: _AnalysisState) -> None:
    """Harmonize with ``preserve = confounds`` and record the site-interest diagnostic.

    The tested variable is deliberately omitted from ComBat so the
    harmonization fit never sees the labels that the permutation will
    reshuffle (Nygaard 2016 label-leak avoidance). Mild attenuation
    when corr(site, interest) > 0; that's the accepted trade-off.
    """
    if state.strategy not in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM):
        return
    if state.preserve is not None:
        state.flags.append(
            f"preserve= overridden by {state.strategy}: "
            "ComBat preserve is set to confounds only."
        )
    state.preserve = _as_2d(state.confounds)
    state.flags.append(
        f"{state.strategy}: preserve excludes interest."
    )
    if state.sites is None:
        return
    assert state.Y is not None
    res = combat_harmonize(
        state.Y, sites=cast(Sequence[Any], np.asarray(state.sites)),
        preserve=state.preserve, missing=state.missing,
    )
    state.Y = res.Y_adjusted
    combat_diag = dict(res.diagnostics)
    combat_diag["strategy"] = state.strategy
    combat_diag["preserve_columns"] = "confounds_only"
    ratio = combat_diag.get("between_site_variance_ratio_after_over_before")
    if ratio is not None and ratio > 0.5:
        state.flags.append(
            f"ComBat removed less than half of between-site variance "
            f"(ratio after/before = {ratio:.3f})."
        )
    lost = combat_diag.get("values_not_harmonized", 0)
    if lost:
        state.flags.append(
            f"ComBat could not harmonize {lost} value(s) (fewer than 2 subjects "
            "of their site have them); they are treated as missing."
        )
    # Site-interest confounding diagnostic. ComBat never sees `interest`
    # (label-leak avoidance), so a site effect that is correlated with the
    # tested variable is partly removed as if it were site noise, which
    # attenuates the interest effect. Quantify how much site and interest
    # covary so the trade-off is visible in flags and combat_diagnostics.
    association = _site_interest_association(state.sites, state.interest)
    if association is not None:
        combat_diag["site_interest_corr"] = association["R"]
        combat_diag["site_interest_R2"] = association["R2"]
        combat_diag["site_interest_column"] = association["column"]
        if len(association["per_column_R2"]) > 1:
            combat_diag["site_interest_R2_per_column"] = dict(association["per_column_R2"])
            column_note = f" (max over interest columns: '{association['column']}')"
        else:
            column_note = ""
        state.flags.append(
            f"site_interest_association: R={association['R']:.3f} "
            f"(R^2={association['R2']:.3f}) between site and interest{column_note}; "
            "higher site-interest confounding increases attenuation under "
            "label-leak-free ComBat."
        )
    state.combat_diag = combat_diag


# ---------------------------------------------------------------------------
# 7. Site dummies → GLM nuisance for the site-GLM strategies
# ---------------------------------------------------------------------------
def _add_site_dummies(state: _AnalysisState) -> None:
    """Append drop-first site indicators to the nuisance design.

    combat_site_dummies_glm: ComBat harmonized the matrix but did not see
    interest; site dummies in the GLM still help in case any site-correlated
    noise survives. site_dummies_glm: no ComBat ran at all; site dummies are
    the only site adjustment.
    """
    state.confound_names = None
    state.n_site_dummies = 0
    if not (
        state.glm_mode
        and state.sites is not None
        and state.strategy in (COMBAT_SITE_DUMMIES_GLM, SITE_DUMMIES_GLM)
    ):
        return
    site_dum = _site_dummies(np.asarray(state.sites))
    if site_dum.shape[1] > 0:
        n_user = 0 if state.confounds is None else _as_2d(state.confounds).shape[1]
        state.n_site_dummies = int(site_dum.shape[1])
        site_names = [f"site[{u}]" for u in np.unique(np.asarray(state.sites))[1:]]
        state.confound_names = [f"confound_{k + 1}" for k in range(n_user)] + site_names
        if state.confounds is None:
            state.confounds = site_dum
        else:
            state.confounds = np.column_stack([_as_2d(state.confounds), site_dum])
    if state.strategy == COMBAT_SITE_DUMMIES_GLM:
        state.flags.append(
            "combat_site_dummies_glm: site dummies appended to GLM nuisance."
        )
    if state.strategy == SITE_DUMMIES_GLM:
        if state.preserve is not None:
            state.flags.append(
                "preserve= ignored by site_dummies_glm because "
                "ComBat does not run."
            )
            state.preserve = None
        if state.harmonize == "auto" and not state.had_user_confounds:
            state.flags.append(
                "site_dummies_glm: no ComBat (harmonize='auto' without confounds= "
                "skips ComBat because no non-site covariates were given to "
                "preserve); site dummies appended to GLM nuisance."
            )
        else:
            state.flags.append(
                "site_dummies_glm: no ComBat; site dummies appended to GLM nuisance."
            )
        state.combat_diag = {
            "strategy": SITE_DUMMIES_GLM,
            "preserve_columns": None,
        }


# ---------------------------------------------------------------------------
# 8. Strata
# ---------------------------------------------------------------------------
def _resolve_strata(state: _AnalysisState) -> None:
    """Auto-stratify the permutation engine on ``sites`` (PALM ``-eb`` semantics).

    For two-sample / GLM / paired_interest this constrains label/row
    permutations within site; for paired / one-sample sign-flip permutations,
    the sign-flip null is stratum-invariant by construction and ``strata`` is
    recorded for provenance and per-subject site checks. An explicit
    ``strata=`` passed via ``**method_kwargs`` wins.
    """
    state.signflip_mode = state.ttest_mode and state.test_type in ("paired", "one-sample")
    if "strata" in state.method_kwargs:
        user_strata = state.method_kwargs.pop("strata")
        auto_strata = (
            np.asarray(user_strata) if user_strata is not None else None
        )
    else:
        auto_strata = np.asarray(state.sites) if state.sites is not None else None
        if auto_strata is not None:
            if (
                state.ttest_mode
                and state.test_type == "paired"
                and state.group1 is not None
                and auto_strata.shape[0] != state.group1.shape[0]
            ):
                raise ValueError(
                    "For paired inference, sites/strata must provide one label per "
                    f"paired subject ({state.group1.shape[0]}), got {auto_strata.shape[0]}."
                )
            if state.signflip_mode:
                state.flags.append(
                    "strata= auto-set to `sites` (validated and recorded for "
                    "provenance; within-subject/one-sample sign-flip permutations "
                    "are stratum-invariant by construction); pass strata= "
                    "explicitly (or strata=None) to override."
                )
            else:
                state.flags.append(
                    "strata= auto-set to `sites` (restricts label/row permutations "
                    "within site); pass strata= explicitly (or strata=None) to "
                    "override."
                )
    state.auto_strata = auto_strata


def _strata_note_text(auto_strata: Optional[npt.NDArray[np.float64]], signflip_mode: bool,
                      paired_interest: bool) -> Optional[str]:
    if auto_strata is None:
        return None
    if signflip_mode:
        return (
            "Within-subject / one-sample sign-flip permutations are "
            "stratum-invariant by construction; strata is validated and "
            "recorded for provenance without altering the sign-flip null."
        )
    if paired_interest:
        return (
            "paired_interest tests a between-subject predictor on (B - A); "
            "strata constrains between-subject Freedman-Lane permutations "
            "within site/stratum."
        )
    return "Constrains between-subject permutations within each stratum/site."


def _glm_column_roles(col_names: Sequence[str], interest_cols: Sequence[str],
                      n_user_confounds: int, n_site_dummies: int) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    interest_set = set(interest_cols)
    for idx, nm in enumerate(col_names):
        if idx == 0 and nm == "intercept":
            roles[nm] = "intercept"
        elif nm in interest_set:
            roles[nm] = "interest"
        elif idx > n_user_confounds and idx <= n_user_confounds + n_site_dummies:
            roles[nm] = "site_dummy"
        else:
            roles[nm] = "confound"
    return roles


# ---------------------------------------------------------------------------
# 9. Inference dispatch
# ---------------------------------------------------------------------------
def _inference_kwargs(state: _AnalysisState) -> Dict[str, Any]:
    """Keyword arguments shared by every ``compute_p_val*`` call."""
    return dict(
        method=state.method, n_permutations=state.n_permutations,
        two_tailed=state.effective_two_tailed,
        acceleration=state.acceleration,
        e=state.e, h=state.h, n=state.n, rng=state.rng, verbose=state.verbose, use_mp=state.use_mp,
        progress=state.progress, cancel=state.cancel, n_processes=state.n_processes,
        strata=cast(Optional[Sequence[Any]], state.auto_strata), missing=state.missing,
        min_coverage=state.min_coverage,
        atlas=state.atlas, roi_ids=state.roi_ids, groupings=state.groupings, units=state.units,
    )


def _thread_design_provenance(state: _AnalysisState, res: Any) -> None:
    """Write the analysis-scale / paired / centering facts onto ``res.design``."""
    if getattr(res, "design", None) is None:
        return
    res.design["scale"] = state.analysis_scale
    if state.derived_transform:
        res.design["response_transform"] = state.derived_transform
        res.design["paired_difference"] = dict(state.paired_details or {})
    if state.centered_means is not None:
        res.design["predictor_centering_means"] = dict(state.centered_means)
    if state.paired_alignment is not None:
        res.design["paired_alignment"] = deepcopy(state.paired_alignment)
    update_analysis_scale(
        res, state.analysis_scale,
        state.units or _estimator_aware_units(state.analysis_scale, state.preprocessing_record)
        or ("Fisher z difference (B - A)" if state.derived_scale else None),
    )
    _thread_preprocessing(res, state.preprocessing_record)
    _thread_fisher(res, state.fisher_record)


def _attach_net_labels_mapping(res: Any, net_labels_mapping: Optional[Dict[str, int]]) -> None:
    if net_labels_mapping is None:
        return
    if getattr(res, "analysis_provenance", None) is None:
        res.analysis_provenance = {}
    if isinstance(res.analysis_provenance, dict):
        res.analysis_provenance["net_labels_mapping"] = dict(net_labels_mapping)


def _run_multi_inference(state: _AnalysisState) -> Dict[str, _Inferred]:
    """Several predictors in one design, each against its own null.

    Build X = [intercept, confounds(+site dummies), *interest cols] and a unit
    contrast per predictor. ComBat strategies already ran above with
    preserve=confounds, excluding *all* tested columns.
    """
    if state.contrast_id is not None:
        raise ValueError("contrast_id is set by each predictor name in multi-contrast mode")
    interest = state.interest
    assert isinstance(interest, dict)
    interest_keys = list(interest.keys())
    X, contrasts = _build_multi_design(interest, state.confounds)
    multi_col_names = _design_names(state.confounds, state.confound_names, interest_keys)
    multi_roles = _glm_column_roles(multi_col_names, interest_keys, state.n_user_confounds, state.n_site_dummies)
    results = compute_p_val_glm_multi(
        state.Y, design_matrix=X, contrasts=contrasts,
        design_columns=multi_col_names,
        **_inference_kwargs(state),
        **state.method_kwargs,
    )
    strata_note = _strata_note_text(state.auto_strata, state.signflip_mode, state.paired_interest)
    out: Dict[str, _Inferred] = {}
    for nm, res in results.items():
        _thread_design_provenance(state, res)
        if state.tested_tail is not None:
            res = res._restricted_to_tail(state.tested_tail, selection="prespecified")
        res.harmonized = (
            state.strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM)
            and state.sites is not None
        )
        res.preserve_provided = state.preserve is not None
        res.combat_diagnostics = state.combat_diag
        design_diag = analyze_design_diagnostics(
            design_kind="paired_interest_multi" if state.paired_interest else "glm_multi",
            unit_of_observation="subject_pair_difference" if state.paired_interest else "subject",
            X=X,
            names=multi_col_names,
            column_roles=multi_roles,
            contrast=contrasts,
            focus_contrast=nm,
            sites=state.sites,
            strata=cast(Optional[Sequence[Any]], state.auto_strata),
            interest_name=nm,
            interest_values=np.asarray(interest[nm], dtype=np.float64),
            missing_report=getattr(res, "missing_report", None),
            design_note=(
                "Repeated-measures GLM on B - A across named predictors."
                if state.paired_interest
                else "Multi-contrast GLM; each predictor is tested adjusted for intercept, confounds, and all other predictors."
            ),
            strata_note=strata_note,
        )
        res_flags = list(state.flags)
        if getattr(res, "missing_report", None) is not None:
            from conninfpy.stats._missing_pairwise import describe
            res_flags.append(describe(res.missing_report))
        res.resolved_strategy = state.backend_resolved_strategy
        res.design_diagnostics = design_diag
        res.flags = list(res_flags)
        _attach_net_labels_mapping(res, state.net_labels_mapping)
        if state.motion_qc_diag is not None:
            res.motion_qc = state.motion_qc_diag
        out[nm] = _Inferred(res, design_diag, res_flags)
    return out


def _run_single_inference(
    state: _AnalysisState,
) -> Tuple[Union[InferenceResult, OmnibusInferenceResult], Dict[str, Any]]:
    """One contrast: GLM, repeated-measures GLM or the t-test designs."""
    if state.contrast_id is not None:
        state.method_kwargs["contrast_id"] = state.contrast_id
    strata_note = _strata_note_text(state.auto_strata, state.signflip_mode, state.paired_interest)
    if state.glm_mode:
        # Single-predictor mode: a dict ``interest`` is handled by the multi path.
        assert state.interest is not None and not isinstance(state.interest, dict)
        col_names = _design_names(state.confounds, state.confound_names, ["interest"])
        result = compute_p_val_glm(
            state.Y, interest=state.interest, confounds=state.confounds,
            design_columns=col_names,
            **_inference_kwargs(state),
            **state.method_kwargs,
        )
        X_diag, c_diag = build_design_matrix(state.interest, state.confounds)
        design_diag = analyze_design_diagnostics(
            design_kind="paired_interest" if state.paired_interest else "glm",
            unit_of_observation="subject_pair_difference" if state.paired_interest else "subject",
            X=X_diag,
            names=col_names,
            column_roles=_glm_column_roles(col_names, ["interest"], state.n_user_confounds, state.n_site_dummies),
            contrast=c_diag,
            focus_contrast=state.contrast_id or "interest",
            sites=state.sites,
            strata=cast(Optional[Sequence[Any]], state.auto_strata),
            interest_name=state.contrast_id or "interest",
            interest_values=np.asarray(state.interest, dtype=np.float64).ravel(),
            missing_report=getattr(result, "missing_report", None),
            design_note=(
                "Between-subject predictor regressed on within-subject difference B - A (additive subject-constant site effects cancel in B - A; arbitrary site × condition interactions are not controlled unless modeled)."
                if state.paired_interest
                else "Single-contrast Freedman-Lane GLM."
            ),
            strata_note=strata_note,
        )
        return result, design_diag
    if state.paired_glm_mode:
        # Repeated-measures GLM on Δ = group2 − group1, so `positive` keeps
        # the `group2 > group1` orientation of the no-confound paired path.
        group1, group2 = state.group1, state.group2
        assert group1 is not None and group2 is not None
        assert state.confounds_group1 is not None and state.confounds_group2 is not None
        result = compute_p_val_paired_glm(
            group1, group2,
            confounds_A=state.confounds_group1, confounds_B=state.confounds_group2,
            **_inference_kwargs(state),
            **state.method_kwargs,
        )
        ca = np.asarray(state.confounds_group1, dtype=np.float64)
        cb = np.asarray(state.confounds_group2, dtype=np.float64)
        if ca.ndim == 1:
            ca = ca[:, np.newaxis]
            cb = cb[:, np.newaxis]
        delta_c = cb - ca
        X_diag = np.column_stack([np.ones((group1.shape[0], 1), dtype=np.float64), delta_c])
        col_names = ["intercept"] + [f"delta_confound_{k + 1}" for k in range(delta_c.shape[1])]
        c_diag = np.zeros(X_diag.shape[1], dtype=np.float64)
        c_diag[0] = 1.0
        design_diag = analyze_design_diagnostics(
            design_kind="paired_glm",
            unit_of_observation="subject_pair_difference",
            X=X_diag,
            names=col_names,
            column_roles={"intercept": "condition_difference", **{nm: "confound_difference" for nm in col_names[1:]}},
            contrast=c_diag,
            focus_contrast=state.contrast_id or "condition_difference",
            sites=state.sites,
            strata=cast(Optional[Sequence[Any]], state.auto_strata),
            missing_report=getattr(result, "missing_report", None),
            design_note=(
                "Repeated-measures GLM on within-subject difference (group2 - group1) adjusted for "
                "condition-varying confound differences; subject-constant additive site effects cancel "
                "in differences, though arbitrary site × condition interactions are not controlled."
            ),
            strata_note=strata_note,
        )
        return result, design_diag
    group1, group2 = state.group1, state.group2
    result = compute_p_val(
        group1, group2, test_type=state.test_type,
        var_equal=state.var_equal,
        **_inference_kwargs(state),
        **state.method_kwargs,
    )
    if state.test_type == "two-sample":
        assert group1 is not None and group2 is not None
        n1, n2 = group1.shape[0], group2.shape[0]
        X_diag = np.column_stack([
            np.ones(n1 + n2, dtype=np.float64),
            np.concatenate([np.zeros(n1, dtype=np.float64), np.ones(n2, dtype=np.float64)]),
        ])
        design_diag = analyze_design_diagnostics(
            design_kind="two-sample",
            unit_of_observation="subject",
            X=X_diag,
            names=["intercept", "group"],
            column_roles={"intercept": "intercept", "group": "group_indicator"},
            contrast=np.array([0.0, 1.0], dtype=np.float64),
            focus_contrast=state.contrast_id or "group2 - group1",
            sites=state.sites,
            strata=cast(Optional[Sequence[Any]], state.auto_strata),
            group_labels=["group1"] * n1 + ["group2"] * n2,
            missing_report=getattr(result, "missing_report", None),
            var_equal=state.var_equal,
            design_note=(
                f"Two-sample {'Student (pooled variance)' if state.var_equal else 'Welch (separate variances)'} "
                "t-test; indicator matrix [intercept, group] shown for balance/site/rank diagnostics."
            ),
            strata_note=strata_note,
        )
    elif state.test_type == "paired":
        assert group1 is not None
        n_pairs = group1.shape[0]
        design_diag = analyze_design_diagnostics(
            design_kind="paired",
            unit_of_observation="subject_pair_difference",
            X=np.ones((n_pairs, 1), dtype=np.float64),
            names=["intercept"],
            column_roles={"intercept": "condition_difference"},
            contrast=np.array([1.0], dtype=np.float64),
            focus_contrast=state.contrast_id or "group2 - group1",
            sites=state.sites,
            strata=cast(Optional[Sequence[Any]], state.auto_strata),
            missing_report=getattr(result, "missing_report", None),
            design_note=(
                "Paired within-subject sign-flip test on (group2 - group1). "
                "Subject-constant additive site effects cancel in differences, "
                "though arbitrary site × condition interactions are not controlled."
            ),
            strata_note=strata_note,
        )
    else:
        assert group1 is not None
        n1 = group1.shape[0]
        design_diag = analyze_design_diagnostics(
            design_kind="one-sample",
            unit_of_observation="subject",
            X=np.ones((n1, 1), dtype=np.float64),
            names=["intercept"],
            column_roles={"intercept": "intercept"},
            contrast=np.array([1.0], dtype=np.float64),
            focus_contrast=state.contrast_id or "mean != 0",
            sites=state.sites,
            strata=cast(Optional[Sequence[Any]], state.auto_strata),
            missing_report=getattr(result, "missing_report", None),
            design_note="One-sample within-subject sign-flip test against zero.",
            strata_note=strata_note,
        )
    return result, design_diag


# ---------------------------------------------------------------------------
# 10. Provenance threading (single result)
# ---------------------------------------------------------------------------
def _finalize_single(
    state: _AnalysisState,
    result: Union[InferenceResult, OmnibusInferenceResult],
    design_diag: Dict[str, Any],
) -> Union[InferenceResult, OmnibusInferenceResult]:
    """Thread the upstream facts onto the result, then apply a prespecified tail.

    The lower-level compute_p_val* entry-points don't know that analyze()
    ran ComBat upstream; thread that information onto the result so
    downstream consumers (significant_edges, summary_figure, exporters) can
    report what preprocessing applied. With site_dummies_glm, ComBat is
    skipped even when sites= is passed; harmonized reflects whether the
    matrix was actually adjusted.
    """
    missing_report = getattr(result, "missing_report", None)
    if missing_report is not None:
        from conninfpy.stats._missing_pairwise import describe
        state.flags.append(describe(missing_report))
    result.harmonized = (
        state.strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM) and state.sites is not None
    )
    result.preserve_provided = state.preserve is not None
    result.combat_diagnostics = state.combat_diag
    _thread_design_provenance(state, result)
    # The alignment fact also lives in the analysis provenance so a consumer
    # that only reads provenance sees it; design/paired_details are what the
    # bundle carries across to_bundle().
    if state.paired_alignment is not None:
        result_prov = getattr(result, "analysis_provenance", None)
        if isinstance(result_prov, dict):
            result_prov["paired_alignment"] = deepcopy(state.paired_alignment)
    _attach_net_labels_mapping(result, state.net_labels_mapping)
    # A prespecified direction restricts the finished result after the
    # provenance threading above, so the restriction itself is recorded in
    # provenance and exports ('prespecified', never a post hoc relabelling).
    if state.tested_tail is not None:
        # An F-stat omnibus run cannot declare a tail (rejected by the tail policy).
        assert isinstance(result, InferenceResult)
        result = result._restricted_to_tail(state.tested_tail, selection="prespecified")
    result.resolved_strategy = state.backend_resolved_strategy
    result.design_diagnostics = design_diag
    result.flags = list(state.flags)
    if state.motion_qc_diag is not None:
        result.motion_qc = state.motion_qc_diag
    return result


__all__ = [
    "COMBAT_ONLY",
    "COMBAT_SITE_DUMMIES_GLM",
    "SITE_DUMMIES_GLM",
]
