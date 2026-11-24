"""High-level `analyze()` convenience entry-point.

Wraps the most common neuroimaging recipe in one function call:

    input preparation (zero diagonal, Fisher r→z once) → ComBat (optional) →
    GLM with Freedman-Lane permutation → :class:`InferenceResult`.

Use the lower-level :func:`~conninfpy.compute_p_val` /
:func:`~conninfpy.compute_p_val_glm` entry-points when you need
fine-grained control. Use :func:`analyze` when the standard recipe
fits and you want one line of code.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from conninfpy.defaults import DEFAULT_ACCELERATION
from conninfpy.defaults import DEFAULT_EXTENT_EXPONENT
from conninfpy.defaults import DEFAULT_HEIGHT_EXPONENT
from conninfpy.defaults import DEFAULT_N_PERMUTATIONS
from conninfpy.defaults import DEFAULT_N_THRESHOLDS_PERMUTATION
from conninfpy._provenance import update_analysis_scale
from conninfpy._prepare import normalize_fisher_history
from conninfpy.preprocessing import normalize_preprocessing_record
from conninfpy.results import InferenceResult
from conninfpy.results import OmnibusInferenceResult
from conninfpy._prepare import infer_input_kind
from conninfpy._prepare import prepare_connectivity
from conninfpy._shared.rng import RngLike
from conninfpy._shared.alignment import (
    IdAlignmentError,
    alignment_record,
    check_subject_axis,
    paired_id_permutation,
    reorder_subject_axis,
)
from conninfpy.stats._design import build_design_matrix
from conninfpy.stats.glm import compute_p_val_glm
from conninfpy.stats.glm import compute_p_val_glm_multi
from conninfpy.stats.glm import compute_p_val_paired_glm
from conninfpy.harmonize import combat_harmonize
from conninfpy.harmonize.diagnostics import analyze_design_diagnostics
from conninfpy.stats._types import StatMethod
from conninfpy.stats.pairwise import compute_p_val


COMBAT_ONLY = "combat_only"
COMBAT_SITE_DUMMIES_GLM = "combat_site_dummies_glm"
SITE_DUMMIES_GLM = "site_dummies_glm"


@dataclass
class AnalyzeResult:
    """Bundle of inference + harmonization diagnostics from :func:`analyze`."""

    inference: Union[InferenceResult, OmnibusInferenceResult]
    combat_diagnostics: Optional[Dict[str, Any]] = None
    design_diagnostics: Optional[Dict[str, Any]] = None
    flags: list = field(default_factory=list)
    resolved_strategy: Optional[str] = None
    #: Optional descriptive motion / QC-FC side-car (NETSTAT-28); ``None``
    #: unless the caller opted in with ``motion_qc=``. It never affects
    #: inference and carries its own provenance and unavailable reasons.
    motion_qc: Optional[Dict[str, Any]] = None

    def __getitem__(self, key: str) -> Any:
        # Forward dict-style access to the underlying result so
        # `analyze(...)['positive']` or ['omnibus'] works without users
        # unwrapping.
        return self.inference[key]

    @property
    def positive(self) -> npt.NDArray[np.float64]:
        """FWER-corrected p-values, positive tail."""
        if isinstance(self.inference, OmnibusInferenceResult):
            raise AttributeError(
                "AnalyzeResult.positive is undefined for the F-stat omnibus "
                "path; use AnalyzeResult.omnibus instead."
            )
        return self.inference.positive

    @property
    def negative(self) -> npt.NDArray[np.float64]:
        """FWER-corrected p-values, negative tail."""
        if isinstance(self.inference, OmnibusInferenceResult):
            raise AttributeError(
                "AnalyzeResult.negative is undefined for the F-stat omnibus "
                "path; use AnalyzeResult.omnibus instead."
            )
        return self.inference.negative

    @property
    def omnibus(self) -> npt.NDArray[np.float64]:
        """FWER-corrected p-values of the omnibus test."""
        if not isinstance(self.inference, OmnibusInferenceResult):
            raise AttributeError(
                "AnalyzeResult.omnibus is only defined for the F-stat path; "
                "use .positive / .negative for t / β contrasts."
            )
        return self.inference.omnibus

    @property
    def tail_policy(self) -> str:
        """The declared directional family of the wrapped result
        (``'two-sided'``, ``'separate-one-sided'``, ``'one-sided-<tail>'`` or
        ``'omnibus'``); see :attr:`InferenceResult.tail_policy`."""
        return self.inference.tail_policy

    @property
    def tested_tail(self) -> Optional[str]:
        """The one tested direction of a one-sided result, or ``None``."""
        return getattr(self.inference, "tested_tail", None)

    @property
    def tail_selection(self) -> Optional[str]:
        """How ``tested_tail`` was fixed: ``'prespecified'`` (declared before
        the run via ``analyze(..., tested_tail=...)``), ``'post-hoc'``
        (:meth:`one_sided` on a finished result) or ``None``."""
        return getattr(self.inference, "tail_selection", None)

    @property
    def preprocessing(self) -> Optional[Dict[str, Any]]:
        """Known preprocessing facts (each with a source), or ``None``.

        The record captured by ``analyze(preprocessing=...)``: three-state
        (missing/None = unknown), never inferred from the data values. See
        :mod:`conninfpy.preprocessing`. Also preserved in bundle exports.
        """
        provenance = getattr(self.inference, "analysis_provenance", None)
        return provenance.get("preprocessing") if isinstance(provenance, dict) else None

    @property
    def fisher(self) -> Optional[Dict[str, Any]]:
        """Fisher-transform provenance: applied entries + upstream history.

        ``{'applied': [...], 'upstream': [...] | 'unknown' | None}`` — the
        transforms this call actually applied (policy, parameters, clipping
        counts with explicit units) and, for already-transformed input, the
        caller-supplied upstream history (or ``'unknown'``). Never invented;
        preserved in bundle exports.
        """
        provenance = getattr(self.inference, "analysis_provenance", None)
        return provenance.get("fisher") if isinstance(provenance, dict) else None

    @property
    def two_tailed(self) -> bool:
        """Whether both directions were corrected as one two-sided family."""
        return bool(getattr(self.inference, "two_tailed", False))

    def significant_edges(self, *args: Any, **kwargs: Any) -> "Any":
        """Delegate to the underlying inference result's
        :meth:`~conninfpy.InferenceResult.significant_edges`.
        """
        return self.inference.significant_edges(*args, **kwargs)

    def network_summary(self, atlas: Any, **kwargs: Any) -> "Any":
        """Delegate to the underlying inference result's
        :meth:`~conninfpy.InferenceResult.network_summary`: aggregate the
        result over pairs of atlas networks.
        """
        return self.inference.network_summary(atlas, **kwargs)

    def to_bundle(self, **kwargs: Any) -> "Any":
        """Snapshot the completed inference; see ``ResultBundle.from_result``."""
        from conninfpy.post_analysis import ResultBundle

        return ResultBundle.from_result(self, **kwargs)

    def node_summary(self, *args: Any, **kwargs: Any) -> "Any":
        """Delegate to the underlying inference result's ``node_summary``:
        which ROIs gather the detected connections (descriptive; see
        :meth:`~conninfpy.InferenceResult.node_summary`).
        """
        return self.inference.node_summary(*args, **kwargs)

    @property
    def block_report(self) -> Optional[Dict[str, Any]]:
        """The underlying result's cNBS block report (``None`` for other
        methods); see :attr:`~conninfpy.InferenceResult.block_report`.
        """
        return getattr(self.inference, "block_report", None)

    @property
    def component_report(self) -> Optional[Dict[str, Any]]:
        """The underlying result's NBS component report (``None`` for other
        methods); see :attr:`~conninfpy.InferenceResult.component_report`.
        """
        return getattr(self.inference, "component_report", None)

    def component_table(self, *args: Any, **kwargs: Any) -> "Any":
        """Delegate to the underlying inference result's
        :meth:`~conninfpy.InferenceResult.component_table`: the observed NBS
        components with their FWER p-values.
        """
        return self.inference.component_table(*args, **kwargs)

    def to_csv(self, path: "Any", **kwargs: Any) -> None:
        """Delegate to the underlying inference result's
        :meth:`~conninfpy.InferenceResult.to_csv`.
        """
        return self.inference.to_csv(path, **kwargs)

    def one_sided(self, direction: str) -> "AnalyzeResult":
        """Copy restricted to one direction chosen after the run (post hoc);
        see :meth:`~conninfpy.InferenceResult.one_sided`. To fix the direction
        before the run, call :func:`analyze` with ``tested_tail=``."""
        if isinstance(self.inference, OmnibusInferenceResult):
            raise ValueError("An F-stat omnibus result has no direction to restrict.")
        return replace(self, inference=self.inference.one_sided(direction))

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        parts = [
            f"AnalyzeResult(inference={self.inference!r})",
        ]
        if self.combat_diagnostics is not None:
            ratio = self.combat_diagnostics.get(
                "between_site_variance_ratio_after_over_before"
            )
            if ratio is not None:
                parts.append(f"  combat: between-site var ratio={ratio:.3f}")
        if self.flags:
            parts.append("  flags: " + "; ".join(self.flags))
        return "\n".join(parts)


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


def analyze(
    Y: Optional[npt.NDArray[np.float64]] = None,
    *,
    interest: Optional[
        Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]
    ] = None,
    confounds: Optional[npt.NDArray[np.float64]] = None,
    group1: Optional[npt.NDArray[np.float64]] = None,
    group2: Optional[npt.NDArray[np.float64]] = None,
    confounds_group1: Optional[npt.NDArray[np.float64]] = None,
    confounds_group2: Optional[npt.NDArray[np.float64]] = None,
    ids_group1: Optional[Sequence[Any]] = None,
    ids_group2: Optional[Sequence[Any]] = None,
    interactions: Optional[Sequence[tuple[str, str]]] = None,
    test_type: str = "two-sample",
    sites: Optional[Sequence[Any]] = None,
    preserve: Optional[npt.NDArray[np.float64]] = None,
    harmonize: Optional[str] = "auto",
    input: Optional[str] = None,
    fisher_z: Optional[bool] = None,
    r_to_z: bool = True,
    var_equal: bool = False,
    method: Union[str, StatMethod] = "tfnbs",
    two_tailed: Optional[bool] = None,
    tested_tail: Optional[str] = None,
    acceleration: Optional[str] = DEFAULT_ACCELERATION,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    e: Union[float, Sequence[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, Sequence[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: Optional[int] = DEFAULT_N_THRESHOLDS_PERMUTATION,
    rng: RngLike = None,
    verbose: bool = False,
    use_mp: bool = True,
    n_processes: Optional[int] = None,
    progress: Optional[Any] = None,
    cancel: Optional[Callable[[], bool]] = None,
    missing: str = "raise",
    min_coverage: float = 0.8,
    atlas: Optional[Any] = None,
    roi_ids: Optional[Sequence[Any]] = None,
    groupings: Optional[Dict[str, Sequence[Any]]] = None,
    units: Optional[str] = None,
    contrast_id: Optional[str] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    fisher_history: Optional[Sequence[Mapping[str, Any]]] = None,
    motion_qc: Optional[Mapping[str, Any]] = None,
    **method_kwargs: Any,
) -> Union[AnalyzeResult, Dict[str, AnalyzeResult]]:
    """Run the standard prepare → harmonize → GLM/two-sample → infer recipe.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N), optional
        Per-subject connectivity matrices for the GLM path. Pass this
        with ``interest``/``confounds``. Leave as ``None`` (or simply
        omit) when using the two-sample path via ``group1``/``group2``.
    interest : ndarray or dict of ndarray, optional
        GLM regressor(s) of interest. Triggers the GLM pipeline.

        * **Single predictor** — a 1D array of shape ``(n_subjects,)``
          (or ``(n_subjects, 1)``), e.g. ``age`` or a 0/1 group dummy.
          ``analyze()`` returns one :class:`AnalyzeResult`.
        * **Several predictors** — a dict ``{name: vector}`` (e.g.
          ``{'age': age, 'sex': sex}``). Each predictor is tested adjusted
          for the intercept, ``confounds`` and the other predictors,
          against its own Freedman–Lane null, with permutations shared
          across predictors, via
          :func:`~conninfpy.compute_p_val_glm_multi`. ``analyze()`` then
          returns a ``dict`` mapping each name to its own
          :class:`AnalyzeResult`.

        A bare multi-column array or a Python list of regressors is
        rejected (it cannot carry predictor names and would silently test
        only the last column) — pass a dict instead. For a *joint*
        omnibus test of several predictors use a multi-row F-contrast via
        :func:`~conninfpy.compute_p_val_glm` (``stat_type='fstat'``).
    confounds : ndarray, optional
        Nuisance regressors for the GLM pipeline.
    group1, group2 : ndarray, optional
        Two-sample inputs of shape ``(n_i, N, N)``. Triggers the t-test
        pipeline. With ``interest`` and ``test_type='paired'`` they instead
        form the subject-level GLM response B minus A after transforming each
        condition separately to Fisher z. For
        ``test_type='paired'`` they are the two within-subject
        conditions and must be row-aligned (``group1[s]`` and
        ``group2[s]`` are the same subject) unless ``ids_group1`` /
        ``ids_group2`` name the subjects.
    ids_group1, ids_group2 : sequence of hashable scalars, optional
        Subject IDs for the two paired conditions, in the row order of
        ``group1`` / ``group2``. Paired paths only (``test_type='paired'``
        with ``group1``/``group2`` and no ``Y``); passing them to a
        two-sample, one-sample or Y-only call raises.

        The two vectors must form a **strict full bijection**: the same
        unique, non-missing IDs on both sides, matched by Python equality of
        the opaque values. Nothing is normalised, stringified or type-coerced,
        and no ID is ever guessed from the data. ``None``, ``NaN``, ``NaT``,
        ``pd.NA`` and empty strings are rejected as IDs, as are unhashable
        values. The API raises ``ValueError`` for one-sided IDs (only one
        condition named), an ID present in only one condition (missing pair),
        a repeated ID within a condition, an ID vector whose length disagrees
        with its array's subject axis, and for data with mismatched shapes —
        silently dropping a subject or pairing the wrong rows never happens.

        **Index ownership.** ``group1`` (condition A) is the reference row
        order and the result's order. Exactly one permutation moves every
        condition-B subject-axis input — ``group2`` and
        ``confounds_group2`` — into A order *before* the Fisher transform,
        the B − A difference, the condition-confound difference and any
        inference. Inputs that already live in A order are validated against
        A and **never** permuted: ``interest`` (including interaction
        predictors), subject-level ``confounds``, ``sites`` and
        ``confounds_group1``. Edge-axis objects (``edge_mask``,
        ``net_labels``) are invariant under a subject permutation.
        Without IDs the positional behaviour is unchanged.

        The applied alignment is recorded in ``result.design
        ['paired_alignment']`` (``reference``, ``n_subjects``, the
        ``permutation_b_to_a`` and ``identity_permutation``), in ``flags``
        and, on the paired-interest path, in
        ``design['paired_difference']['alignment']``; it survives
        :meth:`~conninfpy.InferenceResult.to_bundle`.
    confounds_group1, confounds_group2 : ndarray, optional
        Condition-varying confounds for the **repeated-measures GLM**
        path, shape ``(n_subjects,)`` or ``(n_subjects, k)``, aligned to
        ``group1`` / ``group2`` respectively. Only valid with
        ``test_type='paired'``; pass both or neither. When given, the
        paired difference ``group2 - group1`` is tested while the
        per-subject confound difference is partialled out
        (Freedman–Lane on the differences, via
        :func:`~conninfpy.compute_p_val_paired_glm`). Use these for a
        confound whose value differs between conditions for the same
        subject (e.g. condition-level motion, arousal, reaction time).
        With paired ``interest``, ``confounds`` instead specifies subject-level
        covariates; both kinds can be supplied and only the condition-specific
        covariates are differenced (B minus A).
    interactions : sequence of pairs of names, optional
        Requires named numeric ``interest`` predictors. Center all named
        predictors by their sample means and add a product column per pair,
        named ``a:b``. Each named effect is tested with all other columns,
        including the main effects, as nuisance. Encode categories explicitly.
        Subject-constant nuisances (and additive site effects) cancel in
        the difference and do not need to be supplied. With no
        condition-varying confounds the paired path stays on the
        sign-flip t-test (exact non-asymptotic null). Either way the
        result keeps the ``positive = group2 > group1`` orientation.
    test_type : str, default ``'two-sample'``
        Test type for the t-test pipeline (also accepts ``'paired'`` and
        ``'one-sample'``). ``'paired'`` combined with
        ``confounds_group1`` / ``confounds_group2`` selects the
        repeated-measures GLM.
    sites : sequence, optional
        Per-subject site labels. If provided, the permutation engine is
        auto-stratified on ``sites`` (within-block exchangeability —
        equivalent to PALM's ``-eb`` option). ComBat runs when
        ``harmonize`` resolves to ``'combat_only'`` or
        ``'combat_site_dummies_glm'``. In GLM mode with sites but no confounds,
        ``'auto'`` resolves to ``'site_dummies_glm'`` (site dummies in the GLM,
        no ComBat). In two-sample mode with sites, ComBat is skipped and
        a flag asks the caller to promote the analysis to GLM with a
        binary ``interest``.
    preserve : ndarray, optional
        Covariates whose effect should be preserved through ComBat.
        Under ``'combat_only'`` and ``'combat_site_dummies_glm'`` this is set
        automatically to ``confounds``; passing ``preserve`` explicitly is
        overridden and emits a flag because the tested variable must remain
        outside the ComBat design. ``'site_dummies_glm'`` does not run ComBat, so
        ``preserve`` is ignored.
    harmonize : str or ``None``, default ``'auto'``
        Selects the ComBat / site-handling strategy. Three named multi-site
        strategies are supported, matching the paper comparison:

        * ``'combat_only'`` — fit ComBat with
          ``preserve = confounds`` only, then test on the harmonized matrix
          without adding site dummies to the GLM. Use this to
          isolate what the ComBat transform itself contributes.
        * ``'combat_site_dummies_glm'`` — primary site-aware recipe. Fit ComBat with ``preserve = confounds``
          only, deliberately excluding the tested variable, then append site
          dummies to the downstream GLM nuisance design. Use this when the
          analysis needs both a harmonized matrix and residual site control
          during inference.
        * ``'site_dummies_glm'`` (or ``None``) — skip ComBat entirely and model site as fixed-effect nuisance dummies in
          the same GLM that tests the variable of interest. Use this when the
          only deliverable is the inference result, when ComBat assumptions
          are questionable, or as the no-ComBat site-aware sensitivity arm.

        ``'auto'`` (default) dispatches based on the call shape:

        - GLM + ``sites`` + ``confounds`` → ``'combat_site_dummies_glm'``.
        - GLM + ``sites``, no ``confounds`` → ``'site_dummies_glm'``.
        - GLM + no ``sites`` → no harmonization.
        - Two-sample + ``sites`` → skip ComBat with a flag asking
          the caller to promote to GLM with binary ``interest`` for
          the ``'combat_site_dummies_glm'`` recipe.
    input : {'r', 'z'} or None, default ``None``
        Scale of ``Y`` (or ``group1``/``group2``). ``'r'``: correlations,
        Fisher r→z is applied once. ``'z'``: already on the analysis scale
        (e.g. Fisher z), left unchanged. ``None`` detects it: a diagonal of
        ones means ``'r'``, any off-diagonal value beyond ±1 means ``'z'``;
        anything else is ambiguous and raises ``ValueError`` rather than
        risk transforming Fisher-z data twice. The diagonal is always
        zeroed; the caller's arrays are never modified.
    fisher_z : bool, optional
        Legacy alias: ``True`` means ``input='r'``, ``False`` means
        ``input='z'``. Prefer ``input``.
    r_to_z : bool, default True
        Whether correlations (``input='r'``, given or detected) are Fisher
        transformed before inference. ``False`` analyses the correlations
        as they are, as the MATLAB NBS toolbox does; the result then records
        ``design['scale'] = 'correlation'`` instead of ``'fisher_z'``. No
        effect for ``input='z'``.
    var_equal : bool, default False
        Two-sample t-test path only (``group1``/``group2`` with
        ``test_type='two-sample'``): False uses Welch's t, True Student's t
        with a pooled variance (the MATLAB NBS toolbox t-test). The GLM path
        always uses the pooled residual variance, and paired / one-sample
        tests have one group, so ``var_equal=True`` raises there.
    method : str, default ``'tfnbs'``
        Enhancement method.
    two_tailed : bool or None, default None
        Directional family of the corrected p-values. ``None`` (the default)
        and ``True`` test **one two-sided family**: positive and negative
        directions are corrected against the joint max null, so one nominal
        alpha covers both directions. ``False`` keeps the legacy contract of
        two separate one-sided families, each corrected at the nominal alpha
        — two families at α=0.05 do **not** promise a combined-direction FWER
        of 0.05 (use α/2 per tail for that). F-stat omnibus runs have no
        direction and ignore this setting. The low-level
        :func:`~conninfpy.compute_p_val` / :func:`~conninfpy.compute_p_val_glm`
        entry-points keep their own ``two_tailed=False`` default; passing
        ``two_tailed=False`` here explicitly reproduces them exactly.
    tested_tail : {'positive', 'negative'} or None, default None
        A directional hypothesis fixed **before** the run: the inference uses
        per-tail nulls (``two_tailed=False`` at the kernel level) and the
        result is restricted to that direction — the other direction is not
        tested (p = 1, no detections). The result records
        ``tail_policy = 'one-sided-<tested_tail>'`` and
        ``tail_selection = 'prespecified'``, which provenance and exports
        report honestly. Cannot be combined with ``two_tailed=True`` or with
        an F-stat contrast (F is non-directional). Choosing a direction only
        after inspecting a finished run is a post hoc selection: use
        :meth:`~conninfpy.InferenceResult.one_sided` on that result, which is
        marked ``tail_selection = 'post-hoc'`` instead.
    acceleration : str or None, default ``None``
        ``None`` gives exact empirical permutation p-values. ``'gpd'`` /
        ``'gamma'`` replace the p-values that fewer than 10 permutations
        reach with a fitted tail and keep all others exact — with ~200
        permutations, a setting for exploratory runs.
    n_permutations : int, default ``1000``
        Permutations. The same defaults apply to every entry point
        (:mod:`conninfpy.defaults`).
    e, h : float or sequence of float, default ``0.4``, ``3.0``
        TFNBS exponents. Pass equal-length sequences to evaluate a
        whole ``(E, H)`` grid in one call. Exact integration evaluates
        each parameter pair; cost increases with the number of pairs.
        When arrays are passed the returned :class:`InferenceResult`
        carries the parameter axis (``result.is_grid == True``,
        ``result.e_grid``, ``result.h_grid``); use
        ``result.select(param_idx)`` or pass ``param_idx=`` to
        ``significant_edges`` / ``to_csv`` to project to a single
        cell. Default ``(0.4, 3.0)``.
    n : int or None, default None
        Exact integral if None; a positive integer uses the historical right grid.
    rng : int, ``numpy.random.Generator``, or ``None``
        Reproducibility handle.
    verbose : bool, default ``False``
        Show a progress bar during permutation.
    use_mp : bool, default ``True``
        Use multiprocessing for the permutation loop.
    n_processes : int, optional
        Worker processes for the permutation loop (capped at the number of
        planned permutations). ``None`` uses the available logical CPUs.
        Each worker runs with one BLAS/OpenMP thread. The run plan records
        the requested value; the effective pool size is resolved inside the
        engine and may be smaller.
    progress : callable or ProgressSession, optional
        Parent-side progress reporting. A callable receives
        ``(done, total)`` at permutation chunk boundaries for permutations
        that were actually folded; ``total`` is the number of permutation
        executions planned for the call (see
        :class:`conninfpy.stats._workers.ProgressSession`). Nothing is sent
        to the workers and no numerical result changes.
    cancel : callable, optional
        Zero-argument predicate evaluated in the parent process at chunk
        boundaries. When it returns true the permutation loop stops at that
        boundary, the pool is terminated and joined, private task files are
        removed and :class:`~conninfpy.InferenceCancelled` is raised — no
        partial p-values and no ``InferenceResult`` are returned. A
        ``threading.Event`` is therefore never pickled into a worker.
    missing : {'raise', 'available_case'}, default ``'raise'``
        Missing connectivity values (NaN), e.g. a region without signal in
        one subject. ``'raise'`` stops with a report of the affected
        subjects and regions. ``'available_case'`` tests each edge with the
        subjects who have it — no subject is dropped and nothing is filled
        in; edges kept by fewer than ``min_coverage`` of a group's subjects
        are not tested (p = NaN). Works for two-sample, paired and
        one-sample designs and for GLM designs (per missingness pattern),
        including the ComBat strategies (per-feature available-case ComBat).
        The coverage report is in ``result.inference.missing_report`` and
        summarised in ``flags``.
    min_coverage : float, default ``0.8``
        Minimum share of each group's subjects (or of the pairs) an edge
        needs under ``missing='available_case'``.
    atlas : AtlasInfo, optional
        ROI identities and network/hemisphere metadata in input matrix order.
    roi_ids : sequence, optional
        Explicit original ROI IDs when the input has no suitable atlas IDs.
    groupings : dict of str to sequence, optional
        Additional named per-ROI groupings retained alongside atlas groupings.
    units : str, optional
        Response units for the saved effect; defaults to the declared scale or
        ``'unknown response units'`` for low-level calls without a scale.
    contrast_id : str, optional
        Original contrast/predictor name for a single result. Multi-contrast
        results use their dictionary keys instead.
    preprocessing : dict, optional
        Known preprocessing facts to attach to the result's provenance
        (three-state: each fact carries a value and a source; ``None`` or a
        missing fact means unknown, never "not applied"). Typically
        ``dataset.provenance['preprocessing']`` from a loader; see
        :mod:`conninfpy.preprocessing`. Nothing is inferred from the
        connectivity values themselves.
    fisher_history : sequence of dicts, optional
        Structured provenance of Fisher transforms applied to the input
        *before* this call (e.g. the app's ingestion option or design-step
        button). Only meaningful with already-transformed (``input='z'``)
        data; the entries are preserved verbatim as the result's upstream
        history — this call never re-transforms or re-counts them.
        ``input='z'`` without a history records the upstream transform as
        unknown; ``input='r'`` transforms here and records its own entry.
    motion_qc : mapping, optional
        Opt-in descriptive motion / QC-FC side-car diagnostics, computed on
        the native input **before** any Fisher/ComBat step and before
        inference (see :mod:`conninfpy.interpret.motion_qc`). Off by default:
        with ``None`` nothing is computed and the result is unchanged.

        Recognised keys: ``fd`` (phenotype column name or a per-observation
        vector aligned to the analysis observations: ``Y`` for the GLM path,
        ``group1 + group2`` concatenated for the two-sample path, ``group1``
        for a paired/one-sample path), ``fd_group1``/``fd_group2`` (paired
        conditions; ``fd_group2`` follows the original condition-B row order
        and is moved into condition A's order with the same ID permutation as
        the data), ``pheno`` (for FD column lookup and subject-ID checks),
        ``subject_ids`` (exact identity check, no normalisation), ``input``
        (override the resolved ``'r'``/``'z'`` scale), ``nuisance`` (an
        aligned design matrix **including an intercept**; only this enables
        residualisation), ``nuisance_names``, ``coords``/``atlas``/``space``
        (known ROI coordinates for the distance part), ``roi_ids``,
        ``pool`` (paired pooling, only with verified identity),
        ``condition_a``/``condition_b``, ``fd_units``, ``pheno_subject_column``.

        The diagnostics are returned as ``result.motion_qc`` (also attached
        to the wrapped inference result for report/export), with an explicit
        ``status`` and every unavailable reason. They add one ``flags`` entry
        and nothing else: p-maps, statistics, masks, sampling and RNG
        consumption are bit-identical with and without them.
    **method_kwargs
        Forwarded to the underlying pipeline (e.g. ``net_labels``,
        ``min_cluster_size``, ``threshold``).

    Returns
    -------
    AnalyzeResult or dict of AnalyzeResult
        For a single predictor / two-sample / paired call, one
        :class:`AnalyzeResult` carrying ``.inference`` (the
        :class:`~conninfpy.InferenceResult`) plus optional ComBat /
        design diagnostics and any plain-English warning flags. For a
        dict ``interest`` (several predictors), a ``dict`` mapping each
        predictor name to its own :class:`AnalyzeResult`; the shared
        ComBat diagnostics and flags are attached to every entry.
    """
    if method_kwargs:
        # The same controls can arrive through **method_kwargs (e.g. from a
        # wrapper that forwards the compute-function signature); keep one
        # explicit value instead of failing on a duplicate keyword.
        if n_processes is None:
            n_processes = method_kwargs.pop("n_processes", None)
        if progress is None:
            progress = method_kwargs.pop("progress", None)
        if cancel is None:
            cancel = method_kwargs.pop("cancel", None)
    flags: list = []
    combat_diag: Optional[Dict[str, Any]] = None
    # Opt-in, additive motion / QC-FC side-car (NETSTAT-28). ``None`` means the
    # caller did not opt in: no diagnostic is computed and no result changes.
    motion_qc_diag: Optional[Dict[str, Any]] = None
    motion_qc_spec: Optional[Dict[str, Any]] = dict(motion_qc) if motion_qc is not None else None
    # Known preprocessing facts (three-state, with sources); normalized once so
    # every result and export carries the same record. None stays unknown.
    preprocessing_record = normalize_preprocessing_record(preprocessing)
    # Upstream Fisher transforms (applied before this call, e.g. by the app);
    # validated once. Whether they are *applicable* depends on the resolved
    # input scale below.
    fisher_upstream = normalize_fisher_history(fisher_history)
    fisher_applied: List[Dict[str, Any]] = []
    fisher_upstream_state: Any = None  # list of entries | "unknown" | None
    fisher_upstream_decided = False     # paired-interest resolves it on its own

    # ---- Tail policy: two-sided by default, prespecified direction opt-in ----
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
        stat_type_arg = method_kwargs.get("stat_type", "tstat")
        stat_type_arg = getattr(stat_type_arg, "value", stat_type_arg)
        if str(stat_type_arg) == "fstat":
            raise ValueError(
                "tested_tail=... is undefined for an F-stat omnibus contrast: "
                "F is non-negative and has no direction."
            )
    effective_two_tailed = (
        False if tested_tail is not None else (True if two_tailed is None else bool(two_tailed))
    )
    if tested_tail is not None:
        flags.append(
            f"prespecified one-sided test ({tested_tail}): the direction was fixed "
            "before the run and the other direction was not tested."
        )
    elif two_tailed is False:
        flags.append(
            "two_tailed=False: positive and negative are corrected as separate "
            "one-sided families, each at the nominal alpha; the combined-direction "
            "FWER is not the nominal alpha (use alpha/2 per tail for that)."
        )

    # ---- Optional subject-ID alignment for paired designs (NETSTAT-31) ----
    # Condition A (group1) is the reference row order. When the caller names
    # the subjects, verify the A<->B bijection once and move every
    # condition-B subject-axis input into A order *before* the Fisher
    # transform, the B-A differences and any inference. A-order inputs
    # (interest, subject-level confounds, sites, confounds_group1) are
    # validated against A and never permuted; edge-axis masks are invariant.
    paired_alignment = None
    if ids_group1 is not None or ids_group2 is not None:
        if Y is not None or group1 is None or group2 is None or test_type != "paired":
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
            perm = paired_id_permutation(
                ids_group1, ids_group2,
                n_group1=n_a, n_group2=int(b_data.shape[0]),
                name_group1="ids_group1", name_group2="ids_group2",
            )
            a_order = ["interest"] if interest is not None else []
            if isinstance(interest, dict):
                for name, values in interest.items():
                    check_subject_axis(values, f"interest[{name!r}]", n_a, condition="group1")
            else:
                check_subject_axis(interest, "interest", n_a, condition="group1")
            for a_name, a_value in (("confounds", confounds), ("sites", sites),
                                    ("confounds_group1", confounds_group1)):
                if a_value is not None:
                    check_subject_axis(a_value, a_name, n_a, condition="group1")
                    a_order.append(a_name)
            had_condition_confounds = confounds_group2 is not None
            confounds_group2 = reorder_subject_axis(
                confounds_group2, perm, "confounds_group2"
            )
        except IdAlignmentError as exc:
            raise ValueError(str(exc)) from exc
        group2 = b_data[perm]
        reordered = ["group2"] + (["confounds_group2"] if had_condition_confounds else [])
        if motion_qc_spec is not None and motion_qc_spec.get("fd_group2") is not None:
            # The side-car is descriptive, but its condition-B FD must follow the
            # exact same rows as condition B: reorder it with the one permutation.
            from conninfpy.interpret.motion_qc import _resolve_fd_vector

            fd_b = _resolve_fd_vector(
                motion_qc_spec["fd_group2"], motion_qc_spec.get("pheno"),
                int(b_data.shape[0]), "motion_qc['fd_group2']",
            )
            motion_qc_spec["fd_group2"] = fd_b[perm]
            motion_qc_spec["fd_group2_alignment"] = (
                "condition-B FD moved into condition A's row order with the same "
                "subject-ID permutation as condition B"
            )
        if motion_qc_spec is not None and ids_group1 is not None:
            motion_qc_spec["subject_ids"] = tuple(ids_group1)
            # analyze verified the A<->B bijection and moved condition B (and its
            # FD) into A order; the descriptive side-car records that fact.
            motion_qc_spec["id_alignment_verified"] = "exact_id_match"
        paired_alignment = alignment_record(
            perm,
            n_subjects=n_a,
            reference="group1 (condition A)",
            reordered_inputs=reordered,
            a_order_inputs=a_order,
        )
        if paired_alignment["identity_permutation"]:
            flags.append(
                f"paired subject IDs checked: {n_a} subjects form a one-to-one "
                "match between group1 and group2; condition B already followed "
                "condition A's row order (no permutation applied)."
            )
        else:
            flags.append(
                f"paired subject IDs aligned: {n_a} subjects matched one-to-one by "
                f"ID; {', '.join(reordered)} permuted into condition A's row order "
                "before the Fisher transform and the B - A difference. "
                "A-order inputs (interest, subject-level confounds, sites) were "
                "validated against A and not permuted."
            )

    # A repeated-measures regression has one observation per subject.  Do
    # the nonlinear transform before subtraction, never on the difference.
    paired_interest = interest is not None and (group1 is not None or group2 is not None)
    derived_scale = None
    derived_transform = None
    paired_details = None
    centered_means = None
    if paired_interest:
        if Y is not None or group1 is None or group2 is None or test_type != "paired":
            raise ValueError("paired interest requires group1 and group2, test_type='paired', and no Y")
        a, b = np.asarray(group1), np.asarray(group2)
        if a.ndim != 3 or a.shape != b.shape:
            raise ValueError("paired conditions must have identical (subjects, regions, regions) shapes and subject order")
        original_kind = _resolve_input_kind(input, fisher_z, [a, b])
        if original_kind == "r" and not r_to_z:
            raise ValueError("paired interest with correlations requires Fisher transformation; r_to_z=False is not supported")
        if harmonize not in (None, "auto"):
            raise ValueError("paired interest supports harmonize=None or 'auto'; harmonization of condition differences is not automatic")
        if motion_qc_spec is not None:
            motion_qc_diag = _motion_qc_paired_diagnostics(
                motion_qc_spec, a, b, original_kind, atlas
            )
            flags.append(_motion_qc_flag(motion_qc_diag))
        if (confounds_group1 is None) != (confounds_group2 is None):
            raise ValueError("supply both condition-specific confounds or neither")
        n_subject_confounds = 0 if confounds is None else _as_2d(confounds).shape[1]
        n_condition_confounds = 0
        if confounds_group1 is not None:
            ca, cb = _as_2d(confounds_group1), _as_2d(confounds_group2)
            if ca.shape != cb.shape or ca.shape[0] != a.shape[0]:
                raise ValueError("condition-specific confounds must align with paired subjects and have equal shapes")
            delta_c = cb.astype(float) - ca.astype(float)
            n_condition_confounds = delta_c.shape[1]
            confounds = delta_c if confounds is None else np.column_stack((_as_2d(confounds), delta_c))
        paired_details = {"input": original_kind, "orientation": "B - A",
                          "subject_confound_columns": n_subject_confounds,
                          "condition_confound_columns": n_condition_confounds,
                          "condition_confound_transform": "B - A"}
        if paired_alignment is not None:
            paired_details["alignment"] = deepcopy(paired_alignment)
        allow_missing_pair = missing == "available_case"
        # Fisher is applied to each condition separately, never to the
        # difference; the captured entries record exactly that (stage
        # 'analysis_prepare_before_difference', scopes group1/group2).
        b_pre, fisher_entry_b = prepare_connectivity(
            b, original_kind, "group2", allow_missing=allow_missing_pair,
            stage="analysis_prepare_before_difference", return_report=True)
        a_pre, fisher_entry_a = prepare_connectivity(
            a, original_kind, "group1", allow_missing=allow_missing_pair,
            stage="analysis_prepare_before_difference", return_report=True)
        Y = b_pre - a_pre
        fisher_applied = [e for e in (fisher_entry_a, fisher_entry_b) if e is not None]
        if original_kind == "z":
            fisher_upstream_state = fisher_upstream if fisher_upstream is not None else "unknown"
        fisher_upstream_decided = True
        derived_scale = "difference_fisher_z"
        derived_transform = "Fisher r-to-z separately for each condition, then B minus A" if original_kind == "r" else "B minus A of Fisher z conditions"
        group1 = group2 = confounds_group1 = confounds_group2 = None
        input, fisher_z = "z", None

    if interactions is not None:
        if not isinstance(interest, dict) or not interest:
            raise ValueError("interactions require a nonempty named interest dictionary of numeric predictors")
        centered = {}
        centered_means = {}
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
        interest = centered

    glm_mode = interest is not None
    multi_mode = glm_mode and isinstance(interest, dict)
    ttest_mode = group1 is not None or group2 is not None
    if glm_mode and ttest_mode:
        raise ValueError(
            "analyze() takes either (interest, [confounds]) for the GLM path or "
            "(group1, group2) for the t-test path, not both."
        )
    if not glm_mode and not ttest_mode:
        raise ValueError(
            "analyze() requires either an `interest` regressor or `group1`/`group2`."
        )
    if glm_mode and Y is None:
        raise ValueError(
            "analyze() requires Y when using the GLM path (interest=, "
            "[confounds=]). Pass Y as the first positional argument."
        )
    if var_equal and not (ttest_mode and test_type == "two-sample"):
        raise ValueError(
            "var_equal applies to the two-sample t-test (group1=, group2=, "
            "test_type='two-sample'). The GLM t uses a pooled residual variance "
            "by construction; paired and one-sample tests have one group."
        )
    if multi_mode:
        # Several predictors → compute_p_val_glm_multi. Each value must be a
        # single 1D regressor; the dict key names the per-predictor result.
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
    elif glm_mode:
        # `interest` is a SINGLE predictor. A (n, k>1) array or a Python list
        # of regressors would be tested by build_design_matrix's contrast on
        # the *last* column only (silently demoting the rest to nuisance), and
        # a list like [age, sex] is read as shape (k, n) and crashes
        # downstream. Reject both with a pointer to the dict (multi) API.
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
    has_cg1 = confounds_group1 is not None
    has_cg2 = confounds_group2 is not None
    paired_glm_mode = has_cg1 or has_cg2
    if paired_glm_mode:
        if has_cg1 != has_cg2:
            raise ValueError(
                "Pass both confounds_group1 and confounds_group2, or neither — "
                "the repeated-measures GLM needs the confound in both conditions "
                "to form the per-subject difference."
            )
        if not ttest_mode:
            raise ValueError(
                "confounds_group1/confounds_group2 require the paired t-test "
                "inputs (group1=, group2=). For a between-subject continuous "
                "predictor with confounds use Y=, interest=, confounds=."
            )
        if test_type != "paired":
            raise ValueError(
                "confounds_group1/confounds_group2 are only valid with "
                f"test_type='paired'; got test_type={test_type!r}. Condition-"
                "varying confounds are differenced within subject, which only "
                "makes sense for a repeated-measures design."
            )
    if ttest_mode and confounds is not None:
        raise ValueError(
            "confounds= is the GLM-path nuisance design (used with interest=). "
            "For a paired design with condition-varying confounds use "
            "confounds_group1=/confounds_group2=; for a two-sample contrast with "
            "nuisance covariates promote to GLM with a binary interest= column."
        )

    had_user_confounds = confounds is not None
    n_user_confounds = 0 if confounds is None else _as_2d(confounds).shape[1]

    strategy = _resolve_strategy(
        harmonize,
        glm_mode=glm_mode,
        ttest_mode=ttest_mode,
        sites_provided=sites is not None,
        confounds_provided=confounds is not None,
    )
    if paired_interest:
        # Site labels constrain between-subject permutations, while
        # subject-constant additive site effects cancel in B - A.
        strategy = None
        if sites is not None:
            flags.append(
                "paired differences (GLM on B - A): ComBat skipped — "
                "subject-constant additive site effects cancel in B - A "
                "(arbitrary site × condition interactions are not automatically "
                "controlled unless modeled); sites= stratifies between-subject "
                "permutations."
            )

    # Explicit ComBat guards (only fire for explicit requests; 'auto'
    # resolves to combat_site_dummies_glm only when these guards are satisfied).
    if harmonize in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM):
        if sites is None:
            raise ValueError(
                f"{strategy} requires sites=...; otherwise "
                "there is no ComBat step to control."
            )
        if confounds is None:
            raise ValueError(
                f"{strategy} requires confounds=; otherwise "
                "the ComBat preserve design would be empty."
            )
        if not glm_mode:
            raise ValueError(
                f"{strategy} is GLM-only (interest=, confounds=). For "
                "two-sample / paired designs use "
                "harmonize='auto' or None."
            )
    # Two-sample + sites resolves to no-ComBat (no defensible preserve).
    # Flag so the demotion is visible and the caller can promote to GLM.
    if (
        ttest_mode
        and test_type != "paired"
        and sites is not None
        and harmonize in ("auto", None)
    ):
        flags.append(
            "two-sample + sites: ComBat skipped (no interest column to "
            "preserve). For combat_site_dummies_glm, promote to GLM "
            "with a binary interest indicator."
        )
    # Paired + sites: subject-constant additive site effects cancel in
    # group2 - group1 (not arbitrary site × condition effects), and
    # within-subject sign-flip permutations are stratum-invariant by construction.
    if (
        ttest_mode
        and test_type == "paired"
        and sites is not None
        and harmonize in ("auto", None)
    ):
        flags.append(
            "paired + sites: ComBat skipped — subject-constant additive site "
            "effects cancel in the within-subject difference (arbitrary site × "
            "condition interactions are not automatically controlled); "
            "within-subject sign-flip permutations are stratum-invariant by "
            "construction, so sites/strata do not alter the sign-flip null."
        )
    if strategy is None and preserve is not None:
        flags.append(
            "preserve= ignored because no ComBat harmonization strategy is "
            "active."
        )
        preserve = None

    # ---- Input preparation: zero diagonal, Fisher r→z exactly once ----
    stacks = [Y] if glm_mode else [g for g in (group1, group2) if g is not None]
    kind = _resolve_input_kind(input, fisher_z, stacks)
    if motion_qc_spec is not None and motion_qc_diag is None:
        # Described on the caller's native input, before Fisher/ComBat/inference.
        if ttest_mode and test_type == "paired" and group1 is not None and group2 is not None:
            motion_qc_diag = _motion_qc_paired_diagnostics(
                motion_qc_spec, group1, group2, kind, atlas
            )
        else:
            if glm_mode:
                mq_data = Y
            elif test_type == "two-sample" and group2 is not None:
                mq_data = np.concatenate([group1, group2], axis=0)
            else:
                mq_data = group1
            motion_qc_diag = _motion_qc_single_diagnostics(
                motion_qc_spec, mq_data, kind, atlas
            )
        flags.append(_motion_qc_flag(motion_qc_diag))
    # Already-transformed input: nothing is applied here. The upstream
    # transform is the caller's history when supplied, else explicitly
    # unknown — never an invented policy or zero-count clip. (The
    # paired-interest path decided this above on the original conditions.)
    if kind == "z" and not fisher_upstream_decided:
        fisher_upstream_state = fisher_upstream if fisher_upstream is not None else "unknown"
    analysis_scale = "correlation" if (kind == "r" and not r_to_z) else "fisher_z"
    if derived_scale:
        analysis_scale = derived_scale
    if analysis_scale == "correlation":
        # prepare_connectivity(..., 'z') checks and zeroes the diagonal
        # without transforming: the correlations are analysed as given.
        kind = "z"
    if missing not in ("raise", "available_case"):
        raise ValueError(f"missing must be 'raise' or 'available_case'; got {missing!r}.")
    allow_missing = missing == "available_case"
    if glm_mode:
        Y, _fisher_entry_y = prepare_connectivity(
            Y, kind, "Y", allow_missing=allow_missing, return_report=True)
        if _fisher_entry_y is not None:
            fisher_applied.append(_fisher_entry_y)
    else:
        if group1 is not None:
            group1, _fisher_entry_g1 = prepare_connectivity(
                group1, kind, "group1", allow_missing=allow_missing, return_report=True)
            if _fisher_entry_g1 is not None:
                fisher_applied.append(_fisher_entry_g1)
        if group2 is not None:
            group2, _fisher_entry_g2 = prepare_connectivity(
                group2, kind, "group2", allow_missing=allow_missing, return_report=True)
            if _fisher_entry_g2 is not None:
                fisher_applied.append(_fisher_entry_g2)
    fisher_record: Dict[str, Any] = {
        "schema_version": 1,
        "applied": fisher_applied,
        "upstream": fisher_upstream_state,
    }

    # ---- ComBat strategies: preserve = confounds only ----
    # The tested variable is deliberately omitted from ComBat so the
    # harmonization fit never sees the labels that the permutation will
    # reshuffle (Nygaard 2016 label-leak avoidance). Mild attenuation
    # when corr(site, interest) > 0; that's the accepted trade-off.
    if strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM):
        if preserve is not None:
            flags.append(
                f"preserve= overridden by {strategy}: "
                "ComBat preserve is set to confounds only."
            )
        preserve = _as_2d(confounds)
        flags.append(
            f"{strategy}: preserve excludes interest."
        )

    # ---- ComBat (runs for combat_only and combat_site_dummies_glm) ----
    if strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM) and sites is not None:
        assert Y is not None
        res = combat_harmonize(Y, sites=np.asarray(sites), preserve=preserve, missing=missing)
        Y = res.Y_adjusted
        combat_diag = dict(res.diagnostics)
        combat_diag["strategy"] = strategy
        combat_diag["preserve_columns"] = "confounds_only"
        ratio = combat_diag.get("between_site_variance_ratio_after_over_before")
        if ratio is not None and ratio > 0.5:
            flags.append(
                f"ComBat removed less than half of between-site variance "
                f"(ratio after/before = {ratio:.3f})."
            )
        lost = combat_diag.get("values_not_harmonized", 0)
        if lost:
            flags.append(
                f"ComBat could not harmonize {lost} value(s) (fewer than 2 subjects "
                "of their site have them); they are treated as missing."
            )

    confound_names: Optional[List[str]] = None
    n_site_dummies = 0
    # ---- Site dummies → GLM nuisance for the site-GLM strategies ----
    # combat_site_dummies_glm: ComBat harmonized the matrix but did not see interest;
    # site dummies in the GLM still help in case any site-correlated noise
    # survives. site_dummies_glm: no ComBat ran at all; site dummies are the only site
    # adjustment.
    if (
        glm_mode
        and sites is not None
        and strategy in (COMBAT_SITE_DUMMIES_GLM, SITE_DUMMIES_GLM)
    ):
        site_dum = _site_dummies(np.asarray(sites))
        if site_dum.shape[1] > 0:
            n_user = 0 if confounds is None else _as_2d(confounds).shape[1]
            n_site_dummies = int(site_dum.shape[1])
            site_names = [f"site[{u}]" for u in np.unique(np.asarray(sites))[1:]]
            confound_names = [f"confound_{k + 1}" for k in range(n_user)] + site_names
            if confounds is None:
                confounds = site_dum
            else:
                confounds = np.column_stack([_as_2d(confounds), site_dum])
        if strategy == COMBAT_SITE_DUMMIES_GLM:
            flags.append(
                "combat_site_dummies_glm: site dummies appended to GLM nuisance."
            )
        if strategy == SITE_DUMMIES_GLM:
            if preserve is not None:
                flags.append(
                    "preserve= ignored by site_dummies_glm because "
                    "ComBat does not run."
                )
                preserve = None
            if harmonize == "auto" and not had_user_confounds:
                flags.append(
                    "site_dummies_glm: no ComBat (harmonize='auto' without confounds= "
                    "skips ComBat because no non-site covariates were given to "
                    "preserve); site dummies appended to GLM nuisance."
                )
            else:
                flags.append(
                    "site_dummies_glm: no ComBat; site dummies appended to GLM nuisance."
                )
            combat_diag = {
                "strategy": SITE_DUMMIES_GLM,
                "preserve_columns": None,
            }

    # ---- Inference ----
    # Auto-stratify the permutation engine on sites when present (PALM -eb
    # semantics). For two-sample / GLM / paired_interest this constrains
    # label/row permutations within site; for paired / one-sample sign-flip
    # permutations, the sign-flip null is stratum-invariant by construction
    # and `strata` is recorded for provenance and per-subject site checks.
    # An explicit strata= passed via **method_kwargs wins.
    signflip_mode = ttest_mode and test_type in ("paired", "one-sample")
    if "strata" in method_kwargs:
        user_strata = method_kwargs.pop("strata")
        auto_strata = (
            np.asarray(user_strata) if user_strata is not None else None
        )
    else:
        auto_strata = np.asarray(sites) if sites is not None else None
        if auto_strata is not None:
            if (
                ttest_mode
                and test_type == "paired"
                and group1 is not None
                and auto_strata.shape[0] != group1.shape[0]
            ):
                raise ValueError(
                    "For paired inference, sites/strata must provide one label per "
                    f"paired subject ({group1.shape[0]}), got {auto_strata.shape[0]}."
                )
            if signflip_mode:
                flags.append(
                    "strata= auto-set to `sites` (validated and recorded for "
                    "provenance; within-subject/one-sample sign-flip permutations "
                    "are stratum-invariant by construction); pass strata= "
                    "explicitly (or strata=None) to override."
                )
            else:
                flags.append(
                    "strata= auto-set to `sites` (restricts label/row permutations "
                    "within site); pass strata= explicitly (or strata=None) to "
                    "override."
                )

    def _strata_note_text() -> Optional[str]:
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

    def _glm_column_roles(col_names: Sequence[str], interest_cols: Sequence[str]) -> Dict[str, str]:
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

    if multi_mode:
        if contrast_id is not None:
            raise ValueError("contrast_id is set by each predictor name in multi-contrast mode")
        # Several predictors in one design, each against its own null.
        # Build X = [intercept, confounds(+site dummies), *interest cols]
        # and a unit contrast per predictor. ComBat strategies already ran
        # above with preserve=confounds, excluding *all* tested columns.
        assert isinstance(interest, dict)
        interest_keys = list(interest.keys())
        X, contrasts = _build_multi_design(interest, confounds)
        multi_col_names = _design_names(confounds, confound_names, interest_keys)
        multi_roles = _glm_column_roles(multi_col_names, interest_keys)
        results = compute_p_val_glm_multi(
            Y, design_matrix=X, contrasts=contrasts,
            design_columns=multi_col_names,
            method=method, n_permutations=n_permutations,
            two_tailed=effective_two_tailed,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            progress=progress, cancel=cancel, n_processes=n_processes,
            strata=auto_strata, missing=missing, min_coverage=min_coverage,
            atlas=atlas, roi_ids=roi_ids, groupings=groupings, units=units,
            **method_kwargs,
        )
        out: Dict[str, AnalyzeResult] = {}
        for nm, res in results.items():
            if getattr(res, "design", None) is not None:
                res.design["scale"] = analysis_scale
                if derived_transform:
                    res.design["response_transform"] = derived_transform
                    res.design["paired_difference"] = paired_details.copy()
                if centered_means is not None:
                    res.design["predictor_centering_means"] = centered_means.copy()
                if paired_alignment is not None:
                    res.design["paired_alignment"] = deepcopy(paired_alignment)
                update_analysis_scale(res, analysis_scale, units or _estimator_aware_units(analysis_scale, preprocessing_record) or ("Fisher z difference (B - A)" if derived_scale else None))
                _thread_preprocessing(res, preprocessing_record)
                _thread_fisher(res, fisher_record)
            if tested_tail is not None:
                res = res._restricted_to_tail(tested_tail, selection="prespecified")
            res.harmonized = (
                strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM)
                and sites is not None
            )
            res.preserve_provided = preserve is not None
            res.combat_diagnostics = combat_diag
            design_diag = analyze_design_diagnostics(
                design_kind="paired_interest_multi" if paired_interest else "glm_multi",
                unit_of_observation="subject_pair_difference" if paired_interest else "subject",
                X=X,
                names=multi_col_names,
                column_roles=multi_roles,
                contrast=contrasts,
                focus_contrast=nm,
                sites=sites,
                strata=auto_strata,
                interest_name=nm,
                interest_values=np.asarray(interest[nm], dtype=np.float64),
                missing_report=getattr(res, "missing_report", None),
                design_note=(
                    "Repeated-measures GLM on B - A across named predictors."
                    if paired_interest
                    else "Multi-contrast GLM; each predictor is tested adjusted for intercept, confounds, and all other predictors."
                ),
                strata_note=_strata_note_text(),
            )
            res_flags = list(flags)
            if getattr(res, "missing_report", None) is not None:
                from conninfpy.stats._missing_pairwise import describe
                res_flags.append(describe(res.missing_report))
            backend_resolved_strategy = strategy if strategy is not None else "none"
            res.resolved_strategy = backend_resolved_strategy
            res.design_diagnostics = design_diag
            res.flags = list(res_flags)
            if motion_qc_diag is not None:
                res.motion_qc = motion_qc_diag
            out[nm] = AnalyzeResult(
                inference=res,
                combat_diagnostics=combat_diag,
                design_diagnostics=design_diag,
                flags=res_flags,
                resolved_strategy=backend_resolved_strategy,
                motion_qc=motion_qc_diag,
            )
        return out

    if contrast_id is not None:
        method_kwargs["contrast_id"] = contrast_id
    if glm_mode:
        col_names = _design_names(confounds, confound_names, ["interest"])
        result = compute_p_val_glm(
            Y, interest=interest, confounds=confounds,
            design_columns=col_names,
            method=method, n_permutations=n_permutations,
            two_tailed=effective_two_tailed,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            progress=progress, cancel=cancel, n_processes=n_processes,
            strata=auto_strata, missing=missing, min_coverage=min_coverage,
            atlas=atlas, roi_ids=roi_ids, groupings=groupings, units=units,
            **method_kwargs,
        )
        X_diag, c_diag = build_design_matrix(interest, confounds)
        design_diag = analyze_design_diagnostics(
            design_kind="paired_interest" if paired_interest else "glm",
            unit_of_observation="subject_pair_difference" if paired_interest else "subject",
            X=X_diag,
            names=col_names,
            column_roles=_glm_column_roles(col_names, ["interest"]),
            contrast=c_diag,
            focus_contrast=contrast_id or "interest",
            sites=sites,
            strata=auto_strata,
            interest_name=contrast_id or "interest",
            interest_values=np.asarray(interest, dtype=np.float64).ravel(),
            missing_report=getattr(result, "missing_report", None),
            design_note=(
                "Between-subject predictor regressed on within-subject difference B - A (additive subject-constant site effects cancel in B - A; arbitrary site × condition interactions are not controlled unless modeled)."
                if paired_interest
                else "Single-contrast Freedman-Lane GLM."
            ),
            strata_note=_strata_note_text(),
        )
    elif paired_glm_mode:
        # Repeated-measures GLM on Δ = group2 − group1, so `positive` keeps
        # the `group2 > group1` orientation of the no-confound paired path.
        result = compute_p_val_paired_glm(
            group1, group2,
            confounds_A=confounds_group1, confounds_B=confounds_group2,
            method=method, n_permutations=n_permutations,
            two_tailed=effective_two_tailed,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            progress=progress, cancel=cancel, n_processes=n_processes,
            strata=auto_strata, missing=missing, min_coverage=min_coverage,
            atlas=atlas, roi_ids=roi_ids, groupings=groupings, units=units,
            **method_kwargs,
        )
        ca = np.asarray(confounds_group1, dtype=np.float64)
        cb = np.asarray(confounds_group2, dtype=np.float64)
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
            focus_contrast=contrast_id or "condition_difference",
            sites=sites,
            strata=auto_strata,
            missing_report=getattr(result, "missing_report", None),
            design_note=(
                "Repeated-measures GLM on within-subject difference (group2 - group1) adjusted for "
                "condition-varying confound differences; subject-constant additive site effects cancel "
                "in differences, though arbitrary site × condition interactions are not controlled."
            ),
            strata_note=_strata_note_text(),
        )
    else:
        result = compute_p_val(
            group1, group2, test_type=test_type,
            method=method, n_permutations=n_permutations,
            two_tailed=effective_two_tailed,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            progress=progress, cancel=cancel, n_processes=n_processes,
            strata=auto_strata, missing=missing, min_coverage=min_coverage,
            var_equal=var_equal,
            atlas=atlas, roi_ids=roi_ids, groupings=groupings, units=units,
            **method_kwargs,
        )
        if test_type == "two-sample":
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
                focus_contrast=contrast_id or "group2 - group1",
                sites=sites,
                strata=auto_strata,
                group_labels=["group1"] * n1 + ["group2"] * n2,
                missing_report=getattr(result, "missing_report", None),
                var_equal=var_equal,
                design_note=(
                    f"Two-sample {'Student (pooled variance)' if var_equal else 'Welch (separate variances)'} "
                    "t-test; indicator matrix [intercept, group] shown for balance/site/rank diagnostics."
                ),
                strata_note=_strata_note_text(),
            )
        elif test_type == "paired":
            assert group1 is not None
            n_pairs = group1.shape[0]
            design_diag = analyze_design_diagnostics(
                design_kind="paired",
                unit_of_observation="subject_pair_difference",
                X=np.ones((n_pairs, 1), dtype=np.float64),
                names=["intercept"],
                column_roles={"intercept": "condition_difference"},
                contrast=np.array([1.0], dtype=np.float64),
                focus_contrast=contrast_id or "group2 - group1",
                sites=sites,
                strata=auto_strata,
                missing_report=getattr(result, "missing_report", None),
                design_note=(
                    "Paired within-subject sign-flip test on (group2 - group1). "
                    "Subject-constant additive site effects cancel in differences, "
                    "though arbitrary site × condition interactions are not controlled."
                ),
                strata_note=_strata_note_text(),
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
                focus_contrast=contrast_id or "mean != 0",
                sites=sites,
                strata=auto_strata,
                missing_report=getattr(result, "missing_report", None),
                design_note="One-sample within-subject sign-flip test against zero.",
                strata_note=_strata_note_text(),
            )

    if getattr(result, "missing_report", None) is not None:
        from conninfpy.stats._missing_pairwise import describe
        flags.append(describe(result.missing_report))

    # ---- Provenance threading ----
    # The lower-level compute_p_val* entry-points don't know that
    # analyze() ran ComBat upstream; thread that information onto the
    # result so downstream consumers (significant_edges, summary_figure,
    # exporters) can report what preprocessing applied. With site_dummies_glm, ComBat
    # is skipped even when sites= is passed; harmonized reflects whether the
    # matrix was actually adjusted.
    result.harmonized = (
        strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM) and sites is not None
    )
    result.preserve_provided = preserve is not None
    result.combat_diagnostics = combat_diag
    if getattr(result, "design", None) is not None:
        result.design["scale"] = analysis_scale
        if derived_transform:
            result.design["response_transform"] = derived_transform
            result.design["paired_difference"] = paired_details.copy()
        if centered_means is not None:
            result.design["predictor_centering_means"] = centered_means.copy()
        if paired_alignment is not None:
            result.design["paired_alignment"] = deepcopy(paired_alignment)
        update_analysis_scale(result, analysis_scale, units or _estimator_aware_units(analysis_scale, preprocessing_record) or ("Fisher z difference (B - A)" if derived_scale else None))
        _thread_preprocessing(result, preprocessing_record)
        _thread_fisher(result, fisher_record)
    # The alignment fact also lives in the analysis provenance so a consumer
    # that only reads provenance sees it; design/paired_details are what the
    # bundle carries across to_bundle().
    if paired_alignment is not None:
        result_prov = getattr(result, "analysis_provenance", None)
        if isinstance(result_prov, dict):
            result_prov["paired_alignment"] = deepcopy(paired_alignment)
    # A prespecified direction restricts the finished result after the
    # provenance threading above, so the restriction itself is recorded in
    # provenance and exports ('prespecified', never a post hoc relabelling).
    if tested_tail is not None:
        result = result._restricted_to_tail(tested_tail, selection="prespecified")
    # strata_provided was set inside compute_p_val{,_glm} from the strata=
    # argument; preserved here for clarity.
    backend_resolved_strategy = strategy if strategy is not None else "none"
    result.resolved_strategy = backend_resolved_strategy
    result.design_diagnostics = design_diag
    result.flags = list(flags)
    if motion_qc_diag is not None:
        result.motion_qc = motion_qc_diag

    return AnalyzeResult(
        inference=result,
        combat_diagnostics=combat_diag,
        design_diagnostics=design_diag,
        flags=flags,
        resolved_strategy=backend_resolved_strategy,
        motion_qc=motion_qc_diag,
    )


__all__ = ["analyze", "AnalyzeResult"]
