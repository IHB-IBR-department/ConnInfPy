"""High-level `analyze()` convenience entry-point.

Wraps the most common neuroimaging recipe in one function call:

    input preparation (zero diagonal, Fisher r→z once) → ComBat (optional) →
    GLM with Freedman-Lane permutation → :class:`InferenceResult`.

Use the lower-level :func:`~conninfpy.compute_p_val` /
:func:`~conninfpy.compute_p_val_glm` entry-points when you need
fine-grained control. Use :func:`analyze` when the standard recipe
fits and you want one line of code.

The pipeline is decomposed into private stages in
:mod:`conninfpy._analysis_stages`; :func:`analyze` below only normalizes the
keyword aliases, builds the shared state and calls the stages in order.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from conninfpy._analysis_stages import (
    _add_site_dummies,
    _align_paired_subjects,
    _AnalysisState,
    _expand_interactions,
    _finalize_single,
    _prepare_inputs,
    _reduce_paired_interest,
    _resolve_harmonization_strategy,
    _resolve_modes,
    _resolve_strata,
    _resolve_tail_policy,
    _run_combat,
    _run_multi_inference,
    _run_single_inference,
)
# Named strategies and private helpers kept importable from
# ``conninfpy.analysis`` for the CLI and the test-suite (explicit re-exports).
from conninfpy._analysis_stages import COMBAT_ONLY as COMBAT_ONLY
from conninfpy._analysis_stages import COMBAT_SITE_DUMMIES_GLM as COMBAT_SITE_DUMMIES_GLM
from conninfpy._analysis_stages import SITE_DUMMIES_GLM as SITE_DUMMIES_GLM
from conninfpy._analysis_stages import _encode_net_labels as _encode_net_labels
from conninfpy._analysis_stages import _motion_qc_paired_diagnostics as _motion_qc_paired_diagnostics
from conninfpy._analysis_stages import _motion_qc_single_diagnostics as _motion_qc_single_diagnostics
from conninfpy._analysis_stages import _resolve_strategy as _resolve_strategy
from conninfpy._analysis_stages import _site_dummies as _site_dummies
from conninfpy._analysis_stages import _site_interest_association as _site_interest_association
from conninfpy._prepare import normalize_fisher_history
from conninfpy.preprocessing import normalize_preprocessing_record
from conninfpy.results import InferenceResult
from conninfpy.results import OmnibusInferenceResult
from conninfpy._shared.rng import RngLike
from conninfpy.defaults import DEFAULT_ACCELERATION
from conninfpy.defaults import DEFAULT_EXTENT_EXPONENT
from conninfpy.defaults import DEFAULT_HEIGHT_EXPONENT
from conninfpy.defaults import DEFAULT_N_PERMUTATIONS
from conninfpy.defaults import DEFAULT_N_THRESHOLDS_PERMUTATION
from conninfpy.stats._types import StatMethod


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

    def methods_boilerplate(
        self,
        *,
        alpha: float = 0.05,
        atlas: Optional[Any] = None,
        cerebellum: Optional[str] = None,
    ) -> str:
        """Generate a deterministic, provenance-driven Methods section in Markdown."""
        from conninfpy.reporting import methods_boilerplate

        return methods_boilerplate(self, alpha=alpha, atlas=atlas, cerebellum=cerebellum)

    def to_brainnet(
        self,
        atlas: Any,
        out_prefix: "Any",
        *,
        alpha: float = 0.05,
        tail: str = "both",
        param_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Export thresholded edges and nodes to BrainNet Viewer ``.node`` / ``.edge`` files."""
        from conninfpy.reporting import export_brainnet

        return export_brainnet(
            self,
            atlas,
            out_prefix,
            alpha=alpha,
            tail=tail,
            param_idx=param_idx,
        )

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
        entry-points also default to ``two_tailed=True``; pass
        ``two_tailed=False`` explicitly for separate one-sided families.
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
    # ---- Keyword aliases arriving through **method_kwargs -------------------
    # The same controls can arrive through **method_kwargs (e.g. from a
    # wrapper that forwards the compute-function signature); keep one
    # explicit value instead of failing on a duplicate keyword.
    if method_kwargs:
        if n_processes is None:
            n_processes = method_kwargs.pop("n_processes", None)
        if progress is None:
            progress = method_kwargs.pop("progress", None)
        if cancel is None:
            cancel = method_kwargs.pop("cancel", None)

    # ---- Shared state for the stages --------------------------------------
    # Opt-in motion/QC-FC side-car: ``None`` means the caller did not opt in,
    # no diagnostic is computed and no result changes. The preprocessing and
    # Fisher histories are normalized once so every result and export carries
    # the same record (``None`` stays unknown).
    state = _AnalysisState(
        Y=Y,
        interest=interest,
        confounds=confounds,
        group1=group1,
        group2=group2,
        confounds_group1=confounds_group1,
        confounds_group2=confounds_group2,
        sites=sites,
        preserve=preserve,
        input=input,
        fisher_z=fisher_z,
        contrast_id=contrast_id,
        method_kwargs=dict(method_kwargs),
        n_processes=n_processes,
        progress=progress,
        cancel=cancel,
        motion_qc_spec=dict(motion_qc) if motion_qc is not None else None,
        test_type=test_type,
        harmonize=harmonize,
        r_to_z=r_to_z,
        var_equal=var_equal,
        method=method,
        two_tailed=two_tailed,
        tested_tail=tested_tail,
        acceleration=acceleration,
        n_permutations=n_permutations,
        e=e,
        h=h,
        n=n,
        rng=rng,
        verbose=verbose,
        use_mp=use_mp,
        missing=missing,
        min_coverage=min_coverage,
        atlas=atlas,
        roi_ids=roi_ids,
        groupings=groupings,
        units=units,
        preprocessing_record=normalize_preprocessing_record(preprocessing),
        fisher_upstream=normalize_fisher_history(fisher_history),
    )

    # ---- 1. Tail policy: two-sided by default, prespecified direction opt-in
    _resolve_tail_policy(state)

    # ---- 2. Optional subject-ID alignment for paired designs -----------------
    # Condition A (group1) is the reference row order; condition-B inputs move
    # into A order before the Fisher transform, the B-A difference and any
    # inference; A-order inputs are validated and never permuted.
    _align_paired_subjects(state, ids_group1, ids_group2)

    # ---- 3. Paired interest: GLM response B - A (transform each condition
    # separately, never the difference); interactions: centered products.
    _reduce_paired_interest(state)
    _expand_interactions(state, interactions)

    # ---- 4. Modes (GLM / multi-predictor / t-test), method, net labels;
    # then the ComBat / site-dummies strategy for this call shape.
    _resolve_modes(state)
    _resolve_harmonization_strategy(state)

    # ---- 5. Input preparation: zero diagonal, Fisher r→z exactly once, and
    # the opt-in motion/QC-FC side-car on the native input.
    _prepare_inputs(state)

    # ---- 6. ComBat (combat_only / combat_site_dummies_glm) ------------------
    _run_combat(state)

    # ---- 7. Site dummies → GLM nuisance (combat_site_dummies_glm, site_dummies_glm)
    _add_site_dummies(state)

    # ---- 8. Strata: auto-stratify permutations on sites (PALM -eb semantics);
    # an explicit strata= through **method_kwargs wins.
    _resolve_strata(state)

    # ---- 9./10. Inference dispatch and provenance threading -----------------
    if state.multi_mode:
        # Several predictors → one result per predictor, each against its own
        # null with shared permutations; the shared ComBat diagnostics and
        # flags are attached to every entry.
        return {
            nm: AnalyzeResult(
                inference=inferred.result,
                combat_diagnostics=state.combat_diag,
                design_diagnostics=inferred.design_diag,
                flags=inferred.flags,
                resolved_strategy=state.backend_resolved_strategy,
                motion_qc=state.motion_qc_diag,
            )
            for nm, inferred in _run_multi_inference(state).items()
        }

    result, design_diag = _run_single_inference(state)
    result = _finalize_single(state, result, design_diag)
    return AnalyzeResult(
        inference=result,
        combat_diagnostics=state.combat_diag,
        design_diagnostics=design_diag,
        flags=state.flags,
        resolved_strategy=state.backend_resolved_strategy,
        motion_qc=state.motion_qc_diag,
    )


__all__ = ["analyze", "AnalyzeResult"]
