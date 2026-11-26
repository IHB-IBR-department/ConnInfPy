"""Structured return types for the permutation pipelines.

Two levels:

- :func:`conninfpy.analyze` returns an
  :class:`~conninfpy.analysis.AnalyzeResult`: the inference result
  (``.inference``) plus harmonization diagnostics and plain-English
  ``flags``. Dict-style access (``out['positive']``) and the edge-table
  methods forward to ``.inference``.
- The advanced pipelines in :mod:`conninfpy.stats` return the
  :class:`InferenceResult` / :class:`OmnibusInferenceResult` defined here.

:class:`InferenceResult` is a thin enrichment of
:class:`~conninfpy._compat.TailResult` that carries run-time metadata
(method, n_permutations, acceleration, wall-time, optional null
distribution), observed statistic maps, and provenance about any
upstream harmonization step alongside the per-tail p-value matrices.

:class:`OmnibusInferenceResult` is the sibling type returned by the
F-statistic omnibus path (multi-row contrast). It carries the same
metadata + provenance attributes but only a single ``'omnibus'``
p-value map — F has no sign, so the positive/negative split is not
defined for it.

Backward-compatible by design: both classes are ``dict`` subclasses
(via :class:`TailResult`), so existing code that accesses
``result['positive']`` / ``result['negative']`` (or
``result['omnibus']`` for the F path) continues to work. The attribute
APIs (``result.positive``, ``result.stat_signed``, ``result.method``,
``print(result)``) are additive.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from conninfpy._compat import TailResult
from conninfpy._compat import make_tail_result
from conninfpy._export import build_network_summary
from conninfpy._export import build_omnibus_dataframe
from conninfpy._export import build_tailed_dataframe
from conninfpy._export import network_summary_matrix
from conninfpy._provenance import project_analysis_provenance


def _extract_edge_n(result: Any) -> Optional[npt.NDArray[np.float64]]:
    """Return per-edge observation counts from missing_report or provenance, or None."""
    mr = getattr(result, "missing_report", None)
    if isinstance(mr, dict) and mr.get("edge_n") is not None:
        return np.asarray(mr["edge_n"])
    prov = getattr(result, "analysis_provenance", None)
    if isinstance(prov, dict):
        cov = prov.get("coverage")
        if isinstance(cov, dict) and cov.get("edge_n") is not None:
            return np.asarray(cov["edge_n"])
    return None


def summarize_by_network(
    p_maps: Any,
    networks: Any,
    *,
    alpha: float = 0.05,
    as_matrix: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate edge-wise p-value maps over pairs of atlas networks.

    One row per unordered network pair (within-network pairs included):
    the number of possible edges, the number and proportion of edges with
    ``p <= alpha`` per tail, and the ``-log10 p`` mass per tail (sum and
    mean over the block). Network-pair summaries are typically far more
    reproducible across cohorts than individual edges, so report them
    alongside edge tables.

    Parameters
    ----------
    p_maps : dict of str -> ndarray of shape (N, N), or ndarray
        P-value maps keyed by tail, e.g. ``{'positive': p_pos,
        'negative': p_neg}``; a single array is summarized as tail ``'p'``.
        Edges are taken from the upper triangle.
    networks : AtlasInfo or sequence of str
        An atlas (its per-ROI ``networks`` field defines the partition) or
        one network label per ROI. Any user-defined partition works.
    alpha : float, default 0.05
        Significance threshold (``p <= alpha``).
    as_matrix : str, optional
        Return one column as a symmetric K × K DataFrame (networks in
        first-appearance order) instead of the table, e.g.
        ``'sum_neglog10p_positive'`` or ``'signed_proportion'``.

    Returns
    -------
    pandas.DataFrame
        Columns ``network_a``, ``network_b``, ``network_pair``,
        ``within_network``, ``n_edges``, and per tail ``t``: ``n_sig_t``,
        ``prop_sig_t``, ``sum_neglog10p_t``, ``mean_neglog10p_t``; plus
        ``signed_proportion`` = ``(n_sig_positive - n_sig_negative) /
        n_edges`` when both directional tails are given.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.full((4, 4), 0.5); p[0, 1] = p[1, 0] = 0.01
    >>> summarize_by_network({"positive": p}, ["A", "A", "B", "B"])[
    ...     ["network_pair", "n_edges", "n_sig_positive"]]
      network_pair  n_edges  n_sig_positive
    0          A—A        1               1
    1          A—B        4               0
    2          B—B        1               0
    """
    if not isinstance(p_maps, dict):
        p_maps = {"p": p_maps}
    labels = list(getattr(networks, "networks", networks))
    summary = build_network_summary(p_maps, labels, alpha=alpha)
    if as_matrix is not None:
        return network_summary_matrix(summary, as_matrix)
    return summary


class InferenceResult(TailResult):
    """Per-tail p-value maps + metadata from a single permutation run.

    Behaves as a two-key dict (``'positive'`` / ``'negative'``) and also
    exposes named attributes.

    Attributes
    ----------
    positive, negative : ndarray
        Per-tail p-value (or score) matrices.
    method : str
        Enhancement method label, e.g. ``'tfnbs'``, ``'cnbs'``, ``'nbs'``,
        ``'tstat'``, ``'bh_fdr'``.
    n_permutations : int
        Number of permutations used (0 for parametric methods).
    acceleration : str or None
        ``'gpd'``, ``'gamma'``, or ``None`` for empirical p-values.
    wall_time_s : float or None
        Wall-clock seconds for the inference call (when measured).
    null_max_dist : dict[str, ndarray] or None
        Optional max-stat null distribution per tail (set when
        ``store_null=True`` is passed to the pipeline).
    stat_positive, stat_negative : ndarray or None
        Observed one-tail-clipped (non-negative) test statistic maps.
        ``stat_positive[i, j]`` equals ``max(t_ij, 0)``; ``stat_negative``
        equals ``max(-t_ij, 0)``. The original signed statistic map is
        recoverable as ``stat_signed`` (= positive − negative). Populated
        by the pipeline since v2.1; ``None`` on results created without
        stat maps (e.g. pre-v2.1 pipelines).
    stat_type : {'tstat', 'beta'}
        Which statistic the stat maps hold. ``'tstat'`` for the default
        t-statistic path, ``'beta'`` when the GLM is run with
        ``stat_type='beta'``.
    harmonized : bool
        True iff ComBat ran upstream of inference (set by
        :func:`~conninfpy.analyze`).
    preserve_provided : bool
        True iff a non-None ``preserve`` was supplied to ComBat.
    strata_provided : bool
        True iff inference used stratified permutation.
    combat_diagnostics : dict or None
        ComBat diagnostics blob when available.
        ``strategy='combat_only'`` and ``strategy='combat_site_dummies_glm'``
        diagnostics include between-site variance before/after and the explicit
        ``between_site_variance_ratio_after_over_before`` key used by
        :func:`conninfpy.analyze` for residual-site flags.
        ``strategy='site_dummies_glm'`` records provenance but does not run ComBat.
    missing_report : dict or None
        Set by available-case analysis (``missing='available_case'``):
        subjects and regions with missing connectivity, per-edge sample
        sizes (``edge_n``), the edges not tested for lack of coverage
        (their p-values are NaN), and a Fisher test of missingness by group.
    block_report : dict or None
        Set by ``method='cnbs'``: the inference unit is the block. Lists
        every within/between-network block with its edge count, mean signed
        statistic, and uncorrected and corrected p-values per tail; each
        edge's p-value is its block's corrected p-value. ``node_labels`` is
        the partition behind the blocks (the label of every node), which
        :meth:`node_summary` uses to say which blocks a node belongs to.
    component_report : dict or None
        Set by ``method='nbs'``: the inference unit is the connected component
        of the edges above the threshold. Lists every observed component of
        each tail (``tail``, ``component_id``, ``n_edges``, ``n_nodes``,
        ``statistic``: extent or intensity ``sum(t - threshold)``, ``p_fwer``,
        ``nodes``: ROI indices), plus the ``threshold``, the statistic type
        and ``labels``, the component number of every edge. Each edge's
        p-value is its component's ``p_fwer``, taken from the permutation
        distribution of the maximum component statistic; the report adds
        nothing to it. Component ids are numbered within each tail by
        decreasing statistic. :meth:`component_table` returns it as a table,
        with ROI names and networks when given an atlas; ``component_id`` is
        also a column of :meth:`significant_edges`. ``None`` for other methods.
    two_tailed : bool
        Whether positive and negative directions form one two-sided family.
        Each enhancement parameter setting and contrast is corrected separately.
    tested_tail : {'positive', 'negative'} or None
        The one tested direction of a one-sided result: set by
        :meth:`one_sided` (post hoc) or by a prespecified
        ``analyze(..., tested_tail=...)`` run. The other direction was not
        tested and its p-values are reported as 1.
    tail_selection : {'prespecified', 'post-hoc'} or None
        How ``tested_tail`` came about, so a directional restriction can never
        masquerade as a hypothesis fixed in advance. ``'prespecified'``: the
        direction was declared before the inference ran
        (``analyze(..., tested_tail=...)``). ``'post-hoc'``: the direction was
        chosen on a finished result through :meth:`one_sided` after the run.
        ``None`` for unrestricted results and for results saved before the
        marker existed (their policy is reported as it was saved, never
        re-labelled).
    beta : ndarray of shape (p, N, N) or None
        Coefficient maps of the tested model on the response scale (GLM
        coefficients; for pairwise tests the equivalent design: group-1
        mean and group difference, condition-A mean and B - A, or the mean).
        Independent of the enhancement settings.
    design : dict or None
        Columns, column means, contrast, named covariate profiles and the
        comparison used for coupling changes (see :meth:`predicted`,
        :meth:`coupling`), and the response ``scale``.
    analysis_provenance : dict or None
        Producer-captured identities, correction family, validity masks and
        coverage used by :meth:`to_bundle`; ``None`` for legacy results.
    exceedance_counts : dict[str, ndarray] or None
        Raw per-cell permutation exceedance counts (``count`` in
        ``p = (count + 1) / (B + 1)``), genuinely computed by the counting
        code and never reconstructed from p-values. Each array has the shape
        of that tail's p-value map; ``-1`` marks a cell with no genuine count
        (diagonal, outside a declared family mask, NaN p-value, an accelerated
        tail replacement, or a cell that tail does not use). ``None`` for
        legacy results, parametric methods, cNBS and available-case analyses.
    count_kind : str or None
        Which 0/1-per-draw counting rule produced ``exceedance_counts``:
        ``'max_stat'``, ``'joint_max'`` (two-sided run: both tails count
        against the same joint null), ``'edge_wise'`` or
        ``'two_sided_sum_stat'``
        (``bh_fdr_perm``; there the interval belongs to the underlying
        per-edge count, never to the BH-adjusted q value).
    rng_provenance : dict or None
        ``{'seed', 'scheme', 'bit_generator', 'schema_version'}`` describing
        the scheme that actually ran (legacy ``RandomState``/MT19937, one
        uint32 seed per permutation). ``seed`` is ``None`` for a
        non-deterministic run. Recording it draws no random numbers.
    permutation_space : int or None
        Theoretical number of admissible assignments (``2**n`` for sign
        flips, ``comb(n1+n2, n1)`` or its per-stratum product for group
        shuffles). ``None`` where the space is ambiguous (Freedman-Lane
        residuals, available-case, block-level inference); never the number
        of distinct statistic values.
    """

    method: str
    n_permutations: int
    acceleration: Optional[str]
    wall_time_s: Optional[float]
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]]
    stat_positive: Optional[npt.NDArray[np.float64]]
    stat_negative: Optional[npt.NDArray[np.float64]]
    stat_type: str
    harmonized: bool
    preserve_provided: bool
    strata_provided: bool
    combat_diagnostics: Optional[Dict[str, Any]]
    e_grid: Optional[npt.NDArray[np.float64]]
    h_grid: Optional[npt.NDArray[np.float64]]
    missing_report: Optional[Dict[str, Any]]
    block_report: Optional[Dict[str, Any]]
    # A class-level default also covers results pickled before the attribute existed.
    component_report: Optional[Dict[str, Any]] = None
    two_tailed: bool
    tested_tail: Optional[str]
    # Same pickle-compatibility pattern: results saved before this marker
    # reload with None (no selection claim), not with this version's default.
    tail_selection: Optional[str] = None
    beta: Optional[npt.NDArray[np.float64]]
    design: Optional[Dict[str, Any]]
    analysis_provenance: Optional[Dict[str, Any]]
    # Class-level defaults also cover results pickled before these attributes
    # existed: they reload as None ("unavailable"), never as an invention.
    exceedance_counts: Optional[Dict[str, Optional[npt.NDArray[np.int64]]]] = None
    count_kind: Optional[str] = None
    rng_provenance: Optional[Dict[str, Any]] = None
    permutation_space: Optional[int] = None

    def __new__(
        cls,
        positive: npt.NDArray[np.float64],
        negative: npt.NDArray[np.float64],
        *,
        method: str = "tstat",
        n_permutations: int = 0,
        acceleration: Optional[str] = None,
        wall_time_s: Optional[float] = None,
        null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
        stat_positive: Optional[npt.NDArray[np.float64]] = None,
        stat_negative: Optional[npt.NDArray[np.float64]] = None,
        stat_type: str = "tstat",
        harmonized: bool = False,
        preserve_provided: bool = False,
        strata_provided: bool = False,
        combat_diagnostics: Optional[Dict[str, Any]] = None,
        e_grid: Optional[npt.NDArray[np.float64]] = None,
        h_grid: Optional[npt.NDArray[np.float64]] = None,
        missing_report: Optional[Dict[str, Any]] = None,
        block_report: Optional[Dict[str, Any]] = None,
        two_tailed: bool = False,
        beta: Optional[npt.NDArray[np.float64]] = None,
        design: Optional[Dict[str, Any]] = None,
        analysis_provenance: Optional[Dict[str, Any]] = None,
        tested_tail: Optional[str] = None,
        component_report: Optional[Dict[str, Any]] = None,
        tail_selection: Optional[str] = None,
        exceedance_counts: Optional[Dict[str, Optional[npt.NDArray[np.int64]]]] = None,
        count_kind: Optional[str] = None,
        rng_provenance: Optional[Dict[str, Any]] = None,
        permutation_space: Optional[int] = None,
    ) -> "InferenceResult":
        instance = super().__new__(cls)
        TailResult.__init__(
            instance, [("positive", positive), ("negative", negative)]
        )
        instance.method = method
        instance.n_permutations = n_permutations
        instance.acceleration = None if method == "cnbs" else acceleration
        instance.wall_time_s = wall_time_s
        instance.null_max_dist = null_max_dist
        instance.stat_positive = stat_positive
        instance.stat_negative = stat_negative
        instance.stat_type = stat_type
        instance.harmonized = harmonized
        instance.preserve_provided = preserve_provided
        instance.strata_provided = strata_provided
        instance.combat_diagnostics = combat_diagnostics
        instance.e_grid = e_grid
        instance.h_grid = h_grid
        instance.missing_report = missing_report
        instance.block_report = block_report
        instance.component_report = component_report
        instance.two_tailed = two_tailed
        instance.tested_tail = tested_tail
        instance.tail_selection = tail_selection
        instance.beta = beta
        instance.design = design
        instance.analysis_provenance = analysis_provenance
        instance.exceedance_counts = exceedance_counts
        instance.count_kind = count_kind
        instance.rng_provenance = rng_provenance
        instance.permutation_space = permutation_space
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        # __new__ already populated the dict and attributes; suppress the
        # default dict.__init__(*args, **kwargs) which would otherwise wipe
        # the canonical keys when called with kwargs like method="tfnbs".
        return

    def __getnewargs_ex__(self) -> "tuple[tuple, dict]":
        """Rebuild args for pickle (protocol >= 2) and copy/deepcopy.

        Serialization reconstructs the object through
        ``InferenceResult.__new__(cls, positive, negative)`` and then
        restores every metadata attribute from the saved ``__dict__``
        state, so attributes round-trip without being enumerated here
        and future fields are kept automatically. Pickle protocols
        0/1 (legacy text modes) are not supported.
        """
        return ((self["positive"], self["negative"]), {})

    @property
    def positive(self) -> npt.NDArray[np.float64]:
        """Corrected p-values, positive tail (FWER or FDR by method)."""
        return self["positive"]

    @property
    def negative(self) -> npt.NDArray[np.float64]:
        """Corrected p-values, negative tail (FWER or FDR by method)."""
        return self["negative"]

    @property
    def stat_signed(self) -> Optional[npt.NDArray[np.float64]]:
        """Recovered signed statistic map (``stat_positive − stat_negative``).

        Both tail arrays are non-negative one-tail clips of the original
        signed statistic; for any edge exactly one of them is zero, so
        their difference reconstructs the original t/β map.

        Returns ``None`` when stat maps are unavailable (results from
        before v2.1, or objects created without stat maps).
        """
        if self.stat_positive is None or self.stat_negative is None:
            return None
        return self.stat_positive - self.stat_negative

    @property
    def tail_policy(self) -> str:
        """The declared directional family; legacy results default to separate tails."""
        tested = getattr(self, "tested_tail", None)
        if tested is not None:
            return f"one-sided-{tested}"
        return "two-sided" if getattr(self, "two_tailed", False) else "separate-one-sided"

    def mc_report(self, alpha: float = 0.05, param_idx: Optional[int] = None) -> Dict[str, Any]:
        """Monte-Carlo precision and RNG provenance for this result.

        Describes, per tail: the number of permutations ``B``, the numerical
        resolution ``1 / (B + 1)`` (the smallest attainable p-value — a grid
        step, not an error bar), the theoretical permutation space (or
        ``None`` when it is ambiguous), whether the raw exceedance counts
        ``count`` are available, and a conditional binomial (Wilson) interval
        for ``count / B`` wherever one is justified.

        What the numbers mean
        ---------------------
        The reported p-value is ``(count + 1) / (B + 1)``; the interval here is
        for the *conditional exceedance probability* ``count / B`` of a single
        draw, i.e. ``P(T* >= T_obs | H0)``, and never a bound on family-wise or
        FDR error. Because only ``B + 1`` p-values are attainable, a p-value
        within about ``1 / (B + 1)`` of a threshold (alpha) cannot be separated
        from it by this run; that is a resolution limit, not evidence for
        either side.

        Tail policy: a two-sided result uses the joint null
        ``max(positive, negative)``, so both tails count against that same
        null (``count_kind='joint_max'``); their counts, and therefore their
        intervals, are not independent.
        Separate one-sided tails have their own counts. For
        ``bh_fdr_perm`` the reported p-values are BH-adjusted q values; the
        interval belongs to the underlying per-edge empirical count only.

        Honesty rules: counts are the producer's, never reconstructed from
        p-values. Untested cells keep the sentinel ``-1`` (see
        ``exceedance_counts``) and get no interval. Accelerated (GPD/gamma)
        results, cNBS block inference, available-case analyses and legacy
        results report interval status ``'unknown'`` with the actual reason.
        Sampling is i.i.d. *with replacement*, so ``B`` above the permutation
        space only raises an informational flag (duplicate draws expected).

        Parameters
        ----------
        alpha : float, default 0.05
            Two-sided confidence level ``1 - alpha`` for the Wilson interval.
        param_idx : int, optional
            Cell of an (E, H) grid result; omit to describe the whole grid.

        Returns
        -------
        dict
            ``n_permutations``, ``resolution``, ``permutation_space``,
            ``rng``, ``count_kind``, ``flags``, ``notes`` and a ``tails``
            mapping with the counts, tested mask, interval status/reason and
            interval bound arrays (NaN outside the tested cells).
        """
        from conninfpy.stats._mc import build_report

        return build_report(self, alpha=alpha, param_idx=param_idx)

    def one_sided(self, direction: str) -> "InferenceResult":
        """Restrict a finished result to one direction — a post hoc selection.

        .. warning:: The direction is chosen **after** the run, so the copy is
            marked ``tail_selection='post-hoc'`` in its provenance and exports;
            it is not a prespecified hypothesis. For a directional hypothesis
            fixed before seeing the data, declare it at launch with
            :func:`conninfpy.analyze`'s ``tested_tail=`` argument (the run
            then uses per-tail nulls and is marked ``'prespecified'``), or call
            the low-level pipeline with ``two_tailed=False``.

        Only a result computed with ``two_tailed=False`` can be restricted:
        each tail then has its own max-statistic null, and the kept tail's
        p-values are one-sided FWER-corrected at the nominal level. The other
        tail's p-values are set to 1, so tables, summaries and plots report no
        detections in the direction that was not tested; the statistic maps
        are kept unchanged. For ``method='nbs'`` the component report lists
        the tested tail's components only, with the same component ids;
        for ``method='cnbs'`` the block report keeps all blocks of the
        family while marking the tested direction and dropping untested-tail
        p-values.

        Parameters
        ----------
        direction : {'positive', 'negative'}
            The tested direction of the contrast.

        Returns
        -------
        InferenceResult
            A copy with ``tested_tail = direction``,
            ``tail_policy = 'one-sided-<direction>'`` and
            ``tail_selection = 'post-hoc'``.

        Raises
        ------
        ValueError
            For an unknown direction, or for a two-sided result: its
            p-values come from the joint null of both tails, which is not
            the null of a one-sided test.
        """
        return self._restricted_to_tail(direction, selection="post-hoc")

    def _restricted_to_tail(self, direction: str, *, selection: str) -> "InferenceResult":
        """Shared body of :meth:`one_sided` (post hoc) and the prespecified
        ``analyze(..., tested_tail=...)`` restriction. ``selection`` records
        whether the direction was fixed before the run or chosen after it."""
        if direction not in ("positive", "negative"):
            raise ValueError(f"direction must be 'positive' or 'negative'; got {direction!r}.")
        if getattr(self, "two_tailed", False):
            raise ValueError(
                "one_sided() needs a result computed with two_tailed=False: a "
                "two-sided result's p-values come from the joint null of both tails."
            )
        existing_tail = getattr(self, "tested_tail", None)
        if existing_tail not in (None, direction):
            raise ValueError(
                f"result is already restricted to the {existing_tail!r} tail; "
                "call one_sided() on the unrestricted two_tailed=False result."
            )
        from conninfpy.stats._cnbs import restrict_block_report
        from conninfpy.stats._components import restrict_component_report

        other = "negative" if direction == "positive" else "positive"
        maps = {direction: np.array(self[direction], copy=True),
                other: np.ones_like(self[other], dtype=np.float64)}
        return InferenceResult(
            maps["positive"],
            maps["negative"],
            method=self.method,
            n_permutations=self.n_permutations,
            acceleration=self.acceleration,
            wall_time_s=self.wall_time_s,
            null_max_dist=self.null_max_dist,
            stat_positive=self.stat_positive,
            stat_negative=self.stat_negative,
            stat_type=self.stat_type,
            harmonized=self.harmonized,
            preserve_provided=self.preserve_provided,
            strata_provided=self.strata_provided,
            combat_diagnostics=self.combat_diagnostics,
            e_grid=self.e_grid,
            h_grid=self.h_grid,
            missing_report=self.missing_report,
            block_report=restrict_block_report(
                getattr(self, "block_report", None), direction, selection=selection
            ),
            two_tailed=False,
            beta=getattr(self, "beta", None),
            design=getattr(self, "design", None),
            tested_tail=direction,
            tail_selection=selection,
            analysis_provenance=project_analysis_provenance(
                getattr(self, "analysis_provenance", None), tested_tail=direction,
                tail_selection=selection,
            ),
            component_report=restrict_component_report(
                getattr(self, "component_report", None), direction, selection=selection
            ),
            exceedance_counts=(
                {direction: self.exceedance_counts[direction]}
                if getattr(self, "exceedance_counts", None)
                and direction in self.exceedance_counts
                else None
            ),
            count_kind=getattr(self, "count_kind", None),
            rng_provenance=getattr(self, "rng_provenance", None),
            permutation_space=getattr(self, "permutation_space", None),
        )

    @property
    def correction(self) -> str:
        """Multiplicity procedure at the method's native inference unit."""
        if self.method == "cnbs":
            return (self.block_report or {}).get("correction", "bonferroni")
        if self.method in ("bh_fdr", "bh_fdr_perm"):
            return "bh_fdr"
        return "bonferroni" if self.method == "bonferroni" else "permutation_max"

    @property
    def inference_unit(self) -> str:
        """Unit of the null hypothesis represented by the projected edge map."""
        return {"cnbs": "block", "nbs": "component"}.get(self.method, "edge")

    @property
    def is_grid(self) -> bool:
        """True iff this result holds a multi-(E, H) parameter grid.

        When True, ``self['positive']`` and ``self['negative']`` are
        ``(N, N, K)`` tensors and ``self.e_grid`` / ``self.h_grid``
        give the parameter cells. Use :meth:`select` to project to a
        single 2D cell.
        """
        return self["positive"].ndim == 3

    def _project_2d(
        self,
        arr: npt.NDArray[np.float64],
        param_idx: Optional[int],
    ) -> npt.NDArray[np.float64]:
        """Project ``arr`` to a 2D ``(N, N)`` slice.

        For 2D inputs, returns the array unchanged (and rejects
        ``param_idx`` ≠ ``None``). For 3D inputs, slices at
        ``param_idx`` along the last axis.
        """
        if arr.ndim == 2:
            if param_idx is not None and param_idx != 0:
                raise ValueError(
                    f"param_idx={param_idx} given, but this result is "
                    f"not a parameter grid (2D p-maps)."
                )
            return arr
        if param_idx is None:
            raise ValueError(
                f"Result holds a (N, N, K={arr.shape[-1]}) parameter "
                f"grid; pass param_idx to select a cell (see .e_grid "
                f"/ .h_grid), or call .select(param_idx) first."
            )
        return arr[:, :, param_idx]

    def select(self, param_idx: int) -> "InferenceResult":
        """Project a multi-(E, H) result to the single cell ``param_idx``.

        Returns a new :class:`InferenceResult` with 2D p-maps and the
        same metadata; the returned object has ``is_grid is False``
        and ``e_grid`` / ``h_grid`` reduced to single-element arrays
        recording which cell was selected.
        """
        if not self.is_grid:
            raise ValueError(
                "select() requires a multi-(E, H) result; this one is 2D."
            )
        e_sel = self.e_grid[param_idx:param_idx + 1] if self.e_grid is not None else None
        h_sel = self.h_grid[param_idx:param_idx + 1] if self.h_grid is not None else None
        return InferenceResult(
            self["positive"][:, :, param_idx].copy(),
            self["negative"][:, :, param_idx].copy(),
            method=self.method,
            n_permutations=self.n_permutations,
            acceleration=self.acceleration,
            wall_time_s=self.wall_time_s,
            null_max_dist=None,
            stat_positive=self.stat_positive,
            stat_negative=self.stat_negative,
            stat_type=self.stat_type,
            harmonized=self.harmonized,
            preserve_provided=self.preserve_provided,
            strata_provided=self.strata_provided,
            combat_diagnostics=self.combat_diagnostics,
            e_grid=e_sel,
            h_grid=h_sel,
            missing_report=self.missing_report,
            block_report=self.block_report,
            two_tailed=getattr(self, "two_tailed", False),
            beta=getattr(self, "beta", None),
            design=getattr(self, "design", None),
            tested_tail=getattr(self, "tested_tail", None),
            tail_selection=getattr(self, "tail_selection", None),
            analysis_provenance=project_analysis_provenance(
                getattr(self, "analysis_provenance", None), param_idx=param_idx
            ),
            component_report=getattr(self, "component_report", None),
            exceedance_counts=(
                {
                    tail: (
                        arr[:, :, param_idx].copy() if np.asarray(arr).ndim == 3
                        else np.array(arr, copy=True)
                    )
                    for tail, arr in self.exceedance_counts.items()
                    if arr is not None
                }
                if getattr(self, "exceedance_counts", None)
                else None
            ),
            count_kind=getattr(self, "count_kind", None),
            rng_provenance=getattr(self, "rng_provenance", None),
            permutation_space=getattr(self, "permutation_space", None),
        )

    def n_significant(
        self,
        alpha: float = 0.05,
        *,
        param_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Count significant edges per tail at threshold ``alpha``.

        For grid results, pass ``param_idx`` to count one cell;
        omit to get a list of counts per cell along the parameter axis.
        """
        out: Dict[str, Any] = {}
        for tail in ("positive", "negative"):
            arr = self[tail]
            if arr.ndim == 2:
                if arr.shape[0] == arr.shape[1]:
                    iu = np.triu_indices_from(arr, k=1)
                    out[tail] = int(np.sum(arr[iu] <= alpha))
                else:
                    out[tail] = int(np.sum(arr <= alpha))
            else:
                # (N, N, K) grid
                if param_idx is not None:
                    sub = arr[:, :, param_idx]
                    iu = np.triu_indices_from(sub, k=1)
                    out[tail] = int(np.sum(sub[iu] <= alpha))
                else:
                    iu = np.triu_indices(arr.shape[0], k=1)
                    sub = arr[iu[0], iu[1], :]  # (n_edges, K)
                    out[tail] = [int(np.sum(sub[:, k] <= alpha))
                                 for k in range(arr.shape[-1])]
        return out

    def predicted(self, profile: Union[str, Sequence[float]] = "mean") -> npt.NDArray[np.float64]:
        """Predicted connectivity ``x' beta`` (N, N) at a named or explicit covariate profile.

        ``'mean'`` gives the sample mean connectivity for a GLM (the
        prediction at the column means); ``design['profiles']`` lists the
        named profiles (adjusted group means, interest at mean -/+ 1 SD,
        condition means). Never read baselines from the intercept map alone:
        with uncentred covariates it is the connectivity at covariates = 0.
        """
        from conninfpy.stats._estimates import predict

        beta, design = getattr(self, "beta", None), getattr(self, "design", None)
        if beta is None or design is None:
            raise ValueError("this result carries no coefficient maps (beta).")
        x = design["profiles"][profile] if isinstance(profile, str) else profile
        return predict(beta, x)

    def effect_map(self) -> Optional[npt.NDArray[np.float64]]:
        """Contrast effect ``c' beta`` (N, N) in response units, or None.

        The unstandardised effect (e.g. Δz per unit of the predictor, or the
        group difference in Fisher z), not the test statistic.
        """
        from conninfpy.stats._estimates import effect

        beta, design = getattr(self, "beta", None), getattr(self, "design", None)
        if beta is None or design is None:
            return None
        return effect(beta, design["contrast"])

    def to_bundle(self, **kwargs: Any) -> "Any":
        """Snapshot this result for independent post-analysis reporting.

        See :meth:`conninfpy.post_analysis.ResultBundle.from_result`. Captured
        analysis metadata is used automatically; explicit overrides are checked.
        """
        from conninfpy.post_analysis import ResultBundle
        return ResultBundle.from_result(self, **kwargs)

    def coupling(
        self,
        profiles: Optional[Sequence[str]] = None,
        *,
        near_zero: float = 0.0,
    ) -> Dict[str, Any]:
        """How coupling changes between two profiles (default: ``design['comparison']``).

        Returns ``{'profiles', 'a', 'b', 'delta', 'classes'}``: the two
        predicted maps on the analysis scale, their difference in r units
        when the scale is Fisher z (``tanh(b) - tanh(a)``; native units
        otherwise), and the class of each edge: stronger / weaker positive
        coupling, stronger / weaker anticorrelation, sign change, or, with a
        prespecified ``near_zero`` band, emerges from / returns to ~0.
        The classes are descriptive labels on the recorded scale.
        Negative baselines depend on preprocessing (global-signal
        regression both removes and induces anticorrelations); neither
        they nor their classes carry physiological or causal meaning by
        themselves. With partial-correlation input the scale is
        atanh(partial r), a variance-stabilizing transform — not the exact
        standard z of a marginal Pearson r.
        """
        from conninfpy.stats._estimates import coupling_change

        design = getattr(self, "design", None)
        if design is None:
            raise ValueError("this result carries no design / coefficient maps.")
        names = list(profiles) if profiles is not None else design.get("comparison")
        if not names or len(names) != 2:
            raise ValueError("give two profile names; this design defines no comparison.")
        a, b = self.predicted(names[0]), self.predicted(names[1])
        classes, delta = coupling_change(a, b, scale=design.get("scale", "unknown"), near_zero=near_zero)
        return {"profiles": names, "a": a, "b": b, "delta": delta, "classes": classes}

    def significant_edges(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        tail: str = "both",
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        param_idx: Optional[int] = None,
    ) -> "Any":  # pd.DataFrame, kept loose to avoid hard typing import
        """Sorted table of significant edges with optional atlas annotation.

        Parameters
        ----------
        atlas : :class:`~conninfpy.AtlasInfo`, optional
            When supplied, adds ROI names, RS-network labels,
            hemisphere, and a canonical ``network_pair`` column. Length
            must match ``self['positive'].shape[0]``.
        alpha : float, default 0.05
            Edge-wise significance threshold.
        tail : {'both', 'positive', 'negative'}, default 'both'
            Which tail(s) to filter on. ``'both'`` keeps edges where
            either ``p_positive`` or ``p_negative`` falls below
            ``alpha``; the per-row ``tail`` column reports which
            tail(s) actually hit significance for that edge.
        sort : {'p', 'statistic', 'network_pair'}, default 'p'
            Sort key. ``'p'`` — ascending ``min(p_positive, p_negative)``.
            ``'statistic'`` — descending absolute signed t/beta statistic.
            ``'effect_size'`` remains a deprecated alias for ``'statistic'``;
            a t-statistic is not a standardized effect size.
            ``'network_pair'`` — group by canonical network pair,
            then by ``p`` ascending within group; requires ``atlas``.
        include_nonsig : bool, default False
            If True, include all edges (not just significant ones).
        top_k : int, optional
            Keep only the top-K rows after sorting.

        Returns
        -------
        pandas.DataFrame
            Columns (with atlas): ``edge_id, roi_i, roi_i_name,
            roi_i_network, roi_j, roi_j_name, roi_j_network,
            network_pair, t_signed, p_positive, p_negative, p_min,
            tail, hemisphere_i, hemisphere_j``. Without an atlas the
            ``roi_*_name``, ``roi_*_network``, ``hemisphere_*`` and
            ``network_pair`` columns are omitted. A ``method='nbs'`` result
            adds ``component_tail`` and ``component_id`` after ``tail``: the
            component (see :attr:`component_report`) the edge belongs to,
            numbered within its tail; ``<NA>`` for an edge above no
            threshold. Edges of one component share one ``p_*``.

        Notes
        -----
        Interpret detections at ``inference_unit``: cNBS rows project block
        p-values and NBS rows represent component detections; neither gives
        an individual edge localization claim. With separate one-sided
        families, filtering both tails at alpha does not imply joint FWER
        at alpha. Statistical detection alone does not establish a biological
        mechanism.
        For reporting, prefer block-level patterns
        (``roi_i_network`` × ``roi_j_network``) over individual edges
        — use ``sort='network_pair'`` to group them.

        Raises
        ------
        ValueError
            If ``stat_positive`` / ``stat_negative`` are missing (e.g.
            pre-v2.1 results, or objects created without stat maps), or
            if ``sort='network_pair'`` without an atlas.
        """
        report = getattr(self, "component_report", None)
        eff = self.effect_map()
        design = getattr(self, "design", None)
        scale = design.get("scale") if (eff is not None and isinstance(design, dict)) else None
        predicted_maps = None
        if isinstance(design, dict) and getattr(self, "beta", None) is not None:
            comp = design.get("comparison") or []
            profiles = design.get("profiles") or {}
            if comp and profiles:
                predicted_maps = {
                    str(label): self.predicted(label)
                    for label in comp
                    if label in profiles
                }
                if not predicted_maps:
                    predicted_maps = None
        df = build_tailed_dataframe(
            self._project_2d(self["positive"], param_idx),
            self._project_2d(self["negative"], param_idx),
            stat_signed=self.stat_signed,
            atlas=atlas,
            alpha=alpha,
            tail=tail,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
            component_labels=None if report is None else report["labels"],
            n_effective=_extract_edge_n(self),
            effect_map=eff,
            scale=scale,
            predicted_maps=predicted_maps,
        )
        return df

    def component_table(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        tail: str = "both",
        significant_only: bool = False,
    ) -> pd.DataFrame:
        """Components of an NBS result: one row per observed component.

        The inference unit of ``method='nbs'`` is the connected component of
        the edges above the threshold, calibrated against the permutation
        distribution of the maximum component statistic (FWER). Every
        observed component is listed, significant or not, as ``block_report``
        lists every block; ``significant`` is ``p_fwer <= alpha`` and
        ``significant_only=True`` keeps those rows. A component's ``p_fwer``
        is the p-value its edges carry in ``positive`` / ``negative``. After
        :meth:`one_sided` only the tested tail is listed.

        Parameters
        ----------
        atlas : :class:`~conninfpy.AtlasInfo`, optional
            Adds ``roi_names`` and, if the atlas has network labels,
            ``networks`` (with ROI counts) and ``network_pairs`` (with edge
            counts). Without an atlas those columns are absent. Length must
            match the number of ROIs.
        alpha : float, default 0.05
            Threshold of the ``significant`` column.
        tail : {'both', 'positive', 'negative'}, default 'both'
        significant_only : bool, default False

        Returns
        -------
        pandas.DataFrame
            ``tail, component_id, n_edges, n_nodes, statistic,
            statistic_type, p_fwer, significant, nodes`` (+ atlas columns);
            ``statistic`` is the extent or the intensity
            ``sum(t - threshold)``. The threshold and the permutation count
            are in ``df.attrs``. Ids are numbered within each tail by
            decreasing statistic.

        Raises
        ------
        ValueError
            If the result is not from ``method='nbs'``.
        """
        from conninfpy.stats._components import component_table

        return component_table(
            getattr(self, "component_report", None),
            atlas=atlas,
            alpha=alpha,
            tail=tail,
            significant_only=significant_only,
        )

    def node_summary(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        tail: str = "both",
        param_idx: Optional[int] = None,
        sort: str = "detections",
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Which ROIs gather the detected connections: one row per ROI.

        A descriptive summary: it adds no p-value and no new significance and
        leaves the result unchanged. What counts as a detection of a ROI follows
        :attr:`inference_unit`.

        **Edge results** (``tstat``, ``bh_fdr``, ``bh_fdr_perm``, ``tfnbs``,
        ``ni_tfnbs``, ``fbc_tfnbs``; every edge has its own p-value). Per tail:
        ``n_sig_<tail>`` (incident edges with ``p <= alpha``) and
        ``prop_sig_<tail>``; ``n_sig_total`` / ``prop_sig`` over the edges
        significant in either tail; ``n_tested`` is the denominator, the
        incident edges that were *tested*: an edge outside an ``edge_mask``
        family or with too few subjects under ``missing='available_case'`` has
        a NaN p-value and is in neither numerator nor denominator (so it is
        not simply ``N - 1``). ``mean_effect_<tail>`` is the mean contrast
        effect ``c' beta`` (:meth:`effect_map`, the ``effect`` column of
        :meth:`significant_edges`) over the ROI's significant edges of the
        tail; NaN when the result carries no effect map (never replaced by t
        or by an enhanced score) or the ROI has no significant edge.

        **Component results** (``nbs``) and **block results** (``cnbs``). All
        edges of a component or block share one p-value, so a count of
        significant edges or a mass of ``-log10 p`` says nothing about a ROI.
        The summary lists, per tail, the significant (``p_fwer`` / corrected
        ``p <= alpha``) components or blocks that contain the ROI
        (``component_ids_<tail>`` / ``block_ids_<tail>``, blocks as
        ``'A—B'``), their number, and how many of the ROI's tested edges lie
        in them (``n_edges_in_components_<tail>`` /
        ``n_edges_in_blocks_<tail>``; totals without the suffix). The column
        ``inference_unit`` and ``DataFrame.attrs['descriptive']`` mark the
        table as descriptive.

        **Networks.** With an atlas that has network labels, the partner
        networks of the ROI's detections (its significant edges, or its edges
        in the significant components / blocks) are given as
        ``partner_networks`` (``'Net (k)'`` counts), ``n_networks_reached``
        and ``participation``, the participation coefficient of Guimera &
        Amaral (2005), ``P = 1 - sum_s (k_s / k)**2`` over partner networks
        ``s``. It is NaN, not 0, for a ROI without detections, and 0 when all
        its detections go to one network (also when it has a single one:
        read it together with ``n_sig_total`` / ``n_edges_in_*``). Without an
        atlas, or with an atlas without network labels, those columns (and
        ``network``) are absent; nothing is guessed.

        Parameters
        ----------
        atlas : :class:`~conninfpy.AtlasInfo`, optional
            Adds ``roi_name``, ``network`` and ``hemisphere`` where the atlas
            has them. Length must match the number of ROIs.
        alpha : float, default 0.05
            Significance threshold of edges, components or blocks.
        tail : {'both', 'positive', 'negative'}, default 'both'
            Tails to report. After :meth:`one_sided` only the tested tail
            exists; asking for the other raises ``ValueError``. With
            ``two_tailed=True`` results both tails are reported;
            ``attrs['tail_policy']`` records the declared family.
        param_idx : int, optional
            Cell of an (E, H) parameter grid (as in :meth:`significant_edges`).
        sort : {'detections', 'roi'}, default 'detections'
            ``'detections'``: by ``n_sig_total`` (edge results) or by the
            number of the ROI's edges in significant components / blocks,
            descending, ties by ROI index; ``'roi'``: by ROI index.
        top_k : int, optional
            Keep the first ``top_k`` rows after sorting.

        Returns
        -------
        pandas.DataFrame
            One row per ROI: ``roi`` (+ ``roi_name``, ``network``,
            ``hemisphere``), ``inference_unit``, the count columns above,
            ``n_tested`` (+ the network columns). ``DataFrame.attrs`` holds
            ``alpha``, ``inference_unit``, ``tails``, ``tail_policy``,
            ``tested_tail``, ``method``, ``param_idx`` (and ``e`` / ``h`` of a
            grid cell), ``descriptive``, ``denominator`` and ``effect_source``.

        Raises
        ------
        ValueError
            For a grid result without ``param_idx``, a tail that was not
            tested, a wrong atlas length, or a component / block result whose
            report is missing (made before the report was recorded).
        """
        from conninfpy._node_summary import build_node_summary

        if tail not in ("both", "positive", "negative"):
            raise ValueError(f"tail must be 'both', 'positive' or 'negative'; got {tail!r}.")
        tested_tail = getattr(self, "tested_tail", None)
        available = [tested_tail] if tested_tail else ["positive", "negative"]
        if tail == "both":
            tails = available
        elif tail in available:
            tails = [tail]
        else:
            raise ValueError(
                f"the {tail} tail was not tested: this result is one_sided({tested_tail!r})."
            )
        unit = self.inference_unit
        report = {"component": getattr(self, "component_report", None),
                  "block": getattr(self, "block_report", None)}.get(unit)
        attrs: Dict[str, Any] = {
            "method": self.method,
            "tail_policy": self.tail_policy,
            "tested_tail": tested_tail,
            "param_idx": param_idx,
        }
        if self.is_grid and param_idx is not None:
            if self.e_grid is not None:
                attrs["e"] = float(self.e_grid[param_idx])
            if self.h_grid is not None:
                attrs["h"] = float(self.h_grid[param_idx])
        return build_node_summary(
            unit=unit,
            p_maps={t: self._project_2d(np.asarray(self[t]), param_idx) for t in tails},
            tails=tails,
            alpha=alpha,
            effect=self.effect_map(),
            report=report,
            atlas=atlas,
            sort=sort,
            top_k=top_k,
            attrs=attrs,
        )

    def network_summary(
        self,
        atlas: Any,
        *,
        alpha: float = 0.05,
        param_idx: Optional[int] = None,
        as_matrix: Optional[str] = None,
    ) -> pd.DataFrame:
        """Aggregate this result over pairs of atlas networks.

        Shortcut for :func:`summarize_by_network` on the ``positive`` /
        ``negative`` p-value maps; see it for the columns. ``atlas`` is an
        :class:`~conninfpy.atlas.AtlasInfo` (or one network label per
        ROI); ``param_idx`` selects a cell of an (E, H) grid result.
        """
        p_maps = {
            "positive": self._project_2d(np.asarray(self["positive"]), param_idx),
            "negative": self._project_2d(np.asarray(self["negative"]), param_idx),
        }
        return summarize_by_network(p_maps, atlas, alpha=alpha, as_matrix=as_matrix)

    def to_csv(
        self,
        path: "Any",
        *,
        atlas: Optional[Any] = None,
        alpha: float = 0.05,
        tail: str = "both",
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        index: bool = False,
        param_idx: Optional[int] = None,
        **to_csv_kwargs: Any,
    ) -> None:
        """Convenience: ``self.significant_edges(...).to_csv(path, ...)``.

        Same filtering / sorting / atlas keywords as
        :meth:`significant_edges`. Extra keyword arguments are
        forwarded to :meth:`pandas.DataFrame.to_csv`.
        """
        df = self.significant_edges(
            atlas=atlas,
            alpha=alpha,
            tail=tail,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
            param_idx=param_idx,
        )
        df.to_csv(path, index=index, **to_csv_kwargs)

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

    def annotated_edges(
        self,
        rois: Any,
        *,
        alpha: float = 0.05,
        tail: str = "both",
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        param_idx: Optional[int] = None,
        smoothing_fwhm_mm: Optional[float] = None,
    ) -> pd.DataFrame:
        """Significant edges with Brainnetome endpoint anatomy and endpoint relations.

        ``rois`` is a :class:`~conninfpy.interpret.annotate.RoiAnnotation` for the
        result's ROIs, in matrix order (:func:`conninfpy.interpret.annotate.annotate_rois`).
        Adds the columns described in :func:`conninfpy.interpret.annotate.annotate_edges`
        (anatomy of both endpoints, hemispheric relation, circuit class,
        distance, sphere overlap, structural connectivity, network relation,
        shared BrainMap domains) to :meth:`significant_edges`. Description only:
        the p-values and their correction are unchanged.
        """
        from conninfpy.interpret.annotate import annotate_edges

        if len(rois) != len(self["positive"]):
            raise ValueError(f"rois has {len(rois)} ROIs but the result has {len(self['positive'])}.")
        edges = self.significant_edges(alpha=alpha, tail=tail, sort=sort, include_nonsig=include_nonsig,
                                       top_k=top_k, param_idx=param_idx)
        out = annotate_edges(edges, rois, smoothing_fwhm_mm=smoothing_fwhm_mm)
        design = getattr(self, "design", None)
        if design is not None and design.get("comparison") and not out.empty:
            c = self.coupling()
            i, j = out["roi_i"].to_numpy(int), out["roi_j"].to_numpy(int)
            a_name, b_name = c["profiles"]
            out[f"fc[{a_name}]"] = c["a"][i, j]
            out[f"fc[{b_name}]"] = c["b"][i, j]
            # fisher_z is converted back to r; correlation is already r.
            out["delta_r" if design.get("scale") in ("fisher_z", "correlation") else "delta"] = c["delta"][i, j]
            out["coupling_change"] = c["classes"][i, j]
        return out

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        nsig = self.n_significant(0.05)
        wall = (
            f"{self.wall_time_s:.2f}s" if self.wall_time_s is not None else "—"
        )
        accel = self.acceleration if self.acceleration else "empirical"
        prov = []
        if self.harmonized:
            prov.append("harmonized")
        if self.strata_provided:
            prov.append("strata")
        prov_str = (" prov=" + ",".join(prov)) if prov else ""
        return (
            f"InferenceResult(method={self.method!r}, "
            f"n_perms={self.n_permutations}, accel={accel}, "
            f"sig@α=0.05: pos={nsig['positive']}, neg={nsig['negative']}, "
            f"wall_time={wall}{prov_str})"
        )


class OmnibusInferenceResult(TailResult):
    """Single-tail p-value map + metadata for the F-stat omnibus path.

    Behaves as a one-key dict (``'omnibus'``) and exposes the same
    metadata / provenance attributes as :class:`InferenceResult` plus
    the observed F-statistic map (``stat_omnibus``). Sibling — not
    subclass — of :class:`InferenceResult` because F has no sign and
    a subclass that raised on ``.positive``/``.negative`` would
    violate LSP.

    Use ``isinstance(r, OmnibusInferenceResult)`` to dispatch on the
    F-stat path; ``isinstance(r, InferenceResult)`` will be ``False``
    for this type.
    """

    method: str
    n_permutations: int
    acceleration: Optional[str]
    wall_time_s: Optional[float]
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]]
    stat_omnibus: Optional[npt.NDArray[np.float64]]
    stat_type: str
    harmonized: bool
    preserve_provided: bool
    strata_provided: bool
    combat_diagnostics: Optional[Dict[str, Any]]
    analysis_provenance: Optional[Dict[str, Any]]
    # See InferenceResult.component_report (single tail, 'omnibus').
    component_report: Optional[Dict[str, Any]] = None
    # See InferenceResult.exceedance_counts and friends: same contract, one
    # 'omnibus' tail (F has no sign, so there is no two-sided pooling).
    exceedance_counts: Optional[Dict[str, Optional[npt.NDArray[np.int64]]]] = None
    count_kind: Optional[str] = None
    rng_provenance: Optional[Dict[str, Any]] = None
    permutation_space: Optional[int] = None

    def __new__(
        cls,
        omnibus: npt.NDArray[np.float64],
        *,
        method: str = "tstat",
        n_permutations: int = 0,
        acceleration: Optional[str] = None,
        wall_time_s: Optional[float] = None,
        null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
        stat_omnibus: Optional[npt.NDArray[np.float64]] = None,
        stat_type: str = "fstat",
        harmonized: bool = False,
        preserve_provided: bool = False,
        strata_provided: bool = False,
        combat_diagnostics: Optional[Dict[str, Any]] = None,
        missing_report: Optional[Dict[str, Any]] = None,
        block_report: Optional[Dict[str, Any]] = None,
        beta: Optional[npt.NDArray[np.float64]] = None,
        design: Optional[Dict[str, Any]] = None,
        analysis_provenance: Optional[Dict[str, Any]] = None,
        component_report: Optional[Dict[str, Any]] = None,
        exceedance_counts: Optional[Dict[str, Optional[npt.NDArray[np.int64]]]] = None,
        count_kind: Optional[str] = None,
        rng_provenance: Optional[Dict[str, Any]] = None,
        permutation_space: Optional[int] = None,
    ) -> "OmnibusInferenceResult":
        instance = super().__new__(cls)
        # Don't go through TailResult's positive/negative initializer —
        # this type uses a single 'omnibus' key.
        dict.__init__(instance, [("omnibus", omnibus)])
        instance.method = method
        instance.n_permutations = n_permutations
        instance.acceleration = None if method == "cnbs" else acceleration
        instance.wall_time_s = wall_time_s
        instance.null_max_dist = null_max_dist
        instance.stat_omnibus = stat_omnibus
        instance.stat_type = stat_type
        instance.harmonized = harmonized
        instance.preserve_provided = preserve_provided
        instance.strata_provided = strata_provided
        instance.combat_diagnostics = combat_diagnostics
        instance.missing_report = missing_report
        instance.block_report = block_report
        instance.component_report = component_report
        instance.beta = beta
        instance.design = design
        instance.analysis_provenance = analysis_provenance
        instance.exceedance_counts = exceedance_counts
        instance.count_kind = count_kind
        instance.rng_provenance = rng_provenance
        instance.permutation_space = permutation_space
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return

    def __getnewargs_ex__(self) -> "tuple[tuple[npt.NDArray[np.float64]], dict]":
        """Rebuild args for pickle (protocol >= 2) and copy/deepcopy.

        Same mechanism as
        :meth:`InferenceResult.__getnewargs_ex__`: metadata attributes
        are restored from the saved ``__dict__`` state without being
        enumerated here. Pickle protocols 0/1 are not supported.
        """
        return ((self["omnibus"],), {})

    @property
    def omnibus(self) -> npt.NDArray[np.float64]:
        """Corrected p-values of the omnibus test (FWER or FDR by method)."""
        return self["omnibus"]

    def to_bundle(self, **kwargs: Any) -> "Any":
        """Snapshot this unsigned omnibus result for post-analysis reporting."""
        from conninfpy.post_analysis import ResultBundle
        return ResultBundle.from_result(self, **kwargs)

    def mc_report(self, alpha: float = 0.05, param_idx: Optional[int] = None) -> Dict[str, Any]:
        """Monte-Carlo precision and RNG provenance for the omnibus map.

        Same contract as :meth:`InferenceResult.mc_report`, with a single
        ``'omnibus'`` tail: F has no sign, so there is no two-sided pooling and
        the counts come from the one-sided max-F null (one 0/1 indicator per
        draw).
        """
        from conninfpy.stats._mc import build_report

        return build_report(self, alpha=alpha, param_idx=param_idx)

    @property
    def tail_policy(self) -> str:
        return "omnibus"

    @property
    def correction(self) -> str:
        return InferenceResult.correction.fget(self)

    @property
    def inference_unit(self) -> str:
        return InferenceResult.inference_unit.fget(self)

    def n_significant(self, alpha: float = 0.05) -> Dict[str, int]:
        """Edge counts below ``alpha`` per tail."""
        arr = self["omnibus"]
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            iu = np.triu_indices_from(arr, k=1)
            return {"omnibus": int(np.sum(arr[iu] <= alpha))}
        return {"omnibus": int(np.sum(arr <= alpha))}

    def significant_edges(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
    ) -> "Any":
        """Sorted table of significant edges (F-stat omnibus path).

        Same conventions as
        :meth:`InferenceResult.significant_edges` but with a single
        unsigned ``F`` / ``p_omnibus`` pair instead of the tail split.
        ``sort='statistic'`` ranks by ``F`` descending; ``'effect_size'`` is
        a deprecated alias. F is a test statistic, not an effect size.

        Notes
        -----
        Interpret p-values at ``inference_unit``: a projected block or
        component detection is not an individual edge localization claim.
        Detection alone does not establish a biological mechanism. A
        ``method='nbs'`` result adds a ``component_id`` column (the
        component of :attr:`component_report` the edge belongs to).
        """
        report = getattr(self, "component_report", None)
        return build_omnibus_dataframe(
            self["omnibus"],
            stat_omnibus=self.stat_omnibus,
            atlas=atlas,
            alpha=alpha,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
            component_labels=None if report is None else report["labels"].get("omnibus"),
            n_effective=_extract_edge_n(self),
        )

    def component_table(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        significant_only: bool = False,
    ) -> pd.DataFrame:
        """Components of an NBS omnibus result; see
        :meth:`InferenceResult.component_table` (the single tail is
        ``'omnibus'`` and the edge statistic is F)."""
        from conninfpy.stats._components import component_table

        return component_table(
            getattr(self, "component_report", None),
            atlas=atlas,
            alpha=alpha,
            significant_only=significant_only,
        )

    def node_summary(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        sort: str = "detections",
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Which ROIs gather the detected connections (F omnibus, one tail ``'omnibus'``).

        Same conventions as :meth:`InferenceResult.node_summary`, with one
        unsigned tail: ``n_tested``, ``n_sig_total`` and ``prop_sig`` for edge
        results (the denominator is the tested incident edges; NaN-p edges are
        in neither numerator nor denominator), component / block membership for
        ``nbs`` / ``cnbs`` (``n_components``, ``component_ids``,
        ``n_edges_in_components`` or the block counterparts), and the
        participation columns with an atlas that has network labels. F has no
        direction, so there is no effect column. Descriptive: no new p-value
        or significance.
        """
        from conninfpy._node_summary import build_node_summary

        unit = self.inference_unit
        report = {"component": getattr(self, "component_report", None),
                  "block": getattr(self, "block_report", None)}.get(unit)
        return build_node_summary(
            unit=unit,
            p_maps={"omnibus": np.asarray(self["omnibus"])},
            tails=["omnibus"],
            alpha=alpha,
            has_effect=False,
            report=report,
            atlas=atlas,
            sort=sort,
            top_k=top_k,
            attrs={"method": self.method, "tail_policy": self.tail_policy,
                   "tested_tail": None, "param_idx": None},
        )

    def network_summary(
        self,
        atlas: Any,
        *,
        alpha: float = 0.05,
        as_matrix: Optional[str] = None,
    ) -> pd.DataFrame:
        """Aggregate the omnibus p-values over pairs of atlas networks.

        Shortcut for :func:`summarize_by_network` with the single tail
        ``'omnibus'``; see it for the columns.
        """
        return summarize_by_network({"omnibus": np.asarray(self["omnibus"])}, atlas,
                                    alpha=alpha, as_matrix=as_matrix)

    def to_csv(
        self,
        path: "Any",
        *,
        atlas: Optional[Any] = None,
        alpha: float = 0.05,
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        index: bool = False,
        **to_csv_kwargs: Any,
    ) -> None:
        """Convenience: ``self.significant_edges(...).to_csv(path, ...)``."""
        df = self.significant_edges(
            atlas=atlas,
            alpha=alpha,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
        )
        df.to_csv(path, index=index, **to_csv_kwargs)

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

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        nsig = self.n_significant(0.05)["omnibus"]
        wall = (
            f"{self.wall_time_s:.2f}s" if self.wall_time_s is not None else "—"
        )
        accel = self.acceleration if self.acceleration else "empirical"
        prov = []
        if self.harmonized:
            prov.append("harmonized")
        if self.strata_provided:
            prov.append("strata")
        prov_str = (" prov=" + ",".join(prov)) if prov else ""
        return (
            f"OmnibusInferenceResult(method={self.method!r}, "
            f"n_perms={self.n_permutations}, accel={accel}, "
            f"sig@α=0.05: omnibus={nsig}, wall_time={wall}{prov_str})"
        )


def make_inference_result(
    tail_result: TailResult,
    *,
    method: str,
    n_permutations: int = 0,
    acceleration: Optional[str] = None,
    wall_time_s: Optional[float] = None,
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
    stat_positive: Optional[npt.NDArray[np.float64]] = None,
    stat_negative: Optional[npt.NDArray[np.float64]] = None,
    stat_type: str = "tstat",
    harmonized: bool = False,
    preserve_provided: bool = False,
    strata_provided: bool = False,
    combat_diagnostics: Optional[Dict[str, Any]] = None,
    e_grid: Optional[npt.NDArray[np.float64]] = None,
    h_grid: Optional[npt.NDArray[np.float64]] = None,
    two_tailed: bool = False,
) -> InferenceResult:
    """Promote a :class:`TailResult` into a metadata-enriched
    :class:`InferenceResult` without copying the p-value arrays."""
    return InferenceResult(
        tail_result["positive"],
        tail_result["negative"],
        method=method,
        n_permutations=n_permutations,
        acceleration=acceleration,
        wall_time_s=wall_time_s,
        null_max_dist=null_max_dist,
        stat_positive=stat_positive,
        stat_negative=stat_negative,
        stat_type=stat_type,
        harmonized=harmonized,
        preserve_provided=preserve_provided,
        strata_provided=strata_provided,
        combat_diagnostics=combat_diagnostics,
        e_grid=e_grid,
        h_grid=h_grid,
        two_tailed=two_tailed,
    )


__all__ = [
    "InferenceResult",
    "OmnibusInferenceResult",
    "make_inference_result",
    "summarize_by_network",
]
