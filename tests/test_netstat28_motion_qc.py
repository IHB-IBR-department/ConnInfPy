"""NETSTAT-28: additive, opt-in descriptive motion / QC-FC side-car diagnostics.

Synthetic data only. The diagnostics are descriptive (Spearman without
p-values), computed on the native input before inference, and must leave the
inference path bit-identical.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conninfpy import analyze
from conninfpy.analysis import AnalyzeResult
from conninfpy.interpret.motion_qc import (
    detect_fd_column,
    motion_qc_fc,
    motion_qc_fc_paired,
    resolve_fd,
)
from conninfpy.interpret.report import motion_qc_markdown


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def fd_vector(n: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.6, n)


def corr_stack(
    n: int,
    N: int = 6,
    *,
    seed: int = 0,
    fd: np.ndarray | None = None,
    beta: float = 0.0,
    distance: np.ndarray | None = None,
    noise: float = 0.2,
) -> np.ndarray:
    """Native correlation matrices with a known edge~FD (and edge~distance) trend."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(N, 1)
    m = len(iu[0])
    values = rng.normal(0.0, noise, (n, m))
    if beta and fd is not None:
        values = values + beta * ((fd - fd.mean()) / fd.std())[:, None]
    if distance is not None:
        values = values + distance[None, :]
    out = np.zeros((n, N, N))
    out[:, iu[0], iu[1]] = values
    out = out + out.transpose(0, 2, 1)
    out[:, np.arange(N), np.arange(N)] = 1.0
    return out


def edge_matrix(Y: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(Y.shape[1], 1)
    return Y[:, iu[0], iu[1]]


def coords_line(N: int, step: float = 12.0) -> np.ndarray:
    return np.column_stack([np.arange(N) * step, np.zeros(N), np.zeros(N)])


# ----------------------------------------------------------------------
# 1. Known FD~FC dependence is recovered; the null is flat
# ----------------------------------------------------------------------
def test_known_fd_fc_dependence_is_recovered_and_null_is_flat():
    n, N = 48, 6
    fd = fd_vector(n)
    signal = motion_qc_fc(corr_stack(n, N, fd=fd, beta=0.6), fd=fd, scale="r",
                         coords=coords_line(N))
    q = signal["qc_fc"]["quantiles"]
    assert signal["status"] == "available"
    assert q["median"] > 0.5 and q["q25"] > 0.3
    assert signal["qc_fc"]["n_edges_with_association"] == signal["n_edges"]
    assert signal["qc_fc"]["subject_level"]["status"] == "available"

    # A multiplicative motion effect raises the subject-level mean |r| with FD.
    Y_amp = corr_stack(n, N, fd=fd, beta=0.6, seed=8)
    E = edge_matrix(Y_amp) * (1.0 + 3.0 * fd)[:, None]
    Y_amp2 = np.zeros_like(Y_amp)
    iu = np.triu_indices(N, 1)
    Y_amp2[:, iu[0], iu[1]] = E
    Y_amp2 = Y_amp2 + Y_amp2.transpose(0, 2, 1)
    Y_amp2[:, np.arange(N), np.arange(N)] = 1.0
    amp = motion_qc_fc(Y_amp2, fd=fd, scale="r")
    assert amp["qc_fc"]["subject_level"]["rho"] > 0.3
    assert amp["qc_fc"]["quantiles"]["median"] > 0.3

    null = motion_qc_fc(corr_stack(n, N, fd=fd, beta=0.0), fd=fd, scale="r",
                        coords=coords_line(N))
    assert abs(null["qc_fc"]["quantiles"]["median"]) < 0.1
    assert abs(null["distance"]["spearman_mean_vs_distance"] or 0.0) < 0.3


def test_distance_association_follows_known_coordinates():
    n, N = 40, 6
    fd = fd_vector(n)
    coords = coords_line(N, step=10.0)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)[np.triu_indices(N, 1)]
    Y = corr_stack(n, N, fd=fd, beta=0.2, distance=-0.02 * dist)
    diag = motion_qc_fc(Y, fd=fd, scale="r", coords=coords)
    assert diag["distance"]["status"] == "available"
    assert diag["distance"]["spearman_mean_vs_distance"] < -0.5
    assert diag["distance"]["n_edges_used"] == diag["n_edges"]
    assert "post-inference distance_qc" in diag["distance"]["note"]


# ----------------------------------------------------------------------
# 2. FD source resolution and explicit unavailable statuses
# ----------------------------------------------------------------------
def test_no_fd_source_reports_fd_unavailable_and_fabricates_nothing():
    n, N = 20, 5
    Y = corr_stack(n, N)
    diag = motion_qc_fc(Y, scale="r")
    assert diag["status"] == "fd_unavailable"
    assert diag["fd"]["status"] == "unavailable"
    assert "no FD source" in diag["fd"]["reason"]
    assert "values" not in diag["fd"]  # the raw vector is not duplicated in the side-car
    assert diag["qc_fc"]["status"] == "fd_unavailable"
    assert all(v is None for v in diag["subjects"]["fd"])
    assert diag["subjects"]["mean_abs_r"] is not None
    assert all(v is None for v in diag["qc_fc"]["quantiles"].values())
    assert diag["unavailable"][0]["field"] == "qc_fc"


def test_constant_fd_is_an_explicit_status():
    n, N = 20, 5
    diag = motion_qc_fc(corr_stack(n, N), fd=np.full(n, 0.21), scale="r", subject_ids=list(range(n)))
    assert diag["status"] == "constant_fd"
    assert diag["qc_fc"]["status"] == "constant_fd"
    assert diag["qc_fc"]["subject_level"]["status"] == "constant"
    assert diag["fd"]["status"] == "available"


def test_fd_auto_detection_over_explicit_column_and_ids():
    n = 12
    fd1 = fd_vector(n, seed=1)
    fd2 = fd_vector(n, seed=2)
    pheno = pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(n)],
        "func_mean_fd": fd2,
        "Motion_FD": fd1,
    })
    assert detect_fd_column(pheno) == "Motion_FD"  # strict candidate order
    auto = resolve_fd(None, pheno=pheno, n=n, subject_ids=[f"s{i:02d}" for i in range(n)])
    assert auto["source"] == {"kind": "pheno_column", "name": "Motion_FD",
                              "detail": "auto-detected from the strict candidate list"}
    np.testing.assert_allclose(auto["values"], fd1)
    assert auto["id_alignment"] == "exact_id_match"

    explicit = resolve_fd("func_mean_fd", pheno=pheno, n=n)
    assert explicit["source"]["name"] == "func_mean_fd"
    np.testing.assert_allclose(explicit["values"], fd2)
    assert explicit["id_alignment"] == "positional"

    only_mean = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)], "func_mean_fd": fd2})
    assert detect_fd_column(only_mean) == "func_mean_fd"
    assert resolve_fd(None, pheno=pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)]}), n=n)["status"] == "unavailable"


def test_fd_contract_errors_are_explicit():
    n = 5
    pheno = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)], "Motion_FD": fd_vector(n)})
    with pytest.raises(ValueError, match="not a column"):
        resolve_fd("nope", pheno=pheno, n=n)
    with pytest.raises(ValueError, match="but the subject axis"):
        resolve_fd(np.zeros(n + 1), n=n)
    with pytest.raises(ValueError, match="do not match subject_ids"):
        resolve_fd(None, pheno=pheno, n=n, subject_ids=["x"] * n)
    with pytest.raises(ValueError, match="must be passed as pheno"):
        resolve_fd("Motion_FD", n=n)


# ----------------------------------------------------------------------
# 3. Native scale handling
# ----------------------------------------------------------------------
def test_unknown_units_get_a_neutral_label_and_no_transform():
    n, N = 24, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.3)
    diag = motion_qc_fc(Y, fd=fd, scale=None)
    assert diag["input"]["scale"] == "unknown"
    assert diag["input"]["mean_label"] == "mean_abs_value"
    assert "mean_abs_r" not in diag["subjects"]
    assert "none" in diag["input"]["transform"]
    expected = np.nanmean(np.abs(edge_matrix(Y)), axis=1)
    np.testing.assert_allclose(diag["subjects"]["mean_abs_value"], expected, rtol=1e-12)


def test_known_fisher_z_applies_documented_tanh():
    n, N = 24, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.3)
    Z = np.zeros_like(Y)
    iu = np.triu_indices(N, 1)
    Z[:, iu[0], iu[1]] = np.arctanh(np.clip(Y[:, iu[0], iu[1]], -0.999, 0.999))
    Z = Z + Z.transpose(0, 2, 1)
    diag = motion_qc_fc(Z, fd=fd, scale="z")
    assert diag["input"]["scale"] == "fisher_z"
    assert diag["input"]["mean_label"] == "mean_abs_r"
    assert "tanh" in diag["input"]["transform"]
    expected = np.nanmean(np.abs(np.tanh(edge_matrix(Z))), axis=1)
    np.testing.assert_allclose(diag["subjects"]["mean_abs_r"], expected, rtol=1e-12)
    with pytest.raises(ValueError, match="scale must be one of"):
        motion_qc_fc(Z, fd=fd, scale="banana")


# ----------------------------------------------------------------------
# 4. Denominators, n < 3, complete-pair status
# ----------------------------------------------------------------------
def test_nan_and_negative_shares_report_numerators_and_denominators():
    n, N = 16, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.1, seed=4)
    Y[0, 0, 1] = Y[0, 1, 0] = np.nan
    Y[1, 0, 1] = Y[1, 1, 0] = -0.8
    diag = motion_qc_fc(Y, fd=fd, scale="r")
    m = N * (N - 1) // 2
    subj = diag["subjects"]
    assert subj["nan_denominator"] == m
    assert subj["nan_count"][0] == 1 and subj["nan_count"][1] == 0
    assert subj["nan_share"][0] == pytest.approx(1.0 / m)
    manual = edge_matrix(Y)
    finite = np.isfinite(manual)
    np.testing.assert_array_equal(subj["negative_count"], np.where(finite, manual < 0, False).sum(1))
    np.testing.assert_array_equal(subj["negative_denominator"], finite.sum(1))
    np.testing.assert_allclose(
        [v if v is not None else np.nan for v in subj["negative_share"]],
        np.where(finite.sum(1) > 0,
                 np.where(finite, manual < 0, False).sum(1) / np.maximum(finite.sum(1), 1),
                 np.nan),
        equal_nan=True,
    )


def test_fewer_than_three_observations_is_explicitly_insufficient():
    n, N = 2, 4
    diag = motion_qc_fc(corr_stack(n, N), fd=fd_vector(n), scale="r", subject_ids=["a", "b"])
    assert diag["status"] == "insufficient_data"
    assert diag["qc_fc"]["status"] == "insufficient_data"
    assert diag["qc_fc"]["n_edges_insufficient_pairs"] == diag["n_edges"]
    assert diag["qc_fc"]["n_edges_with_association"] == 0
    assert diag["qc_fc"]["subject_level"]["status"] == "insufficient_data"


def test_constant_edge_on_complete_pairs_is_counted_not_interpreted():
    n, N = 12, 4
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.2)
    Y[:, 0, 1] = Y[:, 1, 0] = 0.42  # constant edge across observations
    diag = motion_qc_fc(Y, fd=fd, scale="r")
    assert diag["qc_fc"]["n_edges_constant_edge"] >= 1
    assert diag["qc_fc"]["n_edges_with_association"] == diag["n_edges"] - diag["qc_fc"]["n_edges_constant_edge"]


# ----------------------------------------------------------------------
# 5. Optional residualisation
# ----------------------------------------------------------------------
def test_residualisation_is_fc_only_with_per_edge_df_and_required_intercept():
    n, N = 30, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.3)
    X = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    diag = motion_qc_fc(Y, fd=fd, scale="r", nuisance=X, nuisance_names=["intercept", "x"])
    res = diag["adjustment"]["residualization"]
    assert diag["adjustment"]["mode"] == "fc_residualized_not_partial"
    assert res["status"] == "applied"
    assert res["interpretation"] == "fc_only_residualization_not_partial_spearman"
    assert res["partial_spearman"] is False
    assert "NOT a partial Spearman" in res["interpretation_note"]
    assert res["matrix_rank"] == 2 and res["residual_df"] == n - 2
    assert res["residual_df_min"] == res["residual_df_max"] == n - 2
    assert res["intercept_present"] is True
    assert res["intercept_columns"] == ["intercept"]
    assert res["n_edges_insufficient_df"] == 0
    assert res["column_names"] == ["intercept", "x"]
    assert "residuals" not in res  # the (n, m) residual matrix is never serialised
    assert diag["qc_fc"]["adjustment"] == "fc_residualized_not_partial"

    tiny = motion_qc_fc(corr_stack(2, N), fd=fd_vector(2), scale="r", nuisance=np.ones((2, 1)))
    assert tiny["adjustment"]["mode"] == "unavailable"
    assert "residual df" in tiny["adjustment"]["reason"]
    assert tiny["status"] == "unavailable"
    assert tiny["adjustment"]["residualization"]["residual_df"] < 2

    # No intercept column: the residualisation is refused, never silently added.
    no_intercept = motion_qc_fc(Y, fd=fd, scale="r", nuisance=np.linspace(-1.0, 1.0, n)[:, None])
    assert no_intercept["adjustment"]["residualization"]["intercept_present"] is False
    assert no_intercept["adjustment"]["mode"] == "unavailable"
    assert "intercept" in no_intercept["adjustment"]["reason"]
    assert no_intercept["status"] == "unavailable"


def test_zero_residual_variance_is_unavailable():
    n, N = 8, 4
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.2)
    Y[:, :, :] = 0.1  # every edge constant across observations
    Y[:, np.arange(N), np.arange(N)] = 1.0
    diag = motion_qc_fc(Y, fd=fd, scale="r", nuisance=np.ones((n, 1)))
    res = diag["adjustment"]["residualization"]
    assert diag["adjustment"]["mode"] == "unavailable"
    assert res["status"] == "unavailable"
    assert "non-zero residual variance" in res["reason"]
    assert res["n_edges_zero_residual_variance"] == N * (N - 1) // 2
    assert diag["status"] == "unavailable"


def test_no_nuisance_records_the_unadjusted_mode():
    n, N = 10, 4
    diag = motion_qc_fc(corr_stack(n, N), fd=fd_vector(n), scale="r")
    assert diag["adjustment"]["mode"] == "none"
    assert diag["adjustment"]["residualization"] is None
    assert diag["qc_fc"]["adjustment"] == "none"
    assert "no adjustment is claimed" in diag["adjustment"]["note"]


# ----------------------------------------------------------------------
# 6. Paired A/B: separate arms, FD tied to its own rows, pooling gated on IDs
# ----------------------------------------------------------------------
def test_paired_conditions_are_separate_and_pooling_needs_verified_ids():
    n, N = 10, 4
    fd_a = fd_vector(n, seed=11)
    fd_b = fd_vector(n, seed=12)
    A = corr_stack(n, N, fd=fd_a, beta=0.4, seed=21)
    B = corr_stack(n, N, fd=fd_b, beta=0.4, seed=22)
    ids = [f"s{i}" for i in range(n)]
    diag = motion_qc_fc_paired(A, B, fd_group1=fd_a, fd_group2=fd_b,
                               ids_group1=ids, ids_group2=ids, scale="r")
    assert diag["kind"] == "motion_qc_fc_paired"
    assert diag["pooling"]["status"] == "verified_ids"
    assert diag["conditions"]["A"]["n_observations"] == n
    assert diag["conditions"]["B"]["n_observations"] == n
    assert diag["pooled"]["n_observations"] == 2 * n
    np.testing.assert_allclose(diag["conditions"]["B"]["subjects"]["fd"], fd_b, rtol=0, atol=1e-12)
    assert diag["difference"]["status"] == "not_computed"
    assert "must not be mixed" in diag["difference"]["reason"]

    no_ids = motion_qc_fc_paired(A, B, fd_group1=fd_a, fd_group2=fd_b, scale="r")
    assert no_ids["pooled"] is None
    assert no_ids["pooling"]["status"] == "unavailable"
    assert "no verified subject identity" in no_ids["pooling"]["reason"]


def test_paired_fd_is_reordered_with_the_condition_b_id_permutation():
    n, N = 8, 4
    ids = [f"s{i}" for i in range(n)]
    perm = np.random.default_rng(5).permutation(n)
    fd_by_subject = {ids[i]: 0.01 * (i + 1) for i in range(n)}
    fd_b_original = np.array([fd_by_subject[ids[i]] for i in range(n)])
    A = corr_stack(n, N, seed=1)
    B_rows = corr_stack(n, N, seed=2)[perm]
    diag = motion_qc_fc_paired(
        A, B_rows,
        fd_group1=np.array([fd_by_subject[s] for s in ids]),
        fd_group2=fd_b_original[perm],
        ids_group1=ids, ids_group2=[ids[i] for i in perm], scale="r",
    )
    # Condition B rows are moved into A order, and its FD follows the same rows.
    np.testing.assert_allclose(
        diag["conditions"]["B"]["subjects"]["fd"],
        np.array([fd_by_subject[s] for s in ids]),
    )
    assert diag["id_alignment"] == "exact_id_match"
    broken = list(ids)
    broken[-1] = "someone_else"
    with pytest.raises(ValueError, match="only one condition|has no partner|missing|bijection"):
        motion_qc_fc_paired(A, B_rows, fd_group1=fd_b_original,
                            fd_group2=fd_b_original, ids_group1=ids,
                            ids_group2=broken, scale="r")


# ----------------------------------------------------------------------
# 7. analyze() wiring: opt-in, bit-identical inference, provenance, bundle
# ----------------------------------------------------------------------
def _glm_kwargs(n: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    return dict(
        interest=rng.normal(size=n),
        method="tstat",
        n_permutations=39,
        rng=11,
        use_mp=False,
    )


def test_analyze_sidecar_is_bit_identical_and_opt_in():
    n, N = 16, 5
    Y = corr_stack(n, N, fd=fd_vector(n), beta=0.3, seed=9)
    kwargs = _glm_kwargs(n)
    base = analyze(Y, **kwargs)
    with_sidecar = analyze(
        Y, **kwargs,
        motion_qc={"fd": fd_vector(n), "subject_ids": [f"s{i}" for i in range(n)],
                   "atlas": None},
    )
    assert base.motion_qc is None
    assert getattr(base.inference, "motion_qc", None) is None
    diag = with_sidecar.motion_qc
    assert diag["status"] == "available" and diag["n_observations"] == n
    assert with_sidecar.inference.motion_qc is diag
    for field in ("positive", "negative", "stat_positive", "stat_negative",
                  "stat_signed", "beta"):
        np.testing.assert_array_equal(
            getattr(base.inference, field), getattr(with_sidecar.inference, field)
        )
    assert base.inference.n_permutations == with_sidecar.inference.n_permutations
    assert getattr(base.inference, "rng_provenance", None) == getattr(
        with_sidecar.inference, "rng_provenance", None)
    # Only the additive diagnostic flag differs.
    extra = [f for f in with_sidecar.flags if f not in base.flags]
    assert len(extra) == 1 and "motion/QC-FC side-car" in extra[0]
    assert [f for f in with_sidecar.flags if f not in extra] == base.flags


def test_analyze_sidecar_reaches_bundle_and_workspace(tmp_path: Path):
    n, N = 10, 4
    Y = corr_stack(n, N, fd=fd_vector(n), beta=0.3, seed=12)
    result = analyze(Y, **_glm_kwargs(n), motion_qc={"fd": fd_vector(n), "subject_ids": list(range(n))})
    bundle = result.to_bundle()
    assert bundle.metadata["motion_qc"]["status"] == "available"
    path = tmp_path / "bundle.pkl"
    bundle.save(path)
    reloaded = type(bundle).load(path)
    assert reloaded.metadata["motion_qc"]["n_observations"] == n

    from apps.utils.workspace_io import build_workspace_archive, restore_workspace_archive

    state = {
        "inference_result": result,
        "connectivity_data_kind": "correlation",
        "dataset_info": {"name": "unit", "mode": "custom"},
    }
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        state_doc = json.loads(zf.read("workspace/state.json").decode("utf-8"))
    assert state_doc["workspace_schema_version"] == 2  # no schema bump
    assert "motion_qc" in json.dumps(state_doc)
    target: dict = {}
    restore_workspace_archive(zip_bytes, target, project_root=tmp_path)
    restored = target["inference_result"]
    assert isinstance(restored, AnalyzeResult)
    assert restored.motion_qc is not None
    assert restored.motion_qc["status"] == "available"
    assert restored.motion_qc["fd"]["source"]["detail"] == "explicit per-observation vector in subject-axis order"


def test_analyze_paired_sidecar_keeps_condition_fd_and_inference():
    n, N = 8, 4
    ids = [f"s{i}" for i in range(n)]
    perm = np.random.default_rng(2).permutation(n)
    fd_a = np.linspace(0.05, 0.4, n)
    fd_b = np.linspace(0.5, 0.1, n)
    A = corr_stack(n, N, fd=fd_a, beta=0.3, seed=31)
    B_ordered = corr_stack(n, N, fd=fd_b, beta=0.3, seed=32)
    kwargs = dict(test_type="paired", method="tstat", n_permutations=39, rng=4, use_mp=False)
    ordered = analyze(None, group1=A, group2=B_ordered, **kwargs,
                      motion_qc={"fd_group1": fd_a, "fd_group2": fd_b})
    shuffled = analyze(
        None, group1=A, group2=B_ordered[perm],
        ids_group1=ids, ids_group2=[ids[i] for i in perm], **kwargs,
        motion_qc={"fd_group1": fd_a, "fd_group2": fd_b[perm]},
    )
    for field in ("positive", "negative", "stat_signed"):
        np.testing.assert_array_equal(
            getattr(ordered.inference, field), getattr(shuffled.inference, field))
    np.testing.assert_allclose(
        shuffled.motion_qc["conditions"]["B"]["subjects"]["fd"], fd_b, rtol=0, atol=1e-12)
    assert shuffled.motion_qc["pooling"]["status"] == "verified_ids"
    assert shuffled.motion_qc["status"] == ordered.motion_qc["status"]


def test_analyze_without_fd_source_still_returns_the_sidecar():
    n, N = 10, 4
    Y = corr_stack(n, N, seed=13)
    result = analyze(Y, **_glm_kwargs(n), motion_qc={})
    assert result.motion_qc["status"] == "fd_unavailable"
    assert result.motion_qc["fd"]["status"] == "unavailable"
    assert any("fd_unavailable" in f for f in result.flags)


# ----------------------------------------------------------------------
# 8. Report section
# ----------------------------------------------------------------------
def test_motion_qc_markdown_reports_provenance_and_unavailable_reasons():
    n, N = 12, 4
    fd = fd_vector(n)
    diag = motion_qc_fc(corr_stack(n, N, fd=fd, beta=0.3), fd=fd, scale="r")
    text = motion_qc_markdown(diag)
    assert "Motion / QC-FC side-car" in text
    assert "FD source" in text and "mm" in text
    assert "Power vs Jenkinson" in text or "unknown" in text
    assert "coordinates_unavailable" in text
    assert "no p-values" in text or "without p-values" in text
    assert motion_qc_markdown(None) == ""
    paired = motion_qc_fc_paired(
        corr_stack(6, N, seed=1), corr_stack(6, N, seed=2),
        fd_group1=fd_vector(6), fd_group2=fd_vector(6, seed=2), scale="r")
    text_paired = motion_qc_markdown(paired)
    assert "Condition A" in text_paired and "Pooling:" in text_paired


# ----------------------------------------------------------------------
# 9. CLI opt-in spec
# ----------------------------------------------------------------------
def test_cli_motion_qc_spec_is_opt_in_aligned_and_strict():
    from conninfpy.cli import ConfigError, _motion_qc_spec

    pheno = pd.DataFrame({
        "subject_id": ["a", "b", "c", "d"],
        "group": [0, 0, 1, 1],
        "Motion_FD": [0.1, 0.2, 0.3, 0.4],
        "condition": ["rest", "task", "rest", "task"],
    })
    dataset = SimpleNamespace(pheno=pheno, subject_ids=["a", "b", "c", "d"],
                              condition_column="condition")
    glm_cfg = {"design": {"type": "glm", "interest": "group"}}
    assert _motion_qc_spec(dataset, glm_cfg) is None
    auto = _motion_qc_spec(dataset, glm_cfg, enabled=True)
    np.testing.assert_allclose(auto["fd"], [0.1, 0.2, 0.3, 0.4])
    assert auto["subject_ids"] == ["a", "b", "c", "d"]

    # A missing-value design row is dropped by build_analysis; FD follows the
    # same rows (guard: the FD vector length equals the analysed observations).
    pheno_nan = pheno.copy()
    pheno_nan.loc[1, "Motion_FD"] = np.nan
    nan_dataset = SimpleNamespace(pheno=pheno_nan, subject_ids=list("abcd"),
                                  condition_column="condition")
    nan_cfg = {"design": {"type": "glm", "interest": "group", "confounds": "Motion_FD"}}
    sliced = _motion_qc_spec(nan_dataset, nan_cfg, enabled=True,
                             kwargs={"interest": np.zeros(3), "Y": np.zeros((3, 4, 4))})
    np.testing.assert_allclose(sliced["fd"], [0.1, 0.3, 0.4])

    configured = _motion_qc_spec(
        dataset,
        {"design": {"type": "glm", "interest": "group"},
         "diagnostics": {"motion_qc": {"fd": "Motion_FD", "fd_units": "mm"}}},
    )
    assert configured["fd_units"] == "mm"
    with pytest.raises(ConfigError, match="unknown diagnostics.motion_qc key"):
        _motion_qc_spec(dataset, {"design": {"type": "glm", "interest": "group"},
                                  "diagnostics": {"motion_qc": {"fdx": 1}}})
    with pytest.raises(ConfigError, match="must be a mapping"):
        _motion_qc_spec(dataset, {"design": {"type": "glm", "interest": "group"},
                                  "diagnostics": {"motion_qc": [1, 2]}})
    with pytest.raises(ConfigError, match="not a phenotype column"):
        _motion_qc_spec(dataset, {"design": {"type": "glm", "interest": "group"},
                                  "diagnostics": {"motion_qc": {"fd": "nope"}}})
    with pytest.raises(ConfigError, match="alignment guard"):
        _motion_qc_spec(dataset, glm_cfg, kwargs={"interest": np.zeros(2),
                                                  "Y": np.zeros((2, 3, 3))}, enabled=True)

    paired_cfg = {"design": {"type": "paired", "condition": "condition",
                             "levels": ["rest", "task"], "subject": "subject_id"}}
    paired_spec = _motion_qc_spec(dataset, paired_cfg, enabled=True)
    np.testing.assert_allclose(paired_spec["fd_group1"], [0.1, 0.3])
    np.testing.assert_allclose(paired_spec["fd_group2"], [0.2, 0.4])
    assert paired_spec["condition_a"] == "rest" and paired_spec["condition_b"] == "task"


def test_cli_run_writes_the_sidecar_into_provenance(tmp_path: Path):
    import yaml
    from conninfpy.cli import run

    n, N = 14, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.4, seed=6)
    iu = np.triu_indices(N, 1)  # keep the CLI fixture inside [-1, 1]
    Y[:, iu[0], iu[1]] = np.clip(Y[:, iu[0], iu[1]], -0.95, 0.95)
    Y = Y + Y.transpose(0, 2, 1) / 0.0 if False else Y
    Y[:, iu[1], iu[0]] = Y[:, iu[0], iu[1]]
    rng = np.random.default_rng(4)
    pheno = pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(n)],
        "age": rng.normal(30, 8, n),
        "Motion_FD": fd,
    })
    np.save(tmp_path / "conn.npy", Y)
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    (tmp_path / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "loader": "NumpyLoader",
        "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
    }))
    cfg = {
        "data": "study.yaml",
        "design": {"type": "glm", "interest": "age", "confounds": ["Motion_FD"]},
        "inference": {"method": "tstat", "n_permutations": 20, "use_mp": False, "rng": 0},
        "output": "out",
        "diagnostics": {"motion_qc": {"fd": "Motion_FD"}},
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    written = run(cfg_path)
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
    side_car = provenance["results"]["result"]["motion_qc"]
    assert side_car["status"] == "available"
    assert side_car["n_observations"] == n
    assert side_car["fd"]["source"]["name"] == "Motion_FD"
    assert side_car["fd"]["units"] == "mm"
    assert side_car["input"]["scale"] == "pearson_r"  # native correlations, no silent tanh
    assert "bundle" in written and "provenance" in written


# ----------------------------------------------------------------------
# 10. GUI: the saved side-car renders on the results tab (AppTest)
# ----------------------------------------------------------------------
APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"


def test_results_tab_renders_the_saved_sidecar():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from conninfpy import compute_p_val

    N = 5
    rng = np.random.default_rng(11)

    def sym(a):
        a = (a + a.transpose(0, 2, 1)) / 2
        a[:, np.arange(N), np.arange(N)] = 0.0
        return a

    g1, g2 = sym(rng.normal(size=(9, N, N))), sym(rng.normal(size=(9, N, N)))
    for i in range(3):
        for j in range(3):
            if i != j:
                g2[:, i, j] += 1.5
    inference = compute_p_val(g1, g2, test_type="two-sample", var_equal=True, method="tstat",
                              n_permutations=60, rng=3, use_mp=False, acceleration=None)
    n, N2 = 18, 5
    fd = fd_vector(n)
    diag = motion_qc_fc(np.concatenate([g1, g2], axis=0), fd=fd, scale="r",
                        subject_ids=[f"s{i}" for i in range(n)])
    result = AnalyzeResult(inference=inference, motion_qc=diag)

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((n, N2, N2))
    app.session_state.pheno_df = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)],
                                              "group": [0] * 9 + [1] * 9})
    app.session_state.loaded_reference_atlas = None
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = result.significant_edges()
    app.session_state.run_plan = {"method": "tstat", "alpha": 0.05,
                                  "loaded_settings_hash": "fixture data",
                                  "active_atlas_signature": "fixture atlas",
                                  "question_type": "Group Difference"}
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    markdown = "\n".join(m.value for m in app.markdown)
    assert "Motion / QC–FC Side-Car" in markdown or "Motion / QC–FC Side-Car" in markdown.replace("–", "-")
    assert any("FD source" == m.proto.label for m in app.get("metric"))
    frames = [df.value for df in app.dataframe]
    assert any("mean_abs_r" in df.columns for df in frames)
    labels = [b.proto.label for b in app.get("download_button")]
    assert any("motion/QC–FC report section" in label for label in labels)


def test_design_tab_offers_the_opt_in_and_the_fd_column():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from conninfpy.atlas import AtlasInfo

    N, n = 5, 12
    rng = np.random.default_rng(1)
    Y = np.empty((n, N, N))
    for s in range(n):
        a = rng.standard_normal((N, 30))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        Y[s] = c
    pheno = pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(n)],
        "group": [0] * 6 + [1] * 6,
        "Motion_FD": rng.uniform(0.05, 0.5, n),
    })
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N,
                      coords=np.column_stack([np.arange(N) * 15.0, np.zeros(N), np.zeros(N)]))
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = Y
    app.session_state.connectivity_data_kind = "correlation"
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.active_tab = "2. Design & Inference"
    app.run()
    assert not app.exception
    opt_in = next(
        (c for c in app.checkbox if "motion / QC–FC side-car" in c.proto.label), None
    )
    assert opt_in is not None and opt_in.value is False
    opt_in.set_value(True)
    app.run()
    assert not app.exception
    fd_select = next(
        (s for s in app.selectbox if "FD column" in s.proto.label), None
    )
    assert fd_select is not None
    assert list(fd_select.proto.options) == [
        "auto-detect (Motion_FD / func_mean_fd)", "Motion_FD"
    ]
    assert any("Power vs Jenkinson" in c.value for c in app.caption)


def test_prelaunch_preview_matches_the_saved_diagnostic():
    pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from apps.components.design_inference import _motion_qc_preview
    from conninfpy.atlas import AtlasInfo

    n, N = 14, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.4, seed=17)
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N,
                      coords=coords_line(N))
    spec = {"fd": fd, "subject_ids": [f"s{i}" for i in range(n)], "pheno": None}
    preview = _motion_qc_preview(spec, "glm", Y, None, None, atlas=atlas, scale="r")
    saved = analyze(Y, **_glm_kwargs(n), atlas=atlas,
                    motion_qc={"fd": fd, "subject_ids": [f"s{i}" for i in range(n)]})
    assert preview["status"] == saved.motion_qc["status"] == "available"
    assert preview["qc_fc"]["quantiles"] == saved.motion_qc["qc_fc"]["quantiles"]
    assert preview["distance"]["spearman_mean_vs_distance"] == pytest.approx(
        saved.motion_qc["distance"]["spearman_mean_vs_distance"])
    assert preview["n_edges"] == saved.motion_qc["n_edges"]


# ----------------------------------------------------------------------
# 11. Rework round: ROI-level non-finite, QC-FC~distance, FC-only adjustment
# ----------------------------------------------------------------------
def test_roi_level_nonfinite_counts_and_denominators():
    n, N = 12, 4
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.2, seed=24)
    Y[0, 0, 1] = np.nan
    Y[1, 0, 2] = np.inf
    Y[2, 1, 3] = -np.inf
    diag = motion_qc_fc(Y, fd=fd, scale="r", roi_ids=[f"ROI{i}" for i in range(N)])
    roi = diag["roi_nonfinite"]
    assert roi["denominator"] == n * (N - 1)
    assert roi["n_edges_per_roi"] == N - 1
    assert roi["nan_count"][0] == 1 and roi["inf_count"][0] == 1
    assert roi["nan_count"][1] == 0 and roi["inf_count"][1] == 1
    assert roi["nan_count"][2] == 0 and roi["inf_count"][2] == 0
    assert roi["nonfinite_count"] == [
        a + b for a, b in zip(roi["nan_count"], roi["inf_count"])
    ]
    assert roi["finite_count"][0] == roi["denominator"] - 2
    assert roi["nonfinite_share"][0] == pytest.approx(2 / roi["denominator"])
    assert diag["subjects"]["nan_denominator"] == N * (N - 1) // 2
    text = motion_qc_markdown(diag)
    assert "Per-ROI non-finite entries" in text


def test_qc_fc_vs_distance_is_a_second_measure_with_its_own_status():
    n, N = 40, 7
    fd = fd_vector(n)
    coords = coords_line(N, step=12.0)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)[np.triu_indices(N, 1)]
    # Short edges move with FD, long edges do not -> negative QC-FC~distance.
    E = edge_matrix(corr_stack(n, N, seed=28)) + (
        (0.15 + fd)[:, None] * np.exp(-dist / 25.0)[None, :]
    )
    Y = np.zeros((n, N, N))
    iu = np.triu_indices(N, 1)
    Y[:, iu[0], iu[1]] = E
    Y = Y + Y.transpose(0, 2, 1)
    Y[:, np.arange(N), np.arange(N)] = 1.0
    diag = motion_qc_fc(Y, fd=fd, scale="r", coords=coords)
    block = diag["distance"]["qc_fc_vs_distance"]
    assert block["status"] == "available"
    assert block["measure"].startswith("Spearman(per-edge Spearman(edge, FD), distance)")
    assert block["rho"] < -0.3
    assert block["n_pairs"] == diag["qc_fc"]["n_edges_with_association"] == diag["n_edges"]
    # The two measures are separately labelled and not the same number.
    assert diag["distance"]["spearman_mean_vs_distance_measure"].startswith(
        "Spearman(mean edge value over observations, distance)")
    assert "NOT the distance dependence of QC-FC" in diag["distance"]["measures_note"]
    no_fd = motion_qc_fc(Y, scale="r", coords=coords)
    assert no_fd["distance"]["qc_fc_vs_distance"]["status"] == "unavailable"
    assert "per-edge edge-FD associations are unavailable" in no_fd["distance"]["qc_fc_vs_distance"]["reason"]
    no_coords = motion_qc_fc(Y, fd=fd, scale="r")
    assert no_coords["distance"]["qc_fc_vs_distance"]["status"] == "coordinates_unavailable"


def test_fc_only_adjustment_is_labelled_and_uses_per_edge_df():
    n, N = 24, 5
    fd = fd_vector(n)
    Y = corr_stack(n, N, fd=fd, beta=0.3, seed=29)
    Y[0, 0, 1] = np.nan  # one edge with a different complete-row set
    X = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    diag = motion_qc_fc(Y, fd=fd, scale="r", nuisance=X)
    res = diag["adjustment"]["residualization"]
    assert res["status"] == "applied"
    assert res["interpretation"] == "fc_only_residualization_not_partial_spearman"
    assert res["partial_spearman"] is False
    assert res["mask_kind"] == "per_edge_complete_rows"
    assert res["n_edges_insufficient_df"] == 0
    assert res["residual_df_min"] >= 2
    assert res["residual_df_min"] <= res["residual_df_max"]
    assert "not" in motion_qc_markdown(diag).lower()
    assert "FC-only" in motion_qc_markdown(diag) or "FC-only" in motion_qc_markdown(diag).replace("‑", "-")


# ----------------------------------------------------------------------
# 12. Rework round: CLI alignment for selected groups / interleaved two-sample
# ----------------------------------------------------------------------
def _cli_study(tmp_path: Path, pheno: pd.DataFrame, Y: np.ndarray) -> Path:
    import yaml

    np.save(tmp_path / "conn.npy", Y)
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    (tmp_path / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "loader": "NumpyLoader",
        "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
    }))
    return tmp_path / "study.yaml"


def _valid_corr(Y: np.ndarray) -> np.ndarray:
    N = Y.shape[1]
    iu = np.triu_indices(N, 1)
    Y = Y.copy()
    Y[:, iu[0], iu[1]] = np.clip(Y[:, iu[0], iu[1]], -0.95, 0.95)
    Y[:, iu[1], iu[0]] = Y[:, iu[0], iu[1]]
    return Y


def test_cli_two_sample_interleaved_keeps_fd_and_labels_in_analysis_order(tmp_path: Path):
    import yaml
    from conninfpy.cli import run

    n, N = 16, 5
    fd = np.round(np.linspace(0.05, 0.62, n), 3)
    Y = _valid_corr(corr_stack(n, N, fd=fd, beta=0.3, seed=31))
    # Interleaved rows plus a third level that must not enter the contrast.
    levels = ["case", "ctrl"]
    groups = ["case", "ctrl", "case", "ctrl", "other", "case", "ctrl", "case",
              "ctrl", "case", "ctrl", "other", "case", "ctrl", "case", "ctrl"]
    pheno = pd.DataFrame({"subject_id": [f"s{i:02d}" for i in range(n)],
                          "group": groups,
                          "Motion_FD": fd,
                          "site": ["S1"] * 8 + ["S2"] * 8})
    study = _cli_study(tmp_path, pheno, Y)
    cfg = {
        "data": study.name,
        "design": {"type": "two-sample", "group": "group", "levels": levels},
        "inference": {"method": "tstat", "n_permutations": 20, "use_mp": False, "rng": 0},
        "output": "out",
        "diagnostics": {"motion_qc": {"fd": "Motion_FD"}},
    }
    analysis = tmp_path / "analysis.yaml"
    analysis.write_text(yaml.safe_dump(cfg))
    run(analysis)
    side_car = json.loads((tmp_path / "out" / "provenance.json").read_text())["results"]["result"]["motion_qc"]
    keep_rows = [k for k, g in enumerate(groups) if g in levels]
    expected_order = [k for k in keep_rows if groups[k] == "case"] + [
        k for k in keep_rows if groups[k] == "ctrl"
    ]
    assert side_car["status"] == "available"
    assert side_car["n_observations"] == len(expected_order)
    np.testing.assert_allclose(side_car["subjects"]["fd"], fd[expected_order], rtol=0, atol=1e-9)
    assert side_car["subjects"]["condition"] == (
        ["case"] * sum(1 for k in keep_rows if groups[k] == "case")
        + ["ctrl"] * sum(1 for k in keep_rows if groups[k] == "ctrl")
    )
    assert side_car["subjects"]["index_values"] == expected_order
    assert len(side_car["subjects"]["fd"]) == len(side_car["subjects"]["condition"])


def test_cli_paired_shuffled_b_keeps_condition_fd_with_its_rows(tmp_path: Path):
    import yaml
    from conninfpy.cli import run

    n_sub, N = 8, 5
    rng = np.random.default_rng(7)
    rows = []
    fd_by = {}
    Y_rows = []
    for s_idx in range(n_sub):
        for cond in ("rest", "task"):
            rows.append({"subject_id": f"s{s_idx:02d}", "condition": cond})
            fd_by[(f"s{s_idx:02d}", cond)] = round(0.05 + 0.03 * s_idx + (0.2 if cond == "task" else 0.0), 4)
            Y_rows.append(rng.normal(size=(N, N)))
    Y = np.stack(Y_rows)
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    iu = np.triu_indices(N, 1)
    Y[:, iu[0], iu[1]] = np.clip(Y[:, iu[0], iu[1]], -0.95, 0.95)
    Y[:, iu[1], iu[0]] = Y[:, iu[0], iu[1]]
    Y[:, np.arange(N), np.arange(N)] = 1.0
    pheno = pd.DataFrame(rows)
    pheno["Motion_FD"] = [fd_by[(r.subject_id, r.condition)] for r in pheno.itertuples()]
    order = rng.permutation(len(pheno))  # shuffle the rows, keeping row/FD pairing
    pheno = pheno.iloc[order].reset_index(drop=True)
    Y = Y[order]
    study = _cli_study(tmp_path, pheno, Y)
    cfg = {
        "data": study.name,
        "design": {"type": "paired", "condition": "condition", "levels": ["rest", "task"],
                   "subject": "subject_id"},
        "inference": {"method": "tstat", "n_permutations": 20, "use_mp": False, "rng": 0},
        "output": "out",
        "diagnostics": {"motion_qc": {"fd": "Motion_FD"}},
    }
    analysis = tmp_path / "analysis.yaml"
    analysis.write_text(yaml.safe_dump(cfg))
    run(analysis)
    side_car = json.loads((tmp_path / "out" / "provenance.json").read_text())["results"]["result"]["motion_qc"]
    # Condition A is the reference row order: after the row shuffle it is the
    # shuffled order of the "rest" rows, and condition B (and its FD) follows it.
    ids_a = list(pheno.loc[pheno["condition"] == "rest", "subject_id"])
    assert side_car["kind"] == "motion_qc_fc_paired"
    assert side_car["pooling"]["status"] == "verified_ids"
    assert side_car["id_alignment"] == "exact_id_match"
    arm_a = side_car["conditions"]["rest"]
    arm_b = side_car["conditions"]["task"]
    assert arm_a["subjects"]["subject_ids"] == ids_a
    assert arm_a["subjects"]["fd"] == [
        pytest.approx(fd_by[(s, "rest")], abs=1e-9) for s in ids_a
    ]
    assert arm_b["subjects"]["fd"] == [
        pytest.approx(fd_by[(s, "task")], abs=1e-9) for s in ids_a
    ]
    assert arm_a["subjects"]["condition"] == ["rest"] * n_sub
    assert arm_b["subjects"]["condition"] == ["task"] * n_sub


# ----------------------------------------------------------------------
# 13. Rework round: preview available before the run command; off = nothing
# ----------------------------------------------------------------------
def test_gui_preview_is_available_before_the_run_command():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from conninfpy.atlas import AtlasInfo

    N, n = 5, 12
    rng = np.random.default_rng(1)
    Y = np.empty((n, N, N))
    for s in range(n):
        a = rng.standard_normal((N, 30))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        Y[s] = c
    pheno = pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(n)],
        "group": [0] * 6 + [1] * 6,
        "Motion_FD": rng.uniform(0.05, 0.5, n),
    })
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N,
                      coords=np.column_stack([np.arange(N) * 15.0, np.zeros(N), np.zeros(N)]))
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = Y
    app.session_state.connectivity_data_kind = "correlation"
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.active_tab = "2. Design & Inference"
    app.run()
    assert not app.exception
    assert app.session_state.get("motion_qc_preview") is None
    next(c for c in app.checkbox if "motion / QC–FC side-car" in c.proto.label).set_value(True)
    app.run()
    next(b for b in app.button if "Compute motion / QC–FC preview" in b.proto.label).click()
    app.run()
    assert not app.exception
    preview = app.session_state.get("motion_qc_preview")
    assert preview is not None and preview["status"] in ("available", "fd_unavailable")
    assert app.session_state.get("inference_result") is None  # nothing was inferred
    task = app.session_state.get("inference_task")
    assert task is None or task.status == "idle"
    assert any("Motion / QC-FC side-car" in m.value for m in app.markdown)
    assert any("without running inference" in c.value for c in app.caption)


def test_results_tab_shows_nothing_when_the_diagnostic_was_not_opted_in():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from conninfpy import compute_p_val

    N = 5
    rng = np.random.default_rng(11)

    def sym(a):
        a = (a + a.transpose(0, 2, 1)) / 2
        a[:, np.arange(N), np.arange(N)] = 0.0
        return a

    g1, g2 = sym(rng.normal(size=(9, N, N))), sym(rng.normal(size=(9, N, N)))
    inference = compute_p_val(g1, g2, test_type="two-sample", var_equal=True, method="tstat",
                              n_permutations=60, rng=3, use_mp=False, acceleration=None)
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((18, N, N))
    app.session_state.pheno_df = pd.DataFrame({"subject_id": [f"s{i}" for i in range(18)],
                                              "group": [0] * 9 + [1] * 9})
    app.session_state.loaded_reference_atlas = None
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = AnalyzeResult(inference=inference)  # opt-in off
    app.session_state.edges_df = inference.significant_edges()
    app.session_state.run_plan = {"method": "tstat", "alpha": 0.05,
                                  "loaded_settings_hash": "fixture data",
                                  "active_atlas_signature": "fixture atlas",
                                  "question_type": "Group Difference"}
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    markdown = "\n".join(m.value for m in app.markdown)
    assert "Motion / QC–FC Side-Car" not in markdown
    assert not any("motion/QC–FC" in b.proto.label for b in app.get("download_button"))


def test_root_public_surface_is_unchanged_and_names_live_in_interpret():
    import conninfpy
    import conninfpy.interpret as interpret

    assert len(conninfpy.__all__) <= 21
    for name in ("motion_qc_fc", "motion_qc_fc_paired", "resolve_fd", "detect_fd_column",
                 "motion_qc_markdown"):
        assert name in interpret.__all__
        assert hasattr(interpret, name)
        assert name not in conninfpy.__all__
        assert not hasattr(conninfpy, name)


def test_launch_helpers_align_fd_labels_and_rows_for_every_path():
    pytest.importorskip("streamlit")
    from apps.components.design_inference import (
        _build_design_arrays,
        _motion_qc_preview,
        _motion_qc_spec_from_arrays,
    )

    n, N = 12, 4
    fd = np.round(np.linspace(0.05, 0.5, n), 4)
    Y = corr_stack(n, N, fd=fd, beta=0.3, seed=42)
    pheno = pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(n)],
        "group": ["case", "ctrl", "other", "case", "ctrl", "case",
                  "ctrl", "other", "case", "ctrl", "case", "ctrl"],
        "condition": ["rest", "rest", "task", "task", "rest", "task",
                      "rest", "task", "rest", "task", "task", "rest"],
        "Motion_FD": fd,
        "site": ["S1"] * 6 + ["S2"] * 6,
    })
    # Two-sample: interleaved source rows, group1 then group2 in the arrays.
    pheno_two = pheno.copy()
    pheno_two.loc[4, "site"] = None  # dropped because `site` is in the design
    arrays = _build_design_arrays(
        Y, pheno_two, test_type="two-sample", question_choice="Group Difference",
        group_col="group", ref_group="case", target_group="ctrl", site_col="site",
    )
    assert arrays["n_dropped"] == 1
    keep = list(range(n))
    keep.remove(4)
    expected = [k for k in keep if pheno_two.loc[k, "group"] == "case"] + [
        k for k in keep if pheno_two.loc[k, "group"] == "ctrl"
    ]
    spec = _motion_qc_spec_from_arrays(
        arrays, test_type="two-sample", question_choice="Group Difference",
        fd_col="Motion_FD", subject_col="subject_id", group_col="group",
        ref_group="case", target_group="ctrl",
    )
    np.testing.assert_allclose(spec["fd"], fd[expected])
    assert spec["condition_labels"] == (
        ["case"] * len(arrays["group1"]) + ["ctrl"] * len(arrays["group2"])
    )
    assert spec["index_values"] == expected
    assert spec["subject_ids"] == [f"s{k:02d}" for k in expected]
    preview = _motion_qc_preview(spec, "two-sample", None, arrays["group1"], arrays["group2"],
                                 scale="r")
    assert preview["subjects"]["fd"] == [
        pytest.approx(v, abs=1e-12) for v in fd[expected]
    ]
    assert preview["subjects"]["condition"] == spec["condition_labels"]

    # GLM group difference: only the two selected levels, labels from the group column.
    pheno_glm = pheno.copy()
    pheno_glm.loc[4, "Motion_FD"] = np.nan  # dropped because FD is a confound here
    glm_arrays = _build_design_arrays(
        Y, pheno_glm, test_type="glm", question_choice="Group Difference",
        group_col="group", ref_group="case", target_group="ctrl",
        confound_vars=["Motion_FD"],
    )
    glm_spec = _motion_qc_spec_from_arrays(
        glm_arrays, test_type="glm", question_choice="Group Difference",
        fd_col="Motion_FD", subject_col="subject_id", group_col="group",
        ref_group="case", target_group="ctrl",
    )
    assert len(glm_spec["fd"]) == int((pheno_glm["group"].isin(["case", "ctrl"])).sum()) - 1
    assert set(glm_spec["condition_labels"]) <= {"case", "ctrl"}
    assert len(glm_spec["condition_labels"]) == len(glm_spec["fd"])

    # Paired: condition A defines the order, FD follows each condition's rows.
    paired_arrays = _build_design_arrays(
        Y, pheno, test_type="paired", question_choice="Paired Condition",
        condition_col="condition", subject_col="subject_id",
        baseline_val="rest", target_val="task",
    )
    paired_spec = _motion_qc_spec_from_arrays(
        paired_arrays, test_type="paired", question_choice="Paired Condition",
        fd_col="Motion_FD", subject_col="subject_id",
        baseline_val="rest", target_val="task",
    )
    assert paired_spec["condition_a"] == "rest" and paired_spec["condition_b"] == "task"
    fd_rest = pheno["Motion_FD"].to_numpy()[np.asarray(paired_arrays["idx_baseline"], dtype=int)]
    fd_task = pheno["Motion_FD"].to_numpy()[np.asarray(paired_arrays["idx_target"], dtype=int)]
    np.testing.assert_allclose(paired_spec["fd_group1"], fd_rest, equal_nan=True)
    np.testing.assert_allclose(paired_spec["fd_group2"], fd_task, equal_nan=True)
    assert paired_spec["subject_ids"] == paired_arrays["paired_subjects"]
    paired_preview = _motion_qc_preview(
        paired_spec, "paired", None, paired_arrays["group1"], paired_arrays["group2"], scale="r")
    assert paired_preview["conditions"]["rest"]["subjects"]["subject_ids"] == paired_spec["subject_ids"]
    np.testing.assert_allclose(
        paired_preview["conditions"]["task"]["subjects"]["fd"], fd_task, equal_nan=True)
