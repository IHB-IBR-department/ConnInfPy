"""Tests for NETSTAT-22: design_diagnostics wiring, accurate flags, UI summaries, and provenance round-trip."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conninfpy import analyze
from conninfpy.cli import run as cli_run
from conninfpy.harmonize.diagnostics import design_diagnostics
from conninfpy.post_analysis import ResultBundle
from conninfpy.stats.glm import compute_p_val_glm_multi
from conninfpy.stats.pairwise import compute_p_val


def _make_corr_stack(rng: np.random.Generator, n_sub: int, n_roi: int = 5) -> np.ndarray:
    out = np.empty((n_sub, n_roi, n_roi), dtype=np.float64)
    for s in range(n_sub):
        a = rng.standard_normal((n_roi, 30))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        out[s] = c
    return out


def test_ac1_design_diagnostics_populated_across_all_paths():
    rng = np.random.default_rng(2026)
    Y = _make_corr_stack(rng, 12, 5)
    sites = np.array(["S1"] * 6 + ["S2"] * 6)
    group = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float)
    age = np.linspace(20.0, 50.0, 12) + rng.normal(0.0, 0.5, 12)

    # 1. GLM with binary interest + continuous confound + sites
    res_glm = analyze(
        Y,
        interest=group,
        confounds=age,
        sites=sites,
        harmonize="combat_site_dummies_glm",
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    dd = res_glm.design_diagnostics
    assert isinstance(dd, dict)
    assert dd["design_kind"] == "glm"
    assert dd["unit_of_observation"] == "subject"
    assert dd["n_observations"] == 12
    assert dd["rank_deficient"] is False
    assert dd["contrast_estimable"] is True
    assert "confound_1" in dd["continuous_factors"]
    assert "interest" in dd["categorical_factors"]
    assert dd["categorical_factors"]["interest"]["level_counts"] == {"0": 6, "1": 6}
    assert dd["site_summary"]["n_sites"] == 2
    assert dd["site_summary"]["site_by_interest_table"] == {
        "S1": {"0": 3, "1": 3},
        "S2": {"0": 3, "1": 3},
    }
    assert dd["vif"]["intercept"] == 1.0
    assert dd["scope"] == "global_complete_design"

    # 2. GLM with continuous interest + sites (site × continuous predictor stats)
    res_cont = analyze(
        Y,
        interest=age,
        confounds=group,
        sites=sites,
        harmonize="site_dummies_glm",
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    dd_c = res_cont.design_diagnostics
    assert dd_c is not None
    assert "interest" in dd_c["continuous_factors"]
    assert dd_c["continuous_factors"]["interest"]["n"] == 12
    assert "site_by_interest_stats" in dd_c["site_summary"]
    assert dd_c["site_summary"]["descriptive_r2_site_on_interest"] is not None

    # 3. Multi-contrast GLM
    res_multi = analyze(
        Y,
        interest={"age": age, "group": group},
        sites=sites,
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    assert set(res_multi.keys()) == {"age", "group"}
    for nm, entry in res_multi.items():
        assert entry.design_diagnostics is not None
        assert entry.design_diagnostics["design_kind"] == "glm_multi"
        assert entry.design_diagnostics["focus_contrast"] == nm
        assert entry.design_diagnostics["contrast_estimable"][nm] is True

    # 4. Two-sample, paired, one-sample, paired_glm, paired_interest
    g1 = _make_corr_stack(rng, 6, 5)
    g2 = _make_corr_stack(rng, 6, 5)
    sites_ts = ["S1"] * 3 + ["S2"] * 3 + ["S1"] * 3 + ["S2"] * 3
    res_ts = analyze(
        group1=g1,
        group2=g2,
        test_type="two-sample",
        sites=sites_ts,
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    assert res_ts.design_diagnostics["design_kind"] == "two-sample"
    assert res_ts.design_diagnostics["sample_sizes"]["groups"] == {"group1": 6, "group2": 6}
    assert res_ts.design_diagnostics["site_summary"]["site_by_interest_table"] == {
        "S1": {"group1": 3, "group2": 3},
        "S2": {"group1": 3, "group2": 3},
    }

    res_paired = analyze(
        group1=g1,
        group2=g2,
        test_type="paired",
        sites=["S1", "S1", "S1", "S2", "S2", "S2"],
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    assert res_paired.design_diagnostics["design_kind"] == "paired"
    assert res_paired.design_diagnostics["unit_of_observation"] == "subject_pair_difference"

    res_one = analyze(
        group1=g1,
        test_type="one-sample",
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    assert res_one.design_diagnostics["design_kind"] == "one-sample"

    c1 = rng.standard_normal(6)
    c2 = rng.standard_normal(6)
    res_pglm = analyze(
        group1=g1,
        group2=g2,
        confounds_group1=c1,
        confounds_group2=c2,
        test_type="paired",
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    assert res_pglm.design_diagnostics["design_kind"] == "paired_glm"

    res_pint = analyze(
        group1=g1,
        group2=g2,
        interest=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        test_type="paired",
        sites=["S1", "S1", "S1", "S2", "S2", "S2"],
        method="tstat",
        n_permutations=19,
        rng=1,
        use_mp=False,
    )
    assert res_pint.design_diagnostics["design_kind"] == "paired_interest"


def test_ac1_rank_deficiency_and_raw_vs_standardized_cond():
    rng = np.random.default_rng(42)
    n = 16
    x1 = rng.standard_normal(n)
    x_uncentered = 10000.0 + x1
    X_uncentered = np.column_stack([np.ones(n), x_uncentered])
    diag_unc = design_diagnostics(X_uncentered, names=["intercept", "big_x"])
    assert diag_unc["rank_deficient"] is False
    assert diag_unc["condition_number"] > 1000.0
    assert diag_unc["condition_number_standardized"] < 5.0
    assert "unit" in diag_unc["condition_number_note"].lower()
    assert "intercept" in diag_unc["vif_convention"].lower()

    # Exact rank deficiency in nuisance block while interest contrast remains estimable
    Y = _make_corr_stack(rng, n, 5)
    interest = rng.standard_normal(n)
    c_dup = np.column_stack([x1, 2.0 * x1])
    res_rank = analyze(
        Y,
        interest=interest,
        confounds=c_dup,
        method="tstat",
        n_permutations=19,
        rng=2,
        use_mp=False,
    )
    dd = res_rank.design_diagnostics
    assert dd["rank_deficient"] is True
    assert dd["matrix_rank"] == 3
    assert dd["n_columns"] == 4
    assert dd["nullity"] == 1
    assert dd["contrast_estimable"] is True
    assert dd["condition_number_XtX"] is None or dd["condition_number_XtX"] > 1e8
    assert any("Rank-deficient" in fl for fl in dd["flags"])
    # Strict RFC-8259 JSON compatibility (no NaN/Infinity tokens)
    raw_json = json.dumps(dd, allow_nan=False)
    assert "Rank-deficient" in raw_json


def test_ac3_accurate_flags_for_auto_without_confounds_and_paired_strata():
    rng = np.random.default_rng(7)
    Y = _make_corr_stack(rng, 10, 5)
    sites = np.array(["A"] * 5 + ["B"] * 5)
    interest = rng.standard_normal(10)

    # harmonize="auto" without confounds -> site_dummies_glm explaining why ComBat skipped
    res_auto = analyze(
        Y,
        interest=interest,
        sites=sites,
        harmonize="auto",
        method="tstat",
        n_permutations=19,
        rng=3,
        use_mp=False,
    )
    joined_auto = "\n".join(res_auto.flags)
    assert "site_dummies_glm" in joined_auto
    assert "without confounds=" in joined_auto
    assert "preserve" in joined_auto

    # paired sign-flip + sites -> stratum-invariant by construction, no site × condition claim
    g1 = _make_corr_stack(rng, 6, 5)
    g2 = _make_corr_stack(rng, 6, 5)
    res_paired = analyze(
        group1=g1,
        group2=g2,
        test_type="paired",
        sites=["A", "A", "A", "B", "B", "B"],
        method="tstat",
        n_permutations=19,
        rng=3,
        use_mp=False,
    )
    joined_paired = "\n".join(res_paired.flags)
    assert "stratum-invariant by construction" in joined_paired
    assert "site × condition interactions are not automatically controlled" in joined_paired
    assert "sites= still stratifies" not in joined_paired

    # paired_interest + sites -> stratifies between-subject permutations + site × condition caveat
    res_pint = analyze(
        group1=g1,
        group2=g2,
        interest=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        test_type="paired",
        sites=["A", "A", "A", "B", "B", "B"],
        method="tstat",
        n_permutations=19,
        rng=3,
        use_mp=False,
    )
    joined_pint = "\n".join(res_pint.flags)
    assert "stratifies between-subject permutations" in joined_pint
    assert "site × condition interactions are not automatically controlled" in joined_pint


def test_ac4_bundle_cli_roundtrip_and_zero_numerical_drift(tmp_path: Path):
    rng = np.random.default_rng(11)
    Y = _make_corr_stack(rng, 10, 5)
    # Inject one NaN edge for available_case testing
    Y_ac = Y.copy()
    Y_ac[0, 1, 2] = Y_ac[0, 2, 1] = np.nan
    sites = ["S1"] * 5 + ["S2"] * 5
    age = np.linspace(20.0, 40.0, 10)
    dx = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

    # 1. available_case GLM diagnostics & numerical invariance vs compute_p_val_glm_multi
    multi_out = analyze(
        Y_ac,
        interest={"dx": dx, "age": age},
        input="r",
        r_to_z=False,
        method="tstat",
        n_permutations=19,
        rng=99,
        use_mp=False,
        missing="available_case",
        min_coverage=0.8,
        two_tailed=False,
    )
    dd_ac = multi_out["dx"].design_diagnostics
    assert dd_ac["scope"] == "global_design_with_available_case_edge_patterns"
    assert dd_ac["available_case"]["n_edges_tested"] == 10
    assert dd_ac["available_case"]["edge_n_min"] == 9
    assert dd_ac["available_case"]["edge_n_max"] == 10

    Y_prep = Y_ac.copy()
    for s in range(Y_prep.shape[0]):
        np.fill_diagonal(Y_prep[s], 0.0)
    X_raw = np.column_stack([np.ones(10), dx, age])
    c_raw = {"dx": np.array([0.0, 1.0, 0.0]), "age": np.array([0.0, 0.0, 1.0])}
    low_multi = compute_p_val_glm_multi(
        Y_prep,
        design_matrix=X_raw,
        contrasts=c_raw,
        method="tstat",
        n_permutations=19,
        rng=99,
        use_mp=False,
        missing="available_case",
        min_coverage=0.8,
        two_tailed=False,
    )
    for k in ("dx", "age"):
        np.testing.assert_array_equal(multi_out[k].positive, low_multi[k].positive)
        np.testing.assert_array_equal(multi_out[k].negative, low_multi[k].negative)
        np.testing.assert_array_equal(multi_out[k].inference.stat_signed, low_multi[k].stat_signed)
        np.testing.assert_array_equal(multi_out[k].inference.effect_map(), low_multi[k].effect_map())

    # 2. paired numerical invariance vs compute_p_val
    g1 = _make_corr_stack(rng, 6, 5)
    g2 = _make_corr_stack(rng, 6, 5)
    g1_prep = g1.copy()
    g2_prep = g2.copy()
    for s in range(6):
        np.fill_diagonal(g1_prep[s], 0.0)
        np.fill_diagonal(g2_prep[s], 0.0)
    res_p_an = analyze(
        group1=g1,
        group2=g2,
        test_type="paired",
        input="r",
        r_to_z=False,
        sites=["S1", "S1", "S1", "S2", "S2", "S2"],
        method="tstat",
        n_permutations=19,
        rng=77,
        use_mp=False,
        two_tailed=False,
    )
    res_p_low = compute_p_val(
        g1_prep,
        g2_prep,
        test_type="paired",
        strata=np.array(["S1", "S1", "S1", "S2", "S2", "S2"]),
        method="tstat",
        n_permutations=19,
        rng=77,
        use_mp=False,
        two_tailed=False,
    )
    np.testing.assert_array_equal(res_p_an.positive, res_p_low.positive)
    np.testing.assert_array_equal(res_p_an.negative, res_p_low.negative)
    np.testing.assert_array_equal(res_p_an.inference.stat_signed, res_p_low.stat_signed)
    np.testing.assert_array_equal(res_p_an.inference.effect_map(), res_p_low.effect_map())

    # 3. ResultBundle + pickle + deepcopy round-trip
    res_site = analyze(
        Y,
        interest=dx,
        confounds=age,
        sites=sites,
        harmonize="combat_site_dummies_glm",
        method="tstat",
        n_permutations=19,
        rng=5,
        use_mp=False,
    )
    bundle = res_site.to_bundle()
    assert bundle.metadata["design_diagnostics"] == res_site.design_diagnostics
    assert bundle.metadata["combat_diagnostics"] == res_site.combat_diagnostics
    assert bundle.metadata["flags"] == res_site.flags

    bundle_path = tmp_path / "res.bundle.pkl"
    bundle.save(bundle_path)
    loaded_bundle = ResultBundle.load(bundle_path)
    assert loaded_bundle.metadata["design_diagnostics"] == res_site.design_diagnostics
    assert loaded_bundle.metadata["combat_diagnostics"] == res_site.combat_diagnostics
    assert loaded_bundle.metadata["flags"] == res_site.flags

    unpickled = pickle.loads(pickle.dumps(res_site))
    copied = copy.deepcopy(res_site)
    assert unpickled.design_diagnostics == res_site.design_diagnostics
    assert copied.design_diagnostics == res_site.design_diagnostics
    assert unpickled.inference.design_diagnostics == res_site.design_diagnostics

    # 4. CLI run -> provenance.json + bundle.pkl
    import yaml

    np.save(tmp_path / "conn.npy", Y)
    pd.DataFrame({"dx": dx, "age": age, "site": sites}).to_csv(tmp_path / "pheno.csv", index=False)
    (tmp_path / "study.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "loader": "NumpyLoader",
                "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
            }
        ),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "analysis.yaml"
    out_dir = tmp_path / "cli_out"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": "study.yaml",
                "design": {
                    "type": "glm",
                    "interest": "dx",
                    "confounds": ["age"],
                    "sites": "site",
                },
                "inference": {
                    "method": "tstat",
                    "n_permutations": 19,
                    "use_mp": False,
                    "r_to_z": False,
                },
                "output": "cli_out",
            }
        ),
        encoding="utf-8",
    )
    paths = cli_run(cfg_path, out_dir=out_dir)

    def _reject_nonfinite(tok):
        raise ValueError(f"Non-RFC8259 JSON token: {tok}")

    prov = json.loads(paths["provenance"].read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    per_res = prov["results"]["result"]
    assert per_res["design_diagnostics"] is not None
    assert per_res["combat_diagnostics"] is not None
    assert len(per_res["flags"]) > 0
    cli_bundle = ResultBundle.load(paths["bundle"])
    assert cli_bundle.metadata["design_diagnostics"] == per_res["design_diagnostics"]
    assert cli_bundle.metadata["flags"] == per_res["flags"]


def test_ac2_ui_prelaunch_and_postlaunch_helpers():
    from apps.components.design_inference import (
        _build_prelaunch_design_qc_summary,
        _render_prelaunch_design_qc_summary,
    )
    from apps.components.inference_results import (
        _extract_result_diagnostics_summary,
        _render_analyze_decisions_and_diagnostics,
    )

    rng = np.random.default_rng(123)
    Y = _make_corr_stack(rng, 8, 4)
    Y[0, 1, 2] = Y[0, 2, 1] = np.nan

    # Group Difference pre-launch summary (counts, site × group, QC, rank-deficient preview)
    pheno = pd.DataFrame(
        {
            "group": ["CTRL", "CTRL", "CASE", "CASE", "CTRL", "CTRL", "CASE", "CASE"],
            "site": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
            "age": [20.0, 22.0, 21.0, 23.0, 30.0, 31.0, np.nan, 32.0],
            "age_dup": [40.0, 44.0, 42.0, 46.0, 60.0, 62.0, 60.0, 64.0],
            "n_valid_tr": [120, 45, 110, 130, 140, 150, 160, 170],
            "Motion_FD": [0.1, 0.2, 0.15, 0.12, 0.11, 0.09, 0.14, 0.13],
        }
    )
    pre_grp = _build_prelaunch_design_qc_summary(
        pheno,
        Y,
        question_choice="Group Difference",
        test_type="glm",
        group_col="group",
        ref_group="CTRL",
        target_group="CASE",
        confound_vars=["age", "age_dup"],
        site_col="site",
        harmonization_choice="combat_site_dummies_glm",
    )
    assert pre_grp["n_dropped_na"] == 1
    assert pre_grp["sample_counts"]["Reference (CTRL)"] == 4
    assert pre_grp["sample_counts"]["Target (CASE)"] == 3
    assert pre_grp["site_summary_kind"] == "site_by_group"
    assert pre_grp["connectivity_qc"]["subjects_with_nan"] == 1
    assert any(r["Metric"] == "n_valid_tr" for r in pre_grp["qc_rows"])
    assert any(r["Metric"] == "Motion_FD" for r in pre_grp["qc_rows"])
    assert pre_grp["design_preview"]["rank_deficient"] is True

    # Continuous Predictor pre-launch summary
    pre_cont = _build_prelaunch_design_qc_summary(
        pheno.dropna(),
        Y[:7],
        question_choice="Continuous Predictor",
        test_type="glm",
        interest_var="age",
        confound_vars=[],
        site_col="site",
        harmonization_choice="site_dummies_glm",
    )
    assert pre_cont["interest_stats"]["predictor"] == "age"
    assert pre_cont["site_summary_kind"] == "site_by_interest"
    assert any("R²" in note for note in pre_cont["site_notes"])

    # Paired Condition pre-launch summary with site mismatch across conditions
    pheno_paired = pd.DataFrame(
        {
            "subject": ["sub1", "sub1", "sub2", "sub2", "sub3"],
            "cond": ["pre", "post", "pre", "post", "pre"],
            "site": ["S1", "S1", "S1", "S2", "S1"],
        }
    )
    pre_pair = _build_prelaunch_design_qc_summary(
        pheno_paired,
        _make_corr_stack(rng, 5, 4),
        question_choice="Paired Condition",
        test_type="paired",
        subject_col="subject",
        condition_col="cond",
        baseline_val="pre",
        target_val="post",
        site_col="site",
    )
    assert pre_pair["sample_counts"]["Paired Subjects"] == 2
    assert pre_pair["sample_counts"]["Unpaired Excluded"] == 1
    assert any("Site mismatch" in w for w in pre_pair["warnings"])

    # Post-launch diagnostics extractor on AnalyzeResult and legacy InferenceResult
    res_an = analyze(
        _make_corr_stack(rng, 8, 4),
        interest=np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float),
        confounds=np.linspace(10.0, 20.0, 8),
        sites=["S1"] * 4 + ["S2"] * 4,
        harmonize="combat_site_dummies_glm",
        method="tstat",
        n_permutations=19,
        rng=12,
        use_mp=False,
    )
    extracted = _extract_result_diagnostics_summary(
        res_an,
        run_plan={"effective_site_recipe": "ComBat + site-aware GLM", "harmonize": "combat_site_dummies_glm"},
    )
    assert extracted["resolved_strategy"] == "combat_site_dummies_glm"
    assert extracted["combat_diagnostics"] is not None
    assert extracted["design_diagnostics"] is not None
    assert len(extracted["flags"]) > 0

    legacy_obj = SimpleNamespace(positive=np.ones((4, 4)), negative=np.ones((4, 4)), harmonized=False)
    extracted_legacy = _extract_result_diagnostics_summary(
        legacy_obj,
        run_plan={"harmonize": "auto", "effective_site_recipe": "ComBat + site-aware GLM"},
    )
    assert extracted_legacy["resolved_strategy"] == "unknown / not recorded"
    assert extracted_legacy["harmonize_requested"] == "auto"
    assert extracted_legacy["flags"] == []

    # Smoke-render both UI blocks against a lightweight Streamlit mock
    import apps.components.design_inference as di_mod
    import apps.components.inference_results as ir_mod

    class _DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _DummyCol:
        def __init__(self, logs):
            self.logs = logs

        def metric(self, label, value):
            self.logs.append(("metric", label, value))

    class _DummySt:
        def __init__(self):
            self.logs = []

        def container(self, **kwargs):
            return _DummyCtx()

        def expander(self, title, **kwargs):
            self.logs.append(("expander", title))
            return _DummyCtx()

        def columns(self, n):
            return [_DummyCol(self.logs) for _ in range(n)]

        def markdown(self, text, **kwargs):
            self.logs.append(("markdown", text))

        def caption(self, text):
            self.logs.append(("caption", text))

        def dataframe(self, df, **kwargs):
            self.logs.append(("dataframe", len(df)))

        def warning(self, text):
            self.logs.append(("warning", text))

        def info(self, text):
            self.logs.append(("info", text))

    dummy_st = _DummySt()
    orig_di_st, orig_ir_st = di_mod.st, ir_mod.st
    try:
        di_mod.st = dummy_st
        ir_mod.st = dummy_st
        _render_prelaunch_design_qc_summary(pre_grp)
        _render_analyze_decisions_and_diagnostics(
            res_an,
            run_plan={"effective_site_recipe": "ComBat + site-aware GLM", "harmonize": "combat_site_dummies_glm"},
            method_name="TSTAT",
        )
    finally:
        di_mod.st = orig_di_st
        ir_mod.st = orig_ir_st

    rendered_text = " ".join(str(x) for x in dummy_st.logs)
    assert "combat_site_dummies_glm" in rendered_text
    assert "Raw cond(X'X)" in rendered_text
    assert "Rank-deficient" in rendered_text


def test_welch_vs_student_and_available_case_df_distinction_with_manual_oracle():
    """Welch two-sample and available_case must never falsely claim shared n-rank inference df."""
    from apps.components.inference_results import (
        _format_inference_df_display,
        _render_analyze_decisions_and_diagnostics,
    )

    rng = np.random.default_rng(8022)
    n1, n2, n_roi = 8, 8, 4
    g1 = np.zeros((n1, n_roi, n_roi), dtype=np.float64)
    g2 = np.zeros((n2, n_roi, n_roi), dtype=np.float64)

    # Edge (0, 1): sharply unequal variances -> Welch-Satterthwaite df ~ 7 (far below n - 2 = 14)
    g1_01 = rng.normal(0.0, 0.01, size=n1)
    g2_01 = rng.normal(1.2, 5.0, size=n2)
    g1[:, 0, 1] = g1[:, 1, 0] = g1_01
    g2[:, 0, 1] = g2[:, 1, 0] = g2_01

    # Edge (0, 2): nearly equal variances -> Welch-Satterthwaite df ~ 14
    g1_02 = rng.normal(0.0, 1.0, size=n1)
    g2_02 = rng.normal(0.5, 1.0, size=n2)
    g1[:, 0, 2] = g1[:, 2, 0] = g1_02
    g2[:, 0, 2] = g2[:, 2, 0] = g2_02

    # Fill remaining off-diagonal edges cleanly
    for u, v in [(0, 3), (1, 2), (1, 3), (2, 3)]:
        x1 = rng.normal(0.0, 0.3, size=n1)
        x2 = rng.normal(0.0, 0.3, size=n2)
        g1[:, u, v] = g1[:, v, u] = x1
        g2[:, u, v] = g2[:, v, u] = x2

    def _welch_df(a, b):
        va = float(np.var(a, ddof=1))
        vb = float(np.var(b, ddof=1))
        na, nb = len(a), len(b)
        return ((va / na + vb / nb) ** 2) / (
            ((va / na) ** 2) / (na - 1) + ((vb / nb) ** 2) / (nb - 1)
        )

    df_01 = _welch_df(g1_01, g2_01)
    df_02 = _welch_df(g1_02, g2_02)
    assert df_01 < 7.5  # sharply lower than n - 2 = 14
    assert df_02 > 12.5
    assert abs(df_02 - df_01) > 5.0

    # Manual Welch t-statistic oracle (positive = group2 > group1)
    t_01_oracle = (np.mean(g2_01) - np.mean(g1_01)) / np.sqrt(
        np.var(g1_01, ddof=1) / n1 + np.var(g2_01, ddof=1) / n2
    )

    res_welch = analyze(
        group1=g1,
        group2=g2,
        input="z",
        var_equal=False,
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
    )
    assert np.isclose(res_welch.inference.stat_signed[0, 1], t_01_oracle, atol=1e-12)
    dd_w = res_welch.design_diagnostics
    assert dd_w["scope"] == "global_complete_design"
    assert dd_w["residual_df"] == 14
    assert dd_w["design_residual_df"] == 14
    assert dd_w["inference_df"] is None
    assert dd_w["inference_df_kind"] == "varying_welch_satterthwaite"
    assert "share these observations and residual degrees of freedom" not in dd_w["scope_note"]
    assert "Welch-Satterthwaite" in dd_w["scope_note"]
    assert "varying per-edge Welch-Satterthwaite" in dd_w["inference_df_note"]
    assert _format_inference_df_display(dd_w) == "Per-edge (Welch)"
    json.loads(json.dumps(dd_w, allow_nan=False))

    # Student pooled-variance two-sample DOES share n - rank = 14 across edges
    res_student = analyze(
        group1=g1,
        group2=g2,
        input="z",
        var_equal=True,
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
    )
    dd_s = res_student.design_diagnostics
    assert dd_s["design_residual_df"] == 14
    assert dd_s["inference_df"] == 14
    assert dd_s["inference_df_kind"] == "shared_design_residual_df"
    assert _format_inference_df_display(dd_s) == "14"

    # Available-case Welch and available-case GLM must NOT substitute global n - rank
    g1_miss = g1.copy()
    g1_miss[0, 1, 2] = np.nan
    g1_miss[0, 2, 1] = np.nan
    res_ac_welch = analyze(
        group1=g1_miss,
        group2=g2,
        input="z",
        var_equal=False,
        missing="available_case",
        min_coverage=0.5,
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
    )
    dd_ac_w = res_ac_welch.design_diagnostics
    assert dd_ac_w["design_residual_df"] == 14
    assert dd_ac_w["inference_df"] is None
    assert dd_ac_w["inference_df_kind"] == "available_case_welch_satterthwaite"
    assert _format_inference_df_display(dd_ac_w) == "Per-edge (Welch)"

    Y_glm = np.concatenate([g1_miss, g2], axis=0)
    interest_glm = np.array([0.0] * n1 + [1.0] * n2)
    res_ac_glm = analyze(
        Y_glm,
        interest=interest_glm,
        input="z",
        missing="available_case",
        min_coverage=0.5,
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
    )
    dd_ac_g = res_ac_glm.design_diagnostics
    assert dd_ac_g["design_residual_df"] == 14
    assert dd_ac_g["inference_df"] is None
    assert dd_ac_g["inference_df_kind"] == "available_case_per_edge"
    assert dd_ac_g["available_case"]["edge_residual_df_min"] < 14
    assert _format_inference_df_display(dd_ac_g) != "14"

    # UI rendering check for Welch vs Design df
    import apps.components.inference_results as ir_mod

    class _DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _DummyCol:
        def __init__(self, logs):
            self.logs = logs

        def metric(self, label, value):
            self.logs.append(("metric", label, value))

    class _DummySt:
        def __init__(self):
            self.logs = []

        def expander(self, title, **kwargs):
            return _DummyCtx()

        def columns(self, n):
            return [_DummyCol(self.logs) for _ in range(n)]

        def markdown(self, text, **kwargs):
            self.logs.append(("markdown", text))

        def caption(self, text):
            self.logs.append(("caption", text))

        def dataframe(self, df, **kwargs):
            pass

        def warning(self, text):
            pass

        def info(self, text):
            pass

    dummy_st = _DummySt()
    orig_st = ir_mod.st
    try:
        ir_mod.st = dummy_st
        _render_analyze_decisions_and_diagnostics(res_welch, run_plan={"harmonize": "auto"})
    finally:
        ir_mod.st = orig_st

    metrics = {item[1]: item[2] for item in dummy_st.logs if item[0] == "metric"}
    assert metrics["Design df (n − rank)"] == 14
    assert metrics["Inference df"] == "Per-edge (Welch)"
    assert metrics["Resolved Strategy"] == "none"
    assert metrics["Requested Harmonize"] == "auto"


def test_resolved_strategy_vs_requested_harmonize_across_backend_bundle_cli_and_ui(tmp_path):
    """Resolved strategy comes from backend truth, never guessed from requested run_plan."""
    from apps.components.inference_results import _extract_result_diagnostics_summary
    from conninfpy.cli import run as cli_run
    from conninfpy.post_analysis import ResultBundle

    rng = np.random.default_rng(8023)
    Y = _make_corr_stack(rng, 10, 4)
    sites = ["S1"] * 5 + ["S2"] * 5
    interest = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], dtype=float)
    conf = np.linspace(20.0, 40.0, 10)
    roi_ids = [f"ROI_{i}" for i in range(4)]

    # 1. no sites / harmonize='auto' (two-sample) -> resolved 'none' even with plan={'harmonize': 'auto'}
    r_no_sites = analyze(
        group1=Y[:5],
        group2=Y[5:],
        input="r",
        r_to_z=False,
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
        roi_ids=roi_ids,
    )
    assert r_no_sites.resolved_strategy == "none"
    assert r_no_sites.inference.resolved_strategy == "none"
    assert r_no_sites.inference.harmonized is False
    assert r_no_sites.combat_diagnostics is None
    ex_no_sites = _extract_result_diagnostics_summary(r_no_sites, run_plan={"harmonize": "auto"})
    assert ex_no_sites["resolved_strategy"] == "none"
    assert ex_no_sites["harmonize_requested"] == "auto"
    # Bare inner InferenceResult (TailResult dict subclass) must also read attributes cleanly
    ex_no_sites_inner = _extract_result_diagnostics_summary(r_no_sites.inference, run_plan={"harmonize": "auto"})
    assert ex_no_sites_inner["resolved_strategy"] == "none"
    assert ex_no_sites_inner["design_diagnostics"] is not None

    # 2. sites / no confounds / harmonize='auto' -> resolved 'site_dummies_glm'
    r_site_glm = analyze(
        Y,
        interest=interest,
        sites=sites,
        harmonize="auto",
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
        roi_ids=roi_ids,
    )
    assert r_site_glm.resolved_strategy == "site_dummies_glm"
    assert r_site_glm.inference.resolved_strategy == "site_dummies_glm"
    ex_site_glm = _extract_result_diagnostics_summary(r_site_glm, run_plan={"harmonize": "auto"})
    assert ex_site_glm["resolved_strategy"] == "site_dummies_glm"
    assert ex_site_glm["harmonize_requested"] == "auto"

    # 3. sites + confounds / harmonize='auto' -> resolved 'combat_site_dummies_glm'
    r_combat = analyze(
        Y,
        interest=interest,
        confounds=conf,
        sites=sites,
        harmonize="auto",
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
        roi_ids=roi_ids,
    )
    assert r_combat.resolved_strategy == "combat_site_dummies_glm"
    assert r_combat.inference.resolved_strategy == "combat_site_dummies_glm"
    assert r_combat.inference.harmonized is True
    ex_combat = _extract_result_diagnostics_summary(r_combat, run_plan={"harmonize": "auto"})
    assert ex_combat["resolved_strategy"] == "combat_site_dummies_glm"

    # 4. paired + sites without ComBat -> resolved 'none'
    r_paired_sites = analyze(
        group1=Y[:5],
        group2=Y[5:],
        test_type="paired",
        sites=["S1", "S1", "S2", "S2", "S1"],
        harmonize="auto",
        method="tstat",
        n_permutations=7,
        rng=9,
        use_mp=False,
        roi_ids=roi_ids,
    )
    assert r_paired_sites.resolved_strategy == "none"
    assert r_paired_sites.inference.resolved_strategy == "none"
    assert r_paired_sites.inference.harmonized is False
    ex_paired = _extract_result_diagnostics_summary(r_paired_sites, run_plan={"harmonize": "auto"})
    assert ex_paired["resolved_strategy"] == "none"
    assert ex_paired["harmonize_requested"] == "auto"

    # Bundle save/load preserves recorded resolved_strategy
    bundle_path = tmp_path / "no_sites_bundle.pkl"
    r_no_sites.to_bundle().save(bundle_path)
    loaded_bundle = ResultBundle.load(bundle_path)
    assert loaded_bundle.metadata["resolved_strategy"] == "none"
    assert _extract_result_diagnostics_summary(loaded_bundle, run_plan={"harmonize": "auto"})["resolved_strategy"] == "none"

    # 5. Legacy missing marker (on facade/inner/bundle) -> 'unknown / not recorded', never guessed
    legacy_bundle = ResultBundle.load(bundle_path)
    legacy_bundle.metadata.pop("resolved_strategy", None)
    legacy_bundle.metadata["combat_diagnostics"] = None
    assert (
        _extract_result_diagnostics_summary(
            legacy_bundle,
            run_plan={"harmonize": "auto", "effective_site_recipe": "ComBat + site-aware GLM"},
        )["resolved_strategy"]
        == "unknown / not recorded"
    )

    legacy_inf = SimpleNamespace(harmonized=False)
    assert (
        _extract_result_diagnostics_summary(
            legacy_inf,
            run_plan={"harmonize": "combat_only"},
        )["resolved_strategy"]
        == "unknown / not recorded"
    )

    # CLI provenance records resolved_strategy for no-sites/auto
    import yaml

    np.save(tmp_path / "conn.npy", Y)
    pd.DataFrame({"subject": [f"s{i}" for i in range(10)], "group": ["A"] * 5 + ["B"] * 5}).to_csv(
        tmp_path / "pheno.csv", index=False
    )
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "loader": "NumpyLoader",
                "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
            }
        ),
        encoding="utf-8",
    )
    analysis_yaml = tmp_path / "analysis.yaml"
    out_dir = tmp_path / "cli_out"
    analysis_yaml.write_text(
        yaml.safe_dump(
            {
                "data": "study.yaml",
                "design": {
                    "type": "two-sample",
                    "group": "group",
                    "levels": ["A", "B"],
                    "harmonize": "auto",
                },
                "inference": {
                    "method": "tstat",
                    "n_permutations": 7,
                    "rng": 9,
                    "use_mp": False,
                    "r_to_z": False,
                },
                "output": "cli_out",
            }
        ),
        encoding="utf-8",
    )
    written = cli_run(analysis_yaml, out_dir=out_dir)
    prov = json.loads(written["provenance"].read_text(encoding="utf-8"))
    res_entry = prov["results"]["result"]
    assert res_entry["resolved_strategy"] == "none"
    assert _extract_result_diagnostics_summary(res_entry, run_plan={"harmonize": "auto"})["resolved_strategy"] == "none"
