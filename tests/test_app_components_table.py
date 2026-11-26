"""The NBS component table in the Streamlit results step; fixed small results, no UI inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conninfpy import compute_p_val
from conninfpy.atlas import AtlasInfo

APP = Path(__file__).resolve().parents[1] / "conninfpy" / "app" / "streamlit_app.py"
N = 8
THRESHOLD = 3.5


def _groups():
    rng = np.random.default_rng(11)

    def sym(a):
        a = (a + a.transpose(0, 2, 1)) / 2
        a[:, np.arange(N), np.arange(N)] = 0.0
        return a

    g1, g2 = sym(rng.normal(size=(12, N, N))), sym(rng.normal(size=(12, N, N)))
    for i in range(4):                      # group 2 > group 1 on a 4-node clique
        for j in range(4):
            if i != j:
                g2[:, i, j] += 2.0
    return g1, g2


def _result(method="nbs", threshold=THRESHOLD):
    g1, g2 = _groups()
    kwargs = {"threshold": threshold, "nbs_stat": "extent"} if method == "nbs" else {}
    return compute_p_val(g1, g2, test_type="two-sample", var_equal=True, method=method,
                         n_permutations=60, rng=3, use_mp=False, acceleration=None,
                         two_tailed=False, **kwargs)


ATLAS = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * 2 + ["DMN"] * 2 + ["SMN"] * 4)


def _results_page(result, atlas=ATLAS):
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((24, N, N))
    app.session_state.pheno_df = pd.DataFrame({"subject_id": list("abcd"), "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = result.significant_edges(atlas=atlas)
    app.session_state.run_plan = {
        "method": result.method, "alpha": 0.05, "loaded_settings_hash": "fixture data",
        "active_atlas_signature": "fixture atlas", "question_type": "Group Difference",
        "ref_group": 0, "target_group": 1,
    }
    app.session_state.active_tab = "3. Inference Results"
    return app.run()


def test_nbs_result_shows_the_component_table_and_ids_in_the_edge_table():
    result = _result()
    app = _results_page(result)
    assert not app.exception
    assert any("NBS Components" in m.value for m in app.markdown)
    components = next(df.value for df in app.dataframe if "p (FWER)" in df.value.columns)
    expected = result.component_table(ATLAS)
    assert len(components) == len(expected) >= 1
    assert components["Edges"].tolist() == expected["n_edges"].tolist()
    assert components["ROIs"].tolist() == expected["n_nodes"].tolist()
    np.testing.assert_allclose(components["p (FWER)"], expected["p_fwer"])
    assert components["Component"].tolist() == expected["component_id"].tolist()
    assert "Statistic (extent)" in components.columns
    assert components["ROI names"].iloc[0] == "; ".join(f"ROI{i}" for i in range(4))
    assert "Networks (ROIs)" in components.columns and "Network pairs (edges)" in components.columns
    # the edge table shows the same component ids
    edges = next(df.value for df in app.dataframe if "component_id" in df.value.columns)
    assert set(edges["component_id"].dropna()) == set(expected["component_id"][expected["significant"]])
    # a download of the component table is offered
    labels = [b.proto.label for b in app.get("download_button")]
    assert any("Components CSV" in label for label in labels)


def test_results_without_atlas_names_use_roi_indices():
    result = _result()
    app = _results_page(result, atlas=AtlasInfo(labels=[f"r{i}" for i in range(N)], networks=[""] * N))
    assert not app.exception
    components = next(df.value for df in app.dataframe if "p (FWER)" in df.value.columns)
    assert "Networks (ROIs)" not in components.columns


def test_no_component_above_the_threshold_says_so():
    app = _results_page(_result(threshold=50.0))
    assert not app.exception
    assert any("no component to test" in i.value for i in app.info)
    assert not any("p (FWER)" in df.value.columns for df in app.dataframe)


def test_other_methods_have_no_component_table():
    app = _results_page(_result(method="tstat"))
    assert not app.exception
    assert not any("NBS Components" in m.value for m in app.markdown)
    assert not any("p (FWER)" in df.value.columns for df in app.dataframe)


def test_one_sided_result_lists_the_tested_direction_only():
    result = _result().one_sided("positive")
    app = _results_page(result)
    assert not app.exception
    components = next(df.value for df in app.dataframe if "p (FWER)" in df.value.columns)
    assert len(components) == len(result.component_table())
