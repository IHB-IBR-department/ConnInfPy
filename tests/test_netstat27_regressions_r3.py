"""NETSTAT-27 round 3: launch-signature completeness (R2-1, R2-2, audit).

Scope: the GUI launch signature only. Engine/progress/cancel/CLI code is not
touched in this round (``git diff 0e8745f -- conninfpy/`` is empty).

Covered here:

* R2-1 ``alpha`` reaches the signature and invalidates a running job.
* R2-2 the phenotype **content** fingerprint covers labels/order/index/
  dtypes/missing markers, so a label swap with identical levels/counts
  invalidates a running job and the stale payload cannot publish.
* Audit: every parameter captured by ``_run_inference_background`` maps to
  signature facts (``LAUNCH_FACT_COVERAGE``), every one of those facts is
  present in a real signature, and every signature key is justified by at
  least one worker parameter.
* Structural guard: the signature describes the FINAL kwargs handed to the
  worker (network labels / tested-family mask included), captured from the
  real launch and compared with the recorded signature.

Synthetic data only; no hosted services.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conninfpy.app.components import design_inference as design
from conninfpy.app.components.design_inference import (
    LAUNCH_FACT_COVERAGE,
    InferenceTask,
    _normalize_fact,
    _phenotype_fingerprint,
    _run_inference_background,
    _run_signature,
)

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "conninfpy" / "app" / "streamlit_app.py"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _app_with_task(task):
    from tests.test_netstat27_gui import _app_with_task as base

    return base(task)


def _capture_signature(app):
    captured = []
    original = design._invalidate_superseded_task

    def observe(current, signature):
        captured.append(dict(signature))
        original(current, signature)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(design, "_invalidate_superseded_task", observe)
        app.run()
    assert captured, "render did not compute a launch signature"
    return captured[-1]


def _running_task():
    task = InferenceTask()
    task.status = "running"
    task.generation = 1
    return task


# ----------------------------------------------------------------------
# R2-1: alpha
# ----------------------------------------------------------------------
def test_alpha_is_in_the_launch_signature_and_invalidates_the_job():
    task = _running_task()
    app = _app_with_task(task)
    task.launch_signature = _capture_signature(app)
    assert "alpha" in task.launch_signature
    assert not task.cancel_event.is_set()

    control = next(w for w in app.selectbox if w.label == "Significance level (α)")
    control.set_value(0.1)
    app.run()
    assert not app.exception
    assert task.cancel_event.is_set(), "changing alpha must invalidate the running job"
    assert "changed" in (task.cancel_reason or "")
    assert app.session_state.get("inference_result") is None


# ----------------------------------------------------------------------
# R2-2: phenotype content
# ----------------------------------------------------------------------
def test_phenotype_fingerprint_covers_order_index_dtype_and_missing():
    base = pd.DataFrame(
        {"group": [0, 1, 0, 1], "motion": [0.1, 0.2, 0.3, 0.4]},
        index=["a", "b", "c", "d"],
    )
    same = base.copy()
    assert _phenotype_fingerprint(base) == _phenotype_fingerprint(same)

    swapped = base.copy()
    swapped.loc["a", "group"] = 1
    swapped.loc["b", "group"] = 0
    assert _phenotype_fingerprint(swapped) != _phenotype_fingerprint(base)
    assert base["group"].value_counts().to_dict() == swapped["group"].value_counts().to_dict()

    reordered = base.iloc[[1, 0, 2, 3]]
    assert _phenotype_fingerprint(reordered) != _phenotype_fingerprint(base)

    renamed_index = base.copy()
    renamed_index.index = ["w", "x", "y", "z"]
    assert _phenotype_fingerprint(renamed_index) != _phenotype_fingerprint(base)

    float_dtype = base.astype({"group": float})
    assert _phenotype_fingerprint(float_dtype) != _phenotype_fingerprint(base)

    with_missing = base.copy()
    with_missing.loc["c", "motion"] = np.nan
    assert _phenotype_fingerprint(with_missing) != _phenotype_fingerprint(base)

    renamed_columns = base.rename(columns={"motion": "fd"})
    assert _phenotype_fingerprint(renamed_columns) != _phenotype_fingerprint(base)
    assert _phenotype_fingerprint(None) is None


def test_phenotype_label_swap_invalidates_a_running_job():
    task = _running_task()
    app = _app_with_task(task)
    task.launch_signature = _capture_signature(app)
    previous = app.session_state.pheno_df.copy()

    changed = previous.copy()
    changed.loc[0, "group"] = previous.loc[6, "group"]
    changed.loc[6, "group"] = previous.loc[0, "group"]
    assert not previous["group"].equals(changed["group"]), "labels must actually change"
    assert previous["group"].value_counts().to_dict() == changed["group"].value_counts().to_dict()

    app.session_state.pheno_df = changed
    app.run()
    assert not app.exception
    assert task.cancel_event.is_set(), (
        "a phenotype content change with identical levels/counts must invalidate"
    )
    assert "changed" in (task.cancel_reason or "")
    assert app.session_state.get("inference_result") is None


# ----------------------------------------------------------------------
# Audit: every worker parameter is covered by the launch signature
# ----------------------------------------------------------------------
def test_worker_parameters_and_coverage_table_are_in_step():
    worker_params = set(inspect.signature(_run_inference_background).parameters)
    assert worker_params == set(LAUNCH_FACT_COVERAGE), (
        "LAUNCH_FACT_COVERAGE must list exactly the worker's captured parameters; "
        f"missing={sorted(worker_params - set(LAUNCH_FACT_COVERAGE))} "
        f"extra={sorted(set(LAUNCH_FACT_COVERAGE) - worker_params)}"
    )


def test_every_covered_fact_is_present_and_justified():
    task = _running_task()
    app = _app_with_task(task)
    signature = _capture_signature(app)
    covered = set()
    bookkeeping = {"__worker_bookkeeping__"}
    for param, facts in LAUNCH_FACT_COVERAGE.items():
        assert facts, f"{param} has no coverage entry"
        for fact in facts:
            if fact in bookkeeping:
                continue
            assert fact in signature, (
                f"worker parameter {param!r} is covered by signature fact {fact!r}, "
                "but the launch signature does not contain it"
            )
            covered.add(fact)
    # Every signature key must be justified by at least one worker parameter:
    # no fact is recorded without a captured consumer.
    unexplained = set(signature) - covered
    assert not unexplained, f"signature facts with no worker consumer: {sorted(unexplained)}"


def test_derived_facts_are_documented():
    assert "annotation_atlas" in design.DERIVED_LAUNCH_FACTS
    for key in design.DERIVED_LAUNCH_FACTS:
        assert key.split("/")[0] in LAUNCH_FACT_COVERAGE


# ----------------------------------------------------------------------
# Structural guard: the signature describes the FINAL worker kwargs
# ----------------------------------------------------------------------
def test_launch_signature_matches_the_kwargs_actually_passed_to_the_worker():
    """Real launch: signature's final kwargs == the kwargs the worker got."""
    old_task = InferenceTask()
    old_task.status = "idle"
    app = _app_with_task(old_task)
    app.session_state.inference_generation = 0
    captured_signature = []
    captured_kwargs = {}
    original_invalidate = design._invalidate_superseded_task

    def observe(current, signature):
        captured_signature.append(dict(signature))
        original_invalidate(current, signature)

    def fake_worker(**kwargs):
        captured_kwargs.update(kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(design, "_invalidate_superseded_task", observe)
        mp.setattr(design, "_run_inference_background", fake_worker)
        app.run()
        assert not app.exception
        app.button(key="run_inference_btn").click().run()
    assert not app.exception
    assert captured_kwargs, "the launch did not hand kwargs to the worker"
    assert captured_signature, "no launch signature was computed"

    signature = app.session_state.inference_task.launch_signature
    assert signature == captured_signature[-1], (
        "the recorded launch signature must be the one computed from the final "
        "worker kwargs"
    )
    assert signature["final_op_kwargs"] == _normalize_fact(captured_kwargs["op_kwargs"])
    assert signature["final_comp_kwargs"] == _normalize_fact(captured_kwargs["comp_kwargs"])
    # Independent inputs the audit called out, straight from the launch.
    assert signature["alpha"] == _normalize_fact(captured_kwargs["alpha"])
    assert signature["preprocessing"] == _normalize_fact(captured_kwargs["preprocessing"])
    assert signature["fisher_history"] == _normalize_fact(captured_kwargs["fisher_history"])
    assert signature["effective_site_recipe"] == captured_kwargs["effective_recipe_choice"]
    assert signature["use_sc_prior"] == captured_kwargs["use_sc_prior"]


# ----------------------------------------------------------------------
# Audit gaps: SC labels / upstream transforms change the signature
# ----------------------------------------------------------------------
def test_sc_prior_labels_and_upstream_transforms_are_signature_inputs():
    task = _running_task()
    app = _app_with_task(task)
    base_signature = _capture_signature(app)
    assert "sc_prior_path" in base_signature
    assert "sc_net_labels" in base_signature
    assert "preprocessing" in base_signature
    assert "fisher_history" in base_signature

    # Upstream Fisher history / preprocessing are independent launch inputs.
    app.session_state.dataset_fisher_history = [
        {"stage": "ingest", "from": "r", "to": "z", "applied": True}
    ]
    app.run()
    assert not app.exception
    task.cancel_event.clear()
    with_history = _capture_signature(app)
    task.launch_signature = base_signature
    assert with_history["fisher_history"] != base_signature["fisher_history"]
    assert with_history["fisher_history"] is not None

    app.session_state.dataset_preprocessing = {
        "kind": "preprocessing", "zero_diagonal": True, "r_to_z": True
    }
    app.run()
    assert not app.exception
    task.cancel_event.clear()
    with_preprocessing = _capture_signature(app)
    task.launch_signature = base_signature
    assert with_preprocessing["preprocessing"] != base_signature["preprocessing"]

    # An SC prior path is an independent execution input even before labels are
    # derived from it (invalid file → blocking error, but the fact is captured).
    app.session_state.dataset_preprocessing = None
    app.session_state.dataset_fisher_history = None
    app.run()
    sc_widget = next(
        (w for w in app.text_input if "SC Prior Matrix" in w.label), None
    )
    if sc_widget is not None:
        task.cancel_event.clear()
        task.launch_signature = with_preprocessing
        sc_widget.set_value(str(Path(tempfile.gettempdir()) / "not-a-real-sc.npy"))
        app.run()
        assert not app.exception
        changed = _capture_signature(app)
        assert changed["sc_prior_path"] != base_signature["sc_prior_path"]
        assert task.cancel_event.is_set(), "an SC path change must invalidate the job"


def test_real_sc_prior_labels_case():
    """A valid SC matrix switches the executed labels; the signature must move."""
    from conninfpy.atlas import AtlasInfo

    task = _running_task()
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(5)], networks=["Vis"] * 5)
    app = _app_with_task(task)
    app.session_state.loaded_reference_atlas = atlas
    app.run()
    assert not app.exception
    profile = next(w for w in app.selectbox if "Select method profile" in w.label)
    network_profile = next(
        (o for o in profile.options if "NI-TFNBS" in o or "NETWORK-AWARE" in o.upper()),
        None,
    )
    if network_profile is None:  # pragma: no cover - widget text changed
        pytest.skip("no network-informed method profile available")
    profile.set_value(network_profile)
    app.run()
    assert not app.exception
    task.cancel_event.clear()
    before = _capture_signature(app)
    task.launch_signature = before
    assert "use_sc_prior" in before and "sc_net_labels" in before

    rng = np.random.default_rng(0)
    W = rng.random((5, 5))
    W = (W + W.T) / 2
    sc_path = Path(tempfile.mkdtemp()) / "sc.npy"
    np.save(sc_path, W)
    sc_widget = next(w for w in app.text_input if "SC Prior Matrix" in w.label)
    sc_widget.set_value(str(sc_path))
    app.run()
    assert not app.exception
    after = _capture_signature(app)
    assert after["sc_prior_path"] != before["sc_prior_path"]
    assert after["use_sc_prior"] != before["use_sc_prior"] or after["sc_net_labels"] != before["sc_net_labels"]
    assert task.cancel_event.is_set(), "an SC-label change must invalidate the job"
