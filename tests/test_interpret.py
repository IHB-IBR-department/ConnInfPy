"""Offline interpretation layer: evidence packets from Brainnetome annotation."""
import tempfile

import numpy as np
import pandas as pd
import pytest

from conninfpy.interpret.annotate import annotate_rois, summarize_annotation
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.interpret.evidence import (
    build_annotation_evidence,
    describe_coverage,
    summarize_terms,
    validate_evidence,
)
from conninfpy.results import InferenceResult

POINTS = [(-42, 36, 24), (42, 36, 24), (-22, -4, -18), (-20, -60, -30)]   # dlPFC L/R, amygdala, cerebellum
NAMES = ["dlPFC_L", "dlPFC_R", "Amygdala_L", "Cerebellum_L"]


@pytest.fixture(scope="module")
def rois():
    ref = BrainnetomeReference.load(data_dir=tempfile.mkdtemp(), profiles=False, cache_dir=tempfile.mkdtemp())
    ref.profiles = {"BD": {k: {} for k in range(1, 247)}, "PC": {k: {} for k in range(1, 247)}}
    ref.profiles["BD"][21] = {"Cognition.Memory.Working": 2.5}             # A9/46v_L
    ref.profiles["BD"][22] = {"Cognition.Memory.Working": 1.9}             # A9/46v_R
    ref.profiles["BD"][211] = {"Emotion.Fear": 3.1}                        # mAmyg_L
    return annotate_rois(POINTS, names=NAMES, reference=ref, support="sphere", radius_mm=6)


@pytest.fixture(scope="module")
def evidence(rois):
    edges = pd.DataFrame({"roi_i": [0, 0], "roi_j": [1, 2]})
    return build_annotation_evidence(edges, rois, contrast_name="ABIDE age", source_atlas="test coordinates")


def test_packet_holds_anatomy_and_literature_context(evidence):
    validate_evidence(evidence)
    q = evidence["query"]
    assert (q["contrast"], q["edge_count"], q["roi_ids"], q["inference_unit"]) == ("ABIDE age", 2, [0, 1, 2], "edge")
    rois = {r["roi_name"]: r for r in evidence["rois"]}
    assert rois["dlPFC_L"]["assigned"] == "A9/46v_L" and rois["dlPFC_L"]["n_detected_edges"] == 2
    assert rois["Amygdala_L"]["level"] == "gyrus"
    assert "Cerebellum_L" not in rois                         # not an endpoint
    terms = {(t["roi_name"], t["term"]): t["likelihood_ratio"] for t in evidence["terms"]}
    assert terms[("dlPFC_L", "Cognition.Memory.Working")] == 2.5
    assert terms[("Amygdala_L", "Emotion.Fear")] == 3.1
    assert {p["source"] for p in q["region_pairs"]} | {p["target"] for p in q["region_pairs"]} >= {"A9/46v_L"}


def test_summaries_count_each_roi_once(evidence):
    s = summarize_terms(evidence)
    dom = {d["domain"]: d["n_rois"] for d in s["domains"]}
    assert dom["Cognition"] == 2 and dom["Emotion"] == 1 and dom["Action"] == 0
    assert s["top_endpoint_rois"][0]["roi_name"] == "dlPFC_L"
    cov = describe_coverage(evidence)
    assert cov["n_endpoint_rois"] == 3 and cov["n_with_terms"] == 3
    assert "inference unit: edge" in cov["report_sentence"]


def test_validation_rejects_malformed_packets(evidence):
    bad = dict(evidence)
    del bad["reference"]
    with pytest.raises(ValueError, match="Missing required top-level evidence fields"):
        validate_evidence(bad)
    bad = dict(evidence, rois=[])
    with pytest.raises(ValueError, match="non-empty"):
        validate_evidence(bad)
    validate_evidence(dict(evidence, terms=[]))               # no published terms is allowed


def test_relation_counts_from_the_summary(rois):
    n = len(POINTS)
    p = np.ones((n, n))
    p[0, 1] = p[1, 0] = 0.01
    res = InferenceResult(p, np.ones((n, n)), method="tstat", stat_positive=np.where(p < 1, 4.0, 0.0),
                          stat_negative=np.zeros((n, n)))
    summary = summarize_annotation(res, rois)
    ev = build_annotation_evidence(res.significant_edges(), rois, summary=summary)
    hom = [r for r in ev["relations"] if r["category"] == "homotopic"]
    assert hom and hom[0]["n_detected"] == 1
