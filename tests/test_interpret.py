"""Interpretation layer: evidence packets from Brainnetome annotation, guardrails, narrators."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conninfpy.interpret.annotate import annotate_rois, summarize_annotation
from conninfpy.brainnetome import BrainnetomeReference
from conninfpy.interpret.evidence import (
    build_annotation_evidence,
    describe_coverage,
    summarize_terms,
    validate_evidence,
)
from conninfpy.interpret.llm_narrative import LLMNarrator, check_narrative_terms, load_dotenv_manually
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


def test_guardrail_flags_terms_outside_the_packet(evidence):
    safe = ("Connections link ventral area 9/46 (A9/46v) homologous regions and the amygdala; "
            "BrainMap profiles list Cognition and Emotion terms (memory, working, fear).")
    assert check_narrative_terms(safe, evidence) == []
    flagged = check_narrative_terms("This contrast activates pain and vision networks with executive dysfunction.",
                                    evidence)
    assert "pain" in flagged and "executive" in flagged


def test_mock_narrator(evidence):
    text = LLMNarrator(provider="mock").generate(evidence)
    for heading in ("### Anatomical summary", "### Endpoint regions", "### BrainMap context", "### Cautions"):
        assert heading in text
    assert "dlPFC_L" in text and "Cognition (2 ROIs)" in text


def test_llm_narrator_missing_api_keys(evidence):
    with patch.dict(os.environ, {}, clear=True):
        for provider, msg in (("openai", "OPENAI_API_KEY is not set"),
                              ("gemini", "GEMINI_API_KEY or GOOGLE_API_KEY is not set"),
                              ("openrouter", "OPENROUTER_API_KEY is not set")):
            narrator = LLMNarrator(provider=provider, api_key="")
            narrator.api_key = None
            with pytest.raises(ValueError, match=msg):
                narrator.generate(evidence)


def test_llm_narrator_openrouter_setup():
    with patch.dict(os.environ, {}, clear=True):
        narrator = LLMNarrator(provider="openrouter", api_key="sk-or-v1-testkey", model="meta-llama/test")
        assert narrator.provider == "openrouter"
        assert narrator.api_key == "sk-or-v1-testkey"
        assert narrator.model == "meta-llama/test"


def test_load_dotenv_manually():
    has_env = any(
        d.joinpath(".env").exists()
        for d in [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parents[1]]
    )
    if not has_env:  # e.g. fork PRs / fresh clones without a local .env
        pytest.skip("no local .env with OPENROUTER_API_KEY")
    with patch.dict(os.environ, {}, clear=True):
        load_dotenv_manually()
        assert "OPENROUTER_API_KEY" in os.environ
        assert os.environ["OPENROUTER_API_KEY"].startswith("sk-or-")
