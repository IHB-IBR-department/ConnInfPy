"""Offline producer → save/reload → independent summary example."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from conninfpy import AtlasInfo, ResultBundle, analyze

rng = np.random.default_rng(7)
correlations = np.tanh(rng.normal(0, .3, (10, 3, 3)))
correlations = (correlations + correlations.transpose(0, 2, 1)) / 2
correlations[:, np.arange(3), np.arange(3)] = 1
atlas = AtlasInfo(
    labels=["left_a", "right_a", "left_b"], networks=["A", "A", "B"],
    hemisphere=["L", "R", "L"], label_ids=np.array([1, 5, 12]),
)
result = analyze(
    group1=correlations, test_type="one-sample", input="r", method="tstat",
    n_permutations=20, acceleration=None, use_mp=False, rng=0, atlas=atlas,
)
bundle = result.to_bundle()
with TemporaryDirectory() as temp:
    path = Path(temp) / "result.pkl"
    bundle.save(path)
    loaded = ResultBundle.load(path)
    summary = loaded.summarize(
        grouping="network", measure="contrast_beta",
        contrast_id=loaded.metadata["contrast_id"], tail="positive",
    )
    loaded.export_summary(summary, Path(temp) / "network_summary.csv")
    print(summary.to_string(index=False))
