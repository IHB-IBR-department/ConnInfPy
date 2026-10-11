# Brainnetome interpretation

`brainnetome_interpretation.py` applies Brainnetome anatomy to the existing Schaefer result or performs the separate native Brainnetome contrast. Annotation preserves the Schaefer result arrays; parcel overlap, purity and attribution level constrain anatomical precision. Native parcellation changes the measured signals and edge family.

```bash
python -m examples.open_close.interpretation.brainnetome_interpretation --help
```

Inputs are study time series, saved publication maps and local atlas resources. Default tabular artifacts remain under `examples/open_close/results/brainnetome*`. Use an explicit `--report /path/to/project/vault/_wiki/report.md` for a Markdown narrative. The atlas-stability workflow provides the matched participant/bootstrap comparison; annotation alone is not evidence of improved reproducibility.
