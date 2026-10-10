# Open-Close inference restricted to structurally supported edges

Binarise the Brainnetome structural connectivity (SC). Then compute TFNBS cluster
support and NI-TFNBS block density only over structurally supported edges,
using `compute_p_val(..., edge_mask=mask)` / `analyze(..., edge_mask=mask)`.
FWER is then controlled over that family; edges outside it are untested (NaN).

Plan: vault note `openclose_gsr_sc_followup_plan_2026-10-10`, SC section.

## Design

Data: native BNA-246 (242 regions, GSR); IHB (84) and China EO vs counterbalanced
EC1 (47). Paired tests, paper settings, NI-TFNBS blocks = Brainnetome Yeo-7
crosswalk + subcortex.

| Stage | Question |
|---|---|
| E0 | Are EC−EO effects larger on SC-supported pairs than on unsupported pairs at matched distance (10 mm bin × within/between hemisphere × cortical/subcortical)? Go/no-go, participant bootstrap |
| E2 | SC-restricted family (SC ≥ 0.05 primary; 0.01, 0.1, 0.2) vs the full-family result restricted to the same edges; what the mask removes |
| E3 | SC mask vs random masks matched stratum by stratum, on IHB–China replication (Dice) and detections |
| E4 | FWER within the SC family on null datasets made by random per-participant EO/EC swaps |

## Run

```
python examples/openclose_sc_inference/run_sc_inference.py            # ~2.5-3 h (40 masks, 200 nulls)
python examples/openclose_sc_inference/run_sc_inference.py --quick    # ~10 min
python examples/openclose_sc_inference/run_sc_inference.py --stage e0,e2
```

Outputs go to `results/` (git-ignored): `report.md`, CSV tables per stage, and
p-maps for the primary runs.
