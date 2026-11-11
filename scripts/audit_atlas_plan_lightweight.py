"""Bounded counterexamples for the coordinate-atlas design review; no inference runs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from conninfpy.atlas import AtlasInfo, assign_coordinates
from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.manifest import filter_dataset_rois
from conninfpy.results import summarize_by_network

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, default=Path('/Users/irina/Work/Obsidian/MainVault/Projects/NetworkStatistics/_wiki/atlas_coordinate_rois_implementation_plan_2026-10-07.md'))
parser.add_argument('--output', type=Path, help='Audit evidence path; defaults to the plan folder in the Obsidian vault.')
args = parser.parse_args()
PLAN = args.plan
paths = [ROOT / 'conninfpy/atlas.py', ROOT / 'conninfpy/loaders/manifest.py', ROOT / 'conninfpy/_export.py', PLAN]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
checks = []

def record(name, expected, observed):
    checks.append(dict(name=name, expected=expected, observed=observed))

def image_at(x, label=1, name=None):
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[1, 1, 1] = label
    affine = np.eye(4)
    affine[0, 3] = x
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units('mm')
    return AtlasInfo.from_nifti(img, lut={label: name} if name else None)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    centroid = AtlasInfo(labels=['Cortex'], networks=['Visual'], coords=[[0, 0, 0]])
    t = assign_coordinates([[1000, 0, 0]], centroid, max_distance=5.0)
    record('centroid_distance_cutoff', 'unassigned beyond the declared cutoff, or an explicitly separate unlimited policy', t[['region', 'method', 'distance_mm']].to_dict('records'))

    nearby = image_at(0, name='NearVoxel')
    t = assign_coordinates([[2, 1, 1]], [centroid, nearby], max_distance=5.0)
    record('mixed_reference_fallback_precedence', 'the documented reference priority applied consistently to bounded fallback candidates', t[['region', 'method', 'distance_mm', 'atlas']].to_dict('records'))

    a, b = image_at(0), image_at(20)
    rois = AtlasInfo.from_coordinates([[1, 1, 1], [21, 1, 1]], [a, b], group_by='region')
    record('region_identity_across_atlases', 'two distinct region groups for two distinct atlases', dict(groups=rois.networks, atlas=rois.assignment['atlas'].tolist(), codes=rois.network_index().tolist()))

    override = AtlasInfo.from_coordinates([[1, 1, 1]], image_at(0, name='AnatomicalRegion'), networks=['StudyNetwork'])
    record('assignment_keeps_reference_network', 'reference network and final grouping both retained separately', dict(reference_network='AnatomicalRegion', recorded_network=override.assignment['network'].iloc[0]))

    roi_ds = LoadedDataset(data=np.ones((2, 5, 2)), pheno=pd.DataFrame({'subject_id': ['a', 'b']}), data_kind='timeseries', subject_ids=['a', 'b'], atlas=rois)
    filtered = filter_dataset_rois(roi_ds, [1])
    record('coordinate_assignment_survives_filter', 'one matching assignment row retained', dict(n_rois=filtered.data.shape[-1], assignment_is_none=filtered.atlas.assignment is None))

    label_ds = LoadedDataset(data=np.ones((2, 5, 1)), pheno=pd.DataFrame({'subject_id': ['a', 'b']}), data_kind='timeseries', subject_ids=['a', 'b'], atlas=a)
    filtered = filter_dataset_rois(label_ds, [1])
    record('image_and_label_ids_survive_filter', 'matching label IDs and a correspondingly restricted image retained', dict(image_is_none=filtered.atlas.image is None, label_ids_is_none=filtered.atlas.label_ids is None))

    bad = AtlasInfo.from_coordinates([[np.nan, 0, 0]], networks=['StudyNetwork'])
    record('nonfinite_coordinate_validation', 'early ValueError naming the invalid ROI', dict(accepted=bool(np.isnan(bad.coords[0, 0])), hemisphere=bad.hemisphere))

    data = np.zeros((5, 5, 5), dtype=np.float32)
    data[1, 1, 1] = np.inf
    bad = AtlasInfo.from_nifti(nib.Nifti1Image(data, np.eye(4)))
    record('infinite_label_validation', 'ValueError for nonfinite label data', dict(accepted=True, label_ids=bad.label_ids.tolist()))

    data[1, 1, 1] = 10000.09
    bad = AtlasInfo.from_nifti(nib.Nifti1Image(data, np.eye(4)))
    record('integer_label_validation', 'ValueError for a fractional label, or a declared absolute rounding tolerance', dict(input_value=float(data[1, 1, 1]), accepted_label_ids=bad.label_ids.tolist()))

    data = np.zeros((5, 5, 5, 1), dtype=np.int16)
    data[1, 1, 1, 0] = 1
    single = AtlasInfo.from_nifti(nib.Nifti1Image(data, np.eye(4)))
    record('singleton_4d_contract', 'reject all 4-D images as the plan says, or explicitly normalise singleton input to a 3-D image', dict(accepted=True, retained_image_shape=list(single.image.shape)))

    p = np.ones((2, 2))
    p[1, 0] = 0.01
    summary = summarize_by_network({'positive': p}, ['A', 'B'])
    record('directed_summary_scope', 'directed lower-triangle edge counted, or a clear rejection/undirected-only contract', dict(lower_triangle_p=float(p[1, 0]), n_edges=int(summary['n_edges'].sum()), n_significant=int(summary['n_sig_positive'].sum())))

after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report = dict(date='2026-10-07', scope='Synthetic checks only; images at most 5x5x5x1; no permutations, dataset downloads, registration, or real-data extraction.', versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'nibabel', 'nilearn']}, source_sha256=before, sources_unchanged=before == after, checks=checks)
output = args.output or PLAN.parent / 'atlas_coordinate_rois_review_checks_2026-10-07.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2, allow_nan=False))
