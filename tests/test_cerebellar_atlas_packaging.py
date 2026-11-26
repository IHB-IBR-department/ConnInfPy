"""NETSTAT-48: cerebellar atlas data package and the BNA-preserving composite spike.

Scope of this module is deliberately *test-only*: the prototype helpers below
(``lattice_mapping``, ``gather_exact``, ``composite``, ``crosswalk``,
``companion_modality_status``) are not a public API and are not imported by
``conninfpy``. They pin the facts that a future optional companion layer must
respect; the GO/NO-GO decision itself is recorded in the NETSTAT-48 evidence,
not here.

Tiny fixture (built in ``tmp_path``, never committed): two 4x4x3 NIfTI images on
the *same* 1 mm world lattice but with different orientation/origin, mirroring
the real relationship between BNA-246 1 mm (LAS) and the Buckner maps (RAS,
``qform_code=0``):

* ``bna``   : ids 101 (overlapping a cerebellar voxel -> clipped), 102 (BNA-only);
* ``cb``    : real Diedrichsen ids - 5 ``Left_VI`` (one clipped + one clean voxel),
  7 ``Right_VI`` (two voxels), 6 ``Vermis_VI`` (midline ``M``), 29 ``Left_Dentate``
  (deep nucleus);
* ``net7``  : a Buckner-7-like map on the flipped grid: label 4 under both ``VI``
  hemispheres, label 6 under the vermis and one ``Right_VI`` voxel, nothing under
  the nucleus (-> ``unassigned``);
* one voxel with neither BNA nor cerebellum label (-> 0).
"""
import csv
import gzip
import hashlib
import json
import math

import nibabel as nib
import numpy as np
import pytest

from conninfpy.atlas.resources import atlas_resource_available, atlas_resource_path, bundled_path

ATLAS = {
    "anatom": "atl-Anatom_space-MNI_dseg.nii.gz",
    "buckner7": "atl-Buckner7_space-MNI_dseg.nii.gz",
    "buckner17": "atl-Buckner17_space-MNI_dseg.nii.gz",
    "nettekoven32": "atl-NettekovenSym32_space-MNI_dseg.nii.gz",
}
TABLE = {"anatom": "atl-Anatom.tsv", "buckner7": "atl-Buckner7.tsv",
         "buckner17": "atl-Buckner17.tsv", "nettekoven32": "atl-NettekovenSym32.lut"}
COMMIT = "6857dfa732de6dd60c94da2f84d0b15ccf316618"


# --------------------------------------------------------------------------- prototype
def lattice_mapping(src, dst):
    """Voxel->voxel matrix ``M = inv(affine_dst) @ affine_src`` and whether the two
    grids coincide exactly (signed permutation + integer translation)."""
    M = np.linalg.inv(np.asarray(dst.affine, float)) @ np.asarray(src.affine, float)
    R, t = M[:3, :3], M[:3, 3]
    perm = (np.all(np.abs(R).sum(0) == 1) and np.all(np.abs(R).sum(1) == 1)
            and np.allclose(np.abs(R[R != 0]), 1))
    return M, bool(perm and np.allclose(t, np.rint(t), atol=1e-6))


def gather_exact(src, dst):
    """Exact nearest-neighbour gather of ``src`` onto the grid of ``dst`` (requires
    exact lattice coincidence; raises otherwise instead of silently resampling)."""
    M, exact = lattice_mapping(src, dst)
    if not exact:
        raise ValueError("grids do not coincide exactly; resampling is not allowed here")
    sh = np.asarray(src.shape[:3])
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in sh], indexing="ij")
    hom = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
    r = np.rint((M @ hom)[:3].T).astype(int)
    inside = np.all((r >= 0) & (r < np.asarray(dst.shape[:3])), axis=1)
    out = np.zeros(dst.shape[:3], dtype=np.int16)
    d = np.asanyarray(src.dataobj).ravel()
    sel = inside & (d != 0)
    out[r[sel, 0], r[sel, 1], r[sel, 2]] = d[sel]
    return out


def composite(bna, cb, offset, max_cb=34):
    """BNA has priority; cerebellar ids live in their own offset namespace."""
    c = np.where((cb >= 1) & (cb <= max_cb), cb, 0)
    return np.where(bna > 0, bna, np.where(c > 0, offset + c, 0)).astype(np.int32)


def crosswalk(cb_img, net_img, ids, threshold=0.5):
    """Per cerebellar region: label distribution, normalised weights (sum to 1,
    unassigned included), top label, purity of assigned voxels and status."""
    out = {}
    for i in sorted(ids):
        m = cb_img == i
        n = int(m.sum())
        vals, cnts = np.unique(net_img[m], return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, cnts, strict=True)}
        assigned = n - dist.get(0, 0)
        ranked = sorted(((c, -v) for v, c in dist.items() if v != 0), reverse=True)
        top = -ranked[0][1] if ranked else None
        purity = (ranked[0][0] / assigned) if assigned else None
        status = "unassigned" if assigned == 0 else ("dominant" if purity >= threshold else "mixed")
        out[i] = {"voxels": n, "dist": dist, "weights": {v: c / n for v, c in dist.items()},
                  "unassigned": dist.get(0, 0), "assigned": assigned, "top_label": top,
                  "purity_of_assigned": purity, "status": status}
    return out


def companion_modality_status(atlas_source):
    """SC and BrainMap profiles cover BNA-246 only: NaN + explicit status, never 0
    and never a borrowed BNA profile."""
    if atlas_source == "BNA-246":
        return {"sc": 0.0, "brainmap_status": "available"}
    return {"sc": math.nan, "brainmap_status":
            "not available for companion atlas (BrainMap profiles cover BNA-246 only)"}


def parse_anatom(rows):
    hemi, homolog = {}, {}
    names = {int(r["index"]): r["name"] for r in rows}
    for i, name in names.items():
        for pre, h in (("Left_", "L"), ("Right_", "R"), ("Vermis_", "M")):
            if name.startswith(pre):
                hemi[i] = h
                if h in "LR":
                    other = "Right_" if h == "L" else "Left_"
                    homolog[i] = next(j for j, m in names.items() if m == other + name[len(pre):])
    return names, hemi, homolog


# --------------------------------------------------------------------------- fixture
BNA_AFF = np.array([[-1, 0, 0, 3], [0, 1, 0, -3], [0, 0, 1, -2], [0, 0, 0, 1]], float)   # LAS
NET_AFF = np.array([[1, 0, 0, 0], [0, 1, 0, -3], [0, 0, 1, -2], [0, 0, 0, 1]], float)    # RAS, same lattice


def _nifti(arr, affine, path, qform_code=2):
    img = nib.Nifti1Image(np.asarray(arr, dtype=np.uint8), affine)
    img.header.set_qform(affine, code=qform_code)
    img.header.set_sform(affine, code=2)
    nib.save(img, str(path))
    return nib.load(str(path))


@pytest.fixture
def tiny(tmp_path):
    bna = np.zeros((4, 4, 3), int)
    bna[0, 0, 0], bna[1, 0, 0] = 101, 102
    cb = np.zeros((4, 4, 3), int)
    cb[0, 0, 0], cb[3, 1, 0] = 5, 5       # Left_VI: one clipped, one clean
    cb[2, 0, 0], cb[2, 1, 0] = 7, 7       # Right_VI
    cb[3, 0, 0] = 6                       # Vermis_VI (M)
    cb[0, 1, 0] = 29                      # Left_Dentate (deep nucleus)
    net = np.zeros((4, 4, 3), int)
    # net grid is x-flipped: BNA voxel i -> net voxel 3 - i
    net[3, 0, 0], net[0, 1, 0] = 4, 4     # under Left_VI voxels
    net[1, 0, 0], net[1, 1, 0] = 4, 6     # under Right_VI voxels (one per network -> mixed at >0.5)
    net[0, 0, 0] = 6                      # under Vermis_VI
    # (3,1,0) under the nucleus stays 0 -> unassigned
    return {
        "bna_arr": bna, "cb_arr": cb, "net_arr": net,
        "bna": _nifti(bna, BNA_AFF, tmp_path / "bna.nii.gz"),
        "cb": _nifti(cb, BNA_AFF, tmp_path / "cb.nii.gz"),
        "net": _nifti(net, NET_AFF, tmp_path / "net7.nii.gz", qform_code=0),
    }


# ------------------------------------------------------------------ fixture: grids
def test_flipped_grid_is_an_exact_lattice_not_a_resampling(tiny):
    M, exact = lattice_mapping(tiny["cb"], tiny["net"])
    assert exact
    assert np.array_equal(np.rint(M[:3, :3]), np.diag([-1.0, 1.0, 1.0]))      # x flip only
    assert np.array_equal(np.rint(M[:3, 3]), [3, 0, 0])                       # integer shift
    gathered = gather_exact(tiny["net"], tiny["cb"])
    assert gathered[0, 0, 0] == 4 and gathered[3, 1, 0] == 4          # Left_VI voxels -> network 4
    assert gathered[2, 0, 0] == 4 and gathered[2, 1, 0] == 6          # Right_VI voxels -> 4 and 6
    assert gathered[3, 0, 0] == 6                                    # Vermis_VI -> 6
    assert gathered[0, 1, 0] == 0 and gathered[1, 1, 0] == 0          # nucleus and empty voxel -> unassigned
    # nibabel's order-0 resampling agrees (independent library cross-check)
    from nibabel.processing import resample_from_to
    lib = np.asanyarray(resample_from_to(tiny["net"], (tiny["cb"].shape, BNA_AFF), order=0).dataobj)
    assert np.array_equal(lib, gathered)


def test_non_coincident_grid_is_rejected_not_resampled(tiny, tmp_path):
    shifted = _nifti(tiny["net_arr"], NET_AFF + np.array([[0, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                     tmp_path / "half.nii.gz", qform_code=0)
    with pytest.raises(ValueError, match="coincide exactly"):
        gather_exact(shifted, tiny["cb"])


# -------------------------------------------------------------- fixture: composite
@pytest.mark.parametrize("offset", [246, 1000])
def test_composite_priority_ids_and_bna_identity(tiny, offset):
    comp = composite(tiny["bna_arr"], tiny["cb_arr"], offset)
    bna = tiny["bna_arr"]
    # BNA > 0 + CB > 0 -> BNA wins (clipped), BNA-only untouched, both 0 -> 0
    assert comp[0, 0, 0] == 101 and comp[1, 0, 0] == 102 and comp[1, 1, 0] == 0
    # BNA = 0 + CB > 0 -> offset + CB, including L, R, Vermis M and a deep nucleus
    assert comp[3, 1, 0] == offset + 5 and comp[2, 0, 0] == offset + 7
    assert comp[3, 0, 0] == offset + 6 and comp[0, 1, 0] == offset + 29
    b = bna > 0
    assert np.array_equal(comp[b], bna[b])                                   # bitwise BNA-only identity
    bna_ids = set(np.unique(bna[b]))
    cb_ids = set(int(v) for v in np.unique(comp[~b]) if v > 0)
    assert cb_ids == {offset + 5, offset + 6, offset + 7, offset + 29}       # independent ids
    assert not (bna_ids & cb_ids) and min(cb_ids) > max(bna_ids)             # no collision
    assert set(np.unique(comp)) == bna_ids | cb_ids | {0}


def test_composite_lobules_only_drops_nuclei_and_keeps_offset(tiny):
    comp = composite(tiny["bna_arr"], tiny["cb_arr"], 246, max_cb=28)
    assert comp[0, 1, 0] == 0 and comp.max() == 246 + 7
    comp34 = composite(tiny["bna_arr"], tiny["cb_arr"], 246, max_cb=34)
    assert comp34[0, 1, 0] == 246 + 29 and comp34.max() == 246 + 29
    assert composite(tiny["bna_arr"], tiny["cb_arr"], 1000, max_cb=34).max() == 1000 + 29


def test_composite_without_cerebellum_is_bitwise_bna(tiny):
    comp = composite(tiny["bna_arr"], np.zeros_like(tiny["cb_arr"]), 246)
    assert np.array_equal(comp, tiny["bna_arr"]) and set(np.unique(comp)) == {0, 101, 102}


# ------------------------------------------------------------- fixture: crosswalk
def test_crosswalk_distribution_weights_purity_unassigned(tiny):
    net_on_bna = gather_exact(tiny["net"], tiny["cb"])
    cw = crosswalk(tiny["cb_arr"], net_on_bna, ids=[5, 6, 7, 29])
    assert cw[5]["dist"] == {4: 2} and cw[5]["weights"] == {4: 1.0}
    assert cw[5]["top_label"] == 4 and cw[5]["purity_of_assigned"] == 1.0 and cw[5]["status"] == "dominant"
    assert cw[6]["dist"] == {6: 1} and cw[6]["top_label"] == 6
    assert cw[7]["dist"] == {4: 1, 6: 1} and cw[7]["unassigned"] == 0
    assert cw[7]["purity_of_assigned"] == 0.5 and cw[7]["weights"] == {4: 0.5, 6: 0.5}
    assert cw[7]["top_label"] == 4                                        # tie -> smaller label
    assert cw[7]["status"] == "dominant" and crosswalk(tiny["cb_arr"], net_on_bna, [7], 0.75)[7]["status"] == "mixed"
    assert cw[29]["dist"] == {0: 1} and cw[29]["assigned"] == 0 and cw[29]["status"] == "unassigned"
    assert cw[29]["top_label"] is None and cw[29]["purity_of_assigned"] is None
    for r in cw.values():
        assert math.isclose(sum(r["weights"].values()), 1.0)


def test_crosswalk_never_invents_a_label_for_unassigned_voxels(tiny):
    cw = crosswalk(tiny["cb_arr"], np.zeros_like(tiny["cb_arr"]), ids=[5, 7])
    assert all(r["status"] == "unassigned" and r["top_label"] is None for r in cw.values())


def test_companion_modality_status_is_nan_not_zero_or_bna_profile():
    bna = companion_modality_status("BNA-246")
    cb = companion_modality_status("Diedrichsen_2009/atl-Anatom")
    assert bna["sc"] == 0.0 and bna["brainmap_status"] == "available"
    assert math.isnan(cb["sc"]) and cb["sc"] != 0.0
    assert cb["brainmap_status"].startswith("not available for companion atlas")


# ------------------------------------------------------- packaged files + provenance
HAS_SUIT_CACHE = all(atlas_resource_available(n) for n in (*ATLAS.values(), *TABLE.values()))
HAS_BNA1MM_CACHE = atlas_resource_available("BN_Atlas_246_1mm.nii.gz")
CACHE_SKIP_REASON = "Brainnetome/SUIT atlas files not in local cache; run scripts/download_atlases.py"


@pytest.fixture(scope="module")
def provenance():
    return json.loads(bundled_path("cerebellar_atlases_provenance.json").read_text())


def test_provenance_pins_commit_and_all_nine_files(provenance):
    assert provenance["atlas_set"]["tag"] == "v1.2" and provenance["atlas_set"]["commit"] == COMMIT
    files = provenance["atlas_set"]["files"]
    expected = set(ATLAS.values()) | set(TABLE.values())
    assert set(files) == expected and len(files) == 8
    for name, f in files.items():
        assert f["upstream_url"].startswith(
            f"https://raw.githubusercontent.com/DiedrichsenLab/cerebellar_atlases/{COMMIT}/")
        assert f["git_blob_sha1_matches_tree_at_commit"] is True
        if not atlas_resource_available(name):
            pytest.skip(CACHE_SKIP_REASON)
        path = atlas_resource_path(name)
        raw = path.read_bytes()
        assert len(raw) == f["packaged_bytes"] and hashlib.sha256(raw).hexdigest() == f["packaged_sha256"]


@pytest.mark.skipif(not HAS_SUIT_CACHE, reason=CACHE_SKIP_REASON)
def test_packaged_nifti_is_deterministic_gzip_and_matches_upstream_facts(provenance):
    for name in ATLAS.values():
        f = provenance["atlas_set"]["files"][name]
        path = atlas_resource_path(name)
        raw = path.read_bytes()
        flg, mtime = raw[3], int.from_bytes(raw[4:8], "little")
        assert not (flg & 0x08) and mtime == 0 and f["packaging"]["gzip_header"]["MTIME"] == 0
        assert f["packaging"]["decompressed_equals_upstream_bytes"] is True
        assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == f["upstream_sha256"]
        img = nib.load(str(atlas_resource_path(name)))
        facts = f["nifti"]
        assert list(img.shape) == facts["shape"] and str(img.header.get_data_dtype()) == facts["dtype"]
        assert facts["zooms"] == [1.0, 1.0, 1.0]
        assert np.allclose(img.affine, facts["affine"]) and "".join(nib.aff2axcodes(img.affine)) == facts["axcodes"]


@pytest.mark.skipif(not HAS_SUIT_CACHE, reason=CACHE_SKIP_REASON)
def test_packaged_label_tables_complete_and_upstream_typos_kept(provenance):
    read = lambda n: list(csv.DictReader(open(atlas_resource_path(n), newline=""), delimiter="\t"))
    an = read(TABLE["anatom"])
    assert [int(r["index"]) for r in an] == list(range(1, 35))
    b7 = {int(r["index"]): r["name"] for r in read(TABLE["buckner7"])}
    b17 = {int(r["index"]): r["name"] for r in read(TABLE["buckner17"])}
    assert sorted(b7) == list(range(1, 8)) and sorted(b17) == list(range(1, 18))
    assert b7[6] == "Netowrk6" and b17[8] == "Networks_8" and b17[9] == "Networks_9"   # verbatim, not fixed
    lut = [l.split() for l in atlas_resource_path(TABLE["nettekoven32"]).read_text().splitlines() if l.strip()]
    nk = {int(r[0]): r[4] for r in lut}
    assert lut[0][0] == "0" and lut[0][4] == "0"                        # background row carries no name
    assert sorted(k for k in nk if k != 0) == list(range(1, 33))
    assert all(nk[i][:-1] == nk[i + 16][:-1] and nk[i][-1] == "L" and nk[i + 16][-1] == "R" for i in range(1, 17))
    assert not any(n.endswith("M") for n in nk.values())


@pytest.mark.skipif(not HAS_SUIT_CACHE, reason=CACHE_SKIP_REASON)
def test_anatom_hemisphere_and_homolog_come_from_names_not_id_parity(provenance):
    rows = list(csv.DictReader(open(atlas_resource_path(TABLE["anatom"]), newline=""), delimiter="\t"))
    names, hemi, homolog = parse_anatom(rows)
    assert [hemi[i] for i in (1, 2, 5, 6, 7, 29, 34)] == ["L", "R", "L", "M", "R", "L", "R"]
    assert sorted(hemi.values()) == ["L"] * 13 + ["M"] * 8 + ["R"] * 13
    assert homolog[5] == 7 and homolog[8] == 10 and homolog[29] == 30 and 6 not in homolog
    # the BNA convention (odd = left, homolog = id + 1) does not transfer
    assert not (hemi[6] == "L" and homolog.get(6) == 7)
    assert names[29].endswith("Dentate") and names[34].endswith("Fastigial")


@pytest.mark.skipif(not (HAS_SUIT_CACHE and HAS_BNA1MM_CACHE), reason=CACHE_SKIP_REASON)
def test_real_grids_coincide_exactly_with_bna1mm_and_overlap_is_small():
    bna = nib.load(str(atlas_resource_path("BN_Atlas_246_1mm.nii.gz")))
    assert bna.shape == (182, 218, 182) and np.allclose(bna.affine[:3, 3], [90, -126, -72])
    bna_arr = np.asanyarray(bna.dataobj).astype(np.int16)
    cb_img = nib.load(str(atlas_resource_path(ATLAS["anatom"])))
    cb = gather_exact(cb_img, bna)                              # raises if not an exact lattice
    for name in ATLAS.values():
        assert lattice_mapping(nib.load(str(atlas_resource_path(name))), bna)[1]
    b, c = bna_arr > 0, cb > 0
    clipped, cb_only = int((b & c).sum()), int((~b & c).sum())
    assert clipped > 0 and clipped < 0.02 * c.sum() and (b & c & (cb >= 29)).sum() == 0
    assert cb_only + clipped == int(c.sum()) and int((b & ~c).sum()) + clipped == int(b.sum())
    comp = composite(bna_arr, cb, 1000)
    assert np.array_equal(comp[b], bna_arr[b])
    assert set(int(v) for v in np.unique(comp[~b]) if v > 0) <= set(range(1001, 1035))
