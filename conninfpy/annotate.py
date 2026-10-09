"""Annotate ROIs and connections with the Brainnetome atlas (BNA-246).

Rules (design note ``bna_annotation_neurophysiology_design_2026-10-09``):

* Annotate the tissue that produced the signal (a sphere, a parcel, or a
  point) as overlap fractions over BNA regions rather than one centre label:
  a 6 mm sphere usually covers 2-4 BNA regions, and a parcel's centroid can
  lie outside the parcel.
* Give a hard label only at the finest *pure* level of the hierarchy
  (subregion -> gyrus/nucleus -> lobe); below that the ROI is "mixed".
* Lead with anatomical identity. BrainMap terms are literature context
  (forward-inference likelihood ratios from published activations), never
  "the function" of a region; there is no top-term summary.
* Read a connection as a relation between its endpoints: hemispheric relation
  (homotopic or not), anatomical level, circuit class, distance, shared
  support, population structural connectivity, network relation.

Everything here is description. Nothing tests a new hypothesis, and grouping
or annotating results never changes their correction or inference unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from .brainnetome import BD_DOMAINS, N_REGIONS, BrainnetomeReference

_EXACT_SPACES = {"MNI152NLin6Asym"}
_APPROX_SPACES = {"MNI152", "MNI152NLin2009cAsym", "MNI152NLin2009cSym", "ICBM152", "MNI152Lin"}
_LEVELS = ("subregion", "gyrus", "lobe")
_CIRCUITS = {
    frozenset({"cortex"}): "cortico-cortical",
    frozenset({"cortex", "thalamus"}): "thalamo-cortical",
    frozenset({"cortex", "basal_ganglia"}): "cortico-basal ganglia",
    frozenset({"cortex", "amygdala"}): "amygdalo-cortical",
    frozenset({"cortex", "hippocampus"}): "hippocampo-cortical",
}


@dataclass
class RoiAnnotation:
    """Brainnetome annotation of a set of ROIs, in connectivity-matrix order.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per ROI (see :func:`annotate_rois` for the columns).
    weights : ndarray of shape (n_rois, 246)
        Share of each ROI's support voxels in each BNA region of the
        maximum-probability map (the rest is unlabelled: white matter, CSF,
        or outside the atlas).
    membership : ndarray of shape (n_rois, 246) or None
        Mean probabilistic-map membership over the support (40 reference
        subjects), when ``BNA_PM_4D.nii.gz`` is available.
    reference : BrainnetomeReference
    settings : dict
        Support, space, thresholds and the reference's provenance.
    """

    table: Any
    weights: npt.NDArray[np.float64]
    membership: Optional[npt.NDArray[np.float64]]
    reference: BrainnetomeReference
    settings: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.table)

    def labelled_weights(self) -> npt.NDArray[np.float64]:
        """``weights`` renormalised to the labelled part of each support (NaN rows if none)."""
        cov = self.weights.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cov > 0, self.weights / cov, np.nan)

    def grouping(self, level: str = "lobe") -> List[str]:
        """Group label of each ROI at ``level`` ('subregion', 'gyrus', 'lobe' or 'network').

        ROIs not pure at that level are 'mixed' (or 'unassigned'); they stay
        visible as their own group instead of being forced into one.
        """
        t = self.table
        if level == "network":
            return list(t["yeo7"])
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS + ('network',)}; got {level!r}.")
        depth = {lv: k for k, lv in enumerate(_LEVELS)}
        out = []
        for _, r in t.iterrows():
            if r["level"] == "unassigned":
                out.append("unassigned")
            elif r["level"] == "mixed" or depth[r["level"]] > depth[level]:
                out.append("mixed")
            else:
                out.append({"subregion": r["region"], "gyrus": f"{r['gyrus']}_{r['hemisphere']}",
                            "lobe": f"{r['lobe']}_{r['hemisphere']}"}[level])
        return out

    def to_atlas(self, level: str = "lobe") -> Any:
        """:class:`~conninfpy.AtlasInfo` grouping the ROIs at ``level`` (for network summaries)."""
        from .atlas import AtlasInfo

        t = self.table
        return AtlasInfo(
            labels=list(t["roi"]), networks=self.grouping(level),
            coords=t[["x", "y", "z"]].to_numpy(float),
            hemisphere=[h if h in ("L", "R") else "M" for h in t["hemisphere"]],
            source=f"Brainnetome annotation ({level} level, {self.settings.get('support')} support)",
        )

    def profiles(self, kind: str = "BD", min_weight: Optional[float] = None) -> Any:
        """BrainMap terms of the BNA regions each ROI covers, as a long table.

        One row per (ROI, region, term) for the regions holding at least
        ``min_weight`` of the ROI's labelled support (default: the value used
        at annotation). ``likelihood_ratio`` is the published forward-inference
        value; it is not combined across regions or endpoints.
        """
        import pandas as pd

        ref = self.reference
        if ref.profiles is None:
            raise ValueError("the reference was loaded without BrainMap profiles.")
        mw = self.settings["min_weight"] if min_weight is None else min_weight
        w = self.labelled_weights()
        rows = []
        for r, (_, roi) in enumerate(self.table.iterrows()):
            for k in np.flatnonzero(np.nan_to_num(w[r]) >= mw):
                rid = int(k) + 1
                for term, lr in ref.profile(kind, rid).items():
                    rows.append({"roi": roi["roi"], "roi_index": r, "region": ref.regions.at[rid, "name"],
                                 "region_weight": float(w[r, k]), "term": term,
                                 "domain": ref.domain(term) if kind == "BD" else None,
                                 "likelihood_ratio": float(lr)})
        return pd.DataFrame(rows, columns=["roi", "roi_index", "region", "region_weight", "term",
                                           "domain", "likelihood_ratio"])


# ----------------------------------------------------------------------
# ROI annotation
# ----------------------------------------------------------------------
def annotate_rois(
    coords: Optional[npt.ArrayLike] = None,
    *,
    reference: Optional[BrainnetomeReference] = None,
    names: Optional[Sequence[str]] = None,
    space: str = "MNI152NLin6Asym",
    support: str = "sphere",
    radius_mm: Optional[Union[float, Sequence[float]]] = None,
    parcel_image: Any = None,
    parcel_labels: Optional[Sequence[int]] = None,
    purity: float = 0.5,
    min_coverage: float = 0.5,
    nearest_mm: Optional[float] = None,
    min_weight: float = 0.2,
) -> RoiAnnotation:
    """Describe each ROI by the Brainnetome regions its signal came from.

    Parameters
    ----------
    coords : array of shape (n_rois, 3), optional
        ROI centres in mm, in connectivity-matrix order. Required for
        ``support='point'`` and ``'sphere'``; for ``'parcel'`` the parcel
        centroids are used when omitted.
    reference : BrainnetomeReference, optional
        Loaded with :meth:`BrainnetomeReference.load` when omitted.
    names : sequence of str, optional
        ROI names (default ``ROI_001`` ...).
    space : str
        Template of the coordinates. ``'MNI152NLin6Asym'`` (FSL MNI152, the
        atlas's own grid) is exact; ``'MNI152'``, ``'MNI152NLin2009cAsym'``
        (fMRIPrep) and similar MNI variants differ by about 1-3 mm and are
        flagged approximate. Talairach coordinates must be converted first
        (e.g. Lancaster et al. 2007); the space is never guessed from values.
    support : {'sphere', 'parcel', 'point'}
        What produced each ROI's signal: a sphere of ``radius_mm`` around the
        centre, the voxels of ``parcel_labels`` in ``parcel_image``, or (when
        nothing else is known) the centre point alone, flagged as a point
        estimate.
    radius_mm : float or sequence of float
        Sphere radius (per ROI or one for all).
    parcel_image : nibabel image or path, optional
        Integer label image the ROI signals were extracted with (MNI).
    parcel_labels : sequence of int, optional
        Label value of each ROI in ``parcel_image`` (matrix order).
    purity : float, default 0.5
        A level (subregion, gyrus, lobe; network) is assigned when its largest
        share of the labelled support is at least ``purity``.
    min_coverage : float, default 0.5
        Least labelled share of the support for any assignment; below it the
        ROI is 'unassigned' (white matter, CSF, cerebellum, brainstem or
        outside the brain: BNA-246 covers cerebral cortex and subcortical
        nuclei only).
    nearest_mm : float, optional
        Point support only: an unlabelled centre takes the nearest labelled
        voxel within this distance, flagged with the distance. Off by default.
    min_weight : float, default 0.2
        Regions holding at least this share of the labelled support contribute
        their BrainMap profiles.

    Returns
    -------
    RoiAnnotation
        ``table`` columns: roi, x, y, z, support, radius_mm, n_voxels,
        coverage, level, assigned, region, region_purity, gyrus, lobe,
        hemisphere, structure, description, yeo7, yeo7_purity, yeo17,
        regions, membership, domains, profile_status, flags.
    """
    import pandas as pd

    ref = reference if reference is not None else BrainnetomeReference.load()
    if space in ("Talairach", "TAL", "tal"):
        raise NotImplementedError(
            "Talairach coordinates must be converted to MNI first (e.g. the Lancaster "
            "et al. 2007 tal2icbm transform); the atlas is in MNI152 space.")
    if space not in _EXACT_SPACES | _APPROX_SPACES:
        raise ValueError(f"space must be one of {sorted(_EXACT_SPACES | _APPROX_SPACES)}; got {space!r}.")
    if support not in ("sphere", "parcel", "point"):
        raise ValueError(f"support must be 'sphere', 'parcel' or 'point'; got {support!r}.")
    for nm, v in (("purity", purity), ("min_coverage", min_coverage), ("min_weight", min_weight)):
        if not 0 < v <= 1:
            raise ValueError(f"{nm} must be in (0, 1]; got {v}.")

    supports, coords = _supports(ref, coords, support, radius_mm, parcel_image, parcel_labels)
    n = len(supports)
    names = [f"ROI_{i + 1:03d}" for i in range(n)] if names is None else [str(x) for x in names]
    if len(names) != n:
        raise ValueError(f"names must have length {n}; got {len(names)}.")
    radii = (np.broadcast_to(np.asarray(radius_mm, float), (n,)) if support == "sphere"
             else np.zeros(n))

    weights = np.zeros((n, N_REGIONS))
    has_pm = ref._membership_matrix() is not None
    membership = np.zeros((n, N_REGIONS)) if has_pm else None
    nearest_note: Dict[int, float] = {}
    for r, world in enumerate(supports):
        if support == "point" and nearest_mm is not None and len(world) and ref.labels_at(world)[0] == 0:
            hit = _nearest_labelled(ref, world[0], nearest_mm)
            if hit is not None:
                world, nearest_note[r] = hit[0][None, :], hit[1]
        if len(world):
            lab = ref.labels_at(world)
            counts = np.bincount(lab, minlength=N_REGIONS + 1)
            weights[r] = counts[1:] / len(world)
            if has_pm:
                membership[r] = ref.membership(world)

    regions = ref.regions
    rows = []
    for r in range(n):
        rows.append(_describe(r, names[r], coords[r], supports[r], weights[r],
                              None if membership is None else membership[r], regions, ref,
                              support, radii[r], space, purity, min_coverage, min_weight,
                              nearest_note.get(r)))
    table = pd.DataFrame(rows)
    settings = {"support": support, "space": space, "purity": purity, "min_coverage": min_coverage,
                "min_weight": min_weight, "nearest_mm": nearest_mm, "provenance": dict(ref.provenance)}
    return RoiAnnotation(table=table, weights=weights, membership=membership, reference=ref,
                         settings=settings)


def _supports(ref, coords, support, radius_mm, parcel_image, parcel_labels):
    """World coordinates of each ROI's support voxels, and the ROI centres."""
    if support == "parcel":
        if parcel_image is None or parcel_labels is None:
            raise ValueError("support='parcel' needs parcel_image and parcel_labels.")
        import nibabel as nib

        img = nib.load(str(parcel_image)) if not hasattr(parcel_image, "affine") else parcel_image
        data = np.rint(np.asarray(img.dataobj)).astype(np.int64)
        ijk = np.argwhere(data > 0)
        lab = data[tuple(ijk.T)]
        world_all = np.c_[ijk, np.ones(len(ijk))] @ np.asarray(img.affine, float).T
        supports = []
        for value in parcel_labels:
            sel = lab == int(value)
            if not sel.any():
                raise ValueError(f"parcel label {value} is absent from parcel_image.")
            supports.append(world_all[sel, :3])
        centres = np.array([s.mean(axis=0) for s in supports])
        coords = centres if coords is None else _check_coords(coords, len(supports))
        return supports, coords
    if coords is None:
        raise ValueError(f"support={support!r} needs coords.")
    coords = _check_coords(coords, None)
    if support == "point":
        return [c[None, :] for c in coords], coords
    if radius_mm is None:
        raise ValueError("support='sphere' needs radius_mm (the extraction sphere radius).")
    radii = np.broadcast_to(np.asarray(radius_mm, float), (len(coords),))
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("radius_mm must be positive and finite.")
    return [ref.sphere(c, rad) for c, rad in zip(coords, radii)], coords


def _check_coords(coords, n):
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords must have shape (n_rois, 3); got {c.shape}.")
    if not np.all(np.isfinite(c)):
        raise ValueError("coords must be finite.")
    if n is not None and len(c) != n:
        raise ValueError(f"coords must have {n} rows (one per parcel); got {len(c)}.")
    return c


def _nearest_labelled(ref, point, max_mm):
    from scipy.spatial import cKDTree

    tree = getattr(ref, "_tree", None)
    if tree is None:
        ijk = np.argwhere(ref.labels > 0)
        world = (np.c_[ijk, np.ones(len(ijk))] @ ref.affine.T)[:, :3]
        tree = cKDTree(world)
        ref._tree = tree
    d, k = tree.query(point)
    if d > max_mm:
        return None
    return tree.data[k], float(d)


def _shares(w_lab, keys):
    """Largest summed share over groups of regions and its key."""
    out: Dict[Any, float] = {}
    for k, v in zip(keys, w_lab):
        if v > 0:
            out[k] = out.get(k, 0.0) + float(v)
    if not out:
        return None, 0.0
    key = max(out, key=out.get)
    return key, out[key]


def _describe(r, name, centre, world, w, m, regions, ref, support, radius, space,
              purity, min_coverage, min_weight, nearest):
    ids = np.arange(1, N_REGIONS + 1)
    coverage = float(w.sum())
    flags: List[str] = []
    if space in _APPROX_SPACES:
        flags.append(f"approximate space ({space} vs FSL MNI152, ~1-3 mm)")
    if support == "point":
        flags.append("point estimate (support unknown)")
    if nearest is not None:
        flags.append(f"nearest labelled voxel {nearest:.1f} mm")
    row: Dict[str, Any] = {
        "roi": name, "x": centre[0], "y": centre[1], "z": centre[2], "support": support,
        "radius_mm": float(radius) if support == "sphere" else np.nan, "n_voxels": len(world),
        "coverage": coverage, "level": "unassigned", "assigned": "unassigned", "region": "",
        "region_purity": np.nan, "gyrus": "", "lobe": "", "hemisphere": "", "structure": "",
        "description": "", "yeo7": "", "yeo7_purity": np.nan, "yeo17": "", "regions": "",
        "membership": "", "domains": "", "profile_status": "unassigned",
    }
    if m is not None and m.sum() > 0:
        top = np.argsort(m)[::-1][:3]
        row["membership"] = "; ".join(f"{regions.at[int(k) + 1, 'name']} {m[k]:.2f}" for k in top if m[k] > 0)
    if len(world) == 0 or coverage < min_coverage:
        flags.append("mostly unlabelled: white matter, CSF, or outside BNA-246 "
                     "(cerebellum, brainstem, outside the brain)")
        x = centre[0]
        row["hemisphere"] = "M" if abs(x) < 2 else ("L" if x < 0 else "R")
        row["flags"] = "; ".join(flags)
        return row
    if coverage < 0.75:
        flags.append(f"partial volume: {1 - coverage:.0%} of the support unlabelled")

    w_lab = w / coverage
    top = np.argsort(w)[::-1]
    row["regions"] = "; ".join(f"{regions.at[int(k) + 1, 'name']} {w[k]:.2f}" for k in top[:4] if w[k] >= 0.05)
    dom = int(top[0]) + 1
    sub_p = float(w_lab[dom - 1])
    hemi_l = float(w_lab[(ids % 2) == 1].sum())
    hemi = "L" if hemi_l >= 0.8 else ("R" if hemi_l <= 0.2 else "bilateral")
    if hemi == "bilateral":
        flags.append(f"bilateral support (left {hemi_l:.0%})")
    gy_key, gy_p = _shares(w_lab, [f"{g}_{h}" for g, h in zip(regions.gyrus, regions.hemisphere)])
    lo_key, lo_p = _shares(w_lab, [f"{lb}_{h}" for lb, h in zip(regions.lobe, regions.hemisphere)])
    st_key, st_p = _shares(w_lab, list(regions.structure))
    yeo_keys = [y if s == "cortex" else "not applicable" for y, s in zip(regions.yeo7, regions.structure)]
    y_key, y_p = _shares(w_lab, yeo_keys)
    y17_key, _ = _shares(w_lab, [y if s == "cortex" else "not applicable"
                                 for y, s in zip(regions.yeo17, regions.structure)])

    if sub_p >= purity:
        level, assigned = "subregion", regions.at[dom, "name"]
    elif gy_p >= purity:
        level, assigned = "gyrus", gy_key
    elif lo_p >= purity:
        level, assigned = "lobe", lo_key
    else:
        level, assigned = "mixed", "mixed"
        flags.append("mixed: no lobe holds the purity threshold")
    row.update({
        "level": level, "assigned": assigned, "region": regions.at[dom, "name"], "region_purity": sub_p,
        "gyrus": gy_key.rsplit("_", 1)[0] if level in ("subregion", "gyrus") else "",
        "lobe": lo_key.rsplit("_", 1)[0] if level in ("subregion", "gyrus", "lobe") else "",
        "hemisphere": hemi, "structure": st_key if st_p >= purity else "mixed",
        "description": regions.at[dom, "description"] if level == "subregion" else "",
        "yeo7": y_key if y_p >= purity else "mixed", "yeo7_purity": y_p,
        "yeo17": y17_key if y_p >= purity else "mixed",
    })
    if ref.profiles is not None:
        contrib = [int(k) + 1 for k in np.flatnonzero(w_lab >= min_weight)]
        terms = set().union(*(ref.profile("BD", c).keys() for c in contrib)) if contrib else set()
        row["domains"] = "; ".join(d for d in BD_DOMAINS if d in {ref.domain(t) for t in terms})
        row["profile_status"] = "terms" if terms else "no published terms (BrainMap coverage, not absence of function)"
    else:
        row["profile_status"] = "profiles unavailable"
    row["flags"] = "; ".join(flags)
    return row


# ----------------------------------------------------------------------
# Connections
# ----------------------------------------------------------------------
def annotate_edges(edges: Any, rois: RoiAnnotation, *, smoothing_fwhm_mm: Optional[float] = None) -> Any:
    """Add endpoint anatomy and endpoint relations to an edge table.

    ``edges`` needs ``roi_i`` / ``roi_j`` (matrix indices), e.g. from
    ``result.significant_edges(...)``; its columns (statistics, p-values) are
    kept unchanged. Added columns:

    * ``i_bna`` / ``j_bna`` (assignment at the finest pure level), ``i_lobe``,
      ``j_lobe``, ``i_yeo7``, ``j_yeo7``, ``i_domains``, ``j_domains``;
    * ``hemispheric_relation``: left / right intrahemispheric, homotopic
      (homologous subregions), homotopic gyrus, heterotopic, or
      bilateral/midline endpoint;
    * ``anatomical_relation``: same subregion, same gyrus, same lobe, between
      lobes (hemisphere ignored; see the hemispheric relation), or mixed /
      unassigned;
    * ``circuit``: cortico-cortical, thalamo-cortical, cortico-basal ganglia,
      amygdalo-cortical, hippocampo-cortical, or the subcortical pair;
    * ``distance_mm`` between the ROI centres;
    * ``shared_support``: share of the smaller sphere shared with the other
      (> 0 means overlapping spheres: the edge partly reflects the same
      voxels, not coupling between regions), and with ``smoothing_fwhm_mm``
      ``within_smoothing``: centres closer than r_i + r_j + FWHM;
    * ``sc``: support-weighted population structural connectivity (NaN when
      the tractography maps are unavailable; low values do not show absence
      of a pathway);
    * ``network_relation``: within <Yeo-7 network> / between / n/a;
    * ``shared_domains``: BrainMap domains present at both endpoints
      (descriptive context, not the function of the connection).
    """
    df = edges.copy()
    if df.empty:
        return df
    t = rois.table
    i = df["roi_i"].to_numpy(int)
    j = df["roi_j"].to_numpy(int)
    for side, idx in (("i", i), ("j", j)):
        df[f"{side}_bna"] = t["assigned"].to_numpy()[idx]
        df[f"{side}_lobe"] = t["lobe"].to_numpy()[idx]
        df[f"{side}_yeo7"] = t["yeo7"].to_numpy()[idx]
        df[f"{side}_domains"] = t["domains"].to_numpy()[idx]
    ref = rois.reference
    regions = ref.regions
    rel_h, rel_a, circ, net, shared = [], [], [], [], []
    for a, b in zip(i, j):
        ra, rb = t.iloc[a], t.iloc[b]
        rel_h.append(_hemispheric(ra, rb, regions))
        rel_a.append(_anatomical(ra, rb))
        circ.append(_circuit(ra["structure"], rb["structure"]))
        net.append(_network(ra["yeo7"], rb["yeo7"]))
        da = set(filter(None, str(ra["domains"]).split("; ")))
        db = set(filter(None, str(rb["domains"]).split("; ")))
        shared.append("; ".join(d for d in BD_DOMAINS if d in da & db))
    df["hemispheric_relation"], df["anatomical_relation"] = rel_h, rel_a
    df["circuit"], df["network_relation"], df["shared_domains"] = circ, net, shared
    xyz = t[["x", "y", "z"]].to_numpy(float)
    d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    df["distance_mm"] = d
    rad = np.nan_to_num(t["radius_mm"].to_numpy(float), nan=0.0)
    df["shared_support"] = [_lens_share(di, rad[a], rad[b]) for di, a, b in zip(d, i, j)]
    if smoothing_fwhm_mm is not None:
        df["within_smoothing"] = d < rad[i] + rad[j] + float(smoothing_fwhm_mm)
    sc = ref.structural_connectivity()
    if sc is None:
        df["sc"] = np.nan
    else:
        w = np.nan_to_num(rois.labelled_weights())
        S = np.nan_to_num(sc, nan=0.0)
        off = 1.0 - np.eye(N_REGIONS)
        num = np.einsum("ea,ab,eb->e", w[i], S, w[j])
        den = np.einsum("ea,ab,eb->e", w[i], off, w[j])
        with np.errstate(invalid="ignore", divide="ignore"):
            df["sc"] = np.where(den > 0, num / den, np.nan)
    return df


def _hemispheric(ra, rb, regions) -> str:
    ha, hb = ra["hemisphere"], rb["hemisphere"]
    if ha not in ("L", "R") or hb not in ("L", "R"):
        return "bilateral or midline endpoint"
    if ha == hb:
        return "left intrahemispheric" if ha == "L" else "right intrahemispheric"
    if ra["level"] == rb["level"] == "subregion":
        ia = int(regions.index[regions.name == ra["region"]][0])
        if int(regions.at[ia, "homolog_id"]) == int(regions.index[regions.name == rb["region"]][0]):
            return "homotopic"
    if ra["gyrus"] and ra["gyrus"] == rb["gyrus"]:
        return "homotopic gyrus"
    return "heterotopic"


def _anatomical(ra, rb) -> str:
    la, lb = ra["level"], rb["level"]
    if "unassigned" in (la, lb):
        return "unassigned"
    if "mixed" in (la, lb):
        return "mixed"
    if la == lb == "subregion" and ra["region"][:-2] == rb["region"][:-2]:
        return "same subregion"
    if ra["gyrus"] and ra["gyrus"] == rb["gyrus"]:
        return "same gyrus"
    if ra["lobe"] == rb["lobe"]:
        return "same lobe"
    return "between lobes"


def _circuit(sa: str, sb: str) -> str:
    if not sa or not sb or "mixed" in (sa, sb):
        return "unassigned or mixed"
    key = frozenset({sa, sb})
    if key in _CIRCUITS:
        return _CIRCUITS[key]
    return "subcortical: " + "–".join(sorted(s.replace("_", " ") for s in (sa, sb)))


def _network(a: str, b: str) -> str:
    bad = ("", "mixed", "not applicable")
    if a in bad or b in bad:
        return "n/a"
    return f"within {a}" if a == b else "between"


def _lens_share(d: float, r1: float, r2: float) -> float:
    """Shared volume of two spheres as a share of the smaller one (0 for points)."""
    if r1 <= 0 or r2 <= 0 or d >= r1 + r2:
        return 0.0
    small = min(r1, r2)
    if d <= abs(r1 - r2):
        return 1.0
    lens = np.pi * (r1 + r2 - d) ** 2 * (d * d + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * d)
    return float(lens / (4.0 / 3.0 * np.pi * small ** 3))


# ----------------------------------------------------------------------
# Descriptive summaries
# ----------------------------------------------------------------------
def summarize_annotation(
    result: Any,
    rois: RoiAnnotation,
    *,
    alpha: float = 0.05,
    tail: str = "both",
    param_idx: Optional[int] = None,
    level: str = "lobe",
    distance_bands: Sequence[float] = (30.0, 60.0, 90.0),
    smoothing_fwhm_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """Describe where the detected connections lie, against what was tested.

    Every table puts the detections next to the tested opportunities, because
    composition alone misleads when categories differ in size (there are far
    more cortico-cortical than thalamo-cortical pairs). Nothing is tested: no
    enrichment p-values, and the original correction and inference unit
    (``result.inference_unit``: edge, component or block) are unchanged.

    Returns a dict of DataFrames:

    * ``relations``: by circuit, hemispheric relation, anatomical relation,
      distance band and network relation: n_tested, n_detected, share of
      detections, share of tested, detection rate;
    * ``regions``: per group at ``level``: n_rois, tested and detected
      incident connections, detection rate (ROIs not pure at that level are
      grouped as 'mixed');
    * ``domains``: per BrainMap domain, unique ROIs among detected endpoints
      vs among all ROIs (memberships overlap: shares need not sum to 1);
    * ``distance_qc``: Spearman correlation of the signed and absolute
      statistic with distance over all tested connections, and the share of
      detections vs tested connections shorter than the first band. A
      contrast that covaries with head motion often shows short-range
      increases and long-range decreases (Power et al. 2012).
    """
    import pandas as pd
    from scipy import stats

    edges = result.significant_edges(alpha=alpha, tail="both", include_nonsig=True, param_idx=param_idx)
    edges = edges[np.isfinite(edges["p_min"].to_numpy(float))].reset_index(drop=True)
    ann = annotate_edges(edges, rois, smoothing_fwhm_mm=smoothing_fwhm_mm)
    if tail == "both":
        det = (ann["p_positive"] <= alpha) | (ann["p_negative"] <= alpha)
    elif tail in ("positive", "negative"):
        det = ann[f"p_{tail}"] <= alpha
    else:
        raise ValueError("tail must be 'both', 'positive' or 'negative'.")
    det = det.to_numpy(bool)
    bands = np.r_[0.0, np.asarray(distance_bands, float), np.inf]
    labels = [f"{int(a)}-{int(b)} mm" if np.isfinite(b) else f">{int(a)} mm" for a, b in zip(bands[:-1], bands[1:])]
    ann["distance_band"] = pd.cut(ann["distance_mm"], bands, labels=labels, right=False).astype(str)

    rows = []
    n_tested, n_det = len(ann), int(det.sum())
    for col in ("circuit", "hemispheric_relation", "anatomical_relation", "distance_band", "network_relation"):
        for value, g in ann.groupby(col):
            k = int(det[g.index].sum())
            rows.append({"relation": col, "category": value, "n_tested": len(g), "n_detected": k,
                         "share_of_detections": k / n_det if n_det else np.nan,
                         "share_of_tested": len(g) / n_tested, "detection_rate": k / len(g)})
    relations = pd.DataFrame(rows)

    groups = np.asarray(rois.grouping(level), dtype=object)
    i, j = ann["roi_i"].to_numpy(int), ann["roi_j"].to_numpy(int)
    reg_rows = []
    for g in pd.unique(groups):
        members = groups == g
        inc = members[i] | members[j]
        reg_rows.append({"group": g, "n_rois": int(members.sum()), "n_tested": int(inc.sum()),
                         "n_detected": int((inc & det).sum()),
                         "detection_rate": float((inc & det).sum() / inc.sum()) if inc.any() else np.nan})
    regions = pd.DataFrame(reg_rows).sort_values("group").reset_index(drop=True)

    det_rois = np.unique(np.r_[i[det], j[det]])
    dom = rois.table["domains"].fillna("").map(lambda s: set(filter(None, s.split("; "))))
    status = rois.table["profile_status"].to_numpy()
    dom_rows = []
    for d in BD_DOMAINS:
        has = dom.map(lambda s: d in s).to_numpy(bool)
        dom_rows.append({"domain": d, "n_detected_endpoint_rois": int(has[det_rois].sum()),
                         "share_detected_endpoint_rois": float(has[det_rois].mean()) if det_rois.size else np.nan,
                         "n_all_rois": int(has.sum()), "share_all_rois": float(has.mean())})
    for label, sel in (("no published terms", np.char.startswith(status.astype(str), "no published")),
                       ("unassigned", status == "unassigned")):
        dom_rows.append({"domain": label, "n_detected_endpoint_rois": int(sel[det_rois].sum()),
                         "share_detected_endpoint_rois": float(sel[det_rois].mean()) if det_rois.size else np.nan,
                         "n_all_rois": int(sel.sum()), "share_all_rois": float(sel.mean())})
    domains = pd.DataFrame(dom_rows)

    stat = ann["t_signed"].to_numpy(float)
    dist = ann["distance_mm"].to_numpy(float)
    short = dist < bands[1]
    qc = pd.DataFrame([{
        "spearman_signed_vs_distance": float(stats.spearmanr(stat, dist).statistic) if n_tested > 2 else np.nan,
        "spearman_abs_vs_distance": float(stats.spearmanr(np.abs(stat), dist).statistic) if n_tested > 2 else np.nan,
        "short_range_mm": float(bands[1]),
        "share_detections_short": float(short[det].mean()) if n_det else np.nan,
        "share_tested_short": float(short.mean()),
        "n_overlapping_supports": int((ann["shared_support"] > 0).sum()),
    }])
    meta = {"alpha": alpha, "tail": tail, "level": level, "n_tested": n_tested, "n_detected": n_det,
            "inference_unit": getattr(result, "inference_unit", "edge"),
            "note": "descriptive; no enrichment test; original correction unchanged"}
    return {"relations": relations, "regions": regions, "domains": domains, "distance_qc": qc,
            "edges": ann, "meta": meta}


__all__ = ["RoiAnnotation", "annotate_rois", "annotate_edges", "summarize_annotation"]
