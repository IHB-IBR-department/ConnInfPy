"""Brainnetome Atlas (BNA-246) reference data for annotating connectivity results.

The Brainnetome atlas (Fan et al., 2016) divides cortex and subcortex into 246
subregions defined by connectivity, with an anatomical hierarchy (subregion ->
gyrus or nucleus -> lobe), homologous left/right pairs (odd ID = left, the next
even ID = its right homolog), cytoarchitectonic descriptions, a crosswalk to the
Yeo 7/17 cortical networks, population structural-connectivity maps and
BrainMap behavioural-domain (BD) and paradigm-class (PC) profiles.

What is bundled and what is loaded
----------------------------------
* bundled: maximum-probability label images (1 and 2 mm, FSL MNI152 grid) and
  the hierarchy table ``bna246_hierarchy.csv``;
* bundled: the BrainMap profiles (BDf_FDR05.json, PCf_FDR05.json;
  :func:`conninfpy.atlas.fetch_bna_behavioral_decoding`);
* local files, optional, looked up in ``data_dir``: ``BNA_PM_4D.nii.gz``
  (probabilistic maps, 40 subjects) and ``BNA_SC_3D_246.zip`` (population
  tractography maps), from atlas.brainnetome.org/download.html.

BrainMap profiles are forward-inference likelihood ratios that passed FDR
q < 0.05 in the BrainMap database: they describe which kinds of tasks activate
a region in the literature, not what a region or a connection does. A region
without terms has too few BrainMap experiments there, not "no function".

Brainnetome data are distributed for non-commercial use with attribution and
share-alike (``conninfpy/atlas_data/NOTICE_BRAINNETOME.txt``). Cite Fan et al.
(2016), and Laird et al. (2005, 2011) for the profiles.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union, cast

if TYPE_CHECKING:  # nibabel is imported lazily at runtime; only the stub type is needed here
    from nibabel.spatialimages import SpatialImage

import numpy as np
import numpy.typing as npt

from conninfpy.atlas.resources import (
    _default_cache,
    _sha256,
    brainnetome_data_dir,
    brainnetome_resource_path,
    read_bna_hierarchy,
)

logger = logging.getLogger(__name__)

PM_FILE = "BNA_PM_4D.nii.gz"
SC_FILE = "BNA_SC_3D_246.zip"
N_REGIONS = 246
BD_DOMAINS = ("Action", "Cognition", "Emotion", "Interoception", "Perception")

_YEO7_NAMES = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
}
_ANATOM_MACRO_LOBE = {
    1: ("Anterior", "Anterior Lobe"),
    2: ("Anterior", "Anterior Lobe"),
    3: ("Anterior", "Anterior Lobe"),
    4: ("Anterior", "Anterior Lobe"),
    26: ("Flocculonodular", "Flocculonodular Lobe"),
    28: ("Flocculonodular", "Flocculonodular Lobe"),
}


def _spatial_image(img: Any) -> "SpatialImage":
    """``nib.load`` result narrowed to the spatial image every caller reads.

    ``nibabel.load`` is typed as returning the ``FileBasedImage`` base class,
    which has no ``shape`` / ``affine`` / ``dataobj``; the atlas, label and
    probabilistic-map images loaded here always are spatial images. The cast
    documents that and has no runtime effect.
    """
    return cast("SpatialImage", img)


def _lattice_mapping(src: Any, dst: Any) -> tuple[npt.NDArray[np.float64], bool]:
    """Voxel->voxel matrix M = inv(affine_dst) @ affine_src and exact-lattice flag."""
    M = np.linalg.inv(np.asarray(dst.affine, float)) @ np.asarray(src.affine, float)
    R, t = M[:3, :3], M[:3, 3]
    perm = (
        np.all(np.abs(R).sum(0) == 1)
        and np.all(np.abs(R).sum(1) == 1)
        and np.allclose(np.abs(R[R != 0]), 1)
    )
    return M, bool(perm and np.allclose(t, np.rint(t), atol=1e-6))


def _gather_exact(src: Any, dst: Any) -> npt.NDArray[np.int64]:
    """Exact nearest-neighbour gather of ``src`` onto ``dst`` (requires exact lattice)."""
    M, exact = _lattice_mapping(src, dst)
    if not exact:
        raise ValueError("grids do not coincide exactly; resampling is not allowed here")
    sh = np.asarray(src.shape[:3])
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in sh], indexing="ij")
    hom = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
    r = np.rint((M @ hom)[:3].T).astype(int)
    inside = np.all((r >= 0) & (r < np.asarray(dst.shape[:3])), axis=1)
    out = np.zeros(dst.shape[:3], dtype=np.int64)
    d = np.asanyarray(src.dataobj).ravel()
    sel = inside & (d != 0)
    out[r[sel, 0], r[sel, 1], r[sel, 2]] = d[sel]
    return out


def _crosswalk(
    cb_arr: npt.NDArray[np.int64],
    net_arr: npt.NDArray[np.int64],
    ids: Sequence[int],
    threshold: float = 0.5,
) -> Dict[int, Dict[str, Any]]:
    """Per cerebellar region: label distribution, top label, purity, status."""
    out: Dict[int, Dict[str, Any]] = {}
    for i in sorted(ids):
        m = cb_arr == i
        n = int(m.sum())
        vals, cnts = np.unique(net_arr[m], return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, cnts, strict=True)}
        assigned = n - dist.get(0, 0)
        ranked = sorted(((c, -v) for v, c in dist.items() if v != 0), reverse=True)
        top = -ranked[0][1] if ranked else None
        purity = (ranked[0][0] / assigned) if assigned else None
        status = (
            "unassigned"
            if assigned == 0 or purity is None
            else ("dominant" if purity >= threshold else "mixed")
        )
        out[i] = {"top_label": top, "purity_of_assigned": purity, "status": status}
    return out


@dataclass
class BrainnetomeReference:
    """BNA-246 anatomy, profiles and optional probabilistic / structural maps.

    Use :meth:`load`. ``regions`` is the hierarchy table indexed by the
    original BNA ID (1-246); ``label_image`` the maximum-probability map.
    """

    regions: Any  # pandas.DataFrame indexed by roi_id
    label_image: Any  # nibabel image (MPM)
    data_dir: Optional[Path] = None
    cache_dir: Path = field(default_factory=_default_cache)
    profiles: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    _labels: Optional[npt.NDArray[np.int64]] = field(default=None, repr=False)
    _pm: Any = field(default=None, repr=False)
    _sc: Optional[npt.NDArray[np.float64]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        data_dir: Optional[Union[str, Path]] = None,
        *,
        resolution_mm: int = 1,
        profiles: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        cerebellum: Optional[str] = None,
    ) -> "BrainnetomeReference":
        """Load the bundled atlas and, when available, the BrainMap profiles.

        Parameters
        ----------
        data_dir : path, optional
            Folder holding the optional local files (``BNA_PM_4D.nii.gz``,
            ``BNA_SC_3D_246.zip``). Defaults to ``$CONNINFPY_BNA_DIR`` or
            ``$CONNINFPY_DATA/brainnetome`` (``~/conninfpy_data/brainnetome``).
            Originals are checked against the bundled source manifest when used.
        resolution_mm : {1, 2}
            Bundled label image to use (1 mm by default: spheres of a few
            millimetres need it).
        profiles : bool
            Read the bundled BrainMap BD/PC snapshots without network access.
            If unavailable, annotation proceeds without profiles.
        cache_dir : path, optional
            Cache for derived PM/SC arrays (default
            ``$CONNINFPY_CACHE/bna246`` or ``~/.cache/conninfpy/bna246``).
        cerebellum : {None, 'anatom'}, optional
            Optional opt-in cerebellar companion layer (DiedrichsenLab v1.2
            SUIT ``atl-Anatom`` with Buckner-7/17 crosswalk). Currently
            supported only at ``resolution_mm=1`` (1 mm MNI152 lattice).
            ``None`` (default) keeps pure 246-region BNA-246.
        """
        import nibabel as nib

        from conninfpy.atlas.resources import (
            bna_profile_paths,
            fetch_bna_behavioral_decoding,
            get_bna_246_nifti_path,
        )

        if cerebellum not in (None, "anatom"):
            raise ValueError(f"cerebellum must be None or 'anatom'; got {cerebellum!r}.")
        if cerebellum is not None and resolution_mm != 1:
            raise ValueError(
                f"cerebellum={cerebellum!r} requires resolution_mm=1 (1 mm MNI lattice)."
            )

        regions = read_bna_hierarchy()
        regions["atlas_source"] = "BNA-246"
        img_path = get_bna_246_nifti_path(resolution_mm)
        img = nib.load(str(img_path))
        cache = Path(cache_dir) if cache_dir is not None else _default_cache()
        provenance: Dict[str, Any] = {
            "atlas": "Brainnetome BNA-246 (Fan et al. 2016)",
            "label_image": f"{img_path.name} (FSL MNI152 grid, {resolution_mm} mm)",
            "label_image_sha256": _sha256(img_path),
            "hierarchy": "bna246_hierarchy.csv (BNA_subregions.xlsx Table 1 + Yeo crosswalk)",
            "data_dir": str(brainnetome_data_dir(data_dir)) if brainnetome_data_dir(data_dir) else None,
        }
        if cerebellum == "anatom":
            regions, img, cb_prov = _attach_suit_anatom(regions, img)
            provenance.update(cb_prov)

        ref = cls(
            regions=regions,
            label_image=img,
            data_dir=brainnetome_data_dir(data_dir),
            cache_dir=cache,
        )
        ref.provenance = provenance
        if profiles:
            try:
                bd, pc = fetch_bna_behavioral_decoding()
                ref.profiles = {
                    "BD": {int(k): dict(v) for k, v in bd.items()},
                    "PC": {int(k): dict(v) for k, v in pc.items()},
                }
                ref.provenance["profiles"] = {
                    "source": "BrainMap forward inference, FDR q<0.05 (BDf_FDR05.json, PCf_FDR05.json)",
                    "files": f"local cache ({bna_profile_paths()[0].parent})",
                    "BD_sha256": _sha256(bna_profile_paths()[0]),
                    "PC_sha256": _sha256(bna_profile_paths()[1]),
                }
            except (IOError, OSError) as exc:
                logger.warning(
                    "BrainMap profiles unavailable (%s); annotating without them.", exc
                )
        return ref

    # ------------------------------------------------------------------
    @property
    def n_regions(self) -> int:
        """Total number of regions in the reference (246 for pure BNA-246)."""
        return len(self.regions)

    @property
    def labels(self) -> npt.NDArray[np.int64]:
        """Label array of the MPM image (0 = unlabelled: white matter, CSF, outside)."""
        if self._labels is None:
            self._labels = np.rint(np.asarray(self.label_image.dataobj)).astype(
                np.int64
            )
        return self._labels

    @property
    def affine(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.label_image.affine, dtype=np.float64)

    def names(self, ids: Sequence[int]) -> list:
        return [
            self.regions.at[int(i), "name"] if int(i) > 0 else "unlabelled" for i in ids
        ]

    def labels_at(self, world: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Label at each world (mm) coordinate; 0 outside the image or unlabelled."""
        world = np.atleast_2d(np.asarray(world, dtype=np.float64))
        vox = np.rint(np.c_[world, np.ones(len(world))] @ np.linalg.inv(self.affine).T)[
            :, :3
        ].astype(int)
        shape = np.array(self.labels.shape)
        inside = np.all((vox >= 0) & (vox < shape), axis=1)
        out = np.zeros(len(world), dtype=np.int64)
        out[inside] = self.labels[tuple(vox[inside].T)]
        return out

    def sphere(
        self, center: npt.ArrayLike, radius_mm: float
    ) -> npt.NDArray[np.float64]:
        """World coordinates of the label-image voxels whose centres lie within ``radius_mm``."""
        return _sphere_on_grid(
            np.asarray(center, float), float(radius_mm), self.affine, self.labels.shape
        )

    # ------------------------------------------------------------------
    def homolog(self, roi_id: int) -> Optional[int]:
        val = self.regions.at[int(roi_id), "homolog_id"]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return int(val)

    def profile(self, kind: str, roi_id: int) -> Optional[Dict[str, float]]:
        """BrainMap profile ``{term: likelihood ratio}`` of a region; None without profiles or for companion atlases."""
        if self.profiles is None or int(roi_id) > N_REGIONS:
            return None
        return dict(self.profiles[kind].get(int(roi_id), {}))

    @staticmethod
    def domain(term: str) -> str:
        """Top-level BrainMap behavioural domain of a BD term (``Cognition.Memory.Working`` -> ``Cognition``)."""
        return term.split(".", 1)[0]

    # ------------------------------------------------------------------
    def membership(self, world: npt.ArrayLike) -> Optional[npt.NDArray[np.float64]]:
        """Mean probabilistic-map membership of each region over the given points.

        Returns an array of shape (246,) in [0, 1] (the share of the 40
        reference subjects in whom the points belong to each region, averaged
        over the points), or None when ``BNA_PM_4D.nii.gz`` is not available.
        """
        pm = self._membership_matrix()
        if pm is None:
            return None
        mat, affine, shape = pm
        world = np.atleast_2d(np.asarray(world, dtype=np.float64))
        vox = np.rint(np.c_[world, np.ones(len(world))] @ np.linalg.inv(affine).T)[
            :, :3
        ].astype(int)
        inside = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
        if not inside.any():
            return np.zeros(self.n_regions)
        flat = np.ravel_multi_index(tuple(vox[inside].T), shape)
        vals = np.asarray(mat[flat].sum(axis=0)).ravel() / len(world)
        if self.n_regions > N_REGIONS:
            out = np.zeros(self.n_regions, dtype=vals.dtype)
            out[:N_REGIONS] = vals
            return out
        return vals

    def _membership_matrix(self):
        if self._pm is not None:
            return self._pm
        if self.data_dir is None:
            return None
        path = brainnetome_resource_path(PM_FILE, data_dir=self.data_dir)
        if path is None:
            return None
        import nibabel as nib
        from scipy import sparse

        digest = _sha256(path)
        cache = self.cache_dir / f"pm_sparse_{digest[:16]}.npz"
        img = _spatial_image(nib.load(str(path)))
        shape, affine = img.shape[:3], np.asarray(img.affine, float)
        if cache.exists():
            mat = sparse.load_npz(cache)
        else:
            mat = _build_membership_cache(path, img, cache)
        self._pm = (mat, affine, shape)
        self.provenance["probabilistic_maps"] = {
            "file": PM_FILE,
            "sha256": digest,
            "grid": f"{shape}, {abs(affine[0, 0])} mm (HCP-40 MNI152)",
        }
        return self._pm

    # ------------------------------------------------------------------
    def structural_connectivity(self) -> Optional[npt.NDArray[np.float64]]:
        """Population structural connectivity between BNA regions, (246, 246), or None.

        Derived from the Brainnetome tractography maps (``BNA_SC_3D_246.zip``:
        for each seed region, the share of reference subjects whose
        probabilistic tractography reaches each voxel). Entry (a, b) is the
        mean of seed a's map within region b, averaged with (b, a); the
        diagonal is NaN. This is the population probability of a white-matter
        pathway, with tractography's limits: long association fibres and
        lateral callosal projections are under-detected, so a low value does
        not show that two regions are unconnected.
        """
        if self._sc is not None:
            return self._sc
        if self.data_dir is None:
            return None
        path = brainnetome_resource_path(SC_FILE, data_dir=self.data_dir)
        if path is None:
            return None
        import nibabel as nib

        digest = _sha256(path)
        cache = self.cache_dir / f"sc_derived_{digest[:16]}.npy"
        if cache.exists():
            sc = np.load(cache)
        else:
            from conninfpy.atlas.resources import get_bna_246_nifti_path

            lab_img = _spatial_image(nib.load(str(get_bna_246_nifti_path(2))))
            lab = np.rint(np.asarray(lab_img.dataobj)).astype(np.int64).ravel()
            keep = lab > 0
            counts = np.bincount(lab[keep], minlength=N_REGIONS + 1)[1:]
            sc = np.zeros((N_REGIONS, N_REGIONS))
            with zipfile.ZipFile(path) as zf:
                for s in range(N_REGIONS):
                    with zf.open(f"{s + 1:03d}.nii.gz") as fh:
                        seed = nib.Nifti1Image.from_bytes(_gunzip(fh.read()))
                    if seed.shape != lab_img.shape or not np.allclose(
                        seed.affine, lab_img.affine
                    ):
                        raise ValueError(
                            f"{SC_FILE}: map {s + 1} is not on the 2 mm BNA grid."
                        )
                    m = np.asarray(seed.dataobj, dtype=np.float64).ravel()
                    sc[s] = (
                        np.bincount(
                            lab[keep], weights=m[keep], minlength=N_REGIONS + 1
                        )[1:]
                        / counts
                    )
            sc = (sc + sc.T) / 2.0
            np.fill_diagonal(sc, np.nan)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, sc)
        self._sc = sc
        self.provenance["structural_connectivity"] = {
            "file": SC_FILE,
            "sha256": digest,
            "derivation": "mean seed-map probability within the target region, symmetrized",
        }
        if self.n_regions > N_REGIONS:
            full = np.full((self.n_regions, self.n_regions), np.nan, dtype=np.float64)
            full[:N_REGIONS, :N_REGIONS] = sc
            self._sc = full
        return self._sc


def _build_membership_cache(path: Path, img: Any, cache: Path):
    """Sparse (voxels x regions) membership matrix from the 4-D probabilistic maps.

    Reading volume k of a gzipped 4-D image re-decompresses every volume
    before it, so the file is decompressed once into a temporary
    uncompressed NIfTI that is memory-mapped and read volume by volume.
    """
    import gzip
    import shutil
    import tempfile

    import nibabel as nib
    from scipy import sparse

    logger.warning(
        "Building the sparse probabilistic-map cache once (decompresses ~3.6 GB to a temporary file) ..."
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    shape = img.shape[:3]
    with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
        raw = Path(tmp) / "pm.nii"
        with gzip.open(path, "rb") as src, open(raw, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 24)
        mm = _spatial_image(nib.load(str(raw), mmap=True))
        rows, cols, vals = [], [], []
        for k in range(mm.shape[3]):
            v = np.asarray(mm.dataobj[..., k], dtype=np.float32)
            ijk = np.nonzero(v)
            rows.append(
                np.ravel_multi_index(ijk, shape)
            )  # same convention as membership()
            cols.append(np.full(ijk[0].size, k, dtype=np.int32))
            vals.append(v[ijk])
        del mm
    val_arr = np.concatenate(vals)
    scale = (
        100.0 if val_arr.max() > 1.0 + 1e-6 else 1.0
    )  # the official file stores percent
    mat = sparse.csr_matrix(
        (val_arr / scale, (np.concatenate(rows), np.concatenate(cols))),
        shape=(int(np.prod(shape)), img.shape[3]),
        dtype=np.float32,
    )
    sparse.save_npz(cache, mat)
    return mat


def _gunzip(data: bytes) -> bytes:
    import gzip

    return gzip.decompress(data)


def _sphere_on_grid(
    center: npt.NDArray[np.float64], radius: float, affine, shape
) -> npt.NDArray[np.float64]:
    """World coordinates of grid voxel centres within ``radius`` of ``center``."""
    inv = np.linalg.inv(affine)
    c_vox = (inv @ np.r_[center, 1.0])[:3]
    # Include lattice points outside the reference field of view: labels_at
    # assigns them to background, so coverage uses the whole sphere rather
    # than overstating overlap at an image boundary. Row norms also support
    # rotated and oblique affines.
    half = np.ceil(radius * np.linalg.norm(inv[:3, :3], axis=1)).astype(int) + 1
    lo = np.floor(c_vox).astype(int) - half
    hi = np.ceil(c_vox).astype(int) + half
    if np.any(hi < lo):
        return np.zeros((0, 3))
    grid = np.stack(
        np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi, strict=True)], indexing="ij"), -1
    ).reshape(-1, 3)
    world = np.c_[grid, np.ones(len(grid))] @ np.asarray(affine).T
    world = world[:, :3]
    return world[np.linalg.norm(world - center, axis=1) <= radius + 1e-9]


def _attach_suit_anatom(regions: Any, bna_img: Any) -> tuple[Any, Any, Dict[str, Any]]:
    """Append SUIT atl-Anatom (34 cerebellar regions) with BNA-246 priority."""
    import csv

    import nibabel as nib
    import pandas as pd

    from conninfpy.atlas.resources import atlas_resource_path

    cb_img = nib.load(str(atlas_resource_path("atl-Anatom_space-MNI_dseg.nii.gz")))
    b7_img = nib.load(str(atlas_resource_path("atl-Buckner7_space-MNI_dseg.nii.gz")))
    b17_img = nib.load(str(atlas_resource_path("atl-Buckner17_space-MNI_dseg.nii.gz")))

    cb_arr = _gather_exact(cb_img, bna_img)
    b7_arr = _gather_exact(b7_img, bna_img)
    b17_arr = _gather_exact(b17_img, bna_img)
    bna_arr = np.rint(np.asarray(bna_img.dataobj)).astype(np.int64)

    comp_arr = np.where(
        bna_arr > 0, bna_arr, np.where(cb_arr > 0, N_REGIONS + cb_arr, 0)
    ).astype(np.int64)
    clipped_voxels = int(((bna_arr > 0) & (cb_arr > 0)).sum())
    cerebellar_voxels = int((comp_arr > N_REGIONS).sum())

    comp_img = nib.Nifti1Image(comp_arr.astype(np.int16), bna_img.affine)
    comp_img.header.set_qform(bna_img.affine, code=1)
    comp_img.header.set_sform(bna_img.affine, code=1)

    with open(atlas_resource_path("atl-Anatom.tsv"), newline="") as fh:
        anatom_rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(atlas_resource_path("atl-Buckner17.tsv"), newline="") as fh:
        b17_names = {int(r["index"]): r["name"] for r in csv.DictReader(fh, delimiter="\t")}

    names_by_id = {int(r["index"]): r["name"] for r in anatom_rows}
    cw7 = _crosswalk(cb_arr, b7_arr, ids=list(names_by_id), threshold=0.5)
    cw17 = _crosswalk(cb_arr, b17_arr, ids=list(names_by_id), threshold=0.5)
    aff = np.asarray(bna_img.affine, dtype=np.float64)

    cb_records = []
    for idx, name in sorted(names_by_id.items()):
        roi_id = N_REGIONS + idx
        if name.startswith("Left_"):
            hemi = "L"
            other = "Right_" + name[len("Left_") :]
            hom_idx = next((j for j, m in names_by_id.items() if m == other), None)
        elif name.startswith("Right_"):
            hemi = "R"
            other = "Left_" + name[len("Right_") :]
            hom_idx = next((j for j, m in names_by_id.items() if m == other), None)
        elif name.startswith("Vermis_"):
            hemi = "M"
            hom_idx = None
        else:
            hemi = "M"
            hom_idx = None
        homolog_id = (N_REGIONS + hom_idx) if hom_idx is not None else np.nan

        if idx in _ANATOM_MACRO_LOBE:
            gyrus, gyrus_name = _ANATOM_MACRO_LOBE[idx]
        elif name.startswith("Vermis_"):
            gyrus, gyrus_name = "Vermis", "Cerebellar Vermis"
        elif idx >= 29:
            gyrus, gyrus_name = "Deep Nuclei", "Deep Cerebellar Nuclei"
        else:
            gyrus, gyrus_name = "Posterior", "Posterior Lobe"

        r7 = cw7[idx]
        if r7["status"] == "dominant" and r7["top_label"] is not None:
            y7_id = int(r7["top_label"])
            y7_name = _YEO7_NAMES.get(y7_id, f"Network{y7_id}")
        elif r7["status"] == "mixed":
            y7_id, y7_name = 0, "mixed"
        else:
            y7_id, y7_name = 0, "not applicable"

        r17 = cw17[idx]
        if r17["status"] == "dominant" and r17["top_label"] is not None:
            y17_id = int(r17["top_label"])
            y17_name = b17_names.get(y17_id, f"Network_{y17_id}")
        elif r17["status"] == "mixed":
            y17_id, y17_name = 0, "mixed"
        else:
            y17_id, y17_name = 0, "not applicable"

        ijk = np.argwhere(comp_arr == roi_id)
        if len(ijk) == 0:
            ijk = np.argwhere(cb_arr == idx)
        if len(ijk) > 0:
            xyz = (np.c_[ijk, np.ones(len(ijk))] @ aff.T)[:, :3].mean(axis=0)
            cx, cy, cz = float(xyz[0]), float(xyz[1]), float(xyz[2])
        else:
            cx, cy, cz = np.nan, np.nan, np.nan

        cb_records.append(
            {
                "roi_id": roi_id,
                "name": name,
                "hemisphere": hemi,
                "homolog_id": homolog_id,
                "code": f"SUIT_{idx:02d}",
                "description": f"SUIT cerebellar {name.replace('_', ' ')}",
                "gyrus": gyrus,
                "gyrus_name": gyrus_name,
                "lobe": "Cerebellum",
                "structure": "cerebellum",
                "yeo7_id": y7_id,
                "yeo7": y7_name,
                "yeo17_id": y17_id,
                "yeo17": y17_name,
                "x": cx,
                "y": cy,
                "z": cz,
                "atlas_source": "SUIT",
            }
        )

    cb_df = pd.DataFrame(cb_records).set_index("roi_id")
    combined = pd.concat([regions, cb_df], axis=0)
    prov = {
        "cerebellum_atlas": "DiedrichsenLab cerebellar_atlases v1.2 (atl-Anatom)",
        "clipped_voxels": clipped_voxels,
        "cerebellar_voxels": cerebellar_voxels,
        "cerebellum_n_regions": len(cb_records),
    }
    return combined, comp_img, prov


__all__ = ["BrainnetomeReference", "BD_DOMAINS"]
