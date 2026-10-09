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
* downloaded on first use and cached: the BrainMap profiles
  (:func:`conninfpy.atlas.fetch_bna_behavioral_decoding`);
* local files, optional, looked up in ``data_dir``: ``BNA_PM_4D.nii.gz``
  (probabilistic maps, 40 subjects) and ``BNA_SC_3D_246.zip`` (population
  tractography maps), from atlas.brainnetome.org/download.html.

BrainMap profiles are forward-inference likelihood ratios that passed FDR
q < 0.05 in the BrainMap database: they describe which kinds of tasks activate
a region in the literature, not what a region or a connection does. A region
without terms has too few BrainMap experiments there, not "no function".

Brainnetome data are distributed for non-commercial use with attribution and
share-alike (``conninfpy/atlas_data/NOTICE_BRAINNETOME.txt``). Cite Fan et al.
(2016), and Fox & Laird (2012) for the profiles.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

PM_FILE = "BNA_PM_4D.nii.gz"
SC_FILE = "BNA_SC_3D_246.zip"
N_REGIONS = 246
BD_DOMAINS = ("Action", "Cognition", "Emotion", "Interoception", "Perception")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_cache() -> Path:
    base = os.environ.get("CONNINFPY_CACHE", str(Path.home() / ".cache" / "conninfpy"))
    return Path(base) / "bna246"


def _find_data_dir(data_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    candidates = [data_dir, os.environ.get("CONNINFPY_BNA_DIR"), Path.cwd() / "datasets" / "atlases"]
    for c in candidates:
        if c is not None and Path(c).is_dir():
            return Path(c)
    return None


@dataclass
class BrainnetomeReference:
    """BNA-246 anatomy, profiles and optional probabilistic / structural maps.

    Use :meth:`load`. ``regions`` is the hierarchy table indexed by the
    original BNA ID (1-246); ``label_image`` the maximum-probability map.
    """

    regions: Any                                   # pandas.DataFrame indexed by roi_id
    label_image: Any                               # nibabel image (MPM)
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
    ) -> "BrainnetomeReference":
        """Load the bundled atlas and, when available, the BrainMap profiles.

        Parameters
        ----------
        data_dir : path, optional
            Folder holding the optional local files (``BNA_PM_4D.nii.gz``,
            ``BNA_SC_3D_246.zip``). Defaults to ``$CONNINFPY_BNA_DIR`` or
            ``./datasets/atlases`` when they exist.
        resolution_mm : {1, 2}
            Bundled label image to use (1 mm by default: spheres of a few
            millimetres need it).
        profiles : bool
            Fetch (or read from the cache) the BrainMap BD/PC profiles. Without
            network access and cache, annotation proceeds without them.
        cache_dir : path, optional
            Cache for downloads and derived arrays (default
            ``$CONNINFPY_CACHE/bna246`` or ``~/.cache/conninfpy/bna246``).
        """
        import nibabel as nib
        import pandas as pd

        from .atlas import fetch_bna_behavioral_decoding, get_bna_246_nifti_path

        pkg = resources.files("conninfpy.atlas_data")
        with resources.as_file(pkg / "bna246_hierarchy.csv") as p:
            regions = pd.read_csv(p).set_index("roi_id")
        img_path = get_bna_246_nifti_path(resolution_mm)
        img = nib.load(str(img_path))
        cache = Path(cache_dir) if cache_dir is not None else _default_cache()
        ref = cls(regions=regions, label_image=img, data_dir=_find_data_dir(data_dir), cache_dir=cache)
        ref.provenance = {
            "atlas": "Brainnetome BNA-246 (Fan et al. 2016)",
            "label_image": f"{img_path.name} (FSL MNI152 grid, {resolution_mm} mm)",
            "label_image_sha256": _sha256(img_path),
            "hierarchy": "bna246_hierarchy.csv (BNA_subregions.xlsx Table 1 + Yeo crosswalk)",
            "data_dir": str(ref.data_dir) if ref.data_dir else None,
        }
        if profiles:
            try:
                bd, pc = fetch_bna_behavioral_decoding(cache_dir=cache)
                ref.profiles = {
                    "BD": {int(k): dict(v) for k, v in bd.items()},
                    "PC": {int(k): dict(v) for k, v in pc.items()},
                }
                ref.provenance["profiles"] = {
                    "source": "BrainMap forward inference, FDR q<0.05 (BDf_FDR05.json, PCf_FDR05.json)",
                    "BD_sha256": _sha256(cache / "BDf_FDR05.json"),
                    "PC_sha256": _sha256(cache / "PCf_FDR05.json"),
                }
            except (IOError, OSError) as exc:
                logger.warning("BrainMap profiles unavailable (%s); annotating without them.", exc)
        return ref

    # ------------------------------------------------------------------
    @property
    def labels(self) -> npt.NDArray[np.int64]:
        """Label array of the MPM image (0 = unlabelled: white matter, CSF, outside)."""
        if self._labels is None:
            self._labels = np.rint(np.asarray(self.label_image.dataobj)).astype(np.int64)
        return self._labels

    @property
    def affine(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.label_image.affine, dtype=np.float64)

    def names(self, ids: Sequence[int]) -> list:
        return [self.regions.at[int(i), "name"] if int(i) > 0 else "unlabelled" for i in ids]

    def labels_at(self, world: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Label at each world (mm) coordinate; 0 outside the image or unlabelled."""
        world = np.atleast_2d(np.asarray(world, dtype=np.float64))
        vox = np.rint(np.c_[world, np.ones(len(world))] @ np.linalg.inv(self.affine).T)[:, :3].astype(int)
        shape = np.array(self.labels.shape)
        inside = np.all((vox >= 0) & (vox < shape), axis=1)
        out = np.zeros(len(world), dtype=np.int64)
        out[inside] = self.labels[tuple(vox[inside].T)]
        return out

    def sphere(self, center: npt.ArrayLike, radius_mm: float) -> npt.NDArray[np.float64]:
        """World coordinates of the label-image voxels whose centres lie within ``radius_mm``."""
        return _sphere_on_grid(np.asarray(center, float), float(radius_mm), self.affine, self.labels.shape)

    # ------------------------------------------------------------------
    def homolog(self, roi_id: int) -> int:
        return int(self.regions.at[int(roi_id), "homolog_id"])

    def profile(self, kind: str, roi_id: int) -> Optional[Dict[str, float]]:
        """BrainMap profile ``{term: likelihood ratio}`` of a region; None without profiles."""
        if self.profiles is None:
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
        vox = np.rint(np.c_[world, np.ones(len(world))] @ np.linalg.inv(affine).T)[:, :3].astype(int)
        inside = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
        if not inside.any():
            return np.zeros(N_REGIONS)
        flat = np.ravel_multi_index(tuple(vox[inside].T), shape)
        return np.asarray(mat[flat].sum(axis=0)).ravel() / len(world)

    def _membership_matrix(self):
        if self._pm is not None:
            return self._pm
        if self.data_dir is None or not (self.data_dir / PM_FILE).exists():
            return None
        import nibabel as nib
        from scipy import sparse

        path = self.data_dir / PM_FILE
        digest = _sha256(path)
        cache = self.cache_dir / f"pm_sparse_{digest[:16]}.npz"
        img = nib.load(str(path))
        shape, affine = img.shape[:3], np.asarray(img.affine, float)
        if cache.exists():
            mat = sparse.load_npz(cache)
        else:
            mat = _build_membership_cache(path, img, cache)
        self._pm = (mat, affine, shape)
        self.provenance["probabilistic_maps"] = {"file": PM_FILE, "sha256": digest,
                                                 "grid": f"{shape}, {abs(affine[0, 0])} mm (HCP-40 MNI152)"}
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
        if self.data_dir is None or not (self.data_dir / SC_FILE).exists():
            return None
        import nibabel as nib

        path = self.data_dir / SC_FILE
        digest = _sha256(path)
        cache = self.cache_dir / f"sc_derived_{digest[:16]}.npy"
        if cache.exists():
            sc = np.load(cache)
        else:
            from .atlas import get_bna_246_nifti_path

            lab_img = nib.load(str(get_bna_246_nifti_path(2)))
            lab = np.rint(np.asarray(lab_img.dataobj)).astype(np.int64).ravel()
            keep = lab > 0
            counts = np.bincount(lab[keep], minlength=N_REGIONS + 1)[1:]
            sc = np.zeros((N_REGIONS, N_REGIONS))
            with zipfile.ZipFile(path) as zf:
                for s in range(N_REGIONS):
                    with zf.open(f"{s + 1:03d}.nii.gz") as fh:
                        seed = nib.Nifti1Image.from_bytes(_gunzip(fh.read()))
                    if seed.shape != lab_img.shape or not np.allclose(seed.affine, lab_img.affine):
                        raise ValueError(f"{SC_FILE}: map {s + 1} is not on the 2 mm BNA grid.")
                    m = np.asarray(seed.dataobj, dtype=np.float64).ravel()
                    sc[s] = np.bincount(lab[keep], weights=m[keep], minlength=N_REGIONS + 1)[1:] / counts
            sc = (sc + sc.T) / 2.0
            np.fill_diagonal(sc, np.nan)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, sc)
        self._sc = sc
        self.provenance["structural_connectivity"] = {
            "file": SC_FILE, "sha256": digest,
            "derivation": "mean seed-map probability within the target region, symmetrized",
        }
        return sc


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

    logger.warning("Building the sparse probabilistic-map cache once (decompresses ~3.6 GB to a temporary file) ...")
    cache.parent.mkdir(parents=True, exist_ok=True)
    shape = img.shape[:3]
    with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
        raw = Path(tmp) / "pm.nii"
        with gzip.open(path, "rb") as src, open(raw, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 24)
        mm = nib.load(str(raw), mmap=True)
        rows, cols, vals = [], [], []
        for k in range(mm.shape[3]):
            v = np.asarray(mm.dataobj[..., k], dtype=np.float32)
            ijk = np.nonzero(v)
            rows.append(np.ravel_multi_index(ijk, shape))     # same convention as membership()
            cols.append(np.full(ijk[0].size, k, dtype=np.int32))
            vals.append(v[ijk])
        del mm
    vals = np.concatenate(vals)
    scale = 100.0 if vals.max() > 1.0 + 1e-6 else 1.0          # the official file stores percent
    mat = sparse.csr_matrix((vals / scale, (np.concatenate(rows), np.concatenate(cols))),
                            shape=(int(np.prod(shape)), img.shape[3]), dtype=np.float32)
    sparse.save_npz(cache, mat)
    return mat


def _gunzip(data: bytes) -> bytes:
    import gzip

    return gzip.decompress(data)


def _sphere_on_grid(center: npt.NDArray[np.float64], radius: float, affine, shape) -> npt.NDArray[np.float64]:
    """World coordinates of grid voxel centres within ``radius`` of ``center``."""
    inv = np.linalg.inv(affine)
    c_vox = (inv @ np.r_[center, 1.0])[:3]
    step = np.abs(np.diag(affine)[:3])
    half = np.ceil(radius / step).astype(int) + 1
    lo = np.maximum(np.floor(c_vox).astype(int) - half, 0)
    hi = np.minimum(np.ceil(c_vox).astype(int) + half, np.array(shape) - 1)
    if np.any(hi < lo):
        return np.zeros((0, 3))
    grid = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    world = np.c_[grid, np.ones(len(grid))] @ np.asarray(affine).T
    world = world[:, :3]
    return world[np.linalg.norm(world - center, axis=1) <= radius + 1e-9]


__all__ = ["BrainnetomeReference", "BD_DOMAINS"]
