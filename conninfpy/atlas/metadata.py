"""Atlas / parcellation metadata for connectivity inference.

:class:`AtlasInfo` carries per-ROI labels, resting-state-network (or lobe)
assignments, MNI coordinates, hemisphere, and a free-text provenance
string. It is consumed by the export and plotting helpers
(``InferenceResult.significant_edges``, ``plot_effect_matrix``,
``summary_figure``) so significant edges can be reported with ROI names
and network context rather than bare indices.

Two bundled atlases ship with v2.1:

* :meth:`AtlasInfo.schaefer_100_yeo7`,
  :meth:`AtlasInfo.schaefer_200_yeo7`,
  :meth:`AtlasInfo.schaefer_400_yeo7` — Schaefer 2018 cortical
  parcellations at three resolutions with Yeo 7-network labels and
  MNI152 centroid coordinates.
* :meth:`AtlasInfo.bna_246` — Brainnetome Atlas (Fan et al. 2016) 246
  cortical + subcortical regions, with the Brainnetome Yeo-7 crosswalk by
  default (Yeo-17 or anatomical lobes can also be selected), and MNI152-1mm
  centroids derived from ``BN_Atlas_246_1mm.nii.gz``.

Custom CSVs can be loaded via :meth:`AtlasInfo.from_csv`, any integer
label image via :meth:`AtlasInfo.from_nifti`, and nilearn's deterministic
atlases via :meth:`AtlasInfo.from_nilearn`.

ROIs defined by coordinates (activation peaks, meta-analytic foci, sphere
centres) are placed in a reference atlas with
:meth:`AtlasInfo.from_coordinates`: each ROI gets the region it lies in (or
the nearest one) and that region's network, so edge tables and
``result.network_summary(atlas)`` aggregate coordinate-based ROIs by network
or by atlas region.
"""

from __future__ import annotations

import csv
import dataclasses
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from conninfpy.atlas.resources import (
    bundled_path,
    get_bna_246_nifti_path,
    read_bna_hierarchy,
)

_NetworkIndex = npt.NDArray[np.int_]


@dataclass
class AtlasInfo:
    """Per-ROI metadata for a brain parcellation.

    Attributes
    ----------
    labels : list[str]
        ROI names, length ``N``. Order is significant — index ``i`` in
        an ``(n_subjects, N, N)`` connectivity tensor corresponds to
        ``labels[i]``.
    networks : list[str]
        Per-ROI resting-state-network (or lobe) label.
    coords : ndarray of shape (N, 3) or None
        MNI millimeter centroids ``(x, y, z)``. ``None`` when the
        bundled atlas did not ship coordinates.
    hemisphere : list[str] or None
        Per-ROI hemisphere code (``'L'`` / ``'R'`` / ``'M'`` for medial
        or subcortical structures spanning the midline). Optional.
    source : str or None
        Provenance string. For bundled atlases this points to the
        original publication and the resource the CSV was generated
        from.
    image : nibabel image or None
        Integer label image the ROIs come from (set by :meth:`from_nifti`
        and :meth:`from_nilearn`); enables voxel-exact placement of
        coordinates in :meth:`from_coordinates`.
    label_ids : ndarray of int or None
        Label value of each ROI in ``image`` (same order as ``labels``).
    assignment : pandas.DataFrame or None
        For :meth:`from_coordinates`: where each coordinate was placed —
        region, network, whether it lies inside a region, and the distance
        to it.
    """

    labels: List[str]
    networks: List[str]
    coords: Optional[npt.NDArray[np.float64]] = None
    hemisphere: Optional[List[str]] = None
    source: Optional[str] = None
    image: Optional[Any] = field(default=None, repr=False, compare=False)
    label_ids: Optional[npt.NDArray[np.int64]] = field(
        default=None, repr=False, compare=False
    )
    assignment: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.networks) != n:
            raise ValueError(
                f"labels and networks must be the same length; "
                f"got {n} labels and {len(self.networks)} networks."
            )
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=np.float64)
            if self.coords.shape != (n, 3):
                raise ValueError(
                    f"coords must have shape ({n}, 3); got {self.coords.shape}."
                )
        if self.hemisphere is not None and len(self.hemisphere) != n:
            raise ValueError(
                f"hemisphere must have length {n}; got {len(self.hemisphere)}."
            )
        if self.label_ids is not None:
            self.label_ids = np.asarray(self.label_ids, dtype=np.int64)
            if self.label_ids.shape != (n,):
                raise ValueError(
                    f"label_ids must have length {n}; got {self.label_ids.shape}."
                )

    def __len__(self) -> int:
        return len(self.labels)

    def network_index(self) -> _NetworkIndex:
        """Integer-coded networks for ``block_mass`` / cnbs / ni_tfnbs.

        Two ROIs share an integer iff they share a network label.
        Codes are assigned in order of first appearance.
        """
        seen: dict = {}
        out = np.empty(len(self.networks), dtype=np.int_)
        for i, n in enumerate(self.networks):
            if n not in seen:
                seen[n] = len(seen)
            out[i] = seen[n]
        return out

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        label_col: str = "name",
        network_col: str = "network",
        coord_cols: Optional[Sequence[str]] = ("x", "y", "z"),
        hemisphere_col: Optional[str] = "hemisphere",
        source: Optional[str] = None,
    ) -> "AtlasInfo":
        """Load an atlas from a CSV file.

        Required columns: ``label_col`` (default ``'name'``) and
        ``network_col`` (default ``'network'``). Coordinate columns
        (default ``('x', 'y', 'z')``) and ``hemisphere_col`` (default
        ``'hemisphere'``) are loaded when present and non-empty;
        otherwise ``coords`` / ``hemisphere`` are set to ``None``.
        Rows are taken in file order.
        """
        path = Path(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"{path}: no rows found.")
        if label_col not in rows[0]:
            raise ValueError(f"{path}: missing required column {label_col!r}.")
        if network_col not in rows[0]:
            raise ValueError(f"{path}: missing required column {network_col!r}.")

        labels = [r[label_col] for r in rows]
        networks = [r[network_col] for r in rows]

        coords: Optional[np.ndarray] = None
        if coord_cols is not None and all(c in rows[0] for c in coord_cols):
            raw = np.array([[r[c].strip() for c in coord_cols] for r in rows])
            if (raw == "").all():
                coords = None
            else:
                # NaN-fill any blank cells; numpy will fail otherwise.
                clean = np.where(raw == "", "nan", raw)
                coords = clean.astype(np.float64)

        hemisphere: Optional[List[str]] = None
        if hemisphere_col is not None and hemisphere_col in rows[0]:
            hemisphere = [r[hemisphere_col] for r in rows]
            if all(h == "" for h in hemisphere):
                hemisphere = None

        return cls(
            labels=labels,
            networks=networks,
            coords=coords,
            hemisphere=hemisphere,
            source=source or str(path),
        )

    # ------------------------------------------------------------------
    # Atlases from label images
    # ------------------------------------------------------------------
    @classmethod
    def from_nifti(
        cls,
        img: Any,
        *,
        lut: Any = None,
        networks: Any = None,
        exclude: Sequence[str] = (),
        source: Optional[str] = None,
    ) -> "AtlasInfo":
        """Build an atlas from an integer label image (NIfTI path or nibabel image).

        ROIs are the non-zero label values in ascending order — the order in
        which nilearn's ``NiftiLabelsMasker`` extracts them.

        Parameters
        ----------
        img : str, Path or nibabel image
            3-D image of integer labels (0 = background). Probabilistic
            (4-D) atlases are not label images and are rejected.
        lut : DataFrame, dict, str or Path, optional
            Region names (and optionally networks) per label value: a
            DataFrame or CSV / TSV with an id column (``index``, ``id``,
            ``roi_id`` or ``label``), a ``name`` column and optionally a
            ``network`` column (nilearn's ``lut`` works as is); a
            FreeSurfer-style text LUT (``id name r g b a`` per line); or a
            dict ``{id: name}``. Unnamed labels become ``region_<id>``.
        networks : dict, sequence or callable, optional
            Network per ROI: ``{id or name: network}``, one value per ROI in
            label order, or a function of the region name. Without it the
            LUT's ``network`` column is used, and without that each region is
            its own group (``networks == labels``).
        exclude : sequence of str
            Region names to drop (e.g. white matter or ventricles in the
            Harvard-Oxford subcortical atlas). Their voxels are set to 0 in
            ``atlas.image``, so extraction with that image skips them too.
        source : str, optional
            Provenance string.
        """
        nib_img, data = _load_label_image(img)
        table = _read_lut(lut)
        ids = np.unique(data)
        ids = ids[ids != 0]
        names = [table.get(int(i), {}).get("name") or f"region_{int(i)}" for i in ids]

        drop = {e.lower() for e in exclude}
        if drop:
            unknown = drop - {n.lower() for n in names}
            if unknown:
                raise ValueError(f"exclude: no region named {sorted(unknown)}")
            keep = np.array([n.lower() not in drop for n in names])
            data = np.where(np.isin(data, ids[~keep]), 0, data)
            ids = ids[keep]
            names = [n for n, k in zip(names, keep) if k]
            nib_img = type(nib_img)(
                data.astype(np.int32), nib_img.affine, nib_img.header
            )
            nib_img.set_data_dtype(np.int32)

        if not len(ids):
            raise ValueError("the label image has no non-zero labels.")

        nets = _resolve_networks(networks, ids, names, table)
        coords = _label_centroids(data, ids, nib_img.affine)
        return cls(
            labels=names,
            networks=nets,
            coords=coords,
            hemisphere=_hemisphere(coords[:, 0]),
            source=source
            or (str(img) if isinstance(img, (str, Path)) else "label image"),
            image=nib_img,
            label_ids=ids,
        )

    @classmethod
    def from_nilearn(
        cls,
        atlas: Any,
        *,
        networks: Any = "auto",
        exclude: Sequence[str] = (),
    ) -> "AtlasInfo":
        """Build an atlas from a nilearn ``fetch_atlas_*`` result.

        Needs a deterministic atlas and nilearn >= 0.13 (its ``lut`` table).
        ``networks='auto'`` reads the Yeo network from Schaefer region names
        (``7Networks_LH_Vis_1`` -> ``Visual``, matching
        :meth:`schaefer_100_yeo7`); other atlases have no networks, so each
        region is its own group unless ``networks`` is given (see
        :meth:`from_nifti`).

        Example
        -------
        >>> from nilearn import datasets
        >>> ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
        >>> atlas = AtlasInfo.from_nilearn(ho)
        """
        get = (
            (lambda k: atlas.get(k))
            if isinstance(atlas, Mapping)
            else (lambda k: getattr(atlas, k, None))
        )
        if get("atlas_type") == "probabilistic":
            raise ValueError(
                "probabilistic atlases (maps of weights) are not label images; "
                "use a deterministic atlas or a coordinate set."
            )
        lut = get("lut")
        if lut is None:
            raise ValueError(
                "this atlas has no 'lut' table; nilearn >= 0.13 is required "
                "(or pass the image and names to AtlasInfo.from_nifti)."
            )
        lut = lut[lut["name"].astype(str).str.lower() != "background"]
        if isinstance(networks, str) and networks == "auto":
            networks = _schaefer_networks(list(lut["name"].astype(str)))
        template = get("template")
        return cls.from_nifti(
            get("maps"),
            lut=lut,
            networks=networks,
            exclude=exclude,
            source=f"nilearn atlas{f' ({template})' if template else ''}: {get('maps')}",
        )

    # ------------------------------------------------------------------
    # Coordinate-defined ROIs
    # ------------------------------------------------------------------
    @classmethod
    def from_coordinates(
        cls,
        coords: Any,
        reference: Any = None,
        *,
        names: Optional[Sequence[str]] = None,
        networks: Optional[Sequence[str]] = None,
        group_by: str = "auto",
        max_distance: float = 5.0,
        source: Optional[str] = None,
    ) -> "AtlasInfo":
        """Describe ROIs given by MNI coordinates, placed in a reference atlas.

        Typical use: ROIs centred on activation peaks, with time series
        extracted around them (e.g. spheres). Each coordinate is assigned to
        the reference region that contains it, or to the nearest region
        within ``max_distance`` mm; farther ones are ``'unassigned'``. The
        resulting atlas groups the ROIs for ``network_summary`` and edge
        tables.

        Parameters
        ----------
        coords : array of shape (n_rois, 3)
            MNI coordinates (mm), in the order of the connectivity matrix.
        reference : AtlasInfo, NIfTI path, nibabel image, nilearn atlas, or a list of them
            Atlas to place the coordinates in. With a label image the
            placement is voxel-exact; an :class:`AtlasInfo` without an image
            (e.g. the bundled Schaefer tables) falls back to the nearest
            region centroid, with a warning. A list is tried in order —
            e.g. a cortical atlas, then a subcortical one — and each
            coordinate takes the first atlas that assigns it.
        names : sequence of str, optional
            ROI names; default ``ROI_01``, ``ROI_02``, ...
        networks : sequence of str, optional
            Network per ROI when you already have one (e.g. from the study
            that defined the coordinates). Overrides the grouping from the
            reference; the region assignment is still recorded.
        group_by : {'auto', 'network', 'region'}
            How ROIs are grouped. ``'auto'`` uses the reference's networks
            when it has them, else its regions.
        max_distance : float, default 5.0
            Largest distance (mm) from a coordinate outside every region to
            the nearest labelled voxel for it to be assigned.
        source : str, optional
            Provenance string.

        Returns
        -------
        AtlasInfo
            ``networks`` holds the group of each ROI; ``assignment`` holds
            the placement table (region, network, inside, distance_mm,
            method).
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (n_rois, 3); got {coords.shape}.")
        n = len(coords)
        if names is None:
            width = len(str(n))
            names = [f"ROI_{i + 1:0{width}d}" for i in range(n)]
        names = [str(x) for x in names]
        if len(names) != n:
            raise ValueError(f"names must have length {n}; got {len(names)}.")
        if group_by not in ("auto", "network", "region"):
            raise ValueError("group_by must be 'auto', 'network' or 'region'.")
        if reference is None and networks is None:
            raise ValueError("give a reference atlas or the ROI networks.")

        table = None
        groups: List[str]
        if reference is not None:
            refs = [
                _as_atlas(r)
                for r in (
                    reference if isinstance(reference, (list, tuple)) else [reference]
                )
            ]
            table = assign_coordinates(coords, refs, max_distance=max_distance)
            table.insert(0, "roi", names)
            mode = group_by
            if mode == "auto":
                has_networks = any(list(r.networks) != list(r.labels) for r in refs)
                mode = "network" if has_networks else "region"
            groups = list(table[mode])
        if networks is not None:
            groups = [str(x) for x in networks]
            if len(groups) != n:
                raise ValueError(f"networks must have length {n}; got {len(groups)}.")
            if table is not None:
                table["network"] = groups

        ref_src = (
            " + ".join(str(r.source) for r in refs) if reference is not None else None
        )
        return cls(
            labels=names,
            networks=groups,
            coords=coords,
            hemisphere=_hemisphere(coords[:, 0]),
            source=source
            or (
                f"{n} coordinates placed in {ref_src}"
                if ref_src
                else f"{n} coordinates"
            ),
            assignment=table,
        )

    # ------------------------------------------------------------------
    # Bundled atlases
    # ------------------------------------------------------------------
    @classmethod
    def schaefer_100_yeo7(cls) -> "AtlasInfo":
        """Schaefer-2018 100-parcel atlas with Yeo-7 networks.

        Source: ThomasYeoLab/CBIG Schaefer 2018 release. Labels and
        Yeo-7 network assignments follow the upstream naming
        (``7Networks_LH_Vis_1`` etc.); ``coords`` are MNI152 2mm
        centroids derived from the published parcellation image.
        """
        return _load_bundled("schaefer100_yeo7.csv", source=_SCHAEFER_SRC)

    @classmethod
    def schaefer_200_yeo7(cls) -> "AtlasInfo":
        """Schaefer-2018 200-parcel atlas with Yeo-7 networks.

        Same source and conventions as :meth:`schaefer_100_yeo7`, at
        the 200-parcel resolution.
        """
        return _load_bundled("schaefer200_yeo7.csv", source=_SCHAEFER_SRC)

    @classmethod
    def schaefer_400_yeo7(cls) -> "AtlasInfo":
        """Schaefer-2018 400-parcel atlas with Yeo-7 networks.

        Same source and conventions as :meth:`schaefer_100_yeo7`, at
        the 400-parcel resolution.
        """
        return _load_bundled("schaefer400_yeo7.csv", source=_SCHAEFER_SRC)

    @classmethod
    def bna_246(cls, with_image: bool = False, networks: str = "yeo7") -> "AtlasInfo":
        """Brainnetome Atlas (Fan et al. 2016) 246-region parcellation.

        Names follow ``BN_Atlas_246_LUT.txt``. MNI coordinates are voxel-mass
        centroids computed from the ``BN_Atlas_246_1mm.nii.gz`` parcellation
        (atlas.brainnetome.org).

        ``networks`` selects the grouping carried in ``networks``: the
        Brainnetome division of the regions (``'yeo7'``, default, or
        ``'yeo17'``; from ``bna246_hierarchy.csv``, i.e. the Brainnetome
        Center's subregion-to-network table; subcortical regions, which the
        division leaves unassigned, are 'Subcortical'), or ``'lobe'`` (the
        seven-lobe grouping Frontal / Temporal / Parietal / Insular / Limbic /
        Occipital / Subcortical). Changed in October 2026: the default was
        the lobe grouping; pass ``networks='lobe'`` to keep it.

        ``with_image=True`` also attaches the bundled 1 mm label image, so
        :meth:`from_coordinates` can place coordinates voxel-exactly.
        """
        if networks not in ("yeo7", "yeo17", "lobe"):
            raise ValueError(
                f"networks must be 'yeo7', 'yeo17' or 'lobe'; got {networks!r}."
            )
        grouping = (
            "BNA seven-lobe grouping"
            if networks == "lobe"
            else f"Brainnetome division ({networks}, bna246_hierarchy.csv)"
        )
        if with_image:
            atlas = cls.from_nifti(
                get_bna_246_nifti_path(1),
                lut=bundled_path("bna246.csv"),
                source=f"Fan et al. 2016 Brainnetome Atlas (BN_Atlas_246_1mm.nii.gz); networks = {grouping}.",
            )
        else:
            atlas = _load_bundled(
                "bna246.csv",
                source=(
                    f"Fan et al. 2016 Brainnetome Atlas; networks = {grouping}; centroids "
                    "from BN_Atlas_246_1mm.nii.gz."
                ),
            )
        if networks == "lobe":
            return atlas

        h = read_bna_hierarchy(index="name")
        division = [
            str(h.at[lab, networks])
            if h.at[lab, "structure"] == "cortex"
            else "Subcortical"
            for lab in atlas.labels
        ]
        return dataclasses.replace(atlas, networks=division)


_SCHAEFER_SRC = (
    "Schaefer et al. 2018 (ThomasYeoLab/CBIG release); "
    "centroids from FSLMNI152 2mm parcellation."
)


def _load_bundled(filename: str, *, source: str) -> AtlasInfo:
    """Load a packaged CSV via ``importlib.resources``."""
    return AtlasInfo.from_csv(bundled_path(filename), source=source)


# ----------------------------------------------------------------------
# Helpers: label images, LUTs, coordinate placement
# ----------------------------------------------------------------------
_YEO7_NAMES = {"Vis": "Visual"}  # as in the bundled Schaefer tables
_SCHAEFER_RE = re.compile(r"^(7|17)Networks_(LH|RH)_([A-Za-z]+)")


def _load_label_image(img: Any) -> Tuple[Any, npt.NDArray[np.int64]]:
    """Load a 3-D integer label image; return (nibabel image, int labels)."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - nibabel is a core dependency
        raise ImportError(
            "reading label images needs nibabel (pip install nibabel)."
        ) from exc
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"expected a 3-D label image, got shape {data.shape}; probabilistic "
            "(4-D) atlases are not supported."
        )
    labels = np.rint(data)
    if not np.allclose(data, labels):
        raise ValueError("the label image must hold integer labels.")
    return img, labels.astype(np.int64)


_ID_COLUMNS = ("index", "id", "roi_id", "label", "label_id")


def _read_lut(lut: Any) -> Dict[int, Dict[str, str]]:
    """Normalise a LUT to ``{label id: {'name': ..., 'network': ...}}``."""
    if lut is None:
        return {}
    if isinstance(lut, Mapping):
        out: Dict[int, Dict[str, str]] = {}
        for k, v in lut.items():
            if isinstance(v, (tuple, list)):
                out[int(k)] = {
                    "name": str(v[0]),
                    **({"network": str(v[1])} if len(v) > 1 else {}),
                }
            else:
                out[int(k)] = {"name": str(v)}
        return out

    import pandas as pd

    if isinstance(lut, (str, Path)):
        path = Path(lut)
        if path.suffix.lower() in (".csv", ".tsv"):
            lut = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        else:  # FreeSurfer-style colour LUT: "id name r g b a"
            out = {}
            for line in path.read_text().splitlines():
                parts = line.split()
                if (
                    len(parts) >= 2
                    and not parts[0].startswith("#")
                    and parts[0].lstrip("-").isdigit()
                ):
                    out[int(parts[0])] = {"name": parts[1]}
            return out
    if not isinstance(lut, pd.DataFrame):
        raise TypeError("lut must be a DataFrame, CSV/TSV/LUT path or {id: name} dict.")
    cols = {c.lower(): c for c in lut.columns}
    id_col = next((cols[c] for c in _ID_COLUMNS if c in cols), None)
    if id_col is None or "name" not in cols:
        raise ValueError(
            f"lut needs an id column ({', '.join(_ID_COLUMNS)}) and a 'name' column."
        )
    net_col = cols.get("network")
    out = {}
    for _, row in lut.iterrows():
        entry = {"name": str(row[cols["name"]])}
        if net_col is not None and isinstance(row[net_col], str) and row[net_col]:
            entry["network"] = row[net_col]
        out[int(row[id_col])] = entry
    return out


def _resolve_networks(
    networks: Any,
    ids: npt.NDArray[np.int64],
    names: List[str],
    table: Dict[int, Dict[str, str]],
) -> List[str]:
    """Network per ROI from an explicit spec, the LUT, or the region itself."""
    if networks is None:
        if any("network" in table.get(int(i), {}) for i in ids):
            return [table.get(int(i), {}).get("network", "unassigned") for i in ids]
        return list(names)
    if callable(networks):
        return [str(networks(n)) for n in names]
    if isinstance(networks, Mapping):
        return [
            str(networks.get(int(i), networks.get(n, "unassigned")))
            for i, n in zip(ids, names)
        ]
    nets = [str(x) for x in networks]
    if len(nets) != len(ids):
        raise ValueError(
            f"networks must have one entry per region ({len(ids)}); got {len(nets)}."
        )
    return nets


def _schaefer_networks(names: List[str]) -> Optional[List[str]]:
    """Yeo network from Schaefer names, or None when the names are not Schaefer's."""
    matches = [_SCHAEFER_RE.match(n) for n in names]
    if not names or not all(matches):
        return None
    return [
        _YEO7_NAMES.get(m.group(3), m.group(3)) if m.group(1) == "7" else m.group(3)
        for m in matches
    ]


def _label_centroids(
    data: npt.NDArray[np.int64],
    ids: npt.NDArray[np.int64],
    affine: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Voxel-mass centroid of each label, in world (mm) coordinates."""
    from scipy import ndimage

    vox = np.asarray(
        ndimage.center_of_mass(np.ones_like(data, dtype=np.float64), data, ids)
    )
    return vox @ affine[:3, :3].T + affine[:3, 3]


def _hemisphere(x: npt.NDArray[np.float64], midline_mm: float = 2.0) -> List[str]:
    """'L' / 'R' from the x coordinate (MNI: negative = left); 'M' near the midline."""
    return [
        "M" if abs(v) < midline_mm else ("L" if v < 0 else "R")
        for v in np.asarray(x, float)
    ]


def _as_atlas(reference: Any) -> AtlasInfo:
    """Accept an AtlasInfo, a label-image path / nibabel image, or a nilearn atlas."""
    if isinstance(reference, AtlasInfo):
        return reference
    if isinstance(reference, (str, Path)) or hasattr(reference, "affine"):
        return AtlasInfo.from_nifti(reference)
    if isinstance(reference, Mapping) or hasattr(reference, "maps"):
        return AtlasInfo.from_nilearn(reference)
    raise TypeError("reference must be an AtlasInfo, a label image or a nilearn atlas.")


def assign_coordinates(
    coords: Any,
    reference: Any,
    *,
    max_distance: float = 5.0,
) -> "Any":
    """Place MNI coordinates in a reference atlas (or a list of atlases, in order).

    Returns a table with one row per coordinate: ``x, y, z``, the assigned
    ``region``, its ``network`` (the region itself when the atlas has no
    networks), ``label_id``, ``inside`` (the coordinate lies in that
    region), ``distance_mm`` (0 inside; else to the nearest labelled voxel,
    or to the region centroid without an image), ``method`` (``'voxel'``,
    ``'nearest voxel'``, ``'centroid'`` or ``'unassigned'``) and, for a
    list of atlases, ``atlas`` (the position of the atlas that assigned it).
    With a list, each coordinate takes the first atlas that assigns it
    inside a region, then the first that assigns it within ``max_distance``.
    """
    if isinstance(reference, (list, tuple)):
        tables = [
            assign_coordinates(coords, r, max_distance=max_distance) for r in reference
        ]
        out = tables[0].copy()
        out["atlas"] = np.where(out["method"] == "unassigned", -1, 0)
        for rank in ("voxel", "nearest voxel", "centroid"):
            for k, t in enumerate(tables):
                take = (
                    (out["method"] != "voxel")
                    & (t["method"] == rank)
                    & ((out["method"] == "unassigned") | (rank == "voxel"))
                )
                out.loc[take, t.columns] = t.loc[take]
                out.loc[take, "atlas"] = k
        return out
    return _assign_one(coords, reference, max_distance=max_distance)


def _assign_one(coords: Any, reference: Any, *, max_distance: float) -> "Any":
    """:func:`assign_coordinates` for a single reference atlas."""
    import pandas as pd

    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    ref = _as_atlas(reference)
    n = len(coords)
    idx = np.full(n, -1)
    inside = np.zeros(n, dtype=bool)
    dist = np.full(n, np.nan)
    method = np.array(["unassigned"] * n, dtype=object)

    if ref.image is not None and ref.label_ids is not None:
        _, data = _load_label_image(ref.image)
        affine = ref.image.affine
        pos = {int(v): k for k, v in enumerate(ref.label_ids)}
        vox = np.rint(
            (coords - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
        ).astype(int)
        in_box = np.all((vox >= 0) & (vox < np.array(data.shape)), axis=1)
        lab = np.zeros(n, dtype=np.int64)
        lab[in_box] = data[tuple(vox[in_box].T)]
        for i in range(n):
            if lab[i] in pos:
                idx[i], inside[i], dist[i], method[i] = pos[lab[i]], True, 0.0, "voxel"
        todo = np.flatnonzero(idx < 0)
        if todo.size:
            from scipy.spatial import cKDTree

            labelled = np.argwhere(np.isin(data, ref.label_ids))
            world = labelled @ affine[:3, :3].T + affine[:3, 3]
            d, j = cKDTree(world).query(coords[todo])
            for i, di, ji in zip(todo, d, j):
                if di <= max_distance:
                    idx[i] = pos[int(data[tuple(labelled[ji])])]
                    dist[i], method[i] = di, "nearest voxel"
    elif ref.coords is not None:
        warnings.warn(
            "the reference atlas has no label image; each coordinate is assigned "
            "to the nearest region centroid (see distance_mm). Use "
            "AtlasInfo.from_nifti / from_nilearn for voxel-exact placement.",
            UserWarning,
            stacklevel=2,
        )
        d = np.linalg.norm(coords[:, None, :] - ref.coords[None, :, :], axis=2)
        idx = np.nanargmin(d, axis=1)
        dist = d[np.arange(n), idx]
        method[:] = "centroid"
    else:
        raise ValueError(
            "the reference atlas has neither a label image nor coordinates."
        )

    ok = idx >= 0
    region = np.where(
        ok, np.array(ref.labels, dtype=object)[np.maximum(idx, 0)], "unassigned"
    )
    network = np.where(
        ok, np.array(ref.networks, dtype=object)[np.maximum(idx, 0)], "unassigned"
    )
    label_id = (
        np.where(ok, ref.label_ids[np.maximum(idx, 0)], 0)
        if ref.label_ids is not None
        else np.full(n, 0)
    )
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "region": region,
            "network": network,
            "label_id": label_id,
            "inside": inside,
            "distance_mm": dist,
            "method": method,
        }
    )


__all__ = ["AtlasInfo", "assign_coordinates"]
