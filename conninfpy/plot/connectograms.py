"""Connectograms plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc


if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


_YEO7_COLORS = {
    "Visual": "#781286",
    "Somatomotor": "#4682B4",
    "Dorsal Attention": "#00760E",
    "Ventral Attention": "#C43AFA",
    "Limbic": "#A8C66C",
    "Frontoparietal": "#E69422",
    "Default": "#CD3E4E",
}
_YEO7_ALIASES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Ventral Attention",
    "Salience/Ventral Attention": "Ventral Attention",
    "Cont": "Frontoparietal",
    "Frontoparietal Control": "Frontoparietal",
    "Default Mode": "Default",
    "DMN": "Default",
}
_LOBES = {
    "Frontal": ("FRO", "#D9C7A7"),
    "Insular": ("INS", "#B9DCDC"),
    "Limbic": ("LIM", "#E6C5D6"),
    "Temporal": ("TEM", "#CADDAA"),
    "Parietal": ("PAR", "#AFC8E2"),
    "Occipital": ("OCC", "#D7C9EA"),
    "Subcortical": ("SUB", "#D6D6D6"),
}


def _network_key(name: Any) -> str:
    return _YEO7_ALIASES.get(str(name), str(name))


def _connectogram_layout(
    regions: Any, lobe_gap: float = 1.5, hemi_gap: float = 4.0
) -> Any:
    """Slot of each BNA subregion along its half of the circle, in degrees from the top.

    The right hemisphere runs clockwise from 12 o'clock and the left mirrors it
    (homologs face each other); lobes follow the Brainnetome connectogram order
    and subregions their BNA ID within a lobe, so gyri are contiguous.
    """
    rank = {lobe: k for k, lobe in enumerate(_LOBES)}
    right = regions[regions["hemisphere"] == "R"]
    order = sorted(right.index, key=lambda r: (rank[right.at[r, "lobe"]], r))
    s, prev, slots = hemi_gap / 2, None, {}
    for rid in order:
        lobe = right.at[rid, "lobe"]
        if prev is not None and lobe != prev:
            s += lobe_gap
        slots[rid] = s
        s, prev = s + 1, lobe
    deg = 180.0 / (s + hemi_gap / 2)
    out = regions[["name", "gyrus", "lobe", "hemisphere"]].copy()
    for rid in order:
        out.loc[rid, "s0"] = slots[rid] * deg
        out.loc[int(regions.at[rid, "homolog_id"]), "s0"] = slots[rid] * deg
    out["s1"] = out["s0"] + deg
    return out


def _side_angle(s: Any, hemisphere: str) -> Any:
    """Polar angle (degrees, counter-clockwise from east) of side coordinate ``s``."""
    return 90.0 - s if hemisphere == "R" else 90.0 + s


def plot_connectogram(
    result: Any,
    rois: Any,
    *,
    tail: str = "positive",
    level: str = "gyrus",
    alpha: float = 0.05,
    assign: str = "dominant",
    networks: Optional[Sequence[str]] = None,
    weight: str = "count",
    min_weight: float = 1.0,
    max_links: Optional[int] = None,
    labels: str = "gyrus",
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    param_idx: Optional[int] = None,
) -> Axes:
    """Brainnetome-style connectogram of one tail's detected connections.

    The layout follows the BNA-246 connectogram: left hemisphere on the left,
    right mirrored on the right, lobes from frontal (top) to subcortical
    (bottom). Rings, from outside in: the 246 subregions (pale = not sampled by
    any ROI), the gyri, and the network make-up of each unit at ``level`` (the
    share of its sampled tissue that comes from ROIs of each network).

    Connections are aggregated with
    :func:`conninfpy.interpret.annotate.aggregate_connections`, split by each ROI's
    network label (``networks``, default ``rois.table['network']``: the source
    atlas's division, see :func:`conninfpy.interpret.annotate.annotate_rois`). By default
    each ROI is placed in the unit holding most of its tissue
    (``assign='dominant'``), so every chord is a whole number of connections
    and none is dropped. ``assign='weighted'`` spreads each ROI over the units
    its support covers; that is faithful to mixed ROIs, but splits a
    connection between large parcels into many fractional chords, most below
    ``min_weight`` (the footer gives the share of connections drawn).

    A chord joins two (unit, network) slots; its colour runs from one
    endpoint's network to the other's, and its width scales with the number of
    detected connections (``weight='count'``) or detected / tested
    (``'rate'``). Chords below ``min_weight`` connections are not drawn.

    Descriptive only: the result's correction and inference unit are unchanged.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Wedge

    from conninfpy.interpret.annotate import _unit_weights, aggregate_connections

    if tail not in ("positive", "negative"):
        raise ValueError(
            "tail must be 'positive' or 'negative' (draw one connectogram per tail)."
        )
    if weight not in ("count", "rate"):
        raise ValueError("weight must be 'count' or 'rate'.")
    if labels not in ("gyrus", "subregion", "none"):
        raise ValueError("labels must be 'gyrus', 'subregion' or 'none'.")
    regions = rois.reference.regions
    lay = _connectogram_layout(regions)
    nets = [
        str(x) for x in (rois.grouping("network") if networks is None else networks)
    ]
    W, units = _unit_weights(rois, level, assign)
    agg = aggregate_connections(
        result,
        rois,
        level=level,
        tail=tail,
        alpha=alpha,
        assign=assign,
        networks=nets,
        param_idx=param_idx,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")

    # rings: subregions (sampled or not) and gyri
    sampled = np.nan_to_num(rois.labelled_weights(), nan=0.0).sum(axis=0) >= 0.1
    for rid, r in lay.iterrows():
        a0, a1 = sorted(
            (_side_angle(r.s0, r.hemisphere), _side_angle(r.s1, r.hemisphere))
        )
        on = sampled[int(rid) - 1]
        ax.add_patch(
            Wedge(
                (0, 0),
                1.0,
                a0,
                a1,
                width=0.045,
                lw=0.3,
                facecolor=_LOBES[r.lobe][1] if on else "white",
                edgecolor="white" if on else "#C8C8C8",
            )
        )
    gyri = lay.groupby(["gyrus", "hemisphere"], sort=False).agg(
        s0=("s0", "min"), s1=("s1", "max"), lobe=("lobe", "first")
    )
    for k, ((g, h), r) in enumerate(gyri.sort_values("s0").iterrows()):
        a0, a1 = sorted((_side_angle(r.s0, h), _side_angle(r.s1, h)))
        ax.add_patch(
            Wedge(
                (0, 0),
                0.945,
                a0,
                a1,
                width=0.035,
                lw=0.4,
                edgecolor="white",
                facecolor="#4D4D4D" if k % 2 else "#8A8A8A",
            )
        )

    # network ring at the units of `level`; chords attach to each network's share
    key = lay["name"] if level == "subregion" else lay[level] + "_" + lay["hemisphere"]
    spans = (
        lay.assign(unit=key.to_numpy())
        .groupby("unit")
        .agg(s0=("s0", "min"), s1=("s1", "max"), hemisphere=("hemisphere", "first"))
    )
    net_order = [n for n in _YEO7_COLORS] + [
        n for n in dict.fromkeys(map(_network_key, nets)) if n not in _YEO7_COLORS
    ]
    keyed = np.array([_network_key(x) for x in nets])
    attach = {}
    for u_idx, u in enumerate(units):
        mass = {n: float(W[keyed == n, u_idx].sum()) for n in net_order}
        tot = sum(mass.values())
        if tot <= 0 or u not in spans.index:
            continue
        s0, s1, h = spans.at[u, "s0"], spans.at[u, "s1"], spans.at[u, "hemisphere"]
        pos = s0
        for n in net_order:
            if mass[n] <= 0:
                continue
            width = (s1 - s0) * mass[n] / tot
            a0, a1 = sorted((_side_angle(pos, h), _side_angle(pos + width, h)))
            ax.add_patch(
                Wedge(
                    (0, 0),
                    0.895,
                    a0,
                    a1,
                    width=0.03,
                    lw=0,
                    facecolor=_YEO7_COLORS.get(n, "#8C8C8C"),
                )
            )
            attach[(u, n)] = (pos + width / 2, h, width)
            pos += width

    # chords
    links = agg[agg["detected"] >= min_weight].copy()
    links["value"] = links["detected"] if weight == "count" else links["rate"]
    if max_links is not None:
        links = links.nlargest(int(max_links), "value")
    links = links.sort_values("value")
    r_in, t = 0.86, np.linspace(0.0, 1.0, 60)[:, None]

    def xy(s, h):
        a = np.deg2rad(_side_angle(s, h))
        return np.array([r_in * np.cos(a), r_in * np.sin(a)])

    segs, cols, lws = [], [], []
    vmax = float(links["value"].max()) if len(links) else 1.0
    for row in links.itertuples(index=False):
        na, nb = _network_key(row.network_a), _network_key(row.network_b)
        sa, ha, wa = attach[(row.unit_a, na)]
        sb, hb, wb = attach[(row.unit_b, nb)]
        if (row.unit_a, na) == (row.unit_b, nb):  # within one slot: a small inward loop
            p0, p2 = xy(sa - wa / 4, ha), xy(sa + wa / 4, ha)
            c = 0.7 * xy(sa, ha)
        else:
            p0, p2 = xy(sa, ha), xy(sb, hb)
            c = (
                (p0 + p2)
                / 2
                * (0.15 + 0.6 * (1 - np.linalg.norm(p2 - p0) / (2 * r_in)))
            )
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t**2 * p2
        f = row.value / vmax
        ca, cb = (
            np.array(to_rgba(_YEO7_COLORS.get(na, "#8C8C8C"))),
            np.array(to_rgba(_YEO7_COLORS.get(nb, "#8C8C8C"))),
        )
        mix = 1 / (
            1 + np.exp(-14 * (t[:-1] - 0.5))
        )  # each half keeps its endpoint's colour
        grad = (1 - mix) * ca + mix * cb
        grad[:, 3] = 0.3 + 0.6 * f
        segs.extend(np.stack([pts[:-1], pts[1:]], axis=1))
        cols.extend(grad)
        lws.extend([0.4 + 5.6 * f] * (len(t) - 1))
    if segs:
        ax.add_collection(
            LineCollection(
                segs, colors=cols, linewidths=lws, capstyle="butt", joinstyle="round"
            )
        )

    # labels
    def radial(text, s, h, r, size, **kw):
        a = _side_angle(s, h)
        ax.text(
            r * np.cos(np.deg2rad(a)),
            r * np.sin(np.deg2rad(a)),
            text,
            fontsize=size,
            rotation=a if h == "R" else a - 180,
            rotation_mode="anchor",
            ha="left" if h == "R" else "right",
            va="center",
            **kw,
        )

    r_gyrus = 1.015
    if labels == "subregion":
        for _, r in lay.iterrows():
            radial(
                r["name"].rsplit("_", 1)[0], (r.s0 + r.s1) / 2, r.hemisphere, 1.015, 3.2
            )
        r_gyrus = 1.16
    if labels != "none":
        for (g, h), r in gyri.iterrows():
            radial(g, (r.s0 + r.s1) / 2, h, r_gyrus, 6.0, color="#333333")
    r_lobe = {"gyrus": 1.17, "subregion": 1.33, "none": 1.08}[labels]
    lobes = lay.groupby(["lobe", "hemisphere"], sort=False).agg(
        s0=("s0", "min"), s1=("s1", "max")
    )
    for (lobe, h), r in lobes.iterrows():
        a = np.deg2rad(_side_angle((r.s0 + r.s1) / 2, h))
        ax.text(
            r_lobe * np.cos(a),
            r_lobe * np.sin(a),
            f"{_LOBES[lobe][0]}-{h}",
            fontsize=8.5,
            ha="center",
            va="center",
            color="#333333",
            fontweight="bold",
        )
    lim = r_lobe + 0.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 0.06, lim)

    # title and legends
    design = getattr(result, "design", None) or {}
    comp = design.get("comparison")
    if title is None:
        n_det = int(
            (
                result.significant_edges(
                    alpha=alpha, tail=tail, param_idx=param_idx
                ).shape[0]
            )
        )
        if comp:
            a, b = comp
            title = f"{b} > {a}" if tail == "positive" else f"{a} > {b}"
        else:
            title = f"{tail} tail"
        title = f"{title}: {n_det} connections"
    ax.set_title(title, fontsize=11, pad=2)
    present = [n for n in net_order if any(k[1] == n for k in attach)]
    leg = ax.legend(
        [Line2D([0], [0], color=_YEO7_COLORS.get(n, "#8C8C8C"), lw=3) for n in present],
        present,
        loc="upper left",
        fontsize=6.5,
        frameon=False,
        title="network",
        title_fontsize=7,
    )
    ax.add_artist(leg)
    if len(links):
        if weight == "count":
            ticks = sorted(
                {max(1, round(v)) for v in (vmax, vmax / 2, vmax / 4)}, reverse=True
            )
            fmt = lambda v: f"{v:g} connection{'s' if v != 1 else ''}"  # noqa: E731
        else:
            ticks, fmt = [vmax, vmax / 2, vmax / 4], (lambda v: f"rate {v:.2f}")
        ax.legend(
            [Line2D([0], [0], color="#555555", lw=0.4 + 5.6 * v / vmax) for v in ticks],
            [fmt(v) for v in ticks],
            loc="upper right",
            fontsize=6.5,
            frameon=False,
        )
    total = float(agg["detected"].sum())
    shown = float(links["detected"].sum()) if len(links) else 0.0
    how = (
        "placed in the unit holding most of it"
        if assign == "dominant"
        else "spread over the units it covers"
    )
    ax.text(
        0,
        -lim - 0.03,
        f"Each of {len(rois)} ROIs {how} (Brainnetome {level}); {len(links)} chords ≥ "
        f"{min_weight:g} connection carry {shown:.0f} of {total:.0f} connections ({shown / total:.0%})"
        if total
        else f"Each of {len(rois)} ROIs {how} (Brainnetome {level}); no detected connections",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#555555",
    )
    return ax
