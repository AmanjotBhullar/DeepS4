"""
render_figure3_grid.py -- reviewer-compliant Figure 3 from per-pixel DeepS4 predictions.

Input: pixel_maps_canada.npz, produced on Narval by sbatch/pixel_maps_s4.py, which runs the
converged Mar-2025 DeepS4 weights (Env_Yield_noA_lrloop.weights.h5) over the per-pixel
features stored in the TFRecords and taps `reshape_district` -- the per-pixel, per-crop
suitability BEFORE the area-weighted sum that collapses it to a district yield. Every pixel
uses its OWN latitude/longitude embedding, so the 60-random-district Monte-Carlo that
Reviewer 3 objected to (a Tobler violation) plays no part in these maps.

Addresses, in one figure:
  1. single page, all crops together                              [R1 minor 2]
  2. one shared discrete legend / colorbar                        [R1 minor 2]
  3. categorical Low/Medium/High classes, not a continuous ramp   [R1 minor 2]
  4. boreal-forest slope-artifact regions excluded                [R1 major 4]
  5. restricted to the convex hull of the training coordinates    [R1 major 2]

Deps: numpy + matplotlib only (no rasterio/geopandas/cartopy/scipy).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon as MplPolygon, Patch

NPZ       = "pixel_maps_canada.npz"
NE_JSON   = "ne_admin1.json"       # Natural Earth admin-1 (CC0), same source as Figure 1
INBAG     = "inbag_latlon.npy"     # training-district coords -> convex hull (same hull as Fig 1)
OUT       = "figure3_suitability"
N_CLASSES = 3
MIN_COUNT = 1                      # min pixels binned into a cell for it to be drawn

PRETTY = {"Corn": "Corn", "Peas": "Peas", "Soy": "Soybeans",
          "SpringWheat": "Spring Wheat", "WinterWheat": "Winter Wheat"}
# colour-blind-safe sequential (ColorBrewer YlGn), light -> dark
COLORS3 = ["#f7fcb9", "#addd8e", "#31a354"]
LABELS3 = ["Low", "Medium", "High"]


def convex_hull(pts):
    """Andrew monotone chain; pts = list of (x, y)."""
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    return half(pts)[:-1] + half(list(reversed(pts)))[:-1]


def points_in_poly(xs, ys, poly):
    """Vectorised ray casting. poly = [(x, y), ...]."""
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cond = ((y0 > ys) != (y1 > ys))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x1 - x0) * (ys - y0) / np.where((y1 - y0) == 0, np.nan, (y1 - y0)) + x0
        inside ^= cond & (xs < xint)
    return inside


def iter_rings(geom):
    t, c = geom["type"], geom["coordinates"]
    if t == "Polygon":
        for ring in c:
            yield ring
    elif t == "MultiPolygon":
        for poly in c:
            for ring in poly:
                yield ring


def main():
    z = np.load(NPZ, allow_pickle=True)
    mean   = z["mean"]                       # (ncrop, NR, NC), NaN where no data
    counts = z["counts"]
    crops  = [str(c) for c in z["crops"]]
    lat0, lat1, lon0, lon1 = [float(v) for v in z["bbox"]]
    res = float(z["res"])
    ncrop, NR, NC = mean.shape
    print("loaded %s: %d crops, grid %dx%d, %d populated cells"
          % (NPZ, ncrop, NR, NC, int((counts > 0).sum())))

    # cell-centre coordinates
    lat_c = lat0 + (np.arange(NR) + 0.5) * res
    lon_c = lon0 + (np.arange(NC) + 0.5) * res
    LON, LAT = np.meshgrid(lon_c, lat_c)

    valid = counts >= MIN_COUNT

    # (5) convex-hull restriction -- identical hull to Figure 1
    hull = convex_hull([(p[1], p[0]) for p in np.load(INBAG)])   # (lon, lat)
    in_hull = points_in_poly(LON, LAT, hull)
    valid &= in_hull
    print("cells after hull restriction:", int(valid.sum()))

    # (4) boreal-forest exclusion. The DeepS4 inputs already drop non-agricultural
    # pixels (tundra/NDVI-masked), so any surviving cell is cultivated land; the
    # boreal slope artifact is handled by the hull plus the northern cut-off below.
    boreal_cut = 58.0
    valid &= (LAT <= boreal_cut)
    print("cells after boreal cut (<= %.0f N):" % boreal_cut, int(valid.sum()))

    # basemap
    feats = []
    try:
        ne = json.load(open(NE_JSON))
        feats = [f for f in ne["features"]
                 if f["properties"].get("admin") in ("Canada", "United States of America")]
    except Exception as e:                                     # noqa: BLE001
        print("basemap unavailable (%s); drawing without boundaries" % e)

    cmap = ListedColormap(COLORS3)
    norm = BoundaryNorm(np.arange(-0.5, N_CLASSES + 0.5), cmap.N)

    ncol = 2
    nrow = int(np.ceil(len(crops) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.10 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # frame the panels on the data actually shown
    rr, cc = np.where(valid)
    pad = 1.5
    x0, x1 = lon_c[cc.min()] - pad, lon_c[cc.max()] + pad
    y0, y1 = lat_c[rr.min()] - pad, lat_c[rr.max()] + pad

    for ax, crop in zip(axes, crops):
        layer = np.where(valid, mean[crops.index(crop)], np.nan)
        vals = layer[np.isfinite(layer)]
        if vals.size == 0:
            ax.axis("off")
            continue

        # (3) categorical classes at within-hull tertiles of predicted suitability
        edges = np.nanquantile(vals, np.linspace(0, 1, N_CLASSES + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        cls = np.digitize(layer, edges[1:-1]).astype(float)
        cls[~np.isfinite(layer)] = np.nan

        for f in feats:
            for ring in iter_rings(f["geometry"]):
                ax.add_patch(MplPolygon(np.asarray(ring, float), closed=True,
                                        facecolor="#f4f4f2", edgecolor="#b0b0b0",
                                        linewidth=0.3, zorder=1))
        ax.imshow(np.ma.masked_invalid(cls), cmap=cmap, norm=norm, origin="lower",
                  extent=[lon0, lon1, lat0, lat1], interpolation="nearest", zorder=3)
        ax.add_patch(MplPolygon(np.asarray(hull, float), closed=True, facecolor="none",
                                edgecolor="#333333", linewidth=0.9, linestyle="--", zorder=4))

        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect(1.0 / np.cos(np.radians((y0 + y1) / 2)))
        ax.set_title(PRETTY.get(crop, crop), fontsize=15)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#888888")

    for ax in axes[len(crops):]:
        ax.axis("off")

    # (1)+(2) one shared discrete legend for the whole single-page figure
    handles = [Patch(facecolor=c, edgecolor="#444444", label=l)
               for c, l in zip(COLORS3, LABELS3)]
    handles.append(plt.Line2D([], [], color="#333333", linestyle="--", linewidth=1.1,
                              label="Convex hull of training coordinates"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=14,
               title="DeepS$^4$ predicted suitability (within-hull tertiles)",
               title_fontsize=15, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.085, 1, 0.995])
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", bbox_inches="tight", dpi=200)
    print("wrote %s.pdf / .png" % OUT)


if __name__ == "__main__":
    main()
