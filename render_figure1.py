"""
render_figure1.py -- open-licensed replacement for Figure 1 (train/test coverage).

The submitted Figure 1 was drawn over a proprietary web-map (Google-style) basemap,
which the Scientific Reports editorial office flagged as a CC-BY copyright problem.
This regenerates it from:
  * Natural Earth admin-1 boundaries  (public domain / CC0)   -> ne_admin1.json
  * the authors' own training/test district coordinates       -> fig1_districts.csv
No proprietary basemap tiles are reproduced.

Also overlays the convex hull of the TRAINING coordinates, which is the
interpolation regime the revised Figure 3 is restricted to (Reviewer 1, major 2).

Deps: matplotlib + numpy only (no geopandas / cartopy / scipy).
"""
import json, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

NE_JSON  = "ne_admin1.json"
DIST_CSV = "fig1_districts.csv"
OUT      = "figure1_coverage"

TRAIN_C = "#4477aa"   # blue
TEST_C  = "#cc6677"   # red
HULL_C  = "#222222"


def convex_hull(pts):
    """Andrew monotone chain; pts = list of (x, y). Returns hull in order."""
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0]-out[-2][0])*(p[1]-out[-2][1])
                                     - (out[-1][1]-out[-2][1])*(p[0]-out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    return half(pts)[:-1] + half(reversed(pts))[:-1]


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
    ne = json.load(open(NE_JSON))
    feats = [f for f in ne["features"]
             if f["properties"].get("admin") in ("Canada", "United States of America")]

    raw = list(csv.DictReader(open(DIST_CSV)))
    for r in raw:
        for k in ("lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max"):
            r[k] = float(r[k])
    # the same district recurs once per year/crop -> keep one extent per district
    uniq = {}
    for r in raw:
        uniq[(r["split"], round(r["lat_min"], 3), round(r["lon_min"], 3))] = r
    rows = list(uniq.values())
    train = [r for r in rows if r["split"] == "train"]
    test  = [r for r in rows if r["split"] == "test"]

    hull = convex_hull([(r["lon"], r["lat"]) for r in train])

    def rects(rs):
        return [Rectangle((r["lon_min"], r["lat_min"]),
                          max(r["lon_max"]-r["lon_min"], 0.12),
                          max(r["lat_max"]-r["lat_min"], 0.12)) for r in rs]

    def draw(ax, x0, x1, y0, y1, tag, scale_km, emphasize_ca=False):
        midlat = math.radians((y0 + y1) / 2)
        for f in feats:
            for ring in iter_rings(f["geometry"]):
                ax.add_patch(MplPolygon(np.asarray(ring, float), closed=True,
                                        facecolor="#f2f2f0", edgecolor="#9a9a9a",
                                        linewidth=0.4, zorder=1))
        if emphasize_ca:
            # US districts as muted context; Canadian districts emphasised
            ax.add_collection(PatchCollection(rects([r for r in train if r["country"] == "USA"]),
                                              facecolor="#c4c4c4", edgecolor="none",
                                              alpha=0.55, zorder=3, match_original=False, rasterized=True))
            ax.add_collection(PatchCollection(rects([r for r in train if r["country"] == "CA"]),
                                              facecolor=TRAIN_C, edgecolor="none",
                                              alpha=0.75, zorder=4, match_original=False, rasterized=True))
            ax.add_collection(PatchCollection(rects([r for r in test if r["country"] == "CA"]),
                                              facecolor="none", edgecolor=TEST_C,
                                              linewidths=0.9, zorder=5, match_original=False, rasterized=True))
        else:
            ax.add_collection(PatchCollection(rects(train), facecolor=TRAIN_C, edgecolor="none",
                                              alpha=0.55, zorder=3, match_original=False, rasterized=True))
            ax.add_collection(PatchCollection(rects(test), facecolor="none", edgecolor=TEST_C,
                                              linewidths=0.7, zorder=4, match_original=False, rasterized=True))
        if len(hull) >= 3:
            ax.add_patch(MplPolygon(np.asarray(hull, float), closed=True, facecolor="none",
                                    edgecolor=HULL_C, linewidth=1.6, linestyle="--", zorder=5))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect(1.0 / math.cos(midlat))
        ax.grid(True, linewidth=0.3, color="#cccccc", zorder=2)
        ax.set_xlabel("Longitude ($^\\circ$E)", fontsize=15)
        ax.set_ylabel("Latitude ($^\\circ$N)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.set_title(tag, fontsize=17, loc="left")
        ax.annotate("N", xy=(0.975, 0.90), xytext=(0.975, 0.74), xycoords="axes fraction",
                    ha="center", fontsize=15,
                    arrowprops=dict(arrowstyle="-|>", color="k", linewidth=1.1))
        seg = scale_km / (111.32 * math.cos(midlat))
        xs = x0 + 0.04 * (x1 - x0); ys = y0 + 0.06 * (y1 - y0)
        ax.plot([xs, xs + seg], [ys, ys], "k-", linewidth=3.2, zorder=6)
        ax.text(xs + seg / 2, ys + 0.025 * (y1 - y0), "%d km" % scale_km,
                ha="center", fontsize=13, zorder=6)

    lons = [r["lon"] for r in rows]; lats = [r["lat"] for r in rows]
    ca = [r for r in rows if r["country"] == "CA"]
    calon = [r["lon"] for r in ca]; calat = [r["lat"] for r in ca]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 11.4))
    draw(axes[0], min(lons)-5, max(lons)+5, min(lats)-3, max(lats)+4,
         "(a) Full training and test extent\n(Canada and contiguous United States)", 500)
    n_ca_tr = len([r for r in train if r["country"] == "CA"])
    n_ca_te = len([r for r in test if r["country"] == "CA"])
    draw(axes[1], min(calon)-4, max(calon)+4, min(calat)-2, max(calat)+3,
         "(b) Canadian districts (the region mapped in Figure 3);\n"
         "United States districts shown in grey for context", 400, emphasize_ca=True)

    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=9,
               markerfacecolor=TRAIN_C, alpha=0.7, markeredgecolor="none",
               label="Training districts (n=%d total; %d in Canada)" % (len(train), n_ca_tr)),
        Line2D([], [], marker="s", linestyle="none", markersize=9,
               markerfacecolor="none", markeredgecolor=TEST_C,
               label="Held-out test districts (n=%d total; %d in Canada)" % (len(test), n_ca_te)),
        Line2D([], [], marker="s", linestyle="none", markersize=9,
               markerfacecolor="#c4c4c4", alpha=0.55, markeredgecolor="none",
               label="United States districts (panel b, context only)"),
        Line2D([], [], color=HULL_C, linestyle="--", linewidth=1.6,
               label="Convex hull of training coordinates (interpolation region)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.canvas.draw()
    for ax in axes:
        ax.apply_aspect()
        pos = ax.get_position()
        ax.set_position([(1.0 - pos.width) / 2.0, pos.y0, pos.width, pos.height])
    fig.savefig(OUT + ".pdf", dpi=400)
    fig.savefig(OUT + ".png", dpi=200)
    print("wrote %s.pdf/.png | train=%d test=%d hull_pts=%d"
          % (OUT, len(train), len(test), len(hull)))


if __name__ == "__main__":
    main()
