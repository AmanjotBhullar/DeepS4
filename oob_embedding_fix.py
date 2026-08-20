"""
oob_embedding_fix.py -- geostatistically sound replacement for the random
Monte-Carlo coordinate marginalization in Generate_Maps.ipynb (cells 5-9).

Reviewer 1 (major 2) / Reviewer 3 (item 3): substituting the average of 30-60
RANDOM in-bag coordinates imposes a distant anthropogenic footprint and violates
Tobler's First Law. This module instead:
  * in-hull pixels  -> use the pixel's OWN coordinate embedding (interpolative);
  * out-of-bag pixels -> interpolate the embedding from the k geographically
    NEAREST in-bag training districts, inverse-distance (distance-decay) weighted.

Drop-in usage inside the tile loop (replaces the `for sampled_coord ... ; mean`):

    from oob_embedding_fix import EmbeddingInterpolator
    emb = EmbeddingInterpolator(inbag_latlon, frequency_num=64, k=8, power=2.0)
    c_emb = emb.embed(pixel_lat, pixel_lon)        # (n_pixels, 256)
    ...
    yieldd = newmodel([time_dependent, soil, texture_v, singles,
                       landcover_v2, onehot, area, c_emb[None]])   # single pass, no MC average

Also exposes .in_hull(lat, lon) so the PRIMARY maps can be clipped to the hull.
"""
import numpy as np
import math
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.path import Path


def generate_spr_embeds(latitude, longitude, frequency_num=64):
    """Identical to Generate_Maps.ipynb cell 8 (trigonometric location encoder)."""
    min_radius = (2 * np.pi) / 360.0
    max_radius = (2 * np.pi) / 0.000001
    inc = math.log(max_radius / min_radius) / (frequency_num - 1)
    ts = min_radius * np.exp(np.arange(frequency_num, dtype=float) * inc)
    freq = np.repeat(ts[:, None], 2, axis=1).astype(np.float32)
    lat = np.asarray(latitude, np.float32)[:, None]
    lon = np.asarray(longitude, np.float32)[:, None]
    coords = np.concatenate([lat, lon], -1)
    h = coords.shape[0]
    cm = np.repeat(np.repeat(coords.reshape(h, 2, 1, 1), frequency_num, 2), 2, 3)
    spr = cm * freq[None, None, :, :]
    spr = np.concatenate([np.sin(spr[..., 0::2]), np.cos(spr[..., 1::2])], -1)
    return spr.reshape(h, -1)


class EmbeddingInterpolator:
    def __init__(self, inbag_latlon, frequency_num=64, k=8, power=2.0):
        self.latlon = np.asarray(inbag_latlon, float)          # (N,2) [lat, lon]
        self.freq = frequency_num
        self.k = min(k, len(self.latlon))
        self.power = power
        lonlat = self.latlon[:, ::-1]                          # (lon, lat)
        self._hull = Path(lonlat[ConvexHull(lonlat).vertices])
        self._tree = cKDTree(self.latlon)                      # nn in (lat,lon)
        self._inbag_emb = generate_spr_embeds(self.latlon[:, 0], self.latlon[:, 1], frequency_num)

    def in_hull(self, lat, lon):
        return self._hull.contains_points(np.column_stack([lon, lat]))

    def embed(self, lat, lon):
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        emb = generate_spr_embeds(lat, lon, self.freq)         # own-coordinate embedding
        oob = ~self.in_hull(lat, lon)
        if oob.any():
            d, idx = self._tree.query(np.column_stack([lat[oob], lon[oob]]), k=self.k)
            d = np.atleast_2d(d); idx = np.atleast_2d(idx)
            w = 1.0 / np.power(np.maximum(d, 1e-9), self.power)
            w /= w.sum(1, keepdims=True)                       # distance-decay weights
            # weighted average of the NEAREST in-bag districts' embeddings
            emb[oob] = np.einsum("nk,nkd->nd", w, self._inbag_emb[idx])
        return emb.astype(np.float32)


# --- sensitivity analysis helper (Reviewer 3, item 3) -------------------------
def sensitivity(inbag_latlon, lat, lon, ks=(4, 8, 16), powers=(1.0, 2.0, 3.0), frequency_num=64):
    """Return {(k,power): embedding} so the caller can map pixel-wise variance
    of the frontier predictions across settings and show the maps stabilize."""
    out = {}
    for k in ks:
        for p in powers:
            out[(k, p)] = EmbeddingInterpolator(inbag_latlon, frequency_num, k, p).embed(lat, lon)
    return out
