"""Rasterize each normalized glyph as seen from +z, to read it.

`normalize_glyphs.py` cannot tell on its own whether re-framing a source
mirrored it -- that depends on which way the export's own axes point, which
is not recoverable from the mesh. This draws the result the way the scene's
overhead camera sees it (+x right, +y up, looking down -z), so a mirrored
glyph is obvious and can be added to `normalize_glyphs.MIRROR`.

    uv run python oim/models/xarm6_pusht_tabletop/render_glyphs.py
"""

import os
from typing import List

import numpy as np
from normalize_glyphs import ASSETS, SOURCES, load_obj


def raster(verts: np.ndarray, faces: List[List[int]],
           nx: int = 34, ny: int = 18) -> str:
    """Filled silhouette in the xy-plane, viewed from +z."""
    p = verts[:, :2]
    lo, hi = p.min(0), p.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    xs = (np.arange(nx) + 0.5) / nx * span[0] + lo[0]
    ys = (np.arange(ny) + 0.5) / ny * span[1] + lo[1]
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    hit = np.zeros(pts.shape[0], bool)
    for f in faces:
        for k in range(1, len(f) - 1):
            a, b, c = p[f[0]], p[f[k]], p[f[k + 1]]
            v0, v1 = b - a, c - a
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-16:
                continue
            d = pts - a
            u = (d[:, 0] * v1[1] - v1[0] * d[:, 1]) / den
            v = (v0[0] * d[:, 1] - d[:, 0] * v0[1]) / den
            hit |= (u >= 0) & (v >= 0) & (u + v <= 1)
    g = hit.reshape(ny, nx)
    return "\n".join(
        "   " + "".join("#" if g[j, i] else "." for i in range(nx))
        for j in range(ny - 1, -1, -1)
    )


def main() -> None:
    """Draw every generated glyph."""
    for key in sorted(k for k, _ in SOURCES.values()):
        verts, faces = load_obj(os.path.join(ASSETS, f"glyph_{key}.obj"))
        print(f"\n=== glyph_{key} ===")
        print(raster(verts, faces))


if __name__ == "__main__":
    main()
