"""Normalize raw letter/digit OBJs into the sign row's own convention.

The `icra_sign` row needs every glyph in ONE frame and ONE scale, or the
letters do not read as a single typeface. The raw exports do not agree
about either: the digits extrude along y, `Letra I` along x, and `Letra
A`/`R`/`letter_C` along z, with cap heights of 0.552-0.570 (digits) against
0.512 (letters), all in arbitrary units.

This rewrites each into the convention the scene's mesh geoms assume,
matching the `glyph_*.obj` files it replaces:

    x = glyph width,  y = cap height,  z = extrusion thickness

scaled so every glyph shares `CAP_HEIGHT` and `THICKNESS`, with the width
following each glyph's own proportions (so a `1` stays narrow and an `R`
stays wide), and centred on its own bounding box.

Run from the repo root:

    uv run python oim/models/xarm6_pusht_tabletop/normalize_glyphs.py

THE RAW EXPORTS ARE GONE (deleted 2026-08-19, ~6.9 MB of ImageToStl output
nothing loaded any more). Their names and framing are still recorded in
`SOURCES` below, which is the whole provenance record now; if one is ever
re-added under its old name it is picked up again automatically.

Meanwhile this still re-runs, because the generated glyphs lose nothing:
they carry every vertex and face the export did, only re-framed and
scaled, so each doubles as its own source. When a raw file is missing the
corresponding `glyph_<key>.obj` is read instead, under `_LETTER_AXES`,
since an already-normalized file is by definition in the target
convention. So changing `CAP_HEIGHT` or `THICKNESS` and re-running still
re-cuts the whole set.

The fallback is geometrically lossless but not byte-identical: the
provenance header names the file it actually read, and re-scaling an
already-6-decimal-rounded mesh by a factor a few parts per million from
1.0 can move a last digit. Checked by regenerating the hulls through
`glyph_hulls.py` across the switch -- output identical, so nothing moved
at the 0.1 mm those are quoted to. Repeated fallback runs are then a
fixed point.

It never edits its inputs -- it writes `glyph_<key>.obj` beside them. The
2D convex hulls in `oim/utils/scenes.py` are derived from these outputs, so
re-running this with different constants means regenerating those too (see
`glyph_hulls.py`).
"""

import os
from typing import Dict, List, Tuple

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# The row's existing metrics, kept so this swap does not move the scene:
# the source font ran 0.100-0.103 tall and 0.025 thick, and the glyph slots
# in icra_sign.xml are spaced for that.
CAP_HEIGHT = 0.103
THICKNESS = 0.025

# source file -> (mesh key, (width axis, height axis, thickness axis)).
#
# The axes are STATED, not detected. Two plausible rules both fail:
#   * "thinnest extent is the extrusion" breaks on `Letra I`, which is
#     0.1485 wide, 0.512 tall and 0.2012 thick -- a narrow enough glyph is
#     thinner across the page than through it, and the rule then treats the
#     letter's width as its depth.
#   * "the larger in-plane extent is the cap height" breaks on `Letra A`,
#     which is 0.512 x 0.4741 -- a genuinely wide A -- and comes out lying
#     on its side, apex pointing left.
# Both were caught by rasterising the result and reading it (see
# `render_glyphs.py`), which is the only check that actually works here.
#
# The digits are named positionally by the export (000 = "0" ... 007 = "7")
# and extrude along y; the letters extrude along z.
_DIGIT_AXES = (0, 2, 1)     # width x, cap height z, thickness y
_LETTER_AXES = (0, 1, 2)    # width x, cap height y, thickness z
SOURCES: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "000.obj": ("0", _DIGIT_AXES), "001.obj": ("1", _DIGIT_AXES),
    "002.obj": ("2", _DIGIT_AXES), "003.obj": ("3", _DIGIT_AXES),
    "004.obj": ("4", _DIGIT_AXES), "005.obj": ("5", _DIGIT_AXES),
    "006.obj": ("6", _DIGIT_AXES), "007.obj": ("7", _DIGIT_AXES),
    "Letra A.obj": ("a", _LETTER_AXES),
    "Letra I.obj": ("i", _LETTER_AXES),
    "Letra R.obj": ("r", _LETTER_AXES),
    "Letra T.obj": ("t", _LETTER_AXES),
    "letter_C.obj": ("c", _LETTER_AXES),
}

# Glyphs whose source is a lowercase form and is wanted as a capital.
# `Letra I.obj` is a lowercase `i`: a tittle spanning y in
# [+0.0308, +0.0515] over a stem in [-0.0515, +0.0236], with a 7.2 mm slit
# between them. Both pieces are the same width, so raising the stem's top
# face to meet the tittle turns it into the solid bar a sans-serif capital
# I is, without touching its footprint (0.0299 x 0.103) or its extent.
#
# Worth doing in geometry rather than leaving to the eye: MJX collides the
# mesh as one convex hull and so fills the slit anyway, which would leave
# the render showing a gap the physics does not have -- the same
# render/physics split the pushed C's own comment in `icra_sign.xml` warns
# about, in miniature.
CLOSE_GAP = {"i"}


def load_obj(path: str) -> Tuple[np.ndarray, List[List[int]]]:
    """Vertices (n, 3) and faces (as vertex-index lists, any arity)."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    for line in open(path, errors="ignore"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            faces.append(
                [i - 1 if i > 0 else len(verts) + i for i in idx]
            )
    return np.asarray(verts, float), faces


def normalize(
    verts: np.ndarray, axes: Tuple[int, int, int], mirror: bool
) -> np.ndarray:
    """Re-frame, scale and centre one glyph's vertices.

    Args:
        verts: The raw vertices, (n, 3).
        axes: `(width, height, thickness)` source axis indices, from
            `SOURCES`.
        mirror: Negate the width axis. Whether a re-framing mirrors the
            glyph depends on which way the source's own axes point, which
            no property of the mesh reveals -- it is settled by rasterising
            the result and reading the letter (see `render_glyphs.py`).

            A re-framing that swaps two axes has determinant -1, which is a
            reflection through the glyph's own extrusion plane. That is
            invisible: it exchanges the front and back faces of a shape
            that is symmetric through them, and the letter still reads the
            same way from +z. Only a flip in the WIDTH axis mirrors what
            the camera sees, which is what this flag is for.

    Returns:
        Vertices in the row's convention, (n, 3).
    """
    w_ax, h_ax, t_ax = axes
    out = np.stack(
        [verts[:, w_ax], verts[:, h_ax], verts[:, t_ax]], axis=1
    )
    if mirror:
        out[:, 0] = -out[:, 0]
    extent = out.max(0) - out.min(0)
    # Width follows the cap-height scale so the letterform is preserved;
    # only the thickness is set independently.
    s = CAP_HEIGHT / extent[1]
    out = out * np.array([s, s, THICKNESS / extent[2]])
    return out - (out.max(0) + out.min(0)) / 2.0


def close_vertical_gap(
    verts: np.ndarray, faces: List[List[int]], samples: int = 2000
) -> np.ndarray:
    """Fill the tallest horizontal slit by raising the band below it.

    The slit is found by SCANNING COVERAGE, not by looking for gaps between
    vertex y-values. Those are not the same thing and the difference is not
    subtle: a plain extruded stem has vertices only at its two ends, so its
    solid interior reads as a 68.9 mm gap between vertex levels -- an order
    of magnitude larger than the 7.2 mm slit actually being looked for, and
    the first thing an `argmax` over level gaps finds. This walks a vertical
    line up the glyph's mid-width instead and asks which y are covered by no
    triangle at all, which is the question that was meant.

    Args:
        verts: Normalized vertices, (n, 3).
        faces: Faces as vertex-index lists.
        samples: Scanline resolution.

    Returns:
        The vertices with the slit closed, (n, 3). Unchanged if the glyph
        is already solid.
    """
    p = verts[:, :2]
    x_mid = 0.5 * (p[:, 0].min() + p[:, 0].max())
    ys = np.linspace(p[:, 1].min(), p[:, 1].max(), samples)
    pts = np.stack([np.full(samples, x_mid), ys], axis=1)
    covered = np.zeros(samples, bool)
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
            covered |= (u >= 0) & (v >= 0) & (u + v <= 1)

    # Longest run of uncovered samples strictly inside the glyph.
    best_len = best_start = 0
    run = 0
    for i, hit in enumerate(covered):
        run = 0 if hit else run + 1
        if run > best_len:
            best_len, best_start = run, i - run + 1
    if best_len == 0:
        return verts
    lo, hi = ys[best_start - 1], ys[best_start + best_len]

    out = verts.copy()
    face = np.abs(out[:, 1] - lo) <= (hi - lo) * 0.25
    out[face, 1] = hi
    return out


def write_obj(path: str, verts: np.ndarray, faces: List[List[int]],
              source: str) -> None:
    """Write a v/f-only OBJ.

    Normals and UVs are dropped -- MuJoCo recomputes shading normals and
    nothing here is textured.
    """
    with open(path, "w") as fh:
        fh.write(f"# generated by normalize_glyphs.py from {source}\n")
        fh.write(f"# x=width y=cap-height({CAP_HEIGHT}) z=thickness"
                 f"({THICKNESS}), metres, centred\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write("f " + " ".join(str(i + 1) for i in f) + "\n")


# Settled by rasterising each result and reading it; see the module
# docstring. A glyph absent here is not mirrored.
MIRROR = set()


def resolve_source(src: str, key: str, axes: Tuple[int, int, int]) -> Tuple[
    str, Tuple[int, int, int]
]:
    """The file to read for one glyph, and the axes it is framed in.

    The raw export when it is still there, otherwise the generated glyph,
    which holds the same geometry already in the target convention. See
    the module docstring.

    Raises:
        FileNotFoundError: If neither exists -- better than silently
            skipping a glyph and leaving a stale file in place.
    """
    if os.path.exists(os.path.join(ASSETS, src)):
        return src, axes
    fallback = f"glyph_{key}.obj"
    if os.path.exists(os.path.join(ASSETS, fallback)):
        return fallback, _LETTER_AXES
    raise FileNotFoundError(
        f"glyph {key!r}: neither the raw export {src!r} nor the generated "
        f"{fallback!r} is in {ASSETS}"
    )


def main() -> None:
    """Normalize every source into `assets/glyph_<key>.obj`."""
    print(f"{'source':16s} -> {'output':16s} {'w':>8} {'h':>8} {'t':>8}")
    ordered = sorted(SOURCES.items(), key=lambda kv: kv[1][0])
    for raw, (key, raw_axes) in ordered:
        src, axes = resolve_source(raw, key, raw_axes)
        verts, faces = load_obj(os.path.join(ASSETS, src))
        out = normalize(verts, axes, mirror=key in MIRROR)
        if key in CLOSE_GAP:
            out = close_vertical_gap(out, faces)
        name = f"glyph_{key}.obj"
        write_obj(os.path.join(ASSETS, name), out, faces, src)
        e = out.max(0) - out.min(0)
        print(f"{src:16s} -> {name:16s} {e[0]:8.4f} {e[1]:8.4f} {e[2]:8.4f}")


if __name__ == "__main__":
    main()
