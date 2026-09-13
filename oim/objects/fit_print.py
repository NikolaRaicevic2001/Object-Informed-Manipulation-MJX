"""Turn a 3D-printed object's STL into everything the two repos need.

One STL in, four things out, all from the same geometry so the frames agree:

  * `<name>_centered.obj`  MJX visual mesh in metres, plan bounding box on
                           the origin, underside at z = 0 (`library.apply_
                           to_spec`'s convention).
  * `<Name>.ply`           FoundationPose mesh: ASCII PLY in millimetres with
                           vertex normals, origin at mid-height, the same
                           frame as the raw STL and as meshes/T_block/.
  * axis-aligned boxes     collision cover for a `PushObject`, greedy maximal
                           rectangles on a 2 mm raster of the footprint.
  * coverage               fraction of the true footprint the boxes recover.

    python -m oim.objects.fit_print glyph_t_print.stl --name T_large_block \
        --obj-dir oim/models/xarm6_pusht_tabletop_real/assets --ply-dir /tmp
    python -m oim.objects.fit_print meshes/I_block/I_block.ply --name I_block \
        --obj-dir ...            # FP mesh as the source; no --ply-dir needed

Holes: `boxes_footprint` describes one connected region with no holes, so a
letter with a counter (A, R) has it filled before fitting -- the pusher only
ever meets the outer boundary, and the fill is reported alongside coverage.
Boxes are raster cells, so they may overhang the true outline by up to half
a pitch (1 mm); coverage is scored on the same raster.
"""

import argparse
import os
import struct
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage

RASTER_MM = 2.0      # footprint raster pitch; the YCB entries used the same
MIN_BOX_MM = 8.0     # smallest box side worth keeping, as for YCB
TARGET_COVERAGE = 0.995

Box = Tuple[float, float, float, float]


def read_stl(path: str) -> np.ndarray:
    """Binary STL -> triangles of shape (n, 3, 3), in the file's own units."""
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(
        b[84:84 + 50 * n],
        dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
    )
    return rec["v"].reshape(-1, 3, 3).astype(float)


def read_ply(path: str) -> np.ndarray:
    """ASCII PLY with triangular faces -> triangles (n, 3, 3), file units.

    For objects whose FoundationPose mesh already exists (the I and R,
    ported in July), that mesh IS the physical object as far as tracking
    is concerned, so the MJX side is derived from it rather than from a
    separate STL that might not be the same print.
    """
    lines = open(path).read().splitlines()
    end = lines.index("end_header")
    header = lines[:end]
    n_vert = int(next(l for l in header if l.startswith("element vertex")).split()[2])
    n_face = int(next(l for l in header if l.startswith("element face")).split()[2])
    verts = np.array([list(map(float, l.split()[:3]))
                      for l in lines[end + 1:end + 1 + n_vert]])
    faces = [list(map(int, l.split()[1:4]))
             for l in lines[end + 1 + n_vert:end + 1 + n_vert + n_face]]
    return verts[np.array(faces)]


def read_mesh(path: str) -> np.ndarray:
    """`read_stl` or `read_ply` by extension."""
    return read_ply(path) if path.lower().endswith(".ply") else read_stl(path)


def footprint_mask(tris: np.ndarray, pitch: float):
    """Rasterise the top faces onto a grid; return (mask, x0, y0).

    The prints are extrusions, so the top faces alone trace the plan
    outline. `mask[i, j]` is solid at x = x0 + j*pitch, y = y0 + i*pitch.
    """
    up = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2] > 0
    top = tris[up][:, :, :2]
    lo, hi = top.reshape(-1, 2).min(0), top.reshape(-1, 2).max(0)
    nx, ny = (np.ceil((hi - lo) / pitch)).astype(int) + 1
    xs = lo[0] + pitch * np.arange(nx)
    ys = lo[1] + pitch * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    inside = np.zeros(len(pts), dtype=bool)
    for a, b, c in top:
        # barycentric sign test, tolerant on the edges so seams close
        d1 = (pts - a) @ np.array([b[1] - a[1], a[0] - b[0]])
        d2 = (pts - b) @ np.array([c[1] - b[1], b[0] - c[0]])
        d3 = (pts - c) @ np.array([a[1] - c[1], c[0] - a[0]])
        eps = 1e-9
        inside |= ((d1 >= -eps) & (d2 >= -eps) & (d3 >= -eps)) | (
            (d1 <= eps) & (d2 <= eps) & (d3 <= eps))
    return inside.reshape(ny, nx), lo[0], lo[1]


def largest_rectangle(free: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Largest all-True axis-aligned rectangle: (area, r0, c0, h, w)."""
    ny, nx = free.shape
    heights = np.zeros(nx, dtype=int)
    best = (0, 0, 0, 0, 0)
    for r in range(ny):
        heights = np.where(free[r], heights + 1, 0)
        stack: List[int] = []
        for c in range(nx + 1):
            h = heights[c] if c < nx else 0
            start = c
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                th = heights[top]
                tw = c - (stack[-1] + 1 if stack else 0)
                if th * tw > best[0]:
                    left = stack[-1] + 1 if stack else 0
                    best = (th * tw, r - th + 1, left, th, tw)
            stack.append(c)
    return best


def cover_with_boxes(mask: np.ndarray, pitch: float, x0: float, y0: float,
                     min_side: float, target: float) -> Tuple[List[Box], float]:
    """Greedy maximal-rectangle cover of `mask`; boxes in the raster's units.

    Each box is `(cx, cy, hx, hy)`. Stops once `target` of the mask is
    covered or the next rectangle is thinner than `min_side`.
    """
    free = mask.copy()
    total = mask.sum()
    boxes: List[Box] = []
    min_cells = max(1, int(round(min_side / pitch)))
    while free.sum() > (1.0 - target) * total:
        area, r0, c0, h, w = largest_rectangle(free)
        if area == 0 or min(h, w) < min_cells:
            break
        free[r0:r0 + h, c0:c0 + w] = False
        cx = x0 + pitch * (c0 + (w - 1) / 2.0)
        cy = y0 + pitch * (r0 + (h - 1) / 2.0)
        # Boxes span whole cells, so neighbours that tile the raster share
        # an edge exactly and the union stays one connected region, which
        # `boxes_footprint` requires. The price is up to half a pitch
        # (1 mm) of overhang past a curved or bounding edge -- shrinking
        # to the cell centres instead opens 2 mm gaps at every seam.
        boxes.append((cx, cy, pitch * w / 2.0, pitch * h / 2.0))
    covered = 1.0 - free.sum() / total
    return boxes, covered


def coverage_of(boxes: Sequence[Box], truth: np.ndarray, pitch: float,
                x0: float, y0: float) -> float:
    """Fraction of `truth` cells inside any box."""
    ny, nx = truth.shape
    gx, gy = np.meshgrid(x0 + pitch * np.arange(nx), y0 + pitch * np.arange(ny))
    hit = np.zeros_like(truth)
    for cx, cy, hx, hy in boxes:
        hit |= (np.abs(gx - cx) <= hx + 1e-9) & (np.abs(gy - cy) <= hy + 1e-9)
    return float((hit & truth).sum() / truth.sum())


def components(boxes: Sequence[Box], mask_shape, pitch: float,
               x0: float, y0: float) -> int:
    """How many connected regions the boxes form on the raster.

    `boxes_footprint` can only describe ONE, so anything above 1 means the
    cover has to be refit with a finer floor (`--min-box-mm`) until the
    slivers that join a diagonal or a curve to the body are kept.
    """
    ny, nx = mask_shape
    gx, gy = np.meshgrid(x0 + pitch * np.arange(nx), y0 + pitch * np.arange(ny))
    hit = np.zeros(mask_shape, dtype=bool)
    for cx, cy, hx, hy in boxes:
        hit |= (np.abs(gx - cx) <= hx + 1e-9) & (np.abs(gy - cy) <= hy + 1e-9)
    return int(ndimage.label(hit)[1])


def write_obj(path: str, tris: np.ndarray) -> None:
    """Write triangles as an OBJ with shared vertices (one `v` per point)."""
    flat = tris.reshape(-1, 3)
    verts, inv = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3) + 1
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % tuple(v))
        for a, b, c in faces:
            f.write("f %d %d %d\n" % (a, b, c))


def write_ply(path: str, tris: np.ndarray) -> None:
    """ASCII PLY with per-vertex normals and a flat colour, like T_block.ply.

    Vertices are shared where faces meet; a shared vertex's normal is the
    area-weighted mean of its faces', which is what trimesh reads back.
    """
    flat = tris.reshape(-1, 3)
    verts, inv = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment made by oim.objects.fit_print\n")
        f.write("element vertex %d\n" % len(verts))
        for p in ("x", "y", "z", "nx", "ny", "nz"):
            f.write("property float %s\n" % p)
        for c in ("red", "green", "blue"):
            f.write("property uchar %s\n" % c)
        f.write("element face %d\nproperty list uchar int vertex_indices\n"
                "end_header\n" % len(faces))
        for v, n in zip(verts, vn):
            f.write("%.4f %.4f %.4f %.4f %.4f %.4f 200 200 200\n"
                    % (*v, *n))
        for a, b, c in faces:
            f.write("3 %d %d %d\n" % (a, b, c))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mesh", help="print STL, or an existing FoundationPose PLY")
    ap.add_argument("--name", required=True,
                    help="PushObject key and mesh stem, e.g. T_large_block")
    ap.add_argument("--obj-dir", action="append", default=[],
                    help="Where <name>_centered.obj goes; repeatable")
    ap.add_argument("--ply-dir", default=None,
                    help="Where <name>/<name>.ply goes (FoundationPose)")
    ap.add_argument("--pitch-mm", type=float, default=RASTER_MM)
    ap.add_argument("--min-box-mm", type=float, default=MIN_BOX_MM)
    args = ap.parse_args()

    tris_mm = read_mesh(args.mesh)
    lo, hi = tris_mm.reshape(-1, 3).min(0), tris_mm.reshape(-1, 3).max(0)
    centre = (lo + hi) / 2.0
    centred_mm = tris_mm - centre                     # plan bbox on origin, mid-height at z = 0
    height_mm = hi[2] - lo[2]

    truth, x0, y0 = footprint_mask(centred_mm, args.pitch_mm)
    filled = ndimage.binary_fill_holes(truth)
    counter_mm2 = float((filled & ~truth).sum()) * args.pitch_mm ** 2
    boxes_mm, _ = cover_with_boxes(filled, args.pitch_mm, x0, y0,
                                   args.min_box_mm, TARGET_COVERAGE)
    coverage = coverage_of(boxes_mm, truth, args.pitch_mm, x0, y0)

    for d in args.obj_dir:
        os.makedirs(d, exist_ok=True)
        obj = centred_mm.copy()
        obj[:, :, 2] += height_mm / 2.0                # underside on z = 0
        write_obj(os.path.join(d, f"{args.name}_centered.obj"), obj / 1000.0)
    if args.ply_dir:
        d = os.path.join(args.ply_dir, args.name)
        os.makedirs(d, exist_ok=True)
        write_ply(os.path.join(d, f"{args.name}.ply"), centred_mm)

    print(f"{args.name}: plan {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} mm, "
          f"height {height_mm:.1f} mm  ->  half_height={height_mm/2000:.4f}")
    n_parts = components(boxes_mm, truth.shape, args.pitch_mm, x0, y0)
    print(f"  footprint {truth.sum()*args.pitch_mm**2/100:.1f} cm2, "
          f"counter filled {counter_mm2/100:.1f} cm2, "
          f"{len(boxes_mm)} boxes, coverage {coverage:.3f}, "
          f"{n_parts} connected region{'' if n_parts == 1 else 'S -- NOT USABLE, lower --min-box-mm'}")
    print("  boxes=(  # (cx, cy, hx, hy) [m], body frame")
    for cx, cy, hx, hy in boxes_mm:
        print(f"      ({cx/1000:.4f}, {cy/1000:.4f}, {hx/1000:.4f}, {hy/1000:.4f}),")
    print("  ),")


if __name__ == "__main__":
    main()
