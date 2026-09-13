"""`oim.objects.fit_print`: the STL -> boxes / OBJ / PLY tool, on shapes
whose right answer is known."""

import os
import struct

import numpy as np
import pytest

from oim.objects.fit_print import (
    cover_with_boxes, coverage_of, footprint_mask, read_stl, write_obj,
    write_ply,
)
from scipy import ndimage


def _box_stl(path: str, hx: float, hy: float, hz: float,
             hole: tuple = None) -> None:
    """A binary STL of an axis-aligned block, optionally with a square
    through-hole `(cx, cy, half)` -- enough to exercise fill and cover."""
    def quad(a, b, c, d):
        return [(a, b, c), (a, c, d)]
    tris = []
    def block(x0, x1, y0, y1, z0, z1):
        p = lambda x, y, z: np.array([x, y, z], float)
        tris.extend(quad(p(x0,y0,z1), p(x1,y0,z1), p(x1,y1,z1), p(x0,y1,z1)))  # top
        tris.extend(quad(p(x0,y0,z0), p(x0,y1,z0), p(x1,y1,z0), p(x1,y0,z0)))  # bottom
    if hole is None:
        block(-hx, hx, -hy, hy, -hz, hz)
    else:
        cx, cy, h = hole   # four blocks around the hole, tops only matter
        block(-hx, cx - h, -hy, hy, -hz, hz)
        block(cx + h, hx, -hy, hy, -hz, hz)
        block(cx - h, cx + h, -hy, cy - h, -hz, hz)
        block(cx - h, cx + h, cy + h, hy, -hz, hz)
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a); n /= max(np.linalg.norm(n), 1e-12)
            f.write(struct.pack("<3f", *n) + struct.pack("<9f", *a, *b, *c)
                    + struct.pack("<H", 0))


def test_a_plain_block_is_one_box_at_full_coverage(tmp_path) -> None:
    p = str(tmp_path / "b.stl")
    _box_stl(p, 50.0, 30.0, 10.0)
    tris = read_stl(p)
    mask, x0, y0 = footprint_mask(tris, 2.0)
    boxes, _ = cover_with_boxes(mask, 2.0, x0, y0, 8.0, 0.995)
    assert len(boxes) == 1
    cx, cy, hx, hy = boxes[0]
    assert (cx, cy) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert (hx, hy) == pytest.approx((50.0, 30.0), abs=1.0)   # raster pitch
    assert coverage_of(boxes, mask, 2.0, x0, y0) == pytest.approx(1.0)


def test_a_hole_is_filled_before_fitting_and_excluded_from_coverage(
    tmp_path,
) -> None:
    """A counter would make `boxes_footprint` raise; the tool fills it and
    still scores coverage against the true outline."""
    p = str(tmp_path / "h.stl")
    _box_stl(p, 50.0, 50.0, 10.0, hole=(0.0, 0.0, 10.0))
    tris = read_stl(p)
    truth, x0, y0 = footprint_mask(tris, 2.0)
    filled = ndimage.binary_fill_holes(truth)
    assert filled.sum() > truth.sum()                     # the hole existed
    boxes, _ = cover_with_boxes(filled, 2.0, x0, y0, 8.0, 0.995)
    assert len(boxes) == 1                                # and was filled over
    cov = coverage_of(boxes, truth, 2.0, x0, y0)
    assert cov == pytest.approx(1.0)                      # true outline fully covered


def test_obj_and_ply_share_the_stl_frame(tmp_path) -> None:
    p = str(tmp_path / "b.stl")
    _box_stl(p, 50.0, 30.0, 10.0)
    tris = read_stl(p)
    obj = tris.copy(); obj[:, :, 2] += 10.0
    write_obj(str(tmp_path / "b.obj"), obj / 1000.0)
    write_ply(str(tmp_path / "b.ply"), tris)
    v = np.array([list(map(float, l.split()[1:4]))
                  for l in open(tmp_path / "b.obj") if l.startswith("v ")])
    assert v.min(0) == pytest.approx([-0.05, -0.03, 0.0])
    assert v.max(0) == pytest.approx([0.05, 0.03, 0.02])
    L = open(tmp_path / "b.ply").read().splitlines()
    assert "property float nx" in L and "format ascii 1.0" in L
    i = L.index("end_header"); nv = int([l for l in L if l.startswith("element vertex")][0].split()[2])
    pv = np.array([list(map(float, l.split()[:3])) for l in L[i + 1:i + 1 + nv]])
    assert pv.min(0) == pytest.approx([-50.0, -30.0, -10.0])   # mm, mid-height origin
