"""Emit the 2D hulls `oim/utils/scenes.py` carries for the sign's glyphs.

MJX collides a mesh geom as its CONVEX HULL, so the object-level planner's
obstacle field has to describe that hull rather than the glyph's true
outline -- otherwise the planner routes the block through a notch the
simulator does not have. `oim.utils.scenes` therefore carries one polygon
per glyph, and this generates them from the same normalized meshes the MJCF
loads, so the two cannot drift apart.

The polygons come out in the row's own placed frame: every glyph sits under
`euler="0 0 90"`, which maps mesh (x, y) -> placed (-y, x), so a glyph's
cap height runs along world -x and its width along world y. That yaw is
what makes the sign READ from the recording camera, which sits at +x with
its right axis along +y -- under the old -90 the row was mirrored and
upside down. They are relative to each glyph's own centre; `_glyph()` in
scenes.py translates them into their slot.

    uv run python oim/models/xarm6_pusht_tabletop/glyph_hulls.py

Paste the output over the `_GLYPH_*` block in `oim/utils/scenes.py`.
`tests/test_scenes.py` checks every one against its MJCF geom, so a stale
paste fails there rather than silently mis-planning.
"""

import os
from typing import Dict, List, Tuple

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

# Read from the COMPILED model, not the raw OBJ. MuJoCo re-frames a mesh on
# load (it re-centres, and rotates to principal axes), so the vertices the
# simulator holds are not the ones on disk -- taking the hull from the file
# would describe a differently-placed obstacle than the one MJX collides.
SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "icra_sign.xml")

# Obstacle geom -> (mesh key, authored slot y). `_glyph()` in scenes.py
# adds (0.5, y) back, so the emitted outline is relative to that. The
# pushed C is absent: it is the object, not an obstacle, and its footprint
# comes from `c_shape_footprint`.
ROW: Dict[str, Tuple[str, float]] = {
    "letter_I": ("i", -0.55),
    "letter_R": ("r", -0.25),
    "letter_A": ("a", -0.10),
    "digit_2": ("2", 0.15),
    "digit_0": ("0", 0.30),
    "digit_6": ("6", 0.60),
}
ROW_GLYPHS = tuple(k for k, _ in ROW.values())

# MJX caps each mesh's hull at this many vertices (`maxhullvert` in the
# MJCF), so the polygon here is simplified to the same budget.
MAX_HULL_VERT = 32

# How many vertices to keep in the emitted polygon. The hull of a letter is
# mostly its bounding box, so a handful of points carries it; the existing
# hand-written entries used 4-8. Kept generous enough that the polygon still
# CONTAINS the true hull rather than cutting corners off it -- an inscribed
# simplification would under-report the obstacle to the planner.
TARGET_VERTS = 10


def placed_hull(geom_name: str, slot_y: float) -> np.ndarray:
    """Convex hull of one placed obstacle's footprint, about its own slot.

    Vertices are taken from the compiled model and pushed through the
    geom's own world frame, so the yaw, the re-framing and the placement
    are all the simulator's rather than re-derived here. The authored slot
    is then subtracted, since `scenes._glyph` adds it back.
    """
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = model.geom(geom_name).id
    mid = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[adr:adr + num]
    world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    p = world[:, :2] - np.array([0.5, slot_y])
    return p[ConvexHull(p).vertices]


def simplify(poly: np.ndarray, n: int) -> np.ndarray:
    """Reduce a convex polygon to `n` vertices, staying INSIDE the original.

    Repeatedly drops the vertex whose removal loses the least area, so the
    result is inscribed in the true hull.

    Inscribed, not circumscribed, because that is the contract
    `tests/test_scenes.py::_assert_hull_matches` enforces: a spec polygon
    may not stick more than a millimetre outside the mesh's own hull, and
    must still cover 85% of its area. The reasoning there cuts both ways --
    "too tight and the planner drives the object into a contact it never
    predicted, too loose and it refuses space the simulator allows" -- and
    the obstacle term is a soft exponential in clearance rather than a hard
    constraint, so a hull a hair inside the geometry costs a little margin
    rather than admitting a collision.
    """
    poly = poly.copy()
    while len(poly) > n:
        m = len(poly)
        areas = [
            abs(float(np.cross(poly[i] - poly[i - 1],
                               poly[(i + 1) % m] - poly[i - 1]))) / 2.0
            for i in range(m)
        ]
        poly = np.delete(poly, int(np.argmin(areas)), axis=0)
    return poly


def contains_all(poly: np.ndarray, pts: np.ndarray) -> bool:
    """Whether every point lies inside (or on) the convex polygon."""
    m = len(poly)
    area2 = sum(
        poly[i][0] * poly[(i + 1) % m][1] - poly[(i + 1) % m][0] * poly[i][1]
        for i in range(m)
    )
    sign = 1.0 if area2 >= 0 else -1.0
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        e = b - a
        side = sign * (e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0]))
        if float(np.min(side)) < -1e-9:
            return False
    return True


def emit(geom_name: str) -> Tuple[str, List[Tuple[float, float]]]:
    """The `_GLYPH_<key>` literal for one obstacle."""
    key, slot_y = ROW[geom_name]
    full = placed_hull(geom_name, slot_y)
    poly = simplify(full, TARGET_VERTS)
    # The two halves of tests/test_scenes.py's own contract, checked here
    # so a bad TARGET_VERTS fails at generation rather than in the suite.
    assert contains_all(full, poly), f"{key}: polygon escapes the true hull"
    ratio = ConvexHull(poly).volume / ConvexHull(full).volume
    assert ratio > 0.85, f"{key}: polygon covers only {ratio:.3f} of the hull"
    return key, [(round(float(x), 4), round(float(y), 4)) for x, y in poly]


def main() -> None:
    """Print the whole `_GLYPH_*` block."""
    for geom_name, (key, _slot) in ROW.items():
        name = f"_GLYPH_{key.upper()}"
        _, pts = emit(geom_name)
        span = np.ptp(np.array(pts), axis=0)
        print(f"# {name}: {len(pts)} verts, "
              f"{span[0]:.4f} x {span[1]:.4f} m (cap height x width)")
        print(f"{name} = (")
        for i in range(0, len(pts), 2):
            row = "    " + " ".join(
                f"({x:.4f}, {y:.4f})," for x, y in pts[i:i + 2]
            )
            print(row)
        print(")")


if __name__ == "__main__":
    main()
