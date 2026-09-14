"""The pushable-object registry, and the scene surgery that installs one.

`oim/objects/library.py` derives three descriptions from one list of boxes
-- the MJCF collision geoms, the goal markers that mirror them, and the
analytic footprint the planner reasons about. These check the three cannot
disagree, which is the whole reason the registry stores boxes rather than
storing a footprint and a set of geoms side by side.

`tests/test_scenes.py` already makes the equivalent checks for whatever
object a scene ships built in; this covers the ones swapped in at load.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import ndimage

from oim import ROOT
from oim.objects import c_shape_footprint, t_shape_footprint
from oim.objects.library import (
    PUSH_OBJECTS,
    SCENE_DEFAULT,
    STICK_GEOM,
    object_names,
    push_object,
)
from oim.objects.planar_pushing import boxes_footprint
from oim.tasks.pusht import PushT
from oim.utils.scenes import SCENES

# Scenes the swap has to work in. Every 3D tabletop scene, since the point
# of the mechanism is that the object is independent of the layout.
TABLETOP = ["open_table", "single_obstacle", "shelf_gap", "ycb_clutter",
            "icra_sign"]

_ATOL = 1e-6


# An object can be too big for a scene's own layout, which is a fact about
# the pair and not a bug in either: `icra_sign` parks the block in a slot
# between glyphs, and the lab's 0.25 m hammer is longer than that slot.
# Listed rather than inferred, so growing an object into a scene it no
# longer fits fails here instead of passing quietly.
TOO_BIG = {("icra_sign", "hammer_real")}


def _combos():
    return [(s, o) for s in TABLETOP for o in sorted(PUSH_OBJECTS)
            if (s, o) not in TOO_BIG]


# ----------------------------------------------------------------------
# boxes_footprint: the shared derivation
# ----------------------------------------------------------------------


def test_boxes_footprint_reproduces_the_hand_written_builders() -> None:
    """The generic builder is the two special-cased ones, generalized.

    `t_shape_footprint` and `c_shape_footprint` are the same union-outline
    computation written out for one shape each. If `boxes_footprint`
    disagreed with either, every object built on it would be describing
    something other than its own geoms.
    """

    def canon(poly: np.ndarray) -> np.ndarray:
        p = np.round(np.asarray(poly, dtype=np.float64), 9)
        x, y = p[:, 0], p[:, 1]
        if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
            p = p[::-1]
        return np.roll(p, -int(np.lexsort((p[:, 1], p[:, 0]))[0]), axis=0)

    tee = boxes_footprint(
        [(0.0, 0.0099, 0.0445, 0.0099), (0.0, -0.0397, 0.0099, 0.0397)]
    )
    ref = t_shape_footprint(
        crossbar_half=(0.0445, 0.0099), stem_half=(0.0099, 0.0397),
        crossbar_y=0.0099, stem_y=-0.0397,
    )
    np.testing.assert_allclose(
        canon(tee.vertices), canon(ref.vertices), atol=1e-6
    )

    c = boxes_footprint([
        (-0.0323, 0.0, 0.0160, 0.0515),
        (0.0, 0.0355, 0.0483, 0.0160),
        (0.0, -0.0355, 0.0483, 0.0160),
    ])
    np.testing.assert_allclose(
        canon(c.vertices), canon(c_shape_footprint().vertices), atol=1e-6
    )


def test_boxes_footprint_rejects_a_shape_a_polygon_cannot_hold() -> None:
    """Two disjoint boxes are not one closed loop, and say so.

    A `Polygon` is a single boundary, so a decomposition that fell into
    two pieces (or enclosed a hole) has no valid answer -- worth failing
    loudly rather than silently returning one of the pieces.
    """
    with pytest.raises(ValueError, match="simply-connected"):
        boxes_footprint([(0.0, 0.0, 0.01, 0.01), (0.5, 0.0, 0.01, 0.01)])


def test_boxes_footprint_needs_at_least_one_box() -> None:
    """No boxes is no shape, rather than an empty polygon."""
    with pytest.raises(ValueError, match="at least one box"):
        boxes_footprint([])


# ----------------------------------------------------------------------
# The registry itself
# ----------------------------------------------------------------------


def test_scene_default_is_not_an_entry() -> None:
    """`"scene"` means "change nothing", so it must not name an object."""
    assert SCENE_DEFAULT not in PUSH_OBJECTS
    assert push_object(SCENE_DEFAULT) is None
    assert object_names()[0] == SCENE_DEFAULT
    assert set(object_names()[1:]) == set(PUSH_OBJECTS)


def test_unknown_object_names_the_available_ones() -> None:
    """A typo names what it could have meant instead of KeyError."""
    with pytest.raises(ValueError, match="not in oim.objects.library"):
        push_object("definitely_not_an_object")


@pytest.mark.parametrize("name", sorted(PUSH_OBJECTS))
def test_masses_sum_to_the_object_mass(name: str) -> None:
    """The per-box split is exactly the object's mass.

    It sets `mu * m * g`, the entire friction budget the analytic limit
    surface and every ADMM normalization are built on, so a split that
    lost a few grams would quietly rescale the whole action space.
    """
    obj = PUSH_OBJECTS[name]
    assert sum(obj.masses()) == pytest.approx(obj.mass)
    assert len(obj.masses()) == len(obj.boxes)
    assert all(m > 0.0 for m in obj.masses())


@pytest.mark.parametrize("name", sorted(PUSH_OBJECTS))
def test_footprint_is_the_union_of_the_boxes(name: str) -> None:
    """Same two-way check `test_scenes.py` makes against the MJCF.

    Every footprint vertex lies in some box, and every box corner lies in
    the footprint -- together those pin the outline to the geometry.
    """
    obj = PUSH_OBJECTS[name]
    fp = obj.footprint()
    verts = np.asarray(fp.vertices)

    for v in verts:
        assert any(
            abs(v[0] - cx) <= hx + _ATOL and abs(v[1] - cy) <= hy + _ATOL
            for cx, cy, hx, hy in obj.boxes
        ), f"{name}: footprint vertex {v} is outside every box"

    assert bool(np.all(np.asarray(fp.sdf(jnp.asarray(verts))) <= _ATOL))

    for cx, cy, hx, hy in obj.boxes:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corner = np.array([cx + sx * hx, cy + sy * hy])
                d = float(np.asarray(fp.sdf(jnp.asarray(corner[None, :])))[0])
                assert d <= _ATOL, (
                    f"{name}: box corner {corner} lies {d:.2e} outside the "
                    "footprint"
                )


@pytest.mark.parametrize("name", sorted(PUSH_OBJECTS))
def test_physics_are_plausible_and_measured(name: str) -> None:
    """Mass, friction and the limit-surface radius are real numbers.

    `limit_surface_radius` in particular is measured against the compiled
    model, not chosen, so this only guards the sign and the scale -- that
    it lies inside the object's own footprint, which any patch radius
    must.
    """
    obj = PUSH_OBJECTS[name]
    assert obj.mass > 0.0 and obj.mu > 0.0
    assert obj.half_height > 0.0
    reach = float(np.max(np.abs(np.asarray(obj.footprint().vertices))))
    assert 0.0 < obj.limit_surface_radius < reach, (
        f"{name}: r={obj.limit_surface_radius} is not inside its own "
        f"footprint (reach {reach:.4f})"
    )
    assert 0.0 < obj.coverage <= 1.0


# ----------------------------------------------------------------------
# The swap, against real scenes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scene,name", _combos())
def test_swapped_object_rests_on_the_table_with_the_right_mass(
    scene: str, name: str
) -> None:
    """The installed object is the registry's, in the scene's own world.

    Checks the three things the surgery has to get right and that nothing
    else does: the block weighs what the registry says, its underside sits
    exactly on the tabletop, and `tip_target_z` follows the new object's
    mid-height rather than staying at the T's.
    """
    obj = PUSH_OBJECTS[name]
    task = PushT(
        clutter=True, robot="xarm6", env=scene, planning_dt=0.05,
        push_object=name,
    )
    model = task.mj_model
    block = model.body("block")
    assert float(block.mass[0]) == pytest.approx(obj.mass, rel=1e-6)

    bottoms = {
        round(float(block.pos[2] + model.geom_pos[i][2]
                    - model.geom_size[i][2]), 9)
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == block.id and model.geom_contype[i] != 0
    }
    assert bottoms == {0.0}, f"{scene}/{name}: block bottoms at {bottoms}"
    assert task.tip_target_z == pytest.approx(obj.half_height)


@pytest.mark.parametrize("scene,name", _combos())
def test_swap_leaves_the_scene_alone(scene: str, name: str) -> None:
    """Only the object changes: not the table, obstacles or goal pose.

    This is the whole promise of `--object` -- that a run with a banana is
    the same task as a run with the T, differing in what is pushed. If the
    surgery moved an obstacle or the goal, comparing the two would be
    comparing two experiments.
    """
    spec = SCENES[scene]
    base = PushT(clutter=True, robot="xarm6", env=scene, planning_dt=0.05)
    swapped = PushT(
        clutter=True, robot="xarm6", env=scene, planning_dt=0.05,
        push_object=name,
    )
    np.testing.assert_allclose(
        np.asarray(swapped.object_model.goal),
        np.asarray(spec.goal), atol=_ATOL,
    )
    assert len(swapped.object_model.obstacles.shapes) == len(
        base.object_model.obstacles.shapes
    )
    for a, b in zip(base.object_model.obstacles.shapes,
                    swapped.object_model.obstacles.shapes, strict=True):
        np.testing.assert_allclose(
            np.asarray(a.center), np.asarray(b.center), atol=_ATOL
        )
    table = swapped.mj_model.geom("table")
    np.testing.assert_allclose(
        np.asarray(table.size), np.asarray(base.mj_model.geom("table").size),
        atol=_ATOL,
    )


@pytest.mark.parametrize("scene,name", _combos())
def test_swapped_object_matches_its_own_mjcf_geoms(
    scene: str, name: str
) -> None:
    """The analytic footprint is the union of the INSTALLED geoms.

    The registry's `footprint()` and the geoms `apply_to_spec` writes both
    come from `boxes`, so this is really checking the surgery placed them
    where the registry said -- the failure it catches is a body offset or
    a dropped geom, not a bad outline.
    """
    task = PushT(
        clutter=True, robot="xarm6", env=scene, planning_dt=0.05,
        push_object=name,
    )
    model = task.mj_model
    block_id = model.body("block").id
    installed = sorted(
        (round(float(model.geom_pos[i][0]), 9),
         round(float(model.geom_pos[i][1]), 9),
         round(float(model.geom_size[i][0]), 9),
         round(float(model.geom_size[i][1]), 9))
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == block_id and model.geom_contype[i] != 0
    )
    assert installed == sorted(
        tuple(round(v, 9) for v in b) for b in PUSH_OBJECTS[name].boxes
    )
    # Axis-aligned, the contract `boxes_footprint` is built on.
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == block_id and model.geom_contype[i] != 0:
            q = model.geom_quat[i]
            yaw = math.atan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]),
                1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
            )
            assert abs(yaw) < 1e-6


@pytest.mark.parametrize("scene,name", _combos())
def test_goal_marker_mirrors_the_swapped_block(scene: str, name: str) -> None:
    """The goal ghost is the same shape as the object, so overlap reads.

    Success is judged by eye from the recording, which only works if the
    marker is the object's own outline rather than the T it replaced.
    """
    task = PushT(
        clutter=True, robot="xarm6", env=scene, planning_dt=0.05,
        push_object=name,
    )
    model = task.mj_model

    def plan(body: str):
        bid = model.body(body).id
        return sorted(
            (round(float(model.geom_pos[i][0]), 9),
             round(float(model.geom_pos[i][1]), 9),
             round(float(model.geom_size[i][0]), 9),
             round(float(model.geom_size[i][1]), 9))
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == bid
            and model.geom_type[i] != 7  # mjGEOM_MESH: the visual overlay
        )

    assert plan("goal") == plan("block")
    assert plan("object_plan") == plan("block")


@pytest.mark.parametrize("scene,name", _combos())
def test_swapped_object_starts_clear_of_every_obstacle(
    scene: str, name: str
) -> None:
    """A scene's own start and goal still hold the object it is given.

    The pose files are shared across objects deliberately (one layout, one
    set of variants), so a bigger object has to be checked against them
    rather than assumed to fit -- the hammer is 0.18 m along one axis and
    the power drill's footprint matches the T's.
    """
    spec = SCENES[scene]
    task = PushT(
        clutter=True, robot="xarm6", env=scene, planning_dt=0.05,
        push_object=name,
    )
    obj = task.object_model
    for label, pose in (("start", spec.object_start), ("goal", spec.goal)):
        world = obj.world_boundary(jnp.asarray(np.asarray(pose, dtype=float)))
        worst = min(
            float(np.min(np.asarray(shape.sdf(world))))
            for shape in spec.obstacles.shapes
        )
        assert worst > 0.0, (
            f"{scene}/{name}: the {label} pose overlaps an obstacle by "
            f"{-worst * 100:.1f} cm"
        )


def test_scene_default_changes_nothing() -> None:
    """`--object scene` is byte-for-byte the untouched scene.

    Every recorded run and every other test loads this path, so it has to
    stay the plain `from_xml_path` load rather than a swap that happens to
    reproduce the T.
    """
    plain = PushT(clutter=True, robot="xarm6", env="open_table",
                  planning_dt=0.05)
    named = PushT(clutter=True, robot="xarm6", env="open_table",
                  planning_dt=0.05, push_object=SCENE_DEFAULT)
    assert plain.push_object is None and named.push_object is None
    assert plain.mj_model.ngeom == named.mj_model.ngeom
    np.testing.assert_allclose(
        np.asarray(plain.object_model.footprint.vertices),
        np.asarray(named.object_model.footprint.vertices),
        atol=0,
    )


@pytest.mark.parametrize("name", sorted(PUSH_OBJECTS))
def test_wrench_limit_follows_the_object(name: str) -> None:
    """The friction budget is the object's, not the scene's.

    `mu * m * g` on both force axes and `r * mu * m * g` on the torque
    axis -- if these still read the T's, the planner would be sizing every
    action against the wrong object.
    """
    obj = PUSH_OBJECTS[name]
    task = PushT(clutter=True, robot="xarm6", env="open_table",
                 planning_dt=0.05, push_object=name)
    limit = np.asarray(task.object_model.wrench_limit)
    nominal = obj.mu * obj.mass * 9.81
    np.testing.assert_allclose(limit[:2], [nominal, nominal], rtol=1e-4)
    np.testing.assert_allclose(
        limit[2], obj.limit_surface_radius * nominal, rtol=1e-4
    )


# ----------------------------------------------------------------------
# The cover against the mesh it stands for
# ----------------------------------------------------------------------

# Deepest hole the boxes may leave in the mesh's own plan outline. What
# every entry leaves at 15 mm or less is the rounded shoulder a box cover
# always leaves on a curved outline. What this is here to catch is a whole
# LIMB left out: both hammers' claws once sat at 38 and 53 mm, a diagonal
# strip the maximal-rectangle fit walks past because no large axis-aligned
# rectangle fits inside it. The mesh still drew the claw, so the arm
# pushed through a part of the object that was not there.
MAX_HOLE = 0.020

# Meshes resolve against each scene's own directory (`apply_to_spec`
# writes `assets/<mesh>_centered.obj`); the two families ship the same
# geometry under that name, so reading either one checks both.
MESH_DIR = f"{ROOT}/models/xarm6_pusht_tabletop/assets"


def _outline(path: str, res: float = 0.002) -> tuple:
    """Rasterize an OBJ's plan outline: a boolean grid and its axes."""
    verts, faces = [], []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                fan = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                faces += [(fan[0], fan[i], fan[i + 1])
                          for i in range(1, len(fan) - 1)]
    v = np.asarray(verts)[:, :2]
    lo, hi = v.min(0), v.max(0)
    gx, gy = (lo[k] + (np.arange(int(np.ceil((hi[k] - lo[k]) / res)) + 1)
                       + 0.5) * res for k in (0, 1))
    pts = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    inside = np.zeros(len(pts), bool)
    for a, b, c in v[np.asarray(faces)]:
        lo_t, hi_t = np.minimum.reduce([a, b, c]), np.maximum.reduce([a, b, c])
        cand = np.flatnonzero(
            (pts[:, 0] >= lo_t[0] - res) & (pts[:, 0] <= hi_t[0] + res)
            & (pts[:, 1] >= lo_t[1] - res) & (pts[:, 1] <= hi_t[1] + res)
            & ~inside
        )
        if not len(cand):
            continue
        p = pts[cand]
        # Same-sign cross products against the three edges: the point is
        # in the triangle when none of them disagrees.
        sides = [
            (p[:, 0] - q[0]) * (r[1] - q[1]) - (r[0] - q[0]) * (p[:, 1] - q[1])
            for q, r in ((b, a), (c, b), (a, c))
        ]
        neg = np.any([s < 0 for s in sides], axis=0)
        pos = np.any([s > 0 for s in sides], axis=0)
        inside[cand[~(neg & pos)]] = True
    return inside.reshape(len(gx), len(gy)), gx, gy


@pytest.mark.parametrize(
    "name", sorted(n for n, o in PUSH_OBJECTS.items() if o.mesh is not None)
)
def test_boxes_leave_no_limb_of_the_mesh_uncovered(name: str) -> None:
    """No part of the drawn object is further than `MAX_HOLE` from a box."""
    obj = PUSH_OBJECTS[name]
    occupied, gx, gy = _outline(f"{MESH_DIR}/{obj.mesh}_centered.obj")
    grid_x, grid_y = np.meshgrid(gx + obj.mesh_offset[0],
                                 gy + obj.mesh_offset[1], indexing="ij")

    covered = np.zeros_like(occupied)
    for cx, cy, half_x, half_y in obj.boxes:
        covered |= ((np.abs(grid_x - cx) <= half_x + _ATOL)
                    & (np.abs(grid_y - cy) <= half_y + _ATOL))

    gap = ndimage.distance_transform_edt(~covered) * (gx[1] - gx[0])
    deepest = float(gap[occupied].max()) if (occupied & ~covered).any() else 0.0
    assert deepest <= MAX_HOLE, (
        f"{name}: the mesh reaches {deepest * 1000:.1f} mm beyond the "
        f"nearest collision box -- a limb the cover missed, which the arm "
        f"will push straight through"
    )


# ----------------------------------------------------------------------
# The swap, against the REAL scenes -- the ones `--object` is for
# ----------------------------------------------------------------------

REAL_TABLETOP = ["open_table_real", "single_obstacle_real", "box_clutter_real"]
PRINTED = ["T_large_block", "A_block", "C_block", "I_block", "R_block"]


@pytest.mark.parametrize("scene", REAL_TABLETOP)
@pytest.mark.parametrize("name", PRINTED)
def test_printed_object_swaps_into_a_real_scene(scene: str, name: str) -> None:
    """The lab prints install into the real scenes exactly as into the sim ones.

    The sim-scene tests above never load `xarm6_pusht_tabletop_real/`, whose
    `tee_real.xml` declares its block<->table pairs by geom name and whose
    `assets/` is a different directory -- both of which `apply_to_spec` has
    to get right for the real driver's `--object` to work at all.
    """
    obj = PUSH_OBJECTS[name]
    task = PushT(
        clutter=True, robot="xarm6", env=scene, planning_dt=0.05,
        push_object=name,
    )
    model = task.mj_model
    block = model.body("block")
    assert float(block.mass[0]) == pytest.approx(obj.mass, rel=1e-6)
    assert task.tip_target_z == pytest.approx(obj.half_height)
    # Every box has its own table pair carrying the object's mu, and no
    # pair still names the T's deleted geoms.
    names = {model.geom(i).name for i in range(model.ngeom)}
    for i in range(model.npair):
        g1, g2 = model.geom(model.pair_geom1[i]).name, model.geom(model.pair_geom2[i]).name
        assert g1 in names and g2 in names
    # MuJoCo orders each compiled pair by geom id, so the box may sit on
    # either side of its partner. Two kinds of pair name a box: its table
    # pair (always) and its stick pair (only under `pusher_mu`).
    def partner_pairs(other: str):
        return [
            i for i in range(model.npair)
            if {model.geom(model.pair_geom1[i]).name[:9],
                model.geom(model.pair_geom2[i]).name[:9]}
            == {"block_box", other[:9]}
        ]
    table_pairs = partner_pairs("table")
    assert len(table_pairs) == len(obj.boxes)
    for i in table_pairs:
        assert float(model.pair_friction[i][0]) == pytest.approx(obj.mu)
    stick_pairs = partner_pairs(STICK_GEOM)
    if obj.pusher_mu is None:
        assert stick_pairs == []
    else:
        assert len(stick_pairs) == len(obj.boxes)
        for i in stick_pairs:
            assert float(model.pair_friction[i][0]) == pytest.approx(obj.pusher_mu)
    # The visual mesh resolved from this scene directory's assets/.
    assert model.nmesh >= 1
    assert "pushed_object" in {model.mesh(i).name for i in range(model.nmesh)}


def test_printed_objects_share_one_frame_with_foundationpose() -> None:
    """The OBJ MJX draws and the PLY FoundationPose tracks come from one STL.

    Both are centred on the plan bounding box, and the OBJ's underside is
    at z = 0 while the PLY keeps the STL's mid-height origin -- the body
    sits at `half_height`, so the two origins coincide in the world and a
    FoundationPose pose maps onto the block with no offset.
    """
    import os
    from oim import ROOT
    for name in PRINTED:
        obj = PUSH_OBJECTS[name]
        path = os.path.join(ROOT, "models", "xarm6_pusht_tabletop_real",
                            "assets", f"{name}_centered.obj")
        v = np.array([list(map(float, l.split()[1:4]))
                      for l in open(path) if l.startswith("v ")])
        lo, hi = v.min(0), v.max(0)
        np.testing.assert_allclose((lo + hi)[:2] / 2, 0.0, atol=1e-6)
        assert lo[2] == pytest.approx(0.0, abs=1e-6)
        assert hi[2] == pytest.approx(2 * obj.half_height, abs=1e-6)
        # The box cover lies inside the mesh's plan bounding box, to within
        # the 2 mm raster pitch the boxes were fitted on: a whole-cell box
        # can overhang an outline that is not a multiple of the pitch by
        # up to half a pitch on each side, and the I (72.9 mm wide) does.
        for cx, cy, hx, hy in obj.boxes:
            assert abs(cx) + hx <= hi[0] + 2e-3 and abs(cy) + hy <= hi[1] + 2e-3


def test_printed_objects_flip_axes_match_their_footprint() -> None:
    import os
    from oim import ROOT
    from oim.objects import fit_print
    for name in PRINTED:
        path = os.path.join(ROOT, "models", "xarm6_pusht_tabletop_real",
                            "assets", f"{name}_centered.obj")
        v, f = [], []
        for line in open(path):
            if line.startswith("v "):
                v.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        tris = np.asarray(v)[np.asarray(f)] * 1000.0
        mask, x0, y0 = fit_print.footprint_mask(tris, fit_print.RASTER_MM)
        axes = fit_print.flip_axes(mask, x0, y0, fit_print.RASTER_MM)
        assert axes == PUSH_OBJECTS[name].flip_axes, name


def test_perception_frame_follows_the_pushed_object() -> None:
    from oim.worlds.real3d.build import perception_frame
    for scene in REAL_TABLETOP:
        assert perception_frame(scene, SCENE_DEFAULT) == ((0.0, 0.025), ("y",))
        for name in PRINTED:
            obj = PUSH_OBJECTS[name]
            assert perception_frame(scene, name) == (
                obj.fp_origin_offset, obj.flip_axes)


def test_scene_t_origin_offset_is_minus_its_bounding_box_centre() -> None:
    """meshes/T_block/T_block.ply is centred on its bounding box."""
    for scene in REAL_TABLETOP + ["live_real"]:
        spec = SCENES[scene]
        v = np.asarray(spec.footprint_builder(**spec.footprint_kwargs).vertices)
        centre = (v.min(0) + v.max(0)) / 2.0
        np.testing.assert_allclose(spec.fp_origin_offset, -centre, atol=1e-9)
