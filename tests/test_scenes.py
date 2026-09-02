"""Cross-checks every `SceneSpec` against the MJCF it names.

A scene is described twice: once in MJCF, which is what the simulator
actually runs, and once in `oim/utils/scenes.py`, which is what the ADMM
object subproblem plans against. Nothing at runtime checks the two agree --
a mismatch doesn't crash, it just means the planner is avoiding obstacles
that aren't there, or aiming at a goal the sim doesn't score. These tests
are that check.
"""

import math

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from oim import ROOT
from oim.objects import Box, Circle, Polygon
from oim.utils.scenes import SCENES

# Worldbody geoms that are scenery, not obstacles.
_SCENERY = {"floor", "table"}

# MJCF geoms are float32 out of the compiler and `SceneSpec` shapes are
# float32 out of jax, so nothing here can be compared tighter than this.
_ATOL = 1e-6

# The xArm6 stick tip's maximum planar radius from the base, at the height
# the block sits at (z = 0.025 +- 0.01). Measured by sweeping joints 1-5
# over 13 samples each (371293 poses) on
# models/xarm6_pusht_tabletop/open_table.xml: 0.842 m, and 0.876 m
# ignoring height. A purely kinematic bound -- it
# ignores the table and self-collision -- so it is a necessary condition
# for a solvable scene, not a sufficient one.
_XARM6_PLANAR_REACH = 0.84

# Deviations between an MJCF and its `SceneSpec` that are known, keyed by
# (scene, geom name).
#
# Both entries below are in the original `clutter` scene and predate the
# tabletop conversions; neither is asserted, so that this file records them
# rather than silently passing over them. The tabletop scenes have no
# exemptions.
_KNOWN_DEVIATIONS = {
    # `obs_tri` is a triangle to the planner and its own bounding box
    # (identical centre and half-extents) to MuJoCo, which has no
    # triangular-prism primitive. A deliberate approximation.
    ("clutter", "obs_tri"): "triangle approximated by its bbox in MJCF",
    # BUG, not a simplification: the MJCF says `euler="0 0 0.25"` under
    # `<compiler angle="degree">` -- 0.25 *degrees* -- while the spec says
    # `angle=0.25` *radians* (14.32 degrees). The two disagree by 14
    # degrees on a 0.04 x 0.035 box, so the planner is avoiding a box the
    # simulator has oriented differently. Left alone deliberately: fixing
    # it either way shifts `clutter` results, so which side is
    # authoritative is the user's call. Fix by making the MJCF read
    # `euler="0 0 14.3239"`, or the spec `angle=math.radians(0.25)`.
    ("clutter", "obs_box"): "0.25 deg in MJCF vs 0.25 rad in the spec",
}


def _load(scene: str, robot: str) -> mujoco.MjModel:
    """The compiled MuJoCo model for one scene/embodiment."""
    return mujoco.MjModel.from_xml_path(
        ROOT + "/models/" + SCENES[scene].mjcf_scene(robot)
    )


def _yaw(quat: np.ndarray) -> float:
    """Rotation about z encoded by a wxyz quaternion, in radians."""
    return float(2.0 * math.atan2(quat[3], quat[0]))


def _world_yaw(model: mujoco.MjModel, geom) -> float:  # noqa: ANN001
    """A geom's rotation about z IN THE WORLD, in radians.

    Not `geom_quat`, which is the geom's pose relative to its own body
    and is identity for the real scenes' obstacles: they are geoms inside
    an "obs*" mocap body, so the rotation lives on the BODY. Forward
    kinematics is the only reading that stays true to what the simulator
    collides, whichever of the two carries it.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rot = data.geom_xmat[geom.id].reshape(3, 3)
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def _box_matches(
    half_extents: np.ndarray, yaw: float, shape, tol: float = 1e-4
) -> bool:
    """Whether an MJCF box and a `SceneSpec` Box are the same rectangle.

    Compared as a shape, not as a pair of numbers: a box authored
    `(a, b)` at yaw `t` is the identical rectangle as `(b, a)` at
    `t + 90deg`, and box_clutter_real's obs_3 legitimately uses the
    second form (its ArUco tag sits rotated 90deg against the other two,
    absorbed in the size rather than a body rotation so the tag-to-box
    offset stays one shared transform). Yaw is taken mod 180deg
    throughout -- a rectangle is unchanged by a half turn.
    """
    spec_hs = np.asarray(shape.half_extents, dtype=float)
    hs = np.asarray(half_extents, dtype=float)

    def yaw_close(a: float, b: float) -> bool:
        return abs((a - b + math.pi / 2) % math.pi - math.pi / 2) < tol

    direct = (
        np.allclose(hs, spec_hs, atol=_ATOL)
        and yaw_close(yaw, float(shape.angle))
    )
    swapped = (
        np.allclose(hs[::-1], spec_hs, atol=_ATOL)
        and yaw_close(yaw + math.pi / 2, float(shape.angle))
    )
    return bool(direct or swapped)


def _world_footprint(model: mujoco.MjModel, geom) -> np.ndarray:  # noqa: ANN001
    """A mesh geom's vertices, projected to the world xy plane.

    MuJoCo re-frames a mesh on its centre of mass and principal axes at
    compile time, compensating `geom_pos`/`geom_quat`, so the authored
    numbers no longer describe where the mesh sits. Reading the vertices
    back through forward kinematics is the only description that stays
    true, and it is what the simulator collides.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mesh_id = model.geom_dataid[geom.id]
    adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[adr : adr + num]
    rot = data.geom_xmat[geom.id].reshape(3, 3)
    return (verts @ rot.T + data.geom_xpos[geom.id])[:, :2]


def _world_centre(model: mujoco.MjModel, geom) -> np.ndarray:  # noqa: ANN001
    """An obstacle geom's xy centre in the world, mesh or primitive."""
    if model.geom_dataid[geom.id] >= 0:
        world = _world_footprint(model, geom)
        return (world.min(axis=0) + world.max(axis=0)) / 2.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return np.asarray(data.geom_xpos[geom.id][:2])


def _world_bottom_z(model: mujoco.MjModel, geom) -> float:  # noqa: ANN001
    """The lowest world z the geom occupies, mesh or box."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rot = data.geom_xmat[geom.id].reshape(3, 3)
    pos = data.geom_xpos[geom.id]
    mesh_id = model.geom_dataid[geom.id]
    if mesh_id >= 0:
        adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[adr : adr + num]
    else:
        half = model.geom_size[geom.id]
        local = np.array(
            np.meshgrid([-1, 1], [-1, 1], [-1, 1])
        ).T.reshape(-1, 3) * half
    return float((local @ rot.T + pos)[:, 2].min())


def _assert_hull_matches(
    model: mujoco.MjModel, geom, shape: Polygon, where: str  # noqa: ANN001
) -> None:
    """A `Polygon` obstacle must be the mesh's own convex footprint.

    MJX collides a mesh as its convex hull, so the planner's polygon has to
    match *that*, not the concave outline: too tight and the planner drives
    the object into a contact it never predicted, too loose and it refuses
    space the simulator allows. Checked as a two-way containment with a
    tolerance, since the polygon is deliberately simplified to eight
    vertices for `Polygon.sdf`'s sake.
    """
    world = _world_footprint(model, geom)
    hull = world[ConvexHull(world).vertices]
    poly = np.asarray(shape.vertices)

    # Every spec vertex lies inside the true hull (never wider than the
    # geometry), within a millimetre of numerical slack.
    inside = np.asarray(Polygon(jnp.asarray(hull)).sdf(jnp.asarray(poly)))
    assert inside.max() < 1e-3, (
        f"{where}: spec polygon sticks {inside.max() * 1000:.1f} mm outside "
        f"the mesh's own convex hull"
    )
    # And it covers most of it -- a simplified hull, not a shrunken one.
    area_spec = ConvexHull(poly).volume
    area_hull = ConvexHull(hull).volume
    assert area_spec / area_hull > 0.85, (
        f"{where}: spec polygon covers only "
        f"{area_spec / area_hull:.2f} of the mesh hull's area"
    )


def _obstacle_geoms(model: mujoco.MjModel) -> list:
    """Geoms that stand for obstacles, in declaration order.

    Two shapes count. A geom attached directly to the world, which is
    everything but the floor and tabletop (the pushed object and the goal
    marker each live in their own body), and a geom inside an
    "obs*"-named MOCAP body -- what the real scenes now use so ArUco
    calibration can rewrite the pose at runtime.

    Kept identical to `PushT.obstacle_geoms` in oim/tasks/pusht.py, which
    is the selection the planner's own avoidance cost is built from; the
    "obs" name test is what keeps goal/local_goal -- also mocap -- out.
    """
    return [
        model.geom(i)
        for i in range(model.ngeom)
        if model.geom(i).name not in _SCENERY
        and (
            model.geom_bodyid[i] == 0
            or (
                model.body_mocapid[model.geom_bodyid[i]] >= 0
                and model.body(model.geom_bodyid[i]).name.startswith("obs")
            )
        )
    ]


def _scene_robot_pairs():
    """Every (scene, robot) the registry offers."""
    return [
        (name, robot)
        for name, spec in sorted(SCENES.items())
        for robot in sorted(spec.mjcf_by_robot)
    ]


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_scene_mjcf_loads(scene: str, robot: str) -> None:
    """Every registered scene names an MJCF that compiles."""
    model = _load(scene, robot)
    assert model.body("block").id > 0
    assert model.body("goal").id > 0
    for joint in ("T_x", "T_y", "T_z"):
        assert model.joint(joint) is not None


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_block_anchor_is_the_origin(scene: str, robot: str) -> None:
    """The pushed object's body anchor sits at (0, 0) in xy.

    `PushT._block_pose` reads `qpos` -- the T_x/T_y/T_z joints'
    displacement from this anchor -- and treats it as the world pose. That
    is only true when the anchor is the origin, and a scene that broke it
    would feed a silently wrong pose into every cost term rather than
    fail.
    """
    model = _load(scene, robot)
    np.testing.assert_allclose(model.body("block").pos[:2], [0.0, 0.0], atol=0)


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_goal_pose_matches_the_spec(scene: str, robot: str) -> None:
    """`SceneSpec.goal` is the MJCF goal body's own pose."""
    model = _load(scene, robot)
    spec = SCENES[scene]
    body = model.body("goal")
    np.testing.assert_allclose(
        body.pos[:2], np.asarray(spec.goal)[:2], atol=_ATOL
    )
    expected = float(np.asarray(spec.goal)[2])
    delta = (_yaw(body.quat) - expected + math.pi) % (2 * math.pi) - math.pi
    # 1e-4 rad = 0.006 degrees: loose enough for the 5-decimal quaternions
    # the MJCFs write by hand, tight enough that a wrong quadrant fails.
    assert abs(delta) < 1e-4, f"goal yaw {_yaw(body.quat)} != {expected}"


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_goal_marker_mirrors_the_block(scene: str, robot: str) -> None:
    """The goal marker's geoms match the block's, size for size.

    Overlap is the success criterion by eye, so a marker that isn't the
    block's own shape would make a correct run look wrong (or vice versa).

    Compares what is actually DRAWN -- geoms with a non-zero alpha -- not
    every geom on the body. `icra_sign` draws its C as a visual mesh and
    collides it as three transparent boxes (see that MJCF), so pairing all
    geoms positionally would compare a mesh against a box. Type and mesh id
    are compared alongside size/pos, since `geom_size` carries no meaning
    for a mesh geom and only the mesh id distinguishes two of them.
    """
    model = _load(scene, robot)
    block_id, goal_id = model.body("block").id, model.body("goal").id

    def drawn(body_id: int) -> list:
        return [
            (
                int(model.geom_type[i]),
                int(model.geom_dataid[i]),
                model.geom_size[i],
                model.geom_pos[i],
            )
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == body_id and model.geom_rgba[i][3] > 0.0
        ]

    block, goal = drawn(block_id), drawn(goal_id)
    assert len(block) == len(goal) and block
    for (bt, bd, bs, bp), (gt, gd, gs, gp) in zip(block, goal, strict=True):
        assert (bt, bd) == (gt, gd)
        np.testing.assert_allclose(bs, gs, atol=_ATOL)
        np.testing.assert_allclose(bp, gp, atol=_ATOL)


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_obstacles_match_the_mjcf(scene: str, robot: str) -> None:
    """Each `SceneSpec` obstacle is the geom the simulator collides.

    Centre, half-extents and yaw, not just count -- an obstacle rotated in
    one description and not the other still costs the planner nothing and
    stops the block anyway.

    Geoms are paired with shapes by proximity rather than by declaration
    order, so neither file has to be written in the other's sequence.
    """
    model = _load(scene, robot)
    spec = SCENES[scene]
    geoms = _obstacle_geoms(model)
    # `obstacles_for`, not `obstacles`: that is the field the task builds
    # its costs from, and it already drops the arm base for `point`.
    #
    # For xarm6 the base is still there and deliberately has no matching
    # MJCF obstacle-class geom: it is already collidable as part of the
    # arm body itself (xarm6_link_base), so tagging it `class="obstacle"`
    # too would just duplicate one physical thing as two geoms.
    # Identified by proximity to the scene's own xarm6_base_pos, not by
    # list position, so this does not depend on it being appended last.
    shapes = [
        s
        for s in spec.obstacles_for(robot).shapes
        if spec.xarm6_base_pos is None
        or float(
            np.linalg.norm(
                np.asarray(s.center) - np.asarray(spec.xarm6_base_pos)
            )
        )
        > 0.001
    ]
    assert len(geoms) == len(shapes), (
        f"{scene}/{robot}: MJCF has {len(geoms)} obstacle geoms "
        f"({[g.name for g in geoms]}) but the spec has {len(shapes)}"
    )

    unmatched = list(range(len(shapes)))
    for geom in geoms:
        where = f"{scene}/{robot} obstacle {geom.name}"
        # World centre, not the authored `geom.pos`: MuJoCo re-frames a mesh
        # geom onto its principal axes and compensates pos/quat, so for
        # those two the authored numbers describe nothing.
        centre = _world_centre(model, geom)
        # Loose pairing radius: `clutter`'s triangle has its centroid ~17 mm
        # from its MJCF bounding box's centre, and obstacles in every scene
        # here are far further apart than that.
        j = min(
            unmatched,
            key=lambda k: float(
                np.linalg.norm(centre - np.asarray(shapes[k].center))
            ),
        )
        shape = shapes[j]
        gap = float(np.linalg.norm(centre - np.asarray(shape.center)))
        assert gap < 0.05, (
            f"{where}: no spec obstacle near it (nearest {gap:.3f} m)"
        )
        unmatched.remove(j)

        if (scene, geom.name) in _KNOWN_DEVIATIONS:
            continue
        if isinstance(shape, Polygon):
            # Position is pinned by the hull comparison itself, which is in
            # world coordinates -- a shifted polygon fails containment.
            _assert_hull_matches(model, geom, shape, where)
            continue
        np.testing.assert_allclose(
            centre, np.asarray(shape.center), atol=_ATOL,
            err_msg=f"{where}: centre",
        )
        if isinstance(shape, Circle):
            assert geom.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE, where
            assert abs(float(geom.size[0]) - shape.radius) < _ATOL, where
        elif isinstance(shape, Box):
            assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX, where
            world_yaw = _world_yaw(model, geom)
            assert _box_matches(geom.size[:2], world_yaw, shape), (
                f"{where}: half-extents/yaw -- MJCF "
                f"{np.asarray(geom.size[:2])} at {math.degrees(world_yaw):.1f}deg, "
                f"spec {np.asarray(shape.half_extents)} at "
                f"{math.degrees(float(shape.angle)):.1f}deg"
            )
        else:
            raise AssertionError(f"{where}: unhandled shape {type(shape)}")


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_only_xarm6_pays_for_its_own_base(scene: str, robot: str) -> None:
    """The arm-base obstacle exists for `xarm6` and for nobody else.

    The base disc is not in any `point` MJCF, so a cost on it prices a
    collision the simulator cannot produce: the pusher and the block pass
    straight through, pay, and -- correctly -- decide the detour was worse.
    A soft cost is only meaningful over a region something can actually
    hit, so the disc is dropped from the field rather than reweighted.

    Everything else in the field is untouched: both embodiments run the
    same scene and collide the same obstacle geoms.
    """
    spec = SCENES[scene]
    if spec.xarm6_base_pos is None:
        pytest.skip(f"{scene}: no mounted base")
    base = np.asarray(spec.xarm6_base_pos)

    def at_base(field):
        return [
            s
            for s in field.shapes
            if float(np.linalg.norm(np.asarray(s.center) - base)) < 1e-3
        ]

    # The registry itself always carries it -- pose generation and the
    # real-robot scene check both want the arm's own footprint.
    assert len(at_base(spec.obstacles)) == 1, f"{scene}: base not in the spec"

    shipped = spec.obstacles_for(robot)
    if robot == "xarm6":
        assert len(at_base(shipped)) == 1
        assert len(shipped.shapes) == len(spec.obstacles.shapes)
    else:
        assert not at_base(shipped), (
            f"{scene}/{robot}: still charged for a base that is not there"
        )
        assert len(shipped.shapes) == len(spec.obstacles.shapes) - 1


@pytest.mark.parametrize("scene,robot", _scene_robot_pairs())
def test_footprint_matches_the_block_geoms(scene: str, robot: str) -> None:
    """The analytic footprint is the union of the block's own box geoms.

    Checked by sampling: every footprint vertex must lie inside (or on)
    some block geom, and every block geom's own corners must lie inside
    (or on) the footprint. Together those pin the outline to the geometry
    without assuming either description's vertex order.

    COLLISION geoms only. The analytic footprint stands in for what the
    simulator actually collides, so a visual-only geom is not part of the
    claim -- `icra_sign` draws its C as a mesh and collides it as three
    boxes, and it is the boxes the object block's obstacle term has to
    agree with.
    """
    model = _load(scene, robot)
    footprint = SCENES[scene].footprint()
    block_id = model.body("block").id

    def is_collision(i: int) -> bool:
        return model.geom_bodyid[i] == block_id and model.geom_contype[i] != 0

    boxes = [
        (model.geom_pos[i][:2], model.geom_size[i][:2])
        for i in range(model.ngeom)
        if is_collision(i)
    ]
    assert boxes, f"{scene}: block has no collision geoms"
    # Every block collision geom is axis-aligned in the body frame for
    # these scenes; a rotated one would need its own handling rather than
    # silent wrong answers.
    for i in range(model.ngeom):
        if is_collision(i):
            assert abs(_yaw(model.geom_quat[i])) < 1e-6, (
                f"{scene}: block geom {model.geom(i).name} is rotated in "
                "its body frame; this test assumes axis-aligned boxes"
            )

    tol = _ATOL

    def in_any_box(p: np.ndarray) -> bool:
        return any(
            bool(np.all(np.abs(p - c) <= h + tol)) for c, h in boxes
        )

    verts = np.asarray(footprint.vertices)
    for v in verts:
        assert in_any_box(v), (
            f"{scene}: footprint vertex {v} is outside the block"
        )

    inside = np.asarray(footprint.sdf(verts)) <= tol
    assert bool(np.all(inside))

    for c, h in boxes:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corner = c + np.array([sx * h[0], sy * h[1]])
                d = float(np.asarray(footprint.sdf(corner[None, :]))[0])
                assert d <= tol, (
                    f"{scene}: block corner {corner} lies {d:.2e} outside "
                    "the analytic footprint"
                )


# Every scene under models/xarm6_pusht_tabletop/, i.e. all but `clutter`,
# which predates the tabletop convention: it has no table geom and no
# start keyframe.
_TABLETOP_SCENES = sorted(
    s for s, spec in SCENES.items()
    if any("tabletop/" in p for p in spec.mjcf_by_robot.values())
)


@pytest.mark.parametrize("scene", _TABLETOP_SCENES)
def test_objects_rest_on_the_tabletop(scene: str) -> None:
    """Nothing is spawned floating above, or sunk into, the tabletop.

    The whole task is quasi-static planar pushing; an obstacle hovering a
    centimetre up is not an obstacle, and one buried in the table cannot be
    pushed against. IsaacGym spawns actors 1-5 mm clear and lets them
    settle; these scenes place them resting, so a run starts where it
    looks like it starts.

    Tabletop scenes only: `clutter` has no table geom at all, its objects
    sitting 1 cm above the floor plane.
    """
    model = _load(scene, "xarm6")
    surface_z = float(model.geom("table").pos[2] + model.geom("table").size[2])
    assert abs(surface_z) < _ATOL, (
        f"{scene}: tabletop is at z={surface_z}, not 0"
    )

    for geom in _obstacle_geoms(model):
        # Mesh obstacles included: their lowest vertex in world coordinates
        # is the thing that has to touch the table, and MuJoCo's re-framing
        # means the authored pos/size cannot be read off directly.
        bottom = _world_bottom_z(model, geom)
        assert abs(bottom - surface_z) < 1e-4, (
            f"{scene}: obstacle {geom.name} bottom at z={bottom}, not resting "
            f"on z={surface_z}"
        )

    # Collision geoms only: what rests on the table is what the table can
    # push back on. `icra_sign`'s visual mesh is excluded, and would not be
    # readable this way anyway -- MuJoCo re-frames a mesh geom, so its
    # authored pos/size cannot be differenced like a box's.
    block = model.body("block")
    block_bottom = min(
        float(block.pos[2] + model.geom_pos[i][2] - model.geom_size[i][2])
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == block.id and model.geom_contype[i] != 0
    )
    assert abs(block_bottom - surface_z) < _ATOL, (
        f"{scene}: block bottom at z={block_bottom}, not resting on the table"
    )


@pytest.mark.parametrize("scene", _TABLETOP_SCENES)
def test_start_pose_is_free_and_clear(scene: str) -> None:
    """The "start" keyframe is a valid, useful pose in *this* scene.

    The keyframe is shared by all five tabletop scenes but the obstacles
    are not, so it has to be re-checked per scene: a pose tuned against
    `single_obstacle` alone started 3.7 cm inside `shelf_gap`'s shelf_2,
    which no assertion about `single_obstacle` would ever have caught.

    Tabletop scenes only: `clutter` defines no keyframe and
    oim/worlds/sim3d/build.py falls back to its own tuned constant there.
    """
    model = _load(scene, "xarm6")
    spec = SCENES[scene]
    base = model.body("xarm6_link_base").id
    # `xarm6_base_z` included: it is the measured mount offset, so a scene
    # standing on a real table starts 1.1 cm lower than this check would
    # otherwise place it, and that is exactly the margin the tip-height and
    # penetration assertions below are about.
    model.body_pos[base] = [*spec.xarm6_base_pos, spec.xarm6_base_z]
    yaw = math.radians(spec.xarm6_base_yaw_deg)
    model.body_quat[base] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    assert model.nkey >= 1, f"{scene}: no 'start' keyframe"
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)

    # Nothing may start interpenetrating -- not the table, not an obstacle,
    # not the block.
    penetrating = [
        (
            model.geom(data.contact.geom1[i]).name,
            model.geom(data.contact.geom2[i]).name,
            float(data.contact.dist[i]),
        )
        for i in range(data.ncon)
        if data.contact.dist[i] < -1e-4
    ]
    assert not penetrating, f"{scene}: start pose penetrates {penetrating}"

    tip_site = model.site("xarm6_tip").id
    tip = data.site_xpos[tip_site]
    assert tip[2] > 0.02, f"{scene}: start tip z={tip[2]:.4f} grazes the table"
    # The tip must not start already at the object: ADMM should have to
    # drive the arm out to first contact, as a real deployment would.
    assert float(np.linalg.norm(tip[:2])) > 0.2, (
        f"{scene}: start tip is already on top of the object at {tip[:2]}"
    )
    # The stick should start pointing down, the orientation PushT._tilt
    # rewards, so the run doesn't open by paying to un-flip the wrist.
    z_axis = data.site_xmat[tip_site].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(-z_axis[2], -1.0, 1.0))))
    assert tilt < 30.0, f"{scene}: start tilt {tilt:.1f} deg off vertical"


@pytest.mark.parametrize(
    "scene", [s for s, spec in SCENES.items() if "xarm6" in spec.mjcf_by_robot]
)
def test_xarm6_base_reaches_object_and_goal(scene: str) -> None:
    """Both ends of the task are inside the arm's reach envelope.

    A scene whose goal sits outside it is unsolvable no matter what the
    planner does, which is worth failing loudly rather than inferring from
    a flat error curve. Necessary, not sufficient -- see
    `_XARM6_PLANAR_REACH`.
    """
    spec = SCENES[scene]
    base = np.asarray(spec.xarm6_base_pos)
    for label, point in (
        # `object_start`, not the origin: the block's MJCF *anchor* is the
        # origin, but its start pose lives in the scene's keyframe, and it
        # is the pose the arm actually has to reach.
        ("object start", np.asarray(spec.object_start, dtype=float)[:2]),
        ("goal", np.asarray(spec.goal)[:2]),
    ):
        dist = float(np.linalg.norm(point - base))
        assert dist < _XARM6_PLANAR_REACH, (
            f"{scene}: {label} is {dist:.3f} m from the base, beyond the "
            f"{_XARM6_PLANAR_REACH} m planar reach"
        )
