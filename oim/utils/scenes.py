"""Registry of the 3D scenes `oim.tasks.pusht.PushT` can load.

Each `SceneSpec` supplies everything a scene needs: the MJCF scene file per
embodiment, the object's goal pose, its obstacle field, its footprint
parameterization, and (for xArm6) the arm's ground-mount placement. `PushT`
itself never branches on a scene name or holds scene-specific data -- it
looks up one `SceneSpec` by name and wraps cost functions/ADMM plumbing
around whatever it's handed. A new environment is one new entry here plus
its own MJCF (with its own `<camera>`/`<keyframe>` for recording/starting
pose, read generically), plus a three-line `examples/` script naming it --
never a change to oim/tasks/pusht.py or oim/experiment.py.

Nothing checks at runtime that a `SceneSpec` and its MJCF agree -- the
planner would simply reason about a world the simulator is not running.
`tests/test_scenes.py` does check it, geom by geom, for every scene here.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp

from oim.objects import (
    Box,
    Circle,
    ObstacleField,
    Polygon,
    Shape,
    c_shape_footprint,
    t_shape_footprint,
)


@dataclass(frozen=True)
class SceneSpec:
    """One scene: which MJCF per embodiment, and the object's own facts.

    Args:
        mjcf_by_robot: Scene path (relative to `oim/models/`) for each
            embodiment this scene supports, e.g.
            `{"point": "pusht_clutter/scene.xml"}`. An embodiment not
            listed here isn't available for this scene.
        goal: Goal pose for the object, world-frame SE(2).
        obstacles: Static obstacles, matching the obstacle geoms in the
            MJCF.
        footprint_kwargs: Passed to `footprint_builder` -- empty means that
            builder's own defaults.
        footprint_builder: Builds the pushed object's outline. Defaults to
            `t_shape_footprint`; a scene whose object is not a T (e.g.
            `icra_sign`) names its own.
        xarm6_base_pos: Ground-mounted (x, y) base placement, xArm6 only.
        xarm6_base_yaw_deg: Base yaw about z (degrees), xArm6 only.
        xarm6_base_z: Base height relative to the scene floor, xArm6 only.
            Zero for a simulated scene, whose floor the model already sits
            on; a scene standing on a real table needs the measured offset
            (see the README's bring-up checklist).
        xarm6_arm_start_deg: Arm home configuration (degrees, joints 1-5)
            the mock starts from. Hardware reads its own.
        object_start: The pushed object's start SE(2).
        world_frame: TF frame the planner's world is expressed in.
            "xarm_device" when the arm base *is* the world origin, so no
            world -> base transform has to be published at all.
        mass: Mass of the pushed object.
        mu: Friction coefficient between object and table.
        limit_surface_radius: Limit-surface radius of the object's contact
            patch. Together with `mass`/`mu` this must satisfy
            `mu * mass * g == frictionloss` on the object's MJCF joints, or
            the analytic object model and the simulated one describe
            different physics.
    """

    mjcf_by_robot: Dict[str, str]
    goal: jnp.ndarray
    obstacles: ObstacleField
    footprint_kwargs: Dict[str, object] = field(default_factory=dict)
    footprint_builder: Callable[..., Polygon] = t_shape_footprint
    xarm6_base_pos: Optional[Tuple[float, float]] = None
    xarm6_base_yaw_deg: Optional[float] = None
    xarm6_base_z: float = 0.0
    xarm6_arm_start_deg: Optional[Sequence[float]] = None
    object_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_frame: str = "world"
    mass: float = 2.0
    mu: float = 0.4
    limit_surface_radius: float = 0.06

    def mjcf_scene(self, robot: str) -> str:
        """Scene path (relative to `oim/models/`) for `robot`.

        Raises:
            ValueError: If this scene has no MJCF for `robot`.
        """
        if robot not in self.mjcf_by_robot:
            raise ValueError(
                f"robot={robot!r} has no scene here (available: "
                f"{sorted(self.mjcf_by_robot)})"
            )
        return self.mjcf_by_robot[robot]

    def footprint(self) -> Polygon:
        """The object's footprint outline for this scene."""
        return self.footprint_builder(**self.footprint_kwargs)


# ----------------------------------------------------------------------
# The five tabletop pushing scenes
# ----------------------------------------------------------------------
#
# Converted from the Object-Informed-Manipulation (IsaacGym) repo's
# `conf/task/sim_task01..05.yaml`, matching `models/xarm6_pusht_tabletop/`.
# Positions are IsaacGym's own world-frame numbers, rigidly re-origined in
# xy so the pushed object starts at (0, 0) -- required, because
# `PushT._block_pose` reads `qpos`, the object joints' displacement from
# their body's MJCF anchor, so that anchor has to be the origin or every
# cost term compares the wrong quantity. The shift preserves every real
# relative distance and angle. See
# `models/xarm6_pusht_tabletop/common.xml` for the full derivation,
# including the z convention.
#
# The four T-block scenes share the shift (-0.7, +0.45) (tee_block starts
# at (0.7, -0.45)), the same arm base, block and goal -- they differ *only*
# in obstacles, exactly as sim_task01..04 do. `icra_sign` pushes a letter
# instead and shifts by (-0.7, +0.40).

# IsaacGym's assets/urdf/tee_block/tee_block.urdf (crossbar box
# "0.2 0.05 0.05", stem "0.05 0.1 0.05" at y = -0.075; URDF gives full
# extents, these are halves). Noticeably bigger than the clutter scene's T.
_TABLETOP_TEE_FOOTPRINT = dict(
    crossbar_half=(0.100, 0.025),
    stem_half=(0.025, 0.050),
    crossbar_y=0.0,
    stem_y=-0.075,
)

# conf/actors/block.yaml: a 0.1 m cube at (0.9, 0.05), shared by
# `single_obstacle` and `ycb_clutter`.
_TABLETOP_CUBE = Box(center=[0.2, 0.5], half_extents=[0.05, 0.05])

# Mesh obstacles are carried as the 2D convex hull of their footprint,
# simplified to eight vertices -- because that is what MJX actually
# collides. A mesh geom is convexified at compile time (`maxhullvert` in
# the MJCF caps it), so a planner holding the full concave outline would
# reason about clearance the simulator does not enforce, and one holding
# the bounding box would refuse space the simulator allows. Eight vertices
# keeps 0.90-1.00 of the true hull area while keeping `Polygon.sdf` cheap:
# it is evaluated per footprint sample per rollout.
#
# Generated from the compiled model, not measured by hand -- see
# `tests/test_scenes.py`, which re-derives every one of these from the MJCF
# and fails if the two drift apart.


def _glyph(outline: Sequence[Tuple[float, float]], y: float) -> Polygon:
    """One `icra_sign` glyph's hull, placed at its slot in the row.

    Every glyph sits at x = 0.2 under the same -90 degree yaw, so the
    outlines below are stored once in that placed orientation and only
    translated here. `digit_2` and `digit_2b` share one.

    Args:
        outline: The glyph's hull, relative to its own centre.
        y: Its slot in the row.

    Returns:
        The world-frame footprint.
    """
    return Polygon(jnp.array([[x + 0.2, py + y] for x, py in outline]))


# ycb/spamCan/nontextured.stl, 0.94 of its true footprint hull.
_SPAM_CAN = Polygon(
    jnp.array(
        [
            [-0.1798, 0.7019], [-0.1624, 0.6933], [-0.0985, 0.6964],
            [-0.0818, 0.7139], [-0.0825, 0.7361], [-0.1027, 0.7535],
            [-0.1658, 0.7505], [-0.1838, 0.7334],
        ]
    )
)
# ycb/mustardBottle/nontextured.stl, 0.93 of its true footprint hull.
_MUSTARD_BOTTLE = Polygon(
    jnp.array(
        [
            [0.6483, 0.4850], [0.6798, 0.4691], [0.7163, 0.4680],
            [0.7273, 0.4964], [0.7115, 0.5140], [0.6852, 0.5289],
            [0.6418, 0.5302], [0.6311, 0.5063],
        ]
    )
)

# Glyph hulls, relative to each glyph's own centre, already under the row's
# -90 degree yaw. The I's mesh is a plain 8-vertex slab, so its hull is
# exactly its bounding box -- written as a polygon anyway, so that every
# mesh geom in this scene has a polygon opposite it and the correspondence
# stays one rule rather than one rule with an exception.
_GLYPH_I = (
    (-0.0500, -0.0122), (0.0500, -0.0122),
    (0.0500, 0.0122), (-0.0500, 0.0122),
)
_GLYPH_R = (
    (-0.0500, -0.0356), (0.0233, -0.0312),
    (0.0341, -0.0285), (0.0445, -0.0205),
    (0.0491, -0.0104), (0.0500, 0.0036),
    (0.0500, 0.0356), (-0.0500, 0.0356),
)
_GLYPH_2 = (
    (-0.0508, -0.0317), (-0.0288, -0.0334),
    (0.0210, -0.0320), (0.0421, -0.0238),
    (0.0508, -0.0002), (0.0410, 0.0237),
    (0.0164, 0.0334), (-0.0508, 0.0334),
)
_GLYPH_0 = (
    (-0.0486, -0.0149), (-0.0091, -0.0337),
    (0.0201, -0.0324), (0.0486, -0.0152),
    (0.0486, 0.0149), (0.0192, 0.0326),
    (-0.0202, 0.0325), (-0.0486, 0.0152),
)
_GLYPH_6 = (
    (-0.0429, -0.0239), (-0.0178, -0.0329),
    (0.0233, -0.0324), (0.0515, -0.0020),
    (0.0386, 0.0237), (0.0194, 0.0313),
    (-0.0352, 0.0273), (-0.0515, -0.0007),
)


# Measured directly off oim/models/xarm6/xarm6.xml's own base mesh
# (xarm6_base_shell): max xy vertex radius 0.0912 m. Not guessed.
_ROBOT_BASE_RADIUS = 0.09


def _tee_scene(name: str, obstacles: Sequence[Shape]) -> SceneSpec:
    """A `SceneSpec` for one of the four T-block scenes.

    They differ only in obstacles, so everything else is set here once.

    Args:
        name: The scene's MJCF basename under
            `models/xarm6_pusht_tabletop/`.
        obstacles: That scene's static obstacles.

    Returns:
        The scene spec.
    """
    # IsaacGym's own (0.4, 0) arm base. A rigid shift doesn't change yaw,
    # and IsaacGym never rotates the base (no init_ori on
    # conf/actors/xarm6_stick.yaml), so that stays 0.
    base_pos = (-0.3, 0.45)
    return SceneSpec(
        mjcf_by_robot={
            "xarm6": f"xarm6_pusht_tabletop/{name}.xml",
            "point": f"xarm6_pusht_tabletop/{name}_point.xml",
        },
        # IsaacGym's goal (0.9, 0.30) with quat [0,0,1,0] (xyzw), a
        # 180-degree flip about z from the block's spawn orientation
        # (theta = 0, matching `t_shape_footprint`'s own implicit zero).
        goal=jnp.array([0.2, 0.75, jnp.pi]),
        obstacles=ObstacleField(
            list(obstacles)
            + [
                # The robot's own mounted base was never in the object
                # planner's obstacle field -- physically permanent, but
                # invisible to the analytic path optimizer, which was
                # therefore free to plan straight through it. For
                # shelf_gap specifically, this let the optimizer treat
                # going *around* the gap as cheaper than going through
                # it, since neither route's cost accounted for the base
                # sitting just past the gap's near side either way.
                Circle(center=list(base_pos), radius=_ROBOT_BASE_RADIUS),
            ]
        ),
        footprint_kwargs=dict(_TABLETOP_TEE_FOOTPRINT),
        xarm6_base_pos=base_pos,
        xarm6_base_yaw_deg=0.0,
    )


# The measured lab T-block's plan footprint. A future real scene with a
# different physical object overrides footprint_builder/kwargs (and physics).
_REAL_TEE_FOOTPRINT = dict(
    crossbar_half=(0.0445, 0.0099),
    stem_half=(0.0099, 0.0397),
    crossbar_y=0.0099,
    stem_y=-0.0397,
)


def _real_scene(name, obstacles, goal, object_start, arm_start_deg, *,
                base_z=-0.0111, footprint_builder=t_shape_footprint,
                footprint_kwargs=None, mass=0.1, mu=0.3,
                limit_surface_radius=0.04) -> SceneSpec:
    """A SceneSpec for a real-table scene run on the lab xArm6.

    Fixes what every real scene shares -- the arm base at the world origin
    (world_frame='xarm_device', so no world->base transform) and the lab
    block's physics -- and takes only what varies: the scene MJCF, obstacles,
    goal and start poses. Object shape defaults to the measured T-block; a
    different physical object overrides footprint_builder/kwargs and physics.
    """
    return SceneSpec(
        mjcf_by_robot={"xarm6": f"xarm6_pusht_tabletop_real/{name}.xml"},
        goal=goal,
        obstacles=obstacles,
        footprint_builder=footprint_builder,
        footprint_kwargs=dict(footprint_kwargs or _REAL_TEE_FOOTPRINT),
        xarm6_base_pos=(0.0, 0.0),
        xarm6_base_yaw_deg=0.0,
        xarm6_base_z=base_z,
        xarm6_arm_start_deg=arm_start_deg,
        object_start=object_start,
        world_frame="xarm_device",
        mass=mass,
        mu=mu,
        limit_surface_radius=limit_surface_radius,
    )


SCENES: Dict[str, SceneSpec] = {
    "clutter": SceneSpec(
        mjcf_by_robot={
            "point": "pusht_clutter/scene.xml",
            "xarm6": "xarm6_pusht_clutter/scene.xml",
        },
        # Goal pose (world-frame SE(2)), matching the `goal` mocap body in
        # models/pusht_clutter/pusht_clutter.xml (shared verbatim by the
        # xarm6 scene).
        goal=jnp.array([0.50, 0.48, jnp.pi / 4]),
        # Static obstacles, matching the obstacle geoms in the same MJCF.
        obstacles=ObstacleField(
            [
                Circle(center=[0.08, 0.32], radius=0.04),
                Box(
                    center=[0.38, 0.10], half_extents=[0.04, 0.035],
                    angle=0.25,
                ),
                Polygon(
                    jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])
                ),
                # The robot's own mounted base, previously invisible to
                # the object planner -- see _tee_scene's own comment.
                Circle(center=[0.2, 0.75], radius=_ROBOT_BASE_RADIUS),
            ]
        ),
        # Ground-mounted, chosen via the reach sweep in
        # models/xarm6_pusht_clutter/verify_reach.py; covers the
        # block/goal/obstacle footprint within a few cm.
        xarm6_base_pos=(0.2, 0.75),
        xarm6_base_yaw_deg=-90.0,
    ),
    # The lab's own measurements: the physical T block, three pudding-box
    # obstacles, and the arm base at the origin, so the planner's world *is*
    # the robot base frame -- no world -> base transform, and FoundationPose's
    # TF can be read straight into the planner. The only scene that runs on
    # hardware, which is why it carries the object's real physics rather than
    # the modelled T's.
    "box_clutter": _real_scene(
        "box_clutter",
        obstacles=ObstacleField([
            Box(center=[0.318, 0.178], half_extents=[0.054, 0.0445], angle=jnp.pi / 2),
            Box(center=[0.229, -0.140], half_extents=[0.054, 0.0445], angle=jnp.pi / 2),
            Box(center=[0.521, -0.140], half_extents=[0.054, 0.0445], angle=jnp.pi / 2),
            Circle(center=[0.0, 0.0], radius=_ROBOT_BASE_RADIUS),
        ]),
        goal=jnp.array([0.381, -0.305, jnp.pi / 2]),
        object_start=(0.381, 0.343, 0.0),
        arm_start_deg=[49.2, 34.8, -80.6, 0.0, 45.9],
    ),
    "open_table_real": _real_scene(
        "open_table_real",
        obstacles=ObstacleField([Circle(center=[0.0, 0.0], radius=_ROBOT_BASE_RADIUS)]),
        goal=jnp.array([0.381, -0.305, jnp.pi / 2]),
        object_start=(0.381, 0.343, 0.0),
        arm_start_deg=[49.2, 34.8, -80.6, 0.0, 45.9],
    ),
    "single_obstacle_real": _real_scene(
        "single_obstacle_real",
        obstacles=ObstacleField([
            Box(center=[0.30, 0.02], half_extents=[0.054, 0.0445], angle=jnp.pi / 2),
            Circle(center=[0.0, 0.0], radius=_ROBOT_BASE_RADIUS),
        ]),
        goal=jnp.array([0.381, -0.305, jnp.pi / 2]),
        object_start=(0.381, 0.343, 0.0),
        arm_start_deg=[49.2, 34.8, -80.6, 0.0, 45.9],
    ),
    # sim_task01: "push the tee block". Nothing in the way.
    "open_table": _tee_scene("open_table", []),
    # sim_task02: "... avoiding an obstacle".
    "single_obstacle": _tee_scene("single_obstacle", [_TABLETOP_CUBE]),
    # sim_task03: "... avoiding two shelves". conf/actors/shelf_{1,2}.yaml:
    # 0.2 x 0.25 x 0.2 m boxes at (1.1, 0.05) and (0.7, 0.05), which would
    # put them at x = 0.0 and 0.4 here. Each is moved 0.05 m outward, so
    # the gap is x in [0.05, 0.35] (0.3 m) rather than IsaacGym's [0.1,
    # 0.3] (0.2 m) -- that is exactly the T crossbar's length, a
    # zero-clearance squeeze. Widened about x = 0.2, the goal's own x, so
    # the centred route stays centred. See shelf_gap.xml, whose geoms these
    # must match.
    "shelf_gap": _tee_scene(
        "shelf_gap",
        [
            Box(center=[-0.05, 0.5], half_extents=[0.10, 0.125]),
            Box(center=[0.45, 0.5], half_extents=[0.10, 0.125]),
        ],
    ),
    # sim_task04: "... avoiding multiple obstacles". Two of the three YCB
    # actors are meshes in their URDFs and appear here as their convex
    # hulls; dominoSugar is a box in its own URDF and stays one.
    #
    # RE-LAID OUT 2026-08-17 so the T can pass on *either* side. As
    # converted from IsaacGym, dominoSugar and the cube both sat at
    # x = 0.20 -- a vertical wall rather than a slalom -- and the only
    # gap to the right of it was cube(right edge 0.250) to mustard(left
    # edge 0.436), i.e. 0.186 m. The T is 0.20 x 0.15 m, so with
    # `obstacle_margin` at 0.05 a passable gap needs 0.15 + 2*0.05 =
    # 0.25 m: the clearance hinge fired across the whole width of that
    # corridor and the right-hand route was effectively closed, leaving
    # the 0.360 m left-hand route as the only choice.
    #
    # Two obstacles move, both exclusive to this scene. dominoSugar
    # 0.20 -> 0.08 in x (now left of the start->goal line, while the cube
    # stays right of it, so the field staggers instead of stacking), and
    # mustardBottle +0.195/-0.027 to open the right-hand corridor.
    # Measured after: at y = 0.50 the left gap is 0.360 m and the right
    # 0.381 m; at y = 0.25, 0.393 m and 0.832 m. Both sides clear the
    # 0.25 m threshold, so neither route is forced.
    #
    # The cube is deliberately NOT moved: `_TABLETOP_CUBE` is the same
    # object `single_obstacle` uses, and moving it would silently re-lay
    # out that scene too.
    "ycb_clutter": _tee_scene(
        "ycb_clutter",
        [
            _TABLETOP_CUBE,
            # spamCan and mustardBottle are meshes in their URDFs, so these
            # are their convex hulls -- 0.82x and 0.71x their bounding
            # boxes in plan, which is space the planner used to refuse.
            _SPAM_CAN,
            _MUSTARD_BOTTLE,
            # dominoSugar is *not* a mesh: dominoSugar.urdf declares
            # `<box size="0.06 0.095 0.175"/>`. Laid on its side by
            # init_ori (-90 degrees about y), footprint 0.175 x 0.095.
            Box(center=[0.08, 0.25], half_extents=[0.0875, 0.0475]),
        ],
    ),
    "icra_sign": SceneSpec(
        # sim_task05, respelled: seven fixed glyphs spell "ICRA 2026" in a
        # row at x = 0.2 with the C's own slot left empty, and the goal is
        # that slot. See icra_sign.xml for why the C is the pushed letter.
        mjcf_by_robot={
            "xarm6": "xarm6_pusht_tabletop/icra_sign.xml",
            "point": "xarm6_pusht_tabletop/icra_sign_point.xml",
        },
        # The empty slot, second from the top of the row: (0.9, 0.45) in
        # IsaacGym's coordinates. Orientation is the source's own glyph
        # quat, [0,0,0.7071,-0.7071] (xyzw) = -90 degrees about z. The C
        # spawns unrotated, so this task needs a quarter turn too.
        goal=jnp.array([0.2, 0.85, -jnp.pi / 2]),
        # Six glyphs are the convex hull of their own `*_centered.ply`
        # (scale 0.001), which is what MJX collides the mesh geom as. The
        # A is three boxes matching icra_sign.xml's own -- the source
        # assets ship no A mesh, so there is no hull to take.
        obstacles=ObstacleField(
            [
                _glyph(_GLYPH_I, 1.00),
                _glyph(_GLYPH_R, 0.70),
                # A: three boxes, matching icra_sign.xml's own -- the
                # source assets have no A mesh to take a hull from.
                Box([0.2000, 0.5631], [0.00950, 0.04920], angle=-1.82631),
                Box([0.2000, 0.5369], [0.00950, 0.04920], angle=-1.31528),
                Box([0.1825, 0.5500], [0.02494, 0.00760], angle=-1.57080),
                _glyph(_GLYPH_2, 0.30),
                _glyph(_GLYPH_0, 0.15),
                _glyph(_GLYPH_2, 0.00),
                _glyph(_GLYPH_6, -0.15),
                # The robot's own mounted base, previously invisible to
                # the object planner -- see _tee_scene's own comment.
                Circle(center=[-0.3, 0.40], radius=_ROBOT_BASE_RADIUS),
            ]
        ),
        # The block-letter C's own dimensions, matching icra_sign.xml's
        # three geoms.
        footprint_builder=c_shape_footprint,
        footprint_kwargs=dict(
            half_width=0.0350, half_height=0.0515, half_stroke=0.010
        ),
        # IsaacGym's (0.4, 0), shifted by this scene's own (-0.7, +0.40).
        xarm6_base_pos=(-0.3, 0.40),
        xarm6_base_yaw_deg=0.0,
    ),
}
