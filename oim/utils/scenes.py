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
            patch, setting the analytic torque budget `c * r * mu * m * g`.
            It has to reproduce what the scene actually simulates, and
            there are now two ways a scene carries support friction:

            * the table CONTACT, `mu * N` -- every tabletop scene, real
              and simulated. Friction rises when the block is pressed
              down, and the torque the patch transmits is an OUTCOME of
              the footprint's geometry, so `r` is measured by ramping a
              pure torque to breakaway, not chosen. The measured lab T
              gives 0.0422 m and the larger C 0.0548 m; the shared 0.06
              default fits neither.
            * `frictionloss` on the object's MJCF joints -- a constant
              bound, now used by `clutter` alone. There
              `mu * mass * g == frictionloss` must hold on the slides and
              `r * mu * mass * g` on the hinge.

            Either way, if the two descriptions drift apart the analytic
            object model and the simulated one are different objects.
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

    def obstacles_for(self, robot: str) -> ObstacleField:
        """Planner obstacles for `robot`, without absent ones.

        `obstacles` carries the arm's mounted base (see `_tee_scene`),
        which is a real, permanent obstacle only when the arm is there.
        The `point` embodiment has no base -- nothing in its MJCF occupies
        that disc -- so charging the planner for crossing it prices a
        collision that cannot happen, and both blocks then correctly pay
        it whenever the detour costs more. Dropped rather than reweighted:
        a soft cost on a passable region is exactly the failure mode.

        Identified by proximity to `xarm6_base_pos`, the same criterion
        `tests/test_scenes.py` uses, so it does not depend on the base
        being appended last.

        Args:
            robot: Embodiment name, as in `mjcf_by_robot`.

        Returns:
            The obstacle field this embodiment actually collides with.
        """
        if robot == "xarm6" or self.xarm6_base_pos is None:
            return self.obstacles
        base = jnp.asarray(self.xarm6_base_pos, dtype=float)
        return ObstacleField(
            [
                s
                for s in self.obstacles.shapes
                if float(jnp.linalg.norm(jnp.asarray(s.center) - base))
                > 1e-3
            ]
        )

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
# The four T-block scenes share the same arm base, block and goal -- they
# differ *only* in obstacles, exactly as sim_task01..04 do. `icra_sign`
# pushes a letter instead.
#
# RE-SCALED ONTO THE LAB CELL (2026-08-25). These five used to be IsaacGym's
# own metric world -- a 1.40 x 2.50 m table and a 0.200 x 0.150 m T -- so
# every sim result had to be re-derived before it could be trusted on
# hardware. They now run on the measured lab table with the measured lab
# T-block, in the robot base frame, so a plan made here is a plan the real
# cell can execute. What changed and what deliberately did not:
#
#   table          1.40 x 2.50 m  ->  0.800 x 1.523 m
#   T plan         0.200 x 0.150  ->  0.089 x 0.099 m   (the lab's own)
#   mass / mu      2.0 kg / 0.4   ->  0.1 kg / 0.3
#   corridor       (0,0)->(0.2,0.75)  ->  (0.381,+0.4)->(0.381,-0.4)
#   world frame    world          ->  the robot base frame
#   OBSTACLE SIZES unchanged -- the YCB meshes are scans of real objects
#                  and the cube is a real 0.1 m cube; only the table and
#                  the T were oversized.
#
# The table is 3.7 cm deeper than the lab's own 0.763 m, which buys the
# obstacle field room to clear both the corridor and the arm base;
# `_real_scene` keeps the true one.
#
# LAYOUT RULE, the same for all five: the corridor runs straight down the
# table's long axis from y = +0.4 to y = -0.4, and the obstacle field is
# confined to |y| <= 0.3, the middle third. Both ends of every task are
# therefore at least 0.1 m clear of anything in the way, and obstacles are
# spaced so no lane the block must use is narrower than its own 0.089 m
# crossbar. `icra_sign` is the one exception to the band and says so at
# its own entry.

# The measured lab T-block's plan footprint, shared with the real scenes
# (see `_REAL_TEE_FOOTPRINT`, which is the same numbers for the same
# object). It replaces IsaacGym's tee_block.urdf, 2.2x larger in plan than
# anything the lab actually pushes.
_TABLETOP_TEE_FOOTPRINT = dict(
    crossbar_half=(0.0445, 0.0099),
    stem_half=(0.0099, 0.0397),
    crossbar_y=0.0099,
    stem_y=-0.0397,
)

# conf/actors/block.yaml: a 0.1 m cube, full size, parked mid-table at y = 0
# squarely across the corridor's own x. Shared by `single_obstacle` and
# `ycb_clutter`.
_TABLETOP_CUBE = Box(center=[0.35, 0.0], half_extents=[0.05, 0.05])

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

    Every glyph sits at x = 0.5 under the same +90 degree yaw -- the yaw
    that makes the sign read from the recording camera -- so the outlines
    below are stored once in that placed orientation and only translated
    here. `digit_2` and `digit_2b` share one.

    Args:
        outline: The glyph's hull, relative to its own centre.
        y: Its slot in the row.

    Returns:
        The world-frame footprint.
    """
    return Polygon(jnp.array([[x + 0.5, py + y] for x, py in outline]))


# ycb/spamCan/nontextured.stl, 0.94 of its true footprint hull.
_SPAM_CAN = Polygon(
    jnp.array(
        [
            [0.5530, -0.1815], [0.5704, -0.1901], [0.6343, -0.1870],
            [0.6510, -0.1695], [0.6503, -0.1473], [0.6301, -0.1299],
            [0.5670, -0.1329], [0.5490, -0.1500],
        ]
    )
)
# ycb/mustardBottle/nontextured.stl, 0.93 of its true footprint hull.
_MUSTARD_BOTTLE = Polygon(
    jnp.array(
        [
            [0.1186, -0.1945], [0.1501, -0.2104], [0.1866, -0.2115],
            [0.1976, -0.1831], [0.1818, -0.1655], [0.1555, -0.1506],
            [0.1121, -0.1493], [0.1014, -0.1732],
        ]
    )
)

# Glyph hulls, relative to each glyph's own centre, already under the row's
# -90 degree yaw. GENERATED -- run
# `oim/models/xarm6_pusht_tabletop/glyph_hulls.py` and paste. It reads the
# COMPILED scene rather than the OBJs on disk, so these are the hulls MJX
# actually collides (MuJoCo re-frames a mesh on load) rather than the
# outlines the files happen to store.
#
# Simplified to 10 vertices for `Polygon.sdf`'s sake, dropping the vertex
# that loses the least area each time -- so each polygon stays INSIDE the
# true hull while still covering well over 85% of it, which is the two-way
# contract `tests/test_scenes.py::_assert_hull_matches` enforces. The
# generator asserts both halves itself, so a bad vertex budget fails there
# rather than in the suite.
# _GLYPH_I: 10 verts, 0.1030 x 0.0298 m (cap height x width)
_GLYPH_I = (
    (-0.0320, -0.0149), (0.0503, -0.0149),
    (0.0511, -0.0145), (0.0515, -0.0137),
    (0.0515, 0.0138), (0.0503, 0.0149),
    (-0.0504, 0.0149), (-0.0515, 0.0137),
    (-0.0515, -0.0138), (-0.0504, -0.0149),
)
# _GLYPH_R: 10 verts, 0.1030 x 0.1012 m (cap height x width)
_GLYPH_R = (
    (0.0515, -0.0496), (0.0515, 0.0496),
    (0.0511, 0.0506), (-0.0294, 0.0417),
    (-0.0445, 0.0340), (-0.0502, 0.0208),
    (-0.0515, 0.0035), (-0.0515, -0.0496),
    (-0.0507, -0.0506), (0.0508, -0.0506),
)
# _GLYPH_A: 10 verts, 0.1030 x 0.1112 m (cap height x width)
_GLYPH_A = (
    (-0.0515, 0.0162), (-0.0515, -0.0162),
    (-0.0501, -0.0180), (0.0499, -0.0556),
    (0.0511, -0.0556), (0.0515, -0.0543),
    (0.0515, 0.0543), (0.0511, 0.0556),
    (0.0499, 0.0556), (-0.0501, 0.0180),
)
# _GLYPH_2: 10 verts, 0.1030 x 0.0834 m (cap height x width)
_GLYPH_2 = (
    (0.0515, -0.0362), (0.0515, 0.0417),
    (0.0279, 0.0417), (-0.0219, 0.0386),
    (-0.0345, 0.0352), (-0.0494, 0.0155),
    (-0.0515, -0.0020), (-0.0487, -0.0181),
    (-0.0416, -0.0315), (-0.0302, -0.0417),
)
# _GLYPH_0: 10 verts, 0.1022 x 0.0855 m (cap height x width)
_GLYPH_0 = (
    (-0.0511, -0.0061), (-0.0361, -0.0324),
    (-0.0051, -0.0430), (0.0213, -0.0401),
    (0.0472, -0.0190), (0.0511, 0.0061),
    (0.0361, 0.0324), (-0.0100, 0.0425),
    (-0.0361, 0.0324), (-0.0472, 0.0190),
)
# _GLYPH_6: 10 verts, 0.1017 x 0.0825 m (cap height x width)
_GLYPH_6 = (
    (0.0158, 0.0416), (-0.0433, 0.0390),
    (-0.0503, 0.0219), (-0.0480, -0.0113),
    (-0.0181, -0.0388), (0.0130, -0.0409),
    (0.0425, -0.0255), (0.0514, -0.0002),
    (0.0487, 0.0196), (0.0367, 0.0356),
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
    return SceneSpec(
        mjcf_by_robot={
            "xarm6": f"xarm6_pusht_tabletop/{name}.xml",
            "point": f"xarm6_pusht_tabletop/{name}_point.xml",
        },
        # 0.800 m straight down the table's long axis, +0.4 -> -0.4, with
        # the obstacle field confined to the |y| <= 0.3 band between them.
        # The 180-degree flip about z is IsaacGym's own and is what makes
        # these tasks rotational rather than pure translation.
        goal=jnp.array([0.381, -0.400, jnp.pi]),
        object_start=(0.381, 0.400, 0.0),
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
                Circle(center=[0.0, 0.0], radius=_ROBOT_BASE_RADIUS),
            ]
        ),
        footprint_kwargs=dict(_TABLETOP_TEE_FOOTPRINT),
        # The lab block's own physics, not IsaacGym's 2.0 kg / mu 0.4.
        mass=0.1,
        mu=0.3,
        # MEASURED, not assumed, and the same number the real scenes carry
        # because this is now the same block on the same table under the
        # same contact pairs. Support friction is the table contact's
        # (mu*N, see tee.xml), so the torque the patch can transmit is an
        # OUTCOME of the footprint's geometry: ramping a pure torque to
        # breakaway gives 0.012405 N*m against mu*m*g = 0.2943 N, i.e. an
        # effective radius of 0.0422 m. The old 0.1007 was the 0.200 m T's.
        limit_surface_radius=0.0422,
        # The arm base IS the world origin, so nothing has to publish a
        # world -> base transform and FoundationPose's TF reads straight
        # into the planner. `base_z` is the measured mount offset that puts
        # the model floor on the real table -- see the README's bring-up
        # checklist.
        #
        # The arm home is the standard forward-facing work pose, not the
        # lab's own: it puts the tip over the middle of the table pointing
        # straight down, 0.43 m from the block, so the run has to drive out
        # to first contact. The lab pose parks the tip 4 cm from the block's
        # start, which is where `_real_scene` still wants it.
        xarm6_base_pos=(0.0, 0.0),
        xarm6_base_yaw_deg=0.0,
        xarm6_base_z=-0.0111,
        xarm6_arm_start_deg=[0.0, -45.0, -45.0, 0.0, 90.0],
        world_frame="xarm_device",
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
                limit_surface_radius=0.0422) -> SceneSpec:
    """A SceneSpec for a real-table scene run on the lab xArm6.

    Fixes what every real scene shares -- the arm base at the world origin
    (world_frame='xarm_device', so no world->base transform) and the lab
    block's physics -- and takes only what varies: the scene MJCF, obstacles,
    goal and start poses. Object shape defaults to the measured T-block; a
    different physical object overrides footprint_builder/kwargs and physics.

    `limit_surface_radius` is MEASURED, not chosen. Support friction became
    the table contact's (`mu*N`, see tee_real.xml) rather than a
    `frictionloss` bound on the block's joints, so the torque the patch can
    transmit is now an OUTCOME of the footprint's geometry. Ramping a pure
    torque to breakaway by bisection on a constant applied wrench gives
    0.012405 N*m against `mu*m*g` = 0.2943 N, i.e. r = 0.0422 m -- close to
    the 0.04 the frictionloss era declared, which is a useful check that the
    old number was not far wrong rather than a coincidence. Translation was
    measured the same way and lands at 0.2867 N, 0.974 of the nominal budget,
    so `PlanarPushingObject.wrench_limit` still means what it says.
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
    "box_clutter_real": _real_scene(
        "box_clutter_real",
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
            # ON the straight start->goal line, a third of the way along it
            # and therefore nearer the start. NOT sim's layout: sim's cube
            # sits 68.9% along and 8.3% off to one side, which leaves the
            # direct route open. The point of this scene is that the route is
            # blocked, so the two fractions are 33.3% and 0. The derivation
            # is in single_obstacle_real.xml, whose geom this must match.
            Box(center=[0.381, 0.127], half_extents=[0.054, 0.0445], angle=jnp.pi / 2),
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
    # sim_task03: "... avoiding two shelves". THE GAP IS THE TASK: 0.200 m
    # -- 2.2 T-crossbars, up from the 0.1335 m (1.5x) squeeze -- centred on
    # x = 0.381, the corridor's own x, so straight through stays the centred
    # route. The gate sits at y = 0, 0.4 m from both ends of the corridor.
    #
    # The shelves run PARALLEL TO THE ARM (long axis along x) and are
    # shorter and smaller than before, 0.16 m deep and 0.14 m tall against
    # 0.25 x 0.20. The far one hugs the far table edge at x = 0.75, closing
    # that bypass; the near one cannot hug the near edge, since the arm is
    # bolted to it, so it stops at x = 0.19 and leaves a 0.099 m slot beside
    # the base -- passable by the 0.089 m crossbar with 1 cm to spare, which
    # keeps "go around the outside" alive as the tighter alternative.
    # See shelf_gap.xml, whose geoms these must match.
    "shelf_gap": _tee_scene(
        "shelf_gap",
        [
            Box(center=[0.62, 0.0], half_extents=[0.13, 0.08]),
            Box(center=[0.24, 0.0], half_extents=[0.05, 0.08]),
        ],
    ),
    # sim_task04: "... avoiding multiple obstacles". Two of the three YCB
    # actors are meshes in their URDFs and appear here as their convex
    # hulls; dominoSugar is a box in its own URDF and stays one.
    #
    # Every one keeps its real size -- they are scans of real objects -- and
    # the whole field is confined to |y| <= 0.3, so the corridor's two ends
    # are clear of clutter. Spread across both halves of that band and
    # staggered in x so no two form a second gate. Measured on the compiled
    # scene, the free lanes across the table are 0.183/0.442 m at the sugar
    # box, 0.209/0.350 m at the cube, and 0.152/0.351/0.099 m at the
    # mustard-and-spam row, against an 0.089 m crossbar. The cube must NOT
    # move independently: `_TABLETOP_CUBE` is shared with `single_obstacle`.
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
            Box(center=[0.22, 0.20], half_extents=[0.0875, 0.0475]),
        ],
    ),
    "icra_sign": SceneSpec(
        # sim_task05, respelled: seven fixed glyphs spell "ICRA 2026" in a
        # row at x = 0.5 with the C's own slot left empty, and the goal is
        # that slot. See icra_sign.xml for why the C is the pushed letter.
        #
        # THE ROW READS -y -> +y, so "ICRA" is on the negative half and
        # "2026" on the positive one, and every glyph carries a +90 degree
        # yaw rather than -90. Both together are what make the sign read
        # right way round in the recording: the camera sits at +x with its
        # right axis along +y, so under the old layout the row rendered
        # mirrored and upside down.
        #
        # THE ONE EXCEPTION TO THE |y| <= 0.3 OBSTACLE BAND. Eight slots at
        # 0.15 m spacing is a 1.2 m row, and squeezing that into 0.6 m would
        # leave gaps too narrow for the 0.0966 m C to enter its own slot --
        # the sign's length is the task. It keeps the family's start/goal
        # rule instead: the C starts on the +y half and its slot is at
        # y = -0.40, the same goal line every other scene uses.
        mjcf_by_robot={
            "xarm6": "xarm6_pusht_tabletop/icra_sign.xml",
            "point": "xarm6_pusht_tabletop/icra_sign_point.xml",
        },
        # The empty slot, second letter of "ICRA". The C spawns unrotated,
        # so reaching it needs the row's own +90 degree quarter turn as well
        # as the translation.
        goal=jnp.array([0.5, -0.40, jnp.pi / 2]),
        object_start=(0.3, 0.400, 0.0),
        # Every glyph is the convex hull of its own mesh, which is what
        # MJX collides a mesh geom as.
        obstacles=ObstacleField(
            [
                _glyph(_GLYPH_I, -0.55),
                _glyph(_GLYPH_R, -0.25),
                _glyph(_GLYPH_A, -0.10),
                _glyph(_GLYPH_2, 0.15),
                _glyph(_GLYPH_0, 0.30),
                _glyph(_GLYPH_2, 0.45),
                _glyph(_GLYPH_6, 0.60),
                # The robot's own mounted base, previously invisible to
                # the object planner -- see _tee_scene's own comment.
                Circle(center=[0.0, 0.0], radius=_ROBOT_BASE_RADIUS),
            ]
        ),
        # The block-letter C standing in for the round `glyph_c` mesh,
        # matching icra_sign.xml's three collision boxes. The stroke is the
        # mesh's own: its spine spans x in [-0.0482, -0.0163] at
        # mid-height, i.e. 0.0319 m across a half-width of 0.0483.
        footprint_builder=c_shape_footprint,
        footprint_kwargs=dict(
            half_width=0.0483, half_height=0.0515, half_stroke=0.016
        ),
        # The lab table's physics, and 0.1 kg -- a letter this size at
        # IsaacGym's 2.0 kg would have been eight times the density of
        # steel. mu*m*g = 0.2943 N, the same nominal budget as the T.
        mass=0.1,
        mu=0.3,
        # The effective radius is PURELY GEOMETRIC for a uniform pressure
        # patch -- r = tau_max / (mu*m*g) -- so re-massing the letter does
        # not move it. This C reaches only +/-0.0483 m, and 0.0548 m is what
        # a pure-torque breakaway sweep over that footprint gives.
        limit_surface_radius=0.0548,
        xarm6_base_pos=(0.0, 0.0),
        xarm6_base_yaw_deg=0.0,
        xarm6_base_z=-0.0111,
        xarm6_arm_start_deg=[0.0, -45.0, -45.0, 0.0, 90.0],
        world_frame="xarm_device",
    ),
}
