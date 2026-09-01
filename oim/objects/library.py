"""Registry of the objects `oim.tasks.pusht.PushT` can push.

A scene fixes the table, the obstacles and the goal; this fixes WHAT gets
pushed across it. `run.object` in the robot config (or `--object`) picks
one, so swapping a T for a power drill is a config edit rather than a new
scene, and every scene keeps its own layout unchanged.

Each entry carries ONE list of axis-aligned boxes, and everything else is
derived from it: the MJCF collision geoms, the goal/local_goal markers
that mirror them, and the analytic footprint the object-level planner
reasons about (`boxes_footprint`). That is the point of the design --
those three descriptions cannot drift apart, which is what
`tests/test_scenes.py::test_footprint_matches_the_block_geoms` checks geom
by geom for the scene-default object and `tests/test_objects_library.py`
checks for every entry here.

WHY BOXES AND NOT THE MESH. MJX collides a mesh geom as its CONVEX HULL,
which for a power drill or a banana fills in the very concavity that makes
the shape interesting -- the same reason `icra_sign`'s C is three boxes
rather than its own mesh. A convex decomposition keeps the concavity in
contact. The cost is fidelity: the boxes are strictly INSIDE the true
outline, and each entry records how much of it they recover.

The mesh is still loaded, as a visual-only geom drawn over the boxes, so a
recording shows the real object rather than its decomposition.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from oim.objects.planar_pushing import boxes_footprint
from oim.objects.sdf import Polygon

# The name that means "whatever the scene's own MJCF already declares".
# Not an entry below: the scene files ship the T (or, for `icra_sign`, the
# letter C) built in, and that path swaps nothing at all.
SCENE_DEFAULT = "scene"


@dataclass(frozen=True)
class PushObject:
    """One pushable object: its collision boxes and its physics.

    Args:
        boxes: `(centre_x, centre_y, half_x, half_y)` per collision box, in
            the object's own body frame, centred on its plan bounding box.
            Axis-aligned by contract -- see the module docstring.
        half_height: Half the object's thickness. The body sits at this
            height so its underside rests on the table at z = 0, and
            `PushT` reads it as the object's mid-height (`tip_target_z`).
        mass: Total mass, split across the boxes in proportion to area.
        mu: Friction coefficient against the tabletop.
        limit_surface_radius: Effective radius of the contact patch,
            setting the analytic torque budget `c * r * mu * m * g`.
            MEASURED by ramping a pure torque to breakaway against the
            compiled model, never chosen -- the same procedure the T's
            0.0422 and the C's 0.0548 came from.
        mesh: Visual-only mesh under `models/xarm6_pusht_tabletop/assets/`,
            or None to draw the boxes themselves.
        mesh_offset: Translation applied to the visual mesh so it lines up
            with the boxes -- the vendored OBJs are not centred on their
            own plan bounding box, and the boxes are.
        coverage: Fraction of the mesh's true footprint the boxes recover.
            Recorded rather than asserted: it is a fidelity number for a
            reader, and the boxes are what is actually simulated.
        rgba: Colour for the collision boxes when no mesh is drawn.
    """

    boxes: Tuple[Tuple[float, float, float, float], ...]
    half_height: float
    mass: float
    mu: float
    limit_surface_radius: float
    mesh: Optional[str] = None
    mesh_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    coverage: float = 1.0
    rgba: Tuple[float, float, float, float] = (0.2, 0.45, 0.85, 1.0)

    def footprint(self) -> Polygon:
        """The analytic outline: the exact union of `boxes`."""
        return boxes_footprint(self.boxes)

    def masses(self) -> Tuple[float, ...]:
        """`mass` split across the boxes in proportion to plan area.

        Overlapping boxes double-count a little inertia this way, which is
        deliberate: the alternative is an explicit `<inertial>`, and for a
        quasi-static planar push what has to be right is the TOTAL mass
        (it sets `mu * m * g`, the whole friction budget), not the exact
        yaw inertia of a decomposition that is itself an approximation.
        """
        areas = [4.0 * hx * hy for _, _, hx, hy in self.boxes]
        total = sum(areas)
        return tuple(self.mass * a / total for a in areas)


# YCB meshes, vendored under models/xarm6_pusht_tabletop/assets/. Masses are
# the YCB set's own published values, not guesses; `mu` is the lab table's
# 0.3 throughout, the same coefficient the T is pushed under, so switching
# object changes the object and nothing about the surface.
#
# Boxes were fitted to each mesh's own footprint by greedy maximal-rectangle
# cover at 2 mm resolution, in the mesh's own orientation (checked against
# rotating it first -- for both of these the mesh's own frame was already
# the best axis alignment) with an 8 mm floor on box size so the
# decomposition stays physical rather than collecting slivers.
PUSH_OBJECTS: Dict[str, PushObject] = {
    # 035_power_drill. An L in plan: the barrel along x, the handle hanging
    # off it in -y. 0.184 x 0.188 m, twice the T's footprint, so it does not
    # fit every scene -- see `tests/test_objects_library.py`, which reports
    # which scene/object pairs are geometrically feasible.
    "power_drill": PushObject(
        boxes=(
            (0.0049, 0.0672, 0.0830, 0.0190),
            (0.0389, -0.0018, 0.0150, 0.0900),
            (0.0159, -0.0768, 0.0440, 0.0150),
            (0.0089, 0.0472, 0.0150, 0.0410),
            (0.0279, 0.0602, 0.0480, 0.0280),
        ),
        half_height=0.0287,
        mass=0.895,
        mu=0.3,
        limit_surface_radius=0.0722,
        mesh="power_drill",
        coverage=0.840,
    ),
    # 011_banana. A crescent, and the hardest of the four to describe with
    # axis-aligned boxes -- 75% is what five of them recover. The concavity
    # on its inner edge survives, which is the point; a mesh geom would have
    # been collided as a filled hull and lost it.
    "banana": PushObject(
        boxes=(
            (-0.0315, -0.0212, 0.0130, 0.0540),
            (-0.0155, 0.0458, 0.0130, 0.0170),
            (0.0065, 0.0708, 0.0130, 0.0120),
            (-0.0485, -0.0252, 0.0040, 0.0280),
            (0.0265, 0.0818, 0.0250, 0.0050),
        ),
        half_height=0.0184,
        mass=0.066,
        mu=0.3,
        limit_surface_radius=0.0398,
        mesh="banana",
        coverage=0.748,
    ),
    # 004_sugar_box, laid flat -- the largest face down, which is how a
    # carton rests and the only orientation with a stable base (upright it
    # is a 0.088 m push height on a 0.049 m base). The mesh is rotated
    # +90 deg about y before centring, so plan is 0.176 x 0.094 m and the
    # 0.049 m axis becomes the thickness.
    #
    # The only CONVEX, non-symmetric entry: a rectangle, so one box reaches
    # 96.5% with no cell outside the hull, and unlike `tomato_soup` its
    # orientation is well posed -- a can's theta goal is vacuous.
    "sugar_box": PushObject(
        boxes=((0.0000, 0.0000, 0.0860, 0.0450),),
        half_height=0.0247,
        mass=0.514,
        mu=0.3,
        limit_surface_radius=0.0589,
        mesh="sugar_box",
        coverage=0.965,
    ),
    # 005_tomato_soup_can, standing on its base. Convex, so the boxes are
    # only approximating a circle rather than recovering a concavity -- four
    # of them reach 93% of the disc. The tallest of the four at 0.102 m;
    # it cannot tip, since the block body carries no roll or pitch DoF.
    "tomato_soup": PushObject(
        boxes=(
            (0.0000, -0.0009, 0.0240, 0.0250),
            (0.0000, 0.0001, 0.0320, 0.0140),
            (0.0000, 0.0001, 0.0140, 0.0320),
            (0.0000, -0.0009, 0.0280, 0.0210),
        ),
        half_height=0.0510,
        mass=0.349,
        mu=0.3,
        limit_surface_radius=0.0248,
        mesh="tomato_soup",
        coverage=0.925,
    ),
}


def push_object(name: str) -> Optional[PushObject]:
    """The entry called `name`, or None for `SCENE_DEFAULT`.

    Args:
        name: A key of `PUSH_OBJECTS`, or `SCENE_DEFAULT`.

    Returns:
        The object, or None when the scene's own MJCF should be left alone.

    Raises:
        ValueError: If `name` is neither, naming what is available.
    """
    if name == SCENE_DEFAULT:
        return None
    if name not in PUSH_OBJECTS:
        raise ValueError(
            f"object={name!r} is not in oim.objects.library.PUSH_OBJECTS "
            f"(available: {SCENE_DEFAULT!r}, "
            f"{', '.join(repr(k) for k in sorted(PUSH_OBJECTS))})"
        )
    return PUSH_OBJECTS[name]


def object_names() -> Sequence[str]:
    """Every accepted `--object` value, `SCENE_DEFAULT` first."""
    return (SCENE_DEFAULT, *sorted(PUSH_OBJECTS))


def apply_to_spec(spec: Any, obj: PushObject) -> None:
    """Rebuild a scene's pushed object as `obj`, in place, before compiling.

    The scene MJCFs ship the T built in (`tee.xml`), which is what keeps
    them loadable on their own -- `tests/test_scenes.py` reads them by path
    and expects a `block`. Selecting a different object therefore means
    editing the loaded `mujoco.MjSpec` rather than picking a different
    file, which is also why there is one object include and not one scene
    file per (scene, object) pair.

    What changes: the `block` body's collision geoms, the `goal` and
    `local_goal` markers that mirror them (so overlap stays the success
    criterion by eye), the body heights, and the block-vs-table contact
    pairs that carry the friction. What does NOT change: the table, the
    obstacles, the goal POSE, the arm, the keyframe -- switching object
    leaves the task's layout exactly as the scene declared it.

    Args:
        spec: A `mujoco.MjSpec` loaded from a scene file, not yet compiled.
        obj: The replacement object.
    """
    import mujoco  # noqa: PLC0415 - keeps this module importable without it

    # The old object's pairs must go before its geoms: a pair naming a
    # deleted geom fails to compile. Matched by BODY, not by a `block_`
    # name prefix -- icra_sign's letter calls its geoms `c_spine`,
    # `c_top_bar` and `c_bot_bar`, and a prefix test silently left their
    # table pairs behind pointing at geoms that no longer existed.
    doomed = {g.name for g in spec.body("block").geoms}
    for pair in list(spec.pairs):
        if pair.geomname1 in doomed or pair.geomname2 in doomed:
            spec.delete(pair)
    for body_name in ("block", "goal", "local_goal"):
        for geom in list(spec.body(body_name).geoms):
            spec.delete(geom)

    if obj.mesh is not None:
        mesh = spec.add_mesh()
        mesh.name = "pushed_object"
        # The `_centered` copy, whose plan bounding box is on the origin and
        # whose underside is at z = 0 -- the same convention ycb_clutter's
        # obstacle meshes are vendored under, and what lets the visual geom
        # line up with boxes that are themselves centred.
        mesh.file = f"assets/{obj.mesh}_centered.obj"

    masses = obj.masses()
    for body_name, cls in (
        ("block", None), ("goal", "goal"), ("local_goal", "local_goal")
    ):
        body = spec.body(body_name)
        # Sit the body at its own half-height, so the underside rests on the
        # table at z = 0 and `tip_target_z` reads the true mid-height.
        body.pos = [body.pos[0], body.pos[1], obj.half_height]
        default = spec.find_default(cls) if cls else None
        for i, ((cx, cy, hx, hy), mass) in enumerate(
            zip(obj.boxes, masses, strict=True)
        ):
            geom = body.add_geom(default) if default else body.add_geom()
            geom.name = f"{body_name}_box{i}"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [hx, hy, obj.half_height]
            geom.pos = [cx, cy, 0.0]
            if cls is None:
                geom.mass = mass
                # 3, matching tee.xml's block: collides with the arm (1) and
                # with the floor plane and table (2).
                geom.contype, geom.conaffinity = 3, 3
                geom.rgba = list(obj.rgba)
        if obj.mesh is not None:
            visual = body.add_geom()
            visual.name = f"{body_name}_visual"
            visual.type = mujoco.mjtGeom.mjGEOM_MESH
            visual.meshname = "pushed_object"
            visual.pos = [
                obj.mesh_offset[0],
                obj.mesh_offset[1],
                obj.mesh_offset[2] - obj.half_height,
            ]
            # Visual ONLY, and massless: the boxes carry the whole mass, and
            # a collidable mesh would be collided as its convex hull and
            # undo the decomposition. Same split as icra_sign's `c_visual`.
            visual.contype, visual.conaffinity, visual.mass = 0, 0, 0.0
            visual.rgba = (
                list(obj.rgba) if cls is None else [0.0, 1.0, 0.0, 0.25]
            )

    # Planar friction is the table CONTACT's, mu*N, exactly as tee.xml sets
    # it up -- so it rises when the pusher presses down, and top-riding is
    # priced by physics rather than only by a cost term.
    for i in range(len(obj.boxes)):
        pair = spec.add_pair()
        pair.geomname1, pair.geomname2 = f"block_box{i}", "table"
        pair.condim = 3
        pair.friction = [obj.mu, obj.mu, 0.005, 0.0001, 0.0001]
        pair.solimp = [0.999, 0.9999, 0.0001, 0.5, 2.0]
