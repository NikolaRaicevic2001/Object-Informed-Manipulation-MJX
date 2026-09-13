"""A sign scene: one letter is pushed into its slot, the rest stand in theirs.

`SceneSpec.letter_slots` maps `oim.objects.library` names to SE(2) slots.
Given which letter is pushed, this module derives the three things the
task needs, all from the letters' own `PushObject.boxes`:

  * the goal -- the pushed letter's slot,
  * the planner's obstacles -- every other letter's boxes at its slot,
  * the compiled model's obstacle bodies -- the same boxes, as `obs_<name>`
    mocap bodies, which `PushT` already treats as obstacles by name.

One box list per letter feeds both the pushed and the obstacle role, so a
letter is collided against exactly as it is pushed.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from oim.objects.library import PUSH_OBJECTS, SCENE_DEFAULT, PushObject
from oim.objects.sdf import Box, Shape

Slot = Tuple[float, float, float]


def is_sign(spec: Any) -> bool:
    """Whether `spec` carries a letter slot table."""
    return bool(getattr(spec, "letter_slots", None))


def resolve_letter(spec: Any, push_object: str) -> str:
    """The letter to push: `push_object`, or the scene's default when unset.

    Args:
        spec: A sign `SceneSpec`.
        push_object: The `--object` value; `SCENE_DEFAULT` means unset.

    Returns:
        A key of `spec.letter_slots`.

    Raises:
        ValueError: If `push_object` names something with no slot in this
            sign, listing what has one.
    """
    slots = spec.letter_slots
    if push_object == SCENE_DEFAULT:
        push_object = spec.default_letter or next(iter(slots))
    if push_object not in slots:
        raise ValueError(
            f"object={push_object!r} has no slot in this sign "
            f"(letters: {sorted(slots)}); a sign scene pushes one of its "
            "own letters into place"
        )
    return push_object


def goal_for(spec: Any, letter: str) -> np.ndarray:
    """The pushed letter's slot, as the SE(2) goal."""
    return np.asarray(spec.letter_slots[letter], dtype=float)


def _placed_boxes(obj: PushObject, slot: Slot):
    """`obj.boxes` moved to `slot`: world centre, half extents, yaw."""
    x, y, yaw = slot
    c, s = np.cos(yaw), np.sin(yaw)
    for cx, cy, hx, hy in obj.boxes:
        yield (x + c * cx - s * cy, y + s * cx + c * cy), (hx, hy), yaw


def letter_obstacles(spec: Any, pushed: str) -> List[Shape]:
    """Every letter except `pushed`, as oriented boxes at its slot.

    Boxes rather than the union outline: `Box` is what the obstacle SDF
    already handles, and per-box shapes keep the concavities (the C's
    bowl, the gap under the R's leg) that a hull would fill in.
    """
    shapes: List[Shape] = []
    for name, slot in spec.letter_slots.items():
        if name == pushed:
            continue
        for centre, half, yaw in _placed_boxes(PUSH_OBJECTS[name], slot):
            shapes.append(Box(center=list(centre), half_extents=list(half),
                              angle=float(yaw)))
    return shapes


def apply_to_spec(mj_spec: Any, spec: Any, pushed: str) -> None:
    """Install the standing letters and move the goal marker, before compile.

    Each standing letter becomes an `obs_<name>` mocap body whose geoms are
    its boxes, named so `PushT.obstacle_geoms` picks them up by the same
    "obs*" rule the calibrated clutter boxes use. The `goal` body is moved
    to the pushed letter's slot; `library.apply_to_spec` (run first)
    already gave it that letter's geoms.

    Args:
        mj_spec: The scene's `mujoco.MjSpec`, after `library.apply_to_spec`.
        spec: The sign `SceneSpec`.
        pushed: The letter being pushed.
    """
    import mujoco  # noqa: PLC0415

    x, y, yaw = spec.letter_slots[pushed]
    goal = mj_spec.body("goal")
    goal.pos = [x, y, goal.pos[2]]
    goal.quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    default = mj_spec.find_default("obstacle")
    for name, slot in spec.letter_slots.items():
        if name == pushed:
            continue
        obj = PUSH_OBJECTS[name]
        sx, sy, syaw = slot
        body = mj_spec.worldbody.add_body()
        body.name = f"obs_{name}"
        body.mocap = True
        body.pos = [sx, sy, obj.half_height]
        body.quat = [np.cos(syaw / 2), 0.0, 0.0, np.sin(syaw / 2)]
        for i, (cx, cy, hx, hy) in enumerate(obj.boxes):
            geom = body.add_geom(default) if default else body.add_geom()
            geom.name = f"obs_{name}_box{i}"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [hx, hy, obj.half_height]
            geom.pos = [cx, cy, 0.0]
