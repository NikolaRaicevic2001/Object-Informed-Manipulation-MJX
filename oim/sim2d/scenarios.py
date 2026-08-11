"""Named 2D push scenarios.

Three environments, ordered by what they stress:

* `clutter`  -- route *around* scattered obstacles. Mirrors the MuJoCo
  `pusht_clutter` scene exactly (same goal, same three obstacles), so a 2D
  and a 3D run of the same planner are directly comparable.
* `corridor` -- push *through* a narrow horizontal channel.
* `gate`     -- push through a vertical slot, then reach a rotated goal.

The last two matter because `clutter` never forces the object through an
opening it barely fits: there is always a way round. A planner can look
healthy on `clutter` while being unable to commit to a passage, which is
exactly the non-myopic behaviour the object-level block exists to provide.
Both are sized against the T footprint (~0.15 m tall, ~0.18 m wide) so the
object fits without needing to rotate -- `tests/test_sim2d.py` asserts the
clearances rather than trusting the numbers here.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import jax.numpy as jnp

from oim.objects import Box, Circle, Polygon, Shape, t_shape_footprint


@dataclass
class Scenario:
    """A complete 2D push problem: geometry, goal, and start state."""

    name: str
    description: str
    footprint: Polygon
    goal: Tuple[float, float, float]
    obstacles: List[Shape]
    object_pose0: Tuple[float, float, float]
    robot_pos0: Tuple[float, float]
    view: Tuple[float, float, float, float] = field(
        default=(-0.25, 0.75, -0.25, 0.65)
    )
    """Plot bounds (xmin, xmax, ymin, ymax)."""


def clutter() -> Scenario:
    """Route around three scattered obstacles to a rotated goal.

    Goal and obstacles match `oim/models/pusht_clutter/pusht_clutter.xml`
    and `oim.utils.scenes.SCENES["clutter"]`, so this is the 2D twin of the
    MuJoCo demo.
    """
    return Scenario(
        name="clutter",
        description="Route around scattered obstacles to an SE(2) goal",
        footprint=t_shape_footprint(),
        goal=(0.50, 0.48, float(jnp.pi / 4)),
        obstacles=[
            Circle(center=[0.08, 0.32], radius=0.04),
            Box(center=[0.38, 0.10], half_extents=[0.04, 0.035], angle=0.25),
            Polygon(jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])),
        ],
        object_pose0=(0.0, 0.0, 0.0),
        robot_pos0=(0.0, -0.13),
        view=(-0.25, 0.75, -0.25, 0.65),
    )


def corridor() -> Scenario:
    """Push along a narrow horizontal channel to a goal past the exit.

    The walls leave a 0.24 m vertical gap against a ~0.15 m tall object, so
    it fits upright but has little room to wander. A post sits north of the
    exit, away from the goal, to discourage cutting the corner.
    """
    return Scenario(
        name="corridor",
        description="Push through a narrow horizontal corridor",
        footprint=t_shape_footprint(),
        goal=(0.70, 0.15, 0.0),
        obstacles=[
            # Top wall: y in [0.27, 0.33]; bottom wall: y in [-0.03, 0.03].
            Box(center=[0.22, 0.30], half_extents=[0.20, 0.03]),
            Box(center=[0.22, 0.00], half_extents=[0.20, 0.03]),
            Circle(center=[0.52, 0.34], radius=0.04),
        ],
        object_pose0=(-0.10, 0.15, 0.0),
        robot_pos0=(-0.18, 0.08),
        view=(-0.30, 0.85, -0.15, 0.50),
    )


def gate() -> Scenario:
    """Push through a vertical slot, then reach a lightly rotated goal.

    The pillars leave a 0.22 m slot at x = 0.28. Start and goal are both
    centred on it so the object can pass upright, but the goal's pi/6
    rotation means the planner still has to turn the object *after* the
    passage rather than drifting through at an angle.
    """
    return Scenario(
        name="gate",
        description="Pass through a vertical gate slot, then rotate to goal",
        footprint=t_shape_footprint(),
        goal=(0.62, 0.13, float(jnp.pi / 6)),
        obstacles=[
            # Upper pillar y in [0.24, 0.44]; lower pillar y in [-0.18, 0.02].
            Box(center=[0.28, 0.34], half_extents=[0.035, 0.10]),
            Box(center=[0.28, -0.08], half_extents=[0.035, 0.10]),
            Circle(center=[0.48, 0.38], radius=0.04),
            Polygon(jnp.array([[0.52, -0.18], [0.64, -0.18], [0.58, -0.06]])),
        ],
        object_pose0=(-0.05, 0.13, 0.0),
        robot_pos0=(-0.12, 0.05),
        view=(-0.25, 0.80, -0.30, 0.55),
    )


SCENARIOS: Dict[str, Callable[[], Scenario]] = {
    "clutter": clutter,
    "corridor": corridor,
    "gate": gate,
}

DEFAULT_SCENARIO = "clutter"


def list_scenarios() -> Sequence[str]:
    """Names accepted by `build_scenario`."""
    return sorted(SCENARIOS)


def build_scenario(name: str = DEFAULT_SCENARIO) -> Scenario:
    """Look up a scenario by name.

    Args:
        name: One of `list_scenarios()`.

    Returns:
        The scenario.

    Raises:
        ValueError: If the name is not known.
    """
    key = name.lower().strip()
    if key not in SCENARIOS:
        raise ValueError(
            f"unknown scenario '{name}'; choose from {list_scenarios()}"
        )
    return SCENARIOS[key]()
