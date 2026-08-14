"""Analytic 2D world for narrow-scope debugging of the ADMM formulation.

Same `ADMM` controller, same `ConsensusSpace`, same `ObjectSubproblem` as
the MuJoCo tasks -- only the robot block's physics is swapped, via
`Analytic2DRollout`. Use it to isolate algorithm bugs from simulator bugs:
if a behaviour reproduces here, it is the formulation, not MJX.

    from oim.worlds.sim2d import PushT2D, build_admm_2d

    task = PushT2D(footprint=t_shape_footprint(), goal=[0.5, 0.48, 0.785])
    ctrl, params = build_admm_2d(task)
    state = task.make_data(robot_pos=[-0.1, -0.12])
    params, rollouts = ctrl.optimize(state, params)
"""

from .engine import Analytic2DRollout, Sim2DModel, Sim2DState, resolve_contact
from .run import build_admm_2d, run_2d
from .scenarios import (
    DEFAULT_SCENARIO,
    Scenario,
    build_scenario,
    list_scenarios,
)
from .task import PushT2D

__all__ = [
    "Analytic2DRollout",
    "DEFAULT_SCENARIO",
    "PushT2D",
    "Sim2DModel",
    "Sim2DState",
    "Scenario",
    "build_admm_2d",
    "build_scenario",
    "list_scenarios",
    "resolve_contact",
    "run_2d",
]
