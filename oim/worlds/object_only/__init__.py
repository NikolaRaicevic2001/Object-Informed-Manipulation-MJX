"""The object-level subproblem, run on its own -- no robot, no ADMM.

The smallest of the four worlds: it contains only the object and its
planner. `oim.worlds.sim2d` exists to tell an algorithm bug from an MJX bug
by replacing the *robot* physics with something readable; this goes one
step further and removes the robot entirely, to answer a question neither
of the others can:

    Can the object block route this object to this goal at all?

Nothing here can be blamed on contact, on an embodiment, or on the two
blocks failing to agree -- there is one block. A failure in this world is a
failure of the object-level formulation: its cost weights, its action
bounds, its breakaway threshold, or its sampler budget. A success is the
precondition for the ADMM results meaning anything, since ADMM cannot do
better on the object side than the object block can do unopposed.

    from oim.worlds.object_only import (
        build_object_only,
        build_plant,
        run_object,
    )

    task, block, params, x0 = build_object_only("shelf_gap", "xarm6", cfg)
    log = run_object(task, block, params, x0, max_steps=200)

Which *plant* executes the block's wrench is the one thing that varies:
`AnalyticPlant` (the default) makes the model execute itself, so there is
no model error and the run upper-bounds what the formulation can do;
`MujocoPlant` runs the same wrench through the simulator, so the loop plans
with the limit surface and is graded by MuJoCo. Everything else -- sampler,
costs, projection, warm start -- is the same object either way, so a
difference between two runs is a difference in dynamics. See `plant`.

`examples/object_only.py --plant {analytic,mujoco}` is the front end.
"""

from .build import build_object_only, check_action_budget, report_softmax_ess
from .plant import AnalyticPlant, MujocoPlant, ObjectPlant, build_plant
from .run import run_object

__all__ = [
    "AnalyticPlant",
    "MujocoPlant",
    "ObjectPlant",
    "build_object_only",
    "build_plant",
    "check_action_budget",
    "report_softmax_ess",
    "run_object",
]
