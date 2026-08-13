"""The object-level subproblem, run on its own -- no robot, no simulator.

The third world beside `oim.sim2d` and `oim.sim3d`, and the smallest: it
contains only the analytic object and its planner. `oim.sim2d` exists to
tell an algorithm bug from an MJX bug by replacing the *robot* physics with
something readable; this goes one step further and removes the robot
entirely, to answer a question neither of the others can:

    Can the object block route this object to this goal at all?

Nothing here can be blamed on contact, on an embodiment, or on the two
blocks failing to agree -- there is one block, and the plant it drives is
the very model it plans with. A failure in this world is a failure of the
object-level formulation: its cost weights, its action bounds, its
breakaway deadzone, or its sampler budget. A success is the precondition
for the ADMM results meaning anything, since ADMM cannot do better on the
object side than the object block can do unopposed.

    from oim.simobj import build_object_only, run_object

    task, block, params, x0 = build_object_only("shelf_gap", "xarm6", cfg)
    log = run_object(task, block, params, x0, max_steps=200)

`examples/object_only.py` is the command-line front end.
"""

from .run import (
    build_object_only,
    check_action_budget,
    report_softmax_ess,
    run_object,
)

__all__ = [
    "build_object_only",
    "check_action_budget",
    "report_softmax_ess",
    "run_object",
]
