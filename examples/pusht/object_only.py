"""Run the object-level subproblem on its own -- no robot, no ADMM.

One script for every scene, rather than one per scene as the 3D tasks are:
this world has no MJCF of its own and no embodiment to choose between, so a
scene here is just a key of `oim.utils.scenes.SCENES` and `--scene` says
everything five near-identical files would.

    # can the object planner route the T through the shelf gap at all?
    uv run python examples/object_only.py --scene shelf_gap --robot xarm6

    # watch the plans evolve: one gif frame per control step
    uv run python examples/object_only.py --scene shelf_gap --record

    # same planner, MuJoCo executing the wrench: how good is the model?
    uv run python examples/object_only.py --scene shelf_gap --plant mujoco

    # eager, to breakpoint inside the limit-surface dynamics
    uv run python examples/object_only.py --scene clutter --no-jit --steps 5

Writes its plot (and, with `--record`, its gif) to `oim/recordings/`, and
its run file to `oim/results/object/` -- deliberately *not*
`oim/results/runs/`, which `oim/run_eval.py` globs: these runs have no
robot and no control frequency, so averaging them into a results table
beside real ones would be meaningless. See `Experiment.results_dir`.

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="object")

if __name__ == "__main__":
    main(EXPERIMENT)
