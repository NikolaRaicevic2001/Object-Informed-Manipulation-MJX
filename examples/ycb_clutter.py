"""Push the T through household clutter (IsaacGym sim_task04).

The cube plus a spam can, a sugar box and a mustard bottle,
scattered across the workspace.

    uv run python examples/ycb_clutter.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="ycb_clutter")

if __name__ == "__main__":
    main(EXPERIMENT)
