"""Push the T past one cube (IsaacGym sim_task02).

A single 0.1 m cube sits between the block's start and its
goal, tall enough that the pushing stick can hit it too.

    uv run python examples/single_obstacle.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="single_obstacle")

if __name__ == "__main__":
    main(EXPERIMENT)
