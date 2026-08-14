"""2D: push the T through a narrow horizontal channel.

15 mm clearance -- tighter than `pusht2d_clutter`.

    uv run python examples/pusht2d_corridor.py admm

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="2d", env="corridor")

if __name__ == "__main__":
    main(EXPERIMENT)
