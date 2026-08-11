"""2D: through a vertical slot, then rotate onto the goal.

5 mm clearance, the hardest of the three: the object has to
pass the slot *and* turn afterwards.

    uv run python examples/pusht2d_gate.py admm

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="2d", env="gate")

if __name__ == "__main__":
    main(EXPERIMENT)
