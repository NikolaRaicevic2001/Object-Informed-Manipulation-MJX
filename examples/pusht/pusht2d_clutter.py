"""2D: route the T around three obstacles.

The analytic world -- single-point contact, a disc robot, no
MuJoCo anywhere -- running the same ADMM code as 3D. It exists
to separate algorithm bugs from simulator bugs, so a
disagreement between the two worlds localises the fault.
41 mm clearance.

    uv run python examples/pusht2d_clutter.py admm

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="2d", env="clutter")

if __name__ == "__main__":
    main(EXPERIMENT)
