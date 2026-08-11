"""Push the T through this repo's own clutter layout.

Three obstacles -- a disc, a box and a triangle -- between the
block and a goal a quarter-turn away. The only 3D scene with a
point-mass embodiment as well as the arm, so it is where the
cheap `--robot point` sanity check lives.

    uv run python examples/clutter.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="clutter")

if __name__ == "__main__":
    main(EXPERIMENT)
