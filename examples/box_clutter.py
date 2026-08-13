"""Push the real lab T through the measured pudding-box layout.

The `box_clutter` scene: the physical table the xArm6 driver runs on, so a
sim run here is the matched baseline for a real run of the same scene.

    uv run python examples/box_clutter.py admm --headless
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="box_clutter")

if __name__ == "__main__":
    main(EXPERIMENT)
