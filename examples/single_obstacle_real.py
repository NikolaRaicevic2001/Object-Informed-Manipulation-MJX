"""Real-world table pushing environment with a single obstacle.

This serves as the simulation twin of `pusht_real` for the `--scene single_obstacle_real` configuration.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="single_obstacle_real")

if __name__ == "__main__":
    main(EXPERIMENT)
