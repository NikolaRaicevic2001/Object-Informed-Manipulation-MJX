"""Real-world table pushing environment with a single obstacle.

The simulation twin of `pusht_real --scene single_obstacle_real`.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="single_obstacle_real")

if __name__ == "__main__":
    main(EXPERIMENT)
