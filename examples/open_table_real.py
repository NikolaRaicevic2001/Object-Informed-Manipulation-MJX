"""Real-table open push (no obstacles). sim twin of pusht_real --scene open_table_real."""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="open_table_real")

if __name__ == "__main__":
    main(EXPERIMENT)
