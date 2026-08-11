"""Push the T between two shelves (IsaacGym sim_task03).

The gap between the shelves is 0.2 m and the T's crossbar is
0.2 m long, so straight through is a zero-clearance squeeze and
going around is the long way. Those are the source repo's own
numbers, not a retuning.

    uv run python examples/shelf_gap.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="shelf_gap")

if __name__ == "__main__":
    main(EXPERIMENT)
