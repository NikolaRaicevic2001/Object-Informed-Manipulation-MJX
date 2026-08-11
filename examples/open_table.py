"""Push the T across an empty table (IsaacGym sim_task01).

No obstacles at all: the unobstructed baseline the other
tabletop scenes add clutter to. If a method cannot solve this
one, nothing below it is informative.

    uv run python examples/open_table.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="open_table")

if __name__ == "__main__":
    main(EXPERIMENT)
