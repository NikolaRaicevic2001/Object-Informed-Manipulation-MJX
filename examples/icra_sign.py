"""Push a letter into a sign (IsaacGym sim_task05, respelled).

Seven fixed glyphs spell "ICRA 2026" with the C's slot left
empty; the goal is that slot, so a solved run completes the
word. The only tabletop scene whose object is not the T.

    uv run python examples/icra_sign.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="icra_sign")

if __name__ == "__main__":
    main(EXPERIMENT)
