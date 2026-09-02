"""Push the T through a three-gate slalom.

Three pairs of fins at y = +0.21, 0.00, -0.21 whose 0.200 m openings
alternate near/far/near, so no straight line reaches the goal and the
object has to weave -- while still completing the family's 180-degree
flip. The hardest scene here, and the only one that is not an IsaacGym
task: every reversal forces the pusher to break contact and re-approach
on a different face.

    uv run python examples/pusht/slalom.py admm --headless

Everything besides the declaration below -- the command line, the closed
loop, recording, the run file and the plot -- is `oim/experiment.py`'s.
"""

from oim.experiment import Experiment, main

EXPERIMENT = Experiment(world="3d", scene="slalom")

if __name__ == "__main__":
    main(EXPERIMENT)
