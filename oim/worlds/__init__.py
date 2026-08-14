"""The four worlds a plan can be executed in, and nothing that differs else.

Each world supplies the same three things -- a task, a builder and a
runner -- and takes its sampler, its costs, its consensus space, its
logging and its recording from `oim.algs`, `oim.runtime` and `oim.utils`.
That is the whole point of the split: if two worlds disagree about a
result, the disagreement is in the dynamics, because there is nowhere else
for it to be.

    sim2d/        analytic single contact, no MJX. Tells an algorithm bug
                  from a physics bug by making the robot side readable.
    sim3d/        MJX contact, point mass or xArm6. The headline results.
    object_only/  no robot at all: can the object block route this object
                  to this goal unopposed? Its `plant` chooses between the
                  limit surface executing itself and MuJoCo executing the
                  same wrench, which is the narrowest dynamics-only
                  comparison in the repo.
    real3d/       the physical xArm6 over ROS 2.

`oim/experiment.py` is the front end for all of them; a script under
`examples/` names a world and a scene and gets the same CLI, run file and
plot as every other.
"""
