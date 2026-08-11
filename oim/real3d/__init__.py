"""Real-robot (hardware) closed-loop drivers for the push-T task.

The hardware counterpart of `oim.sim3d`. The ADMM planner, the task cost
and the MJX rollouts are reused verbatim from the simulation path; only the
outer loop's I/O is swapped:

    sim3d:  mjx_data <- mj_data ;  mj_data.ctrl = u ; mujoco.mj_step(...)
    real3d: mjx_data <- sensors ;  publish u to the arm ; read sensors again

See `oim.real3d.interface.RobotWorldInterface` for that I/O boundary and
`oim.real3d.run_real.run_real` for the loop itself.
"""
