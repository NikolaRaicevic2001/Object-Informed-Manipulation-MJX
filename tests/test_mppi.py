import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from oim.algs.mppi import MPPI
from oim.alg_base import Trajectory
from oim.tasks.pendulum import Pendulum


def test_open_loop() -> None:
    """Use MPPI for open-loop pendulum swingup."""
    # Task and optimizer setup
    task = Pendulum()
    opt = MPPI(
        task,
        num_samples=32,
        noise_level=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, _ = jit_opt(state, params)

    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)

    # Roll out the solution, check that it's good enough
    states, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )
    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(opt.ctrl_steps) * task.dt

        ax[0].plot(times, states.qpos[0, :, 0])
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, states.qvel[0, :, 0])
        ax[1].set_ylabel(r"$\dot{\theta}$")

        ax[2].step(times, final_rollout.controls[0], where="post")
        ax[2].axhline(-1.0, color="black", linestyle="--")
        ax[2].axhline(1.0, color="black", linestyle="--")
        ax[2].set_ylabel("u")
        ax[2].set_xlabel("Time (s)")

        time_samples = jnp.linspace(0, times[-1], 100)
        controls = jax.vmap(opt.get_action, in_axes=(None, 0))(
            params, time_samples
        )
        ax[2].plot(time_samples, controls, color="gray", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    test_open_loop()


def _mppi_and_rollouts(costs):
    """An MPPI and a `Trajectory` whose per-sample total costs are `costs`."""
    opt = MPPI(
        Pendulum(),
        num_samples=len(costs),
        noise_level=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )
    knots = jnp.arange(
        len(costs) * opt.num_knots * opt.task.model.nu, dtype=float
    ).reshape(len(costs), opt.num_knots, opt.task.model.nu)
    rollouts = Trajectory(
        controls=knots,
        knots=knots,
        costs=jnp.asarray(costs, dtype=float)[:, None],
        trace_sites=jnp.zeros((len(costs), 1, 3)),
    )
    return opt, opt.init_params(), rollouts


def test_one_nan_rollout_does_not_poison_the_update() -> None:
    """A single non-finite sample is weighted out, not spread everywhere.

    Without the guard `jnp.max` propagates the NaN to every weight, and
    since `mean` is carried across steps the whole run is dead from that
    point on -- which is exactly how a rare NaN in one object rollout
    ends an episode 200 steps in.

    The healthy batch is checked to be BIT-IDENTICAL, so the guard cannot
    be quietly changing the numbers on every ordinary step.
    """
    healthy = [1.0, 2.0, 3.0, 4.0]
    opt, params, rollouts = _mppi_and_rollouts(healthy)
    good = opt.update_params(params, rollouts)
    assert jnp.all(jnp.isfinite(good.mean))

    for bad in (jnp.nan, jnp.inf):
        opt, params, rollouts = _mppi_and_rollouts([1.0, 2.0, bad, 4.0])
        out = opt.update_params(params, rollouts)
        assert jnp.all(jnp.isfinite(out.mean)), f"{bad} leaked into the mean"
        # The bad sample carries ~no weight, so the mean is what the three
        # survivors alone would have produced.
        opt3, params3, rollouts3 = _mppi_and_rollouts([1.0, 2.0, 4.0])
        keep = jnp.delete(rollouts.knots, 2, axis=0)
        ref = opt3.update_params(
            params3, rollouts3.replace(knots=keep, controls=keep)
        )
        assert jnp.allclose(out.mean, ref.mean, atol=1e-5)


def test_an_all_nan_batch_holds_the_previous_nominal() -> None:
    """No usable sample means no information, so the mean must not move.

    The alternative -- what the softmax alone produces once every cost is
    equal -- is the uniform average of knots that are all garbage.
    """
    opt, params, rollouts = _mppi_and_rollouts([jnp.nan] * 4)
    out = opt.update_params(params, rollouts)
    assert jnp.array_equal(out.mean, params.mean)
    assert jnp.array_equal(out.temperature, params.temperature)
