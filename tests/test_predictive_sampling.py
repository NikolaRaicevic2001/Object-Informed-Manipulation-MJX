import jax
import jax.numpy as jnp
from mujoco import mjx

from oim.algs.predictive_sampling import PredictiveSampling
from oim.tasks.pendulum import Pendulum


def test_predictive_sampling() -> None:
    """Test the PredictiveSampling algorithm."""
    task = Pendulum()
    opt = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the policy parameters
    params = opt.init_params()
    assert params.mean.shape == (opt.num_knots, 1)
    assert isinstance(params.rng, jax._src.prng.PRNGKeyArray)

    # Sample control sequences from the policy
    knots, new_params = opt.sample_knots(params)
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    assert controls.shape == (opt.num_samples, opt.ctrl_steps, 1)
    assert knots.shape == (opt.num_samples, opt.num_knots, 1)
    assert new_params.rng != params.rng

    # Roll out the control sequences
    state = mjx.make_data(task.model)
    _, rollouts = opt.eval_rollouts(task.model, state, controls, knots)

    assert rollouts.costs.shape == (
        opt.num_samples,
        opt.ctrl_steps + 1,
    )
    assert rollouts.controls.shape == (
        opt.num_samples,
        opt.ctrl_steps,
        1,
    )
    assert rollouts.knots.shape == (
        opt.num_samples,
        opt.num_knots,
        1,
    )
    assert rollouts.trace_sites.shape == (
        opt.num_samples,
        opt.ctrl_steps + 1,
        len(task.trace_site_ids),
        3,
    )

    # Pick the best rollout
    updated_params = opt.update_params(new_params, rollouts)
    assert updated_params.mean.shape == (opt.num_knots, 1)
    assert jnp.all(updated_params.mean != new_params.mean)




