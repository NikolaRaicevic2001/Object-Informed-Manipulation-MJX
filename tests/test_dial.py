import jax
import jax.numpy as jnp
from mujoco import mjx

from oim.alg_base import Trajectory
from oim.algs.dial import DIAL
from oim.tasks.pendulum import Pendulum


def test_sample_knots_shape() -> None:
    """Test that sample_knots returns the correct shape and updates params."""
    task = Pendulum()
    opt = DIAL(
        task,
        num_samples=20,
        noise_level=0.4,
        beta_opt_iter=1.5,
        beta_horizon=1.5,
        temperature=0.8,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=8,
    )

    params = opt.init_params(seed=123)
    original_rng = params.rng

    knots, updated_params = opt.sample_knots(params)

    # Check shape
    expected_shape = (opt.num_samples, opt.num_knots, task.model.nu)
    assert knots.shape == expected_shape, (
        f"Expected knots shape {expected_shape}, got {knots.shape}"
    )

    # Check that RNG was updated
    assert not jnp.array_equal(original_rng, updated_params.rng), (
        "RNG should be updated after sampling"
    )

    # Check that other parameters remain unchanged
    assert jnp.array_equal(params.mean, updated_params.mean)
    assert jnp.array_equal(params.tk, updated_params.tk)
    assert params.opt_iteration == updated_params.opt_iteration


def test_opt_iteration() -> None:
    """Test that opt_iteration is properly initialized and updated."""
    task = Pendulum()
    controller = DIAL(
        task,
        num_samples=10,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=1.0,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=3,
        iterations=3,
    )

    # Test initial opt_iteration value
    params = controller.init_params()
    assert params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {params.opt_iteration}"
    )

    # Test that opt_iteration is reset after n iterations
    for _ in range(controller.iterations):
        _, params = controller.sample_knots(params)
    assert params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {params.opt_iteration}"
    )

    # Test that opt_iteration is reset after optimization
    state = mjx.make_data(task.model)
    jit_opt = jax.jit(controller.optimize)
    final_params, _ = jit_opt(state, params)
    assert final_params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {final_params.opt_iteration}"
    )


def test_update_params() -> None:
    """Test that update_params correctly updates the mean."""
    task = Pendulum()
    opt = DIAL(
        task,
        num_samples=4,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=1.0,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=3,
    )

    params = opt.init_params(seed=456)
    original_mean = params.mean.copy()

    # Create mock rollouts with different costs
    num_samples = opt.num_samples
    num_knots = opt.num_knots
    nu = task.model.nu
    ctrl_steps = opt.ctrl_steps

    # Create some dummy knots and costs
    knots = jax.random.normal(jax.random.key(789), (num_samples, num_knots, nu))
    costs = jnp.array([10.0, 5.0, 15.0, 8.0])  # Different costs for each sample
    controls = jnp.zeros((num_samples, ctrl_steps, nu))
    trace_sites = jnp.zeros((num_samples, ctrl_steps + 1, 3))

    rollouts = Trajectory(
        controls=controls,
        knots=knots,
        costs=jnp.tile(costs[:, None], (1, ctrl_steps + 1)),
        trace_sites=trace_sites,
    )

    updated_params = opt.update_params(params, rollouts)

    # Check that mean was updated
    assert not jnp.array_equal(original_mean, updated_params.mean), (
        "Mean should be updated after calling update_params"
    )

    # Manually compute expected weighted average
    total_costs = jnp.sum(rollouts.costs, axis=1)
    weights = jax.nn.softmax(-total_costs / opt.temperature, axis=0)
    expected_mean = jnp.sum(weights[:, None, None] * knots, axis=0)

    assert jnp.allclose(updated_params.mean, expected_mean, atol=1e-6), (
        "Updated mean should match manually computed weighted average"
    )


