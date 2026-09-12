import jax.numpy as jnp

from oim.alg_base import Trajectory
from oim.algs.cbo import CBO
from oim.tasks.pendulum import Pendulum


def test_update_params() -> None:
    """Unit test for the CBO particle update."""
    task = Pendulum()
    opt = CBO(
        task,
        num_samples=8,
        initial_noise_level=1.0,
        temperature=0.1,
        consensus_weight=1.0,
        noise_weight=0.0,  # No noise, so the update is deterministic
        step_size=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    params = opt.init_params(seed=42)
    assert params.samples.shape == (8, opt.num_knots, task.model.nu)

    knots, _ = opt.sample_knots(params)
    assert knots.shape == (8, opt.num_knots, task.model.nu)

    # With equal costs the consensus is the average of the particles
    rollouts = Trajectory(
        controls=None,
        knots=knots,
        costs=jnp.zeros((8, 2)),
        trace_sites=None,
    )
    new_params = opt.update_params(params, rollouts)
    assert jnp.allclose(new_params.mean, jnp.mean(knots, axis=0))

    # With σ = 0, each particle moves toward the consensus by a factor of λ Δt
    old_deviation = knots - new_params.mean
    new_deviation = new_params.samples - new_params.mean
    assert jnp.allclose(new_deviation, 0.9 * old_deviation)


