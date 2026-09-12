import jax
import jax.numpy as jnp
from mujoco import mjx

from oim.algs.mppi_cma import MppiCma
from oim.tasks.particle import Particle
from oim.tasks.pendulum import Pendulum


def test_params_update() -> None:
    """Test that the MPPICMA parameter update works."""
    # Task and optimizer setup
    task = Pendulum()
    opt = MppiCma(
        task,
        num_samples=32,
        initial_noise_level=0.1,
        covariance_adaptation_rate=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the system state and policy parameters
    params = opt.init_params()

    assert params.mean.shape == (opt.num_knots, task.model.nu)
    assert params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )

    # Sample from the action distribution, check shapes
    knots, params = opt.sample_knots(params)
    assert knots.shape == (opt.num_samples, opt.num_knots, task.model.nu)

    # Collect some rollouts
    state = mjx.make_data(task.model)
    new_params, rollouts = opt.optimize(state, params)

    assert new_params.mean.shape == (opt.num_knots, task.model.nu)
    assert new_params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )
    assert not jnp.allclose(new_params.covariance, params.covariance)




def test_multi_input() -> None:
    """Check that MPPI-CMA performs as expected on a problem with nu > 1."""
    task = Particle()
    opt = MppiCma(
        task,
        num_samples=32,
        initial_noise_level=0.2,
        minimum_noise_level=0.1,
        covariance_adaptation_rate=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the system state and policy parameters
    params = opt.init_params()

    assert params.mean.shape == (opt.num_knots, task.model.nu)
    assert params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )

    # Sample from the action distribution, check shapes
    knots, params = opt.sample_knots(params)
    assert knots.shape == (opt.num_samples, opt.num_knots, task.model.nu)

    mu_init = params.mean
    cov_init = params.covariance

    state = mjx.make_data(task.model)
    params, _ = jax.jit(opt.optimize)(state, params)

    mu_final = params.mean
    cov_final = params.covariance

    assert mu_init.shape == mu_final.shape
    assert cov_init.shape == cov_final.shape

    # Initial covariance is diagonal, but final is not
    assert jnp.allclose(
        cov_init, jax.vmap(jnp.diag)(jnp.diagonal(cov_init, axis1=-2, axis2=-1))
    )
    assert not jnp.allclose(
        cov_final,
        jax.vmap(jnp.diag)(jnp.diagonal(cov_final, axis1=-2, axis2=-1)),
    )

    # Final covariance has eigenvalues that are not too small
    tol = 1e-6
    eigvals_final = jnp.linalg.eigvalsh(cov_final)
    assert jnp.all(eigvals_final >= opt.minimum_noise_level**2 - tol)


