import jax
import jax.numpy as jnp
from evosax.algorithms.distribution_based.cma_es import CMA_ES
from mujoco import mjx

from oim.algs.evosax import Evosax
from oim.tasks.pendulum import Pendulum


def test_cmaes() -> None:
    """Test the CMAES algorithm."""
    task = Pendulum()
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=32,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the policy parameters
    params = ctrl.init_params()
    assert params.opt_state.C.shape == (ctrl.num_knots, ctrl.num_knots)
    # weights in evosax 0.2.0 stay in params
    assert ctrl.es_params.weights.shape == (32,)

    # Sample control sequences from the policy
    knots, params = ctrl.sample_knots(params)
    assert knots.shape == (32, ctrl.num_knots, 1)

    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
    tq = jnp.linspace(0.0, ctrl.plan_horizon - ctrl.dt, ctrl.ctrl_steps)
    controls = ctrl.interp_func(tq, tk, knots)

    # Roll out the control sequences
    state = mjx.make_data(task.model)
    _, rollouts = ctrl.eval_rollouts(task.model, state, controls, knots)
    assert rollouts.costs.shape == (32, ctrl.ctrl_steps + 1)

    # Update the policy parameters
    params = ctrl.update_params(params, rollouts)
    assert params.mean.shape == (ctrl.num_knots, 1)
    assert jnp.all(params.mean != jnp.zeros((ctrl.num_knots, 1)))
    assert params.opt_state.best_fitness > 0.0


def test_open_loop() -> None:
    """Use CMA-ES for open-loop optimization."""
    # Task and optimizer setup
    task = Pendulum()
    opt = Evosax(
        task,
        CMA_ES,
        num_samples=32,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # elite_ratio was not an argument of the constructor in exosax 0.2.0, it was
    # hard-coded
    opt.elite_ratio = 0.1

    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, final_rollout = jit_opt(state, params)

    # Test consistency of best rollout identification
    best_cost = params.opt_state.best_fitness
    final_costs = jnp.sum(final_rollout.costs, axis=-1)
    best_idx = jnp.argmin(final_costs)
    assert jnp.allclose(best_cost, final_costs[best_idx])

    # rollout the best control sequence and update it once more
    best_knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, best_knots)
    states, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, best_knots
    )



