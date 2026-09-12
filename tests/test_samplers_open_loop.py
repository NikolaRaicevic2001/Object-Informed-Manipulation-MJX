"""Every sampler solves the same open-loop pendulum swingup.

Six files used to hold a copy of this test that differed only in the
optimizer's constructor, the number of optimization steps and the cost it
had to beat -- which is what `_Case` holds. Each sampler's own file keeps
what is specific to it (MPPI's non-finite guard, DIAL's annealing state,
CEM's covariance floor, evosax's parameter plumbing).

The check is deliberately end-to-end and loose: the nominal plan after N
steps, rolled out, must cost less than a budget a working sampler clears
comfortably and a broken one does not. It is a canary for "this optimizer
stopped optimizing", not a benchmark -- the budgets are the ones the
individual files already used, so the bar has not moved.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from oim.algs.cbo import CBO
from oim.algs.cem import CEM
from oim.algs.dial import DIAL
from oim.algs.mppi import MPPI
from oim.algs.mppi_cma import MppiCma
from oim.algs.predictive_sampling import PredictiveSampling
from oim.tasks.pendulum import Pendulum

# Shared plan parameterization: a 1 s horizon on 11 zero-order knots.
SPLINE = {"plan_horizon": 1.0, "spline_type": "zero", "num_knots": 11}


@dataclass(frozen=True)
class _Case:
    """One sampler's swingup.

    Attributes:
        make: Builds the optimizer on the pendulum task.
        steps: Optimization steps before the plan is scored. DIAL anneals
            over `iterations` per step, so it needs fewer.
        budget: Total cost the rolled-out nominal plan must beat.
        check: Optional sampler-specific assertion on the final params.
    """

    make: Callable[[Any], Any]
    steps: int
    budget: float
    check: Optional[Callable[[Any, Any], None]] = None


CASES = {
    "predictive_sampling": _Case(
        lambda task: PredictiveSampling(
            task, num_samples=32, noise_level=0.1, **SPLINE
        ),
        steps=100,
        budget=9.0,
    ),
    "mppi": _Case(
        lambda task: MPPI(
            task, num_samples=32, noise_level=0.1, temperature=0.01, **SPLINE
        ),
        steps=100,
        budget=9.0,
    ),
    "cem": _Case(
        lambda task: CEM(
            task, num_samples=32, num_elites=4, sigma_start=1.0,
            sigma_min=0.1, **SPLINE
        ),
        steps=100,
        budget=9.0,
        # The floor is what keeps CEM from collapsing onto one sample.
        check=lambda opt, params: (
            None if bool(jnp.all(params.cov >= opt.sigma_min))
            else pytest.fail("CEM collapsed below sigma_min")
        ),
    ),
    "cbo": _Case(
        lambda task: CBO(
            task, num_samples=32, initial_noise_level=1.0, temperature=0.1,
            consensus_weight=1.0, noise_weight=1.0, step_size=0.1, **SPLINE
        ),
        steps=100,
        budget=9.0,
    ),
    "dial": _Case(
        lambda task: DIAL(
            task, num_samples=32, noise_level=0.4, beta_opt_iter=1.0,
            beta_horizon=1.0, temperature=0.001, iterations=10, **SPLINE
        ),
        steps=20,
        budget=15.0,
    ),
    "mppi_cma": _Case(
        lambda task: MppiCma(
            task, num_samples=32, initial_noise_level=0.1,
            minimum_noise_level=0.05, covariance_adaptation_rate=0.1,
            temperature=0.01, **SPLINE
        ),
        steps=100,
        budget=9.0,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_open_loop_swingup(name: str) -> None:
    """Optimize from rest, then score the nominal plan."""
    case = CASES[name]
    task = Pendulum()
    opt = case.make(task)
    jit_opt = jax.jit(opt.optimize)

    state = mjx.make_data(task.model)
    params = opt.init_params()
    for _ in range(case.steps):
        params, _ = jit_opt(state, params)

    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    _, rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )

    total_cost = jnp.sum(rollout.costs[0])
    assert total_cost <= case.budget, (
        f"{name} left the pendulum at cost {total_cost:.2f}"
    )
    if case.check is not None:
        case.check(opt, params)
