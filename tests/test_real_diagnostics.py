"""The real driver's diagnostics: device-side reduction and post-run
reconstruction must reproduce the per-step host-side versions exactly."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import mjx_forward

from oim.algs import MPPI, WrenchConsensus, make_object_shim
from oim.algs.admm import ADMM
from oim.tasks.pusht import PushT
from oim.worlds.real3d import run_real as rr

PLAN_DT = 0.05
HORIZON = 6


class _Rollouts:
    def __init__(self, costs: np.ndarray, cv: np.ndarray | None) -> None:
        self.costs = jnp.asarray(costs)
        self.consensus_values = None if cv is None else jnp.asarray(cv)


class _Inner:
    temperature = 3.0


class _Params:
    def __init__(self, oc: np.ndarray, osm: np.ndarray) -> None:
        self.robot_params = _Inner()
        self.object_params = _Inner()
        self.object_costs = jnp.asarray(oc)
        self.object_samples = jnp.asarray(osm)


def _empty_log(admm: bool) -> dict:
    log: dict = {}
    rr._init_sample_stats(log, admm)
    return log


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_reduction_matches_host_statistics(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, h = 64, 8
    costs = rng.normal(100.0, 30.0, size=(n, h + 1)).astype(np.float32)
    costs[3, 2] = np.inf                       # one non-finite sample
    cv = rng.normal(0.0, 0.05, size=(n, h, 3)).astype(np.float32)
    cv[: n // 2] *= 0.0                        # half never touch
    scale = np.array([0.3, 0.3, 0.009])
    oc = rng.normal(20.0, 5.0, size=(256,)).astype(np.float32)
    osm = rng.normal(0.0, 0.01, size=(256, h, 3)).astype(np.float32)
    pose = np.array([0.1, -0.2, 0.3])
    rollouts = _Rollouts(costs, cv)
    params = _Params(oc, osm)

    ref = _empty_log(True)
    rr._log_sample_stats(ref, rollouts, 3.0, scale)
    rr._log_object_stats(ref, params, pose)

    got = _empty_log(True)
    rr._StatsReducer(True, scale)(got, rollouts, params, pose)

    for key in (*rr._SAMPLE_STAT_KEYS, *rr._CONTACT_STAT_KEYS,
                *rr._OBJECT_STAT_KEYS):
        assert len(got[key]) == 1 == len(ref[key]), key
        a, b = got[key][0], ref[key][0]
        if np.isnan(b):
            assert np.isnan(a), key
        elif key == "sample_temp_star":
            assert a == pytest.approx(b, rel=1e-3), key
        else:
            assert a == pytest.approx(b, rel=1e-4, abs=1e-6), key


def test_device_reduction_flat_path() -> None:
    rng = np.random.default_rng(3)
    costs = rng.normal(10.0, 2.0, size=(32, 5)).astype(np.float32)
    rollouts = _Rollouts(costs, None)

    class _Flat:
        temperature = 0.5

    ref = _empty_log(False)
    rr._log_sample_stats(ref, rollouts, 0.5, None)
    got = _empty_log(False)
    rr._StatsReducer(False, None)(got, rollouts, _Flat(), np.zeros(3))
    for key in rr._SAMPLE_STAT_KEYS:
        assert got[key][0] == pytest.approx(ref[key][0], rel=1e-3), key


def _states(task: PushT, n: int) -> list:
    """A few forward-kinematics states with the pusher moved about."""
    base = mjx_forward(task.model, task.make_data())
    out = []
    for i in range(n):
        q = base.qpos.at[task.pusher_dofs[0]].add(0.03 * i)
        q = q.at[task.pusher_dofs[1]].add(-0.02 * i)
        d = base.replace(qpos=q, time=0.4 * i)
        out.append(mjx_forward(task.model, d))
    return out


def test_cost_terms_reconstruction_matches_live() -> None:
    task = PushT(clutter=True, planning_dt=PLAN_DT)
    states = _states(task, 4)
    live = [rr._cost_terms(task, d) for d in states]

    log = {
        "object_pose": [None] * (len(states) + 1),
        "qpos": [np.zeros(task.mj_model.nq)] + [np.asarray(d.qpos) for d in states],
        "qvel": [np.zeros(task.mj_model.nv)] + [np.asarray(d.qvel) for d in states],
        "time": [0.0] + [float(d.time) for d in states],
    }
    rr._init_cost_terms(log)
    rr._reconstruct_cost_terms(task, task.make_data(), log)
    for key in rr._COST_TERM_KEYS:
        assert len(log[key]) == len(states), key
        for got, ref in zip(log[key], live):
            if np.isnan(ref[key]):
                assert np.isnan(got), key
            else:
                assert got == pytest.approx(ref[key], rel=1e-4, abs=1e-5), key


def test_plan_reconstruction_matches_live() -> None:
    task = PushT(clutter=True, planning_dt=PLAN_DT)
    robot_opt = MPPI(
        task, num_samples=8, noise_level=0.4, temperature=1.0,
        plan_horizon=HORIZON * PLAN_DT, spline_type="linear", num_knots=4,
        seed=5,
    )
    shim = make_object_shim(task, dt=PLAN_DT)
    object_opt = MPPI(
        shim, num_samples=8, noise_level=1.0, temperature=1.0,
        plan_horizon=HORIZON * PLAN_DT, spline_type="zero",
        num_knots=HORIZON, seed=5,
    )
    ctrl = ADMM(
        task, robot_opt, object_opt, WrenchConsensus(max_dual=15.0),
        n_admm=1, eps_r=1.0, eps_s=1.0, rho_init=1.0,
    )
    jit_optimize = jax.jit(ctrl.optimize)
    jit_plans = jax.jit(ctrl.nominal_plans)
    states = _states(task, 3)
    params = ctrl.init_params()
    live, knots = [], []
    log = {
        "object_pose": [None] * (len(states) + 1),
        "qpos": [np.zeros(task.mj_model.nq)],
        "qvel": [np.zeros(task.mj_model.nv)],
        "time": [0.0],
    }
    for d in states:
        params, _ = jit_optimize(d, params)
        live.append(tuple(np.asarray(x) for x in jit_plans(d, params)[:2]))
        knots.append(rr._plan_knots(params))
        log["qpos"].append(np.asarray(d.qpos))
        log["qvel"].append(np.asarray(d.qvel))
        log["time"].append(float(d.time))
    assert rr._reconstruct_plans(task, task.make_data(), log, params,
                                 jit_plans, knots)
    for i, (obj, rob) in enumerate(live):
        assert np.allclose(log["object_plan"][i], obj, atol=1e-5)
        assert np.allclose(log["robot_plan"][i], rob, atol=1e-5)


def test_compiled_cost_terms_match_eager() -> None:
    """The console print's compiled evaluator is the eager decomposition."""
    task = PushT(clutter=True, planning_dt=PLAN_DT)
    fn = jax.jit(functools.partial(rr._cost_terms_jnp, task))
    for d in _states(task, 3):
        eager = rr._cost_terms(task, d)
        fast = rr._cost_terms(task, d, fn)
        for key in rr._COST_TERM_KEYS:
            if np.isnan(eager[key]):
                assert np.isnan(fast[key]), key
            else:
                assert fast[key] == pytest.approx(eager[key], rel=1e-5), key
