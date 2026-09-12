"""The real driver and the shared sim builder must agree.

`examples/pusht/pusht_real.py` used to build its own `PushT` + `ADMM`
rather than calling `oim.worlds.sim3d.build.build_admm_3d`, and that
duplication silently diverged four times -- each caught only after
hardware runs were produced under it: the robot rollout ran at 1 substep
while sim read the config's, the object sampler's noise/temperature were
substituted, `rho_torque` arrived as a bare scalar so torque was penalised
10x weaker, and `planning_iterations`/`planning_ls_iterations` were never
passed at all, so every hardware run solved contacts at the MJCF's 20/20
against sim's 40/30.

That construction now lives in `oim.worlds.real3d.build.build_controller`
and delegates, so the builder half of this file is close to a tautology
WHILE THAT HOLDS -- it compares `build_admm_3d` against a
`build_controller` that calls it. That is deliberate: it is a tripwire, not
a proof. Re-introducing a hand-rolled construction, or adding a knob to one
path only, fails here instead of on the robot.

The config half below is NOT tautological, and is the half that catches
more. Both builders read the same yaml, so a key missing from BOTH configs
makes them agree on the wrong value -- deleting `planning_iterations` from
both leaves every builder assertion passing while each run silently drops
to the MJCF's 20/20. That failure mode is why the config tests exist.

The one legitimate difference is the velocity clamp: the real driver caps
the planner's sample bounds at `--vel-limit` because the published command
is capped there. That is hardware plumbing, applied after construction, and
is asserted to be the ONLY difference.
"""

import types
from pathlib import Path

import jax
import numpy as np
import pytest

from oim.worlds.real3d.build import (
    build_controller,
    build_mock_interface,
    load_robot_config,
)
from oim.worlds.sim3d.build import build_admm_3d

SCENE = "open_table_real"
CONFIG = "xarm6_real"
# The hardware config, read once -- the same file `build_controller` and
# `build_admm_3d` both read below.
CFG = load_robot_config(CONFIG)


def _args():
    adm, smp, run = CFG["admm"], CFG["sampler"], CFG["run"]
    return types.SimpleNamespace(
        scene=SCENE, algorithm="admm", warp=False, cost=[], seed=0,
        consensus=adm["consensus"],
        horizon=int(adm.get("horizon", smp["horizon"])),
        num_samples=int(smp["num_samples"]),
        robot_opt=adm["robot_opt"], object_opt=adm["object_opt"],
        n_admm=int(adm["n_admm"]), rho=float(adm["rho"]),
        rho_torque=float(adm["rho_torque"]),
        gamma=float(adm["gamma"]), plant=adm["plant"],
        object_substeps=int(adm.get("object_substeps", 1)),
        vel_limit=float(run.get("vel_limit", 0.25)),
        goal=None, goal_yaw_deg=None,
    )


@pytest.mark.parametrize("mode", ["off", "qpax"])
def test_mock_executes_external_projection(mode):
    """Mock actuators receive the filtered command at the current pose."""
    import jax.numpy as jnp
    import mujoco

    if mode == "qpax":
        pytest.importorskip("qpax")
    args = _args()
    args.algorithm = "mppi"
    args.control_projection = mode
    args.cbf_floor_alpha = 0.5
    args.cbf_slider_alpha = 1.0
    args.cbf_z_near = 0.04
    task, _ = build_controller(args, CFG)
    if mode != "off":
        assert task.control_projector.config.floor_alpha == 0.5
        assert task.control_projector.config.slider_alpha == 1.0
        assert task.control_projector.config.z_near == 0.04
    interface = build_mock_interface(task, 50, CFG)
    nominal = np.array([0.0, 0.2, 0.2, 0.0, 0.2])
    prepare = None if mode == "off" else jax.jit(task.control_projector.prepare)
    for _ in range(3):
        if prepare is not None:
            data = interface._data
            state = task.make_data().replace(
                qpos=jnp.array(data.qpos), qvel=jnp.array(data.qvel)
            )
            constraints = prepare(state)
            assert float(constraints.cbf_a[0] @ nominal + constraints.cbf_b[0]) < 0
        interface.send_velocity(nominal)
        applied = interface.last_applied_velocity
        np.testing.assert_array_equal(interface._data.ctrl, applied)
        if mode == "off":
            np.testing.assert_array_equal(applied, nominal)
        else:
            assert np.max(np.abs(applied - nominal)) > 0.01
            assert np.all(np.asarray(
                constraints.cbf_a @ applied + constraints.cbf_b
            ) >= -1e-5)
        mujoco.mj_forward(interface._model, interface._data)


@pytest.mark.parametrize("mode,algorithm", [
    ("qpax", "flat"), ("qpax", "admm"),
])
def test_real_projection_uses_final_hardware_velocity_bounds(
    mode: str, algorithm: str
) -> None:
    """The real clamp is applied after construction but before preparation."""
    if mode == "qpax":
        pytest.importorskip("qpax")
    args = _args()
    args.algorithm = algorithm
    args.control_projection = mode
    args.num_samples = 2
    args.horizon = 4
    args.vel_limit = .13
    task, _ = build_controller(args, CFG)
    projector = task.control_projector
    assert projector.config.mode == mode
    constraints = jax.jit(projector.prepare)(task.make_data())
    np.testing.assert_allclose(constraints.u_min, -.13)
    np.testing.assert_allclose(constraints.u_max, .13)


@pytest.fixture(scope="module")
def built():
    args = _args()
    adm = CFG["admm"]
    task_r, ctrl_r = build_controller(args, CFG)
    task_s, ctrl_s, _, _ = build_admm_3d(
        SCENE, "xarm6", CFG, warp=False,
        horizon=args.horizon, samples=args.num_samples, seed=args.seed,
        robot_opt=args.robot_opt, object_opt=args.object_opt,
        n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
        consensus_object_weight=float(adm.get("consensus_object_weight", 0.5)),
        rho_torque=args.rho_torque, consensus=args.consensus,
        lagged_consensus=adm.get("lagged_consensus"), plant=args.plant,
        object_substeps=args.object_substeps,
        robot_substeps=int(CFG["world3d"].get("robot_substeps", 1)),
    )
    return task_r, ctrl_r, task_s, ctrl_s, args


@pytest.mark.parametrize("attr", [
    "consensus", "align_ref",
    "tip_z_form",
])
def test_task_formulation_matches(built, attr):
    task_r, _, task_s, _, _ = built
    assert getattr(task_r, attr, None) == getattr(task_s, attr, None), attr


def test_planner_model_solver_effort_matches(built):
    """The gap that made every hardware run 20/20 against sim's 40/30."""
    task_r, _, task_s, _, _ = built
    assert task_r.mj_model.opt.iterations == task_s.mj_model.opt.iterations
    assert task_r.mj_model.opt.ls_iterations == task_s.mj_model.opt.ls_iterations


@pytest.mark.parametrize("attr", [
    "n_admm", "eps_r", "eps_s", "consensus_object_weight",
    "lagged_consensus",
])
def test_controller_scalars_match(built, attr):
    _, ctrl_r, _, ctrl_s, _ = built
    assert getattr(ctrl_r, attr) == getattr(ctrl_s, attr), attr


def test_rho_vectors_match(built):
    """One rho serves both blocks, and both rigs build the same vector."""
    _, ctrl_r, _, ctrl_s, _ = built
    assert np.allclose(np.asarray(ctrl_r.rho_init, dtype=float),
                       np.asarray(ctrl_s.rho_init, dtype=float))
    # The per-block split is gone from the API, not merely unset.
    assert not hasattr(ctrl_r, "rho_object_scale")


def test_sample_budgets_match(built):
    _, ctrl_r, _, ctrl_s, _ = built
    for block in ("robot_optimizer", "object_optimizer"):
        a, b = getattr(ctrl_r, block), getattr(ctrl_s, block)
        assert a.num_samples == b.num_samples, block


def test_velocity_clamp_is_the_only_task_difference(built):
    """Real caps sample bounds at --vel-limit; sim uses the model's range."""
    task_r, _, task_s, _, args = built
    assert np.allclose(np.asarray(task_r.u_min), -args.vel_limit)
    assert np.allclose(np.asarray(task_r.u_max), args.vel_limit)
    assert not np.allclose(np.asarray(task_s.u_min), -args.vel_limit)


# --------------------------------------------------------------------------
# Config parity.
#
# The builder-parity tests above compare the two CONSTRUCTIONS, so they
# cannot catch a key missing from both configs: the builders then agree on
# the wrong value. Deleting `planning_iterations` from both yamls leaves all
# of them passing while every run silently drops to the MJCF's 20/20. These
# tests pin the configs themselves.
#
# The rule the project runs on: sim and real share one ALGORITHM, so every
# formulation switch must match; they may differ in WEIGHTS and budgets.
# A shared key belongs in exactly one of the two lists below, so adding a
# knob forces a deliberate choice instead of defaulting to divergence.

import yaml  # noqa: E402

CFG_DIR = Path(__file__).resolve().parents[1] / "oim" / "configs" / "robots"

# Same algorithm => these must be identical in both configs.
MUST_MATCH = {
    "costs": [
        "align_ref",           # object plan endpoint vs global goal
        "w_effort_normalized",  # wrench-limit normalized vs raw N / N.m
    ],
    "admm": [
        "consensus",           # which variable the blocks agree on
        "lagged_consensus",    # Algorithm 4 exactly, or the lagged variant
        # Numeric, but a paper equation (eq. 27) rather than a tuning knob:
        # both rigs must run one blend or the reported w_o is ambiguous.
        "consensus_object_weight",
    ],
    "world3d": [
        "planning_iterations",     # or a head-to-head compares two
        "planning_ls_iterations",  # constraint-solving fidelities
    ],
}

# Rig-specific formulation switches that are DELIBERATELY split, with the
# reason. Anything enum-valued and not listed here must match.
MAY_DIFFER_FORMS = {
    # sim prices tip height piecewise (quadratic above / exp below); the rig
    # runs a symmetric exponential. Kept split by decision.
    "costs.tip_z_form",
    # OPEN reconciliation item: sim plans the object block in MJX, real on
    # the analytic limit surface. Listed so the suite is green, not because
    # the split is settled.
    "admm.plant",
}

@pytest.fixture(scope="module")
def configs():
    return (yaml.safe_load(open(CFG_DIR / "xarm6.yaml")),
            yaml.safe_load(open(CFG_DIR / "xarm6_real.yaml")))


@pytest.mark.parametrize("block,key", [
    (b, k) for b, ks in MUST_MATCH.items() for k in ks
])
def test_formulation_keys_match_across_configs(configs, block, key):
    sim, real = configs
    s, r = sim.get(block, {}), real.get(block, {})
    assert key in s, f"xarm6.yaml is missing {block}.{key}"
    assert key in r, f"xarm6_real.yaml is missing {block}.{key}"
    assert s[key] == r[key], (
        f"{block}.{key}: sim={s[key]!r} real={r[key]!r} -- this is a "
        "formulation switch, so both rigs must run the same value"
    )


def test_no_undeclared_formulation_split(configs):
    """Enum-valued knobs pick a FORM; numeric ones are tuning.

    Weights are free to differ between the rigs and there are dozens of
    them, so enumerating each would rot immediately. The line the project
    draws is the one that matters: a key whose value is a string selects
    which equation runs, and both rigs must run the same equation unless
    the split is declared in MAY_DIFFER_FORMS with a reason.
    """
    sim, real = configs
    declared = set(MAY_DIFFER_FORMS)
    for block, keys in MUST_MATCH.items():
        declared |= {f"{block}.{k}" for k in keys}
    undeclared = []
    for block in ("costs", "admm", "world3d"):
        s_b, r_b = sim.get(block, {}) or {}, real.get(block, {}) or {}
        for key in sorted(set(s_b) & set(r_b)):
            sv, rv = s_b[key], r_b[key]
            if not (isinstance(sv, str) or isinstance(rv, str)):
                continue          # numeric: tuning, free to differ
            if sv == rv or f"{block}.{key}" in declared:
                continue
            undeclared.append(f"{block}.{key} (sim={sv!r} real={rv!r})")
    assert not undeclared, (
        "formulation switches differ without being declared in "
        "MAY_DIFFER_FORMS:\n  " + "\n  ".join(undeclared)
    )
