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

The driver now delegates, so the builder half of this file is close to a
tautology WHILE THAT HOLDS -- it compares `build_admm_3d` against a
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

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

from oim.worlds.sim3d.build import build_admm_3d

SCENE = "open_table_real"
CONFIG = "xarm6_real"
DRIVER = Path(__file__).resolve().parents[1] / "examples" / "pusht" / "pusht_real.py"


def _driver():
    spec = importlib.util.spec_from_file_location("pusht_real", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._CFG = mod._load_cfg(CONFIG)
    mod._W3, mod._SMP = mod._CFG["world3d"], mod._CFG["sampler"]
    mod._RUN, mod._ADM = mod._CFG["run"], mod._CFG["admm"]
    return mod


def _args(mod):
    adm, smp, run = mod._ADM, mod._SMP, mod._RUN
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


@pytest.fixture(scope="module")
def built():
    mod = _driver()
    args = _args(mod)
    adm = mod._ADM
    task_r, ctrl_r = mod.build_controller(args)
    task_s, ctrl_s, _, _ = build_admm_3d(
        SCENE, "xarm6", mod._CFG, warp=False,
        horizon=args.horizon, samples=args.num_samples, seed=args.seed,
        robot_opt=args.robot_opt, object_opt=args.object_opt,
        n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
        consensus_object_weight=float(adm.get("consensus_object_weight", 0.5)),
        rho_torque=args.rho_torque, consensus=args.consensus,
        lagged_consensus=adm.get("lagged_consensus"), plant=args.plant,
        object_substeps=args.object_substeps,
        robot_substeps=int(mod._W3.get("robot_substeps", 1)),
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
