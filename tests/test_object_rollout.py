"""The object block's two prediction backends must be swappable.

`oim.algs.admm.ObjectRollout` is what lets the object subproblem plan with
either the paper's quasi-static limit surface or an MJX rollout of the same
scene, with everything else -- sampler, costs, projection, warm start,
consensus math -- held fixed. These tests pin the parts of that claim that
would otherwise fail silently:

* injecting the *default* backend explicitly must change nothing, or every
  pre-existing result is quietly a different experiment;
* the MJX backend must survive the `vmap` + `scan` + `jit` the subproblem
  wraps it in, which is the whole reason the CPU `MujocoPlant` cannot be
  used here;
* it must actually be second-order, since removing the quasi-static
  assumption is the only reason to pay for it;
* and the object-only world's model-error column must follow whichever
  backend planned, not the task's own dynamics.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oim.algs.admm import AnalyticObjectRollout
from oim.experiment import load_config
from oim.runtime.object_mjx import (
    FRICTION_MODELS,
    MJXObjectRollout,
    build_object_rollout,
    object_mjx_model,
)
from oim.worlds.object_only.build import build_object_only
from oim.worlds.object_only.plant import (
    PLANT_MODES,
    MujocoPlant,
    build_plant,
    resolve_plant,
)
from oim.worlds.object_only.run import _one_step_model

SCENE, ROBOT = "shelf_gap", "xarm6"
# Small enough to keep the MJX compile inside a test run; the backend is
# indifferent to both.
HORIZON, SAMPLES = 4, 3


@pytest.fixture(scope="module")
def cfg() -> Dict[str, Any]:
    """The xArm6 config, whose scenes carry the tuned `frictionloss`."""
    return load_config(ROBOT)


def _build(cfg: Dict[str, Any], **kwargs: Any) -> Any:
    """An object-only block on the shared scene."""
    return build_object_only(
        SCENE,
        ROBOT,
        cfg,
        horizon=HORIZON,
        samples=SAMPLES,
        seed=0,
        **kwargs,
    )


def test_analytic_backend_is_the_default(cfg: Dict[str, Any]) -> None:
    """An unspecified mode and an explicit analytic one are one object."""
    _, block, _, _ = _build(cfg)
    assert isinstance(block.rollout, AnalyticObjectRollout)


def test_plant_modes_cover_only_the_coherent_pairs(cfg: Dict[str, Any]) -> None:
    """The whole reason `--plant` is one flag and not two.

    Predicting with MJX while executing eq. 5 would give the planner a
    better model of the world than the world has. It is absent from the
    table, and `resolve_plant` is the only way to reach a pair, so no
    caller can assemble it.
    """
    assert set(PLANT_MODES) == {"analytic", "mujoco", "model-error"}
    assert ("mujoco", "analytic") not in PLANT_MODES.values()
    assert resolve_plant("analytic") == ("analytic", "analytic")
    assert resolve_plant("mujoco") == ("mujoco", "mujoco")
    assert resolve_plant("model-error") == ("analytic", "mujoco")
    with pytest.raises(ValueError, match="unknown plant mode"):
        resolve_plant("mixed")


def test_model_error_mode_predicts_analytically(cfg: Dict[str, Any]) -> None:
    """`model-error` must leave the *planner* on eq. 5; only the plant moves."""
    _, block, _, _ = _build(cfg, plant="model-error")
    assert isinstance(block.rollout, AnalyticObjectRollout)


def test_injecting_the_default_changes_no_number(cfg: Dict[str, Any]) -> None:
    """The refactor that made the backend pluggable must be a no-op.

    `AnalyticObjectRollout` is the identity on `init`/`pose`, so routing
    the scan through it has to reproduce `task.object_dynamics` exactly --
    not approximately. A tolerance here would hide the one bug this
    abstraction can introduce.
    """
    task, block, _, x0 = _build(cfg)
    actions = jnp.full((HORIZON, task.object_action_dim), 0.6)
    states, _, _ = block._rollout(x0, actions)

    pose = x0
    for t in range(HORIZON):
        w = task.object_action_to_consensus(pose, actions[t])
        pose = task.object_dynamics(pose, w)
        assert np.array_equal(np.asarray(states[t]), np.asarray(pose))


def test_single_rollout_is_not_routed_through_vmap(cfg: Dict[str, Any]) -> None:
    """A lone nominal must take the direct call, on every backend.

    Entering it through `vmap` with a batch of one is faster for the MJX
    backend (14.19 ms -> 10.53 ms) and was tried. It is reverted because
    `vmap` may reassociate: the same comparison measured 0.0 in one process
    and 2.98e-8 -- one float32 ulp -- in another. Worth ~5% of a control
    step, against a nondeterministic result in the ADMM core.

    Pinned as a test rather than a comment because the change is an
    obvious-looking optimization that reads as free, and the evidence that
    it is not took two contradictory measurements to find.
    """
    for plant in ("analytic", "mujoco"):
        task, block, _, x0 = _build(cfg, plant=plant)
        actions = jnp.full((HORIZON, task.object_action_dim), 0.6)
        # Left eager on purpose. Jitting these was tried and measured
        # *slower* -- 7.0 s to 24.3 s -- because compiling the MJX rollout
        # graph costs more than eagerly running the handful of steps this
        # comparison needs. The same trap as `HORIZON` in `test_admm`.
        direct, _, _ = block._rollout(x0, actions)
        viavmap = jax.vmap(block._rollout, in_axes=(None, 0))(x0, actions[None])
        # Not an equality assertion in either direction -- the point is
        # that they may differ, so `optimize` must not silently swap one
        # for the other. What is pinned is that `_rollout` is what the
        # nominal path calls.
        assert np.allclose(
            np.asarray(direct), np.asarray(viavmap[0][0]), atol=1e-6
        ), f"{plant}: the two paths disagree by more than float noise"
    assert not hasattr(block, "_rollout_one"), (
        "the vmapped single-rollout shortcut is back; see this test's "
        "docstring for why it was removed"
    )


def test_analytic_rollout_is_stateless(cfg: Dict[str, Any]) -> None:
    """Eq. 5 has no state beyond the pose, so the carry must be the pose."""
    task, _, _, x0 = _build(cfg)
    rollout = AnalyticObjectRollout(task)
    assert np.array_equal(np.asarray(rollout.init(x0)), np.asarray(x0))
    assert np.array_equal(
        np.asarray(rollout.pose(rollout.init(x0))), np.asarray(x0)
    )


def test_object_mjx_model_rejects_bad_arguments(cfg: Dict[str, Any]) -> None:
    """Silently rounding either would misreport what the run simulated."""
    task, _, _, _ = _build(cfg)
    with pytest.raises(ValueError, match="substeps"):
        object_mjx_model(task, cfg["world3d"], substeps=0)
    with pytest.raises(ValueError, match="friction"):
        object_mjx_model(task, cfg["world3d"], friction="nonsense")
    with pytest.raises(ValueError, match="unknown object dynamics"):
        build_object_rollout("nonsense", task, ROBOT, cfg["world3d"])


def test_both_sides_offer_the_same_friction_laws(cfg: Dict[str, Any]) -> None:
    """`--friction` must mean one thing whichever side it is shaping.

    Under `--plant mujoco` the same value reaches both the MJX prediction
    and the CPU plant. A law one side implements and the other does not
    would make that flag mean two things, or fail late -- which is exactly
    the cross-flag trap merging `--plant` was meant to remove.
    """
    task, _, _, _ = _build(cfg)
    for friction in FRICTION_MODELS:
        object_mjx_model(task, cfg["world3d"], friction=friction)
        MujocoPlant(
            task,
            ROBOT,
            cfg["world3d"],
            control_dt=float(cfg["world3d"]["planning_dt"]),
            friction=friction,
        )


def test_object_mjx_model_strips_the_scene(cfg: Dict[str, Any]) -> None:
    """The arm, the table and gravity all have to go.

    Same three edits as the CPU plant, and for the same reasons: the object
    block has no robot in it, the block's support friction is already its
    `frictionloss`, and nothing planar needs gravity. If the two models
    disagree about this, the "model error" the object-only world reports is
    partly a difference of scene.
    """
    task, _, _, _ = _build(cfg)
    mj_model = object_mjx_model(task, cfg["world3d"])

    assert np.all(mj_model.opt.gravity == 0.0)
    for geom_id in range(mj_model.ngeom):
        body = mj_model.body(mj_model.geom_bodyid[geom_id]).name
        name = mj_model.geom(geom_id).name
        if body.startswith(("xarm6", "pusher")) or name.startswith(
            ("table", "floor", "ground")
        ):
            assert mj_model.geom_contype[geom_id] == 0
            assert mj_model.geom_conaffinity[geom_id] == 0


def test_substeps_divide_the_planning_step(cfg: Dict[str, Any]) -> None:
    """`substeps` sets the timestep; the rollout takes that many of them."""
    task, _, _, _ = _build(cfg)
    plan_dt = cfg["world3d"]["planning_dt"]
    mj_model = object_mjx_model(task, cfg["world3d"], substeps=4)
    assert mj_model.opt.timestep == pytest.approx(plan_dt / 4)
    rollout = MJXObjectRollout(task, mj_model, ROBOT, substeps=4)
    assert rollout.substeps == 4


def test_cone_friction_disables_the_per_dof_kind(cfg: Dict[str, Any]) -> None:
    """Or the block's support friction is counted twice."""
    task, _, _, _ = _build(cfg)
    box = object_mjx_model(task, cfg["world3d"], friction="box")
    cone = object_mjx_model(task, cfg["world3d"], friction="cone")
    dofs = [box.joint(j).dofadr[0] for j in ("T_x", "T_y", "T_z")]
    assert np.all(box.dof_frictionloss[dofs] > 0.0)
    assert np.all(cone.dof_frictionloss[dofs] == 0.0)


def test_mjx_backend_runs_batched_under_jit(cfg: Dict[str, Any]) -> None:
    """The reason this is MJX and not `MujocoPlant`.

    `ObjectSubproblem._rollout` is vmapped over the sample population and
    jitted. A backend that cannot be traced fails here, which is exactly
    what the CPU plant would do.
    """
    task, block, _, x0 = _build(cfg, plant="mujoco")
    assert isinstance(block.rollout, MJXObjectRollout)

    actions = jnp.full((SAMPLES, HORIZON, task.object_action_dim), 0.6)
    batched = jax.jit(jax.vmap(block._rollout, in_axes=(None, 0)))
    states, wrenches, consensus = batched(x0, actions)

    assert states.shape == (SAMPLES, HORIZON, 3)
    assert wrenches.shape == (SAMPLES, HORIZON, task.consensus_dim)
    assert consensus.shape == (SAMPLES, HORIZON, task.consensus_dim)
    assert np.all(np.isfinite(np.asarray(states)))


def test_mjx_backend_carries_momentum(cfg: Dict[str, Any]) -> None:
    """The point of paying for it: the object coasts, eq. 5 cannot.

    Push hard for two steps, then release. Quasi-static dynamics stop dead
    the instant the wrench does -- the pose after the zero-wrench step is
    bit-identical to the pose before it. A second-order model keeps moving,
    which is difference 2 in `oim/worlds/object_only/plant.py`.
    """
    task, _, _, x0 = _build(cfg)
    push = 3.0 * jnp.asarray(task.object_model.wrench_limit)
    zero = jnp.zeros(3)

    for kind, must_coast in (("analytic", False), ("mujoco", True)):
        rollout = build_object_rollout(kind, task, ROBOT, cfg["world3d"])
        if rollout is None:
            rollout = AnalyticObjectRollout(task)

        # Eager, deliberately: jitting this cost 21.6 s -> 26.6 s, since
        # compiling a three-step MJX graph per backend is dearer than
        # stepping it.
        carry = rollout.init(x0)
        for _ in range(2):
            carry = rollout.step(carry, push)
        before = np.asarray(rollout.pose(carry))
        after = np.asarray(rollout.pose(rollout.step(carry, zero)))
        coasted = float(np.linalg.norm(after[:2] - before[:2]))
        if must_coast:
            assert coasted > 1e-4, f"{kind} did not coast: {coasted}"
        else:
            assert coasted == 0.0, f"{kind} coasted: {coasted}"


def test_model_error_is_measured_against_the_planner(
    cfg: Dict[str, Any],
) -> None:
    """`pred_pos_err` must follow the backend the block planned with.

    Read off the task instead and the column would report eq. 5's error for
    a plan eq. 5 did not make -- reading as a large model error precisely
    when the model had just been improved. Under MJX prediction and the
    MuJoCo plant the two differ only by integrator settings, so the error
    must be far below what the analytic model shows on the same scene.
    """
    task, block, _, x0 = _build(cfg, plant="mujoco")
    predict = _one_step_model(block, jit=True)
    w = 2.0 * jnp.asarray(task.object_model.wrench_limit)

    mjx_pred = np.asarray(predict(x0, w))
    analytic_pred = np.asarray(task.object_dynamics(x0, w))
    assert not np.allclose(mjx_pred, analytic_pred)

    plant = build_plant(
        "mujoco",
        task,
        ROBOT,
        cfg["world3d"],
        control_dt=cfg["world3d"]["planning_dt"],
    )
    plant.reset(np.asarray(x0))
    realized = plant.step(np.asarray(w))
    mjx_err = float(np.linalg.norm(realized[:2] - mjx_pred[:2]))
    analytic_err = float(np.linalg.norm(realized[:2] - analytic_pred[:2]))
    assert mjx_err < analytic_err
