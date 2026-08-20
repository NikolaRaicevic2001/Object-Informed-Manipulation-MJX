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

    That gap is no longer one ulp, and it is worth reading the size of it
    correctly. Measured on shelf_gap/xarm6 as max |direct - vmapped| over
    the horizon, against the trajectory's own scale -- repeatable to every
    digit within a process, three runs each:

        H    iters/ls   substeps    gap        scale     relative
         4     20/20        1       2.5e-4 m   0.157 m     0.2%
         4     20/16        1       2.6e-3     0.159       1.6%
         4     20/20        2       4.4e-4     0.074       0.6%
         4     20/16        2       2.7e-2     0.074      36.1%   <- HERE
        24     20/20        1       3.3e-1     1.239      26.8%
        24     20/16        1       2.3e-1     0.958      23.7%
        24     20/20        2       8.7e-2     0.939       9.3%
        24     20/16        2       1.7e-2     1.086       1.6%   <- shipped

    This test runs the marked row, and it is the worst of the eight in
    relative terms. That is an artifact of `HORIZON` = 4: over four steps
    the block has barely moved, so a fixed absolute perturbation is a large
    fraction of nothing. At the horizon runs actually use, the shipped
    setting is the *best* of the four -- 1.6%, against 9.3% at the scene's
    own ls=20 and 26.8% before `PREDICT_SUBSTEPS`. The gap does not compound
    with horizon; it stays ~1e-2 m absolute while the trajectory grows.

    So the bound below is set from the measured worst case with margin,
    not from float noise. It is loose, deliberately: what it can still
    catch is the nominal being routed somewhere else entirely (a different
    initial state, a dropped step), which is orders larger again. What it
    can no longer do is certify the two paths agree -- they do not, and
    that is the finding.

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
            np.asarray(direct), np.asarray(viavmap[0][0]), atol=5e-2
        ), f"{plant}: the two paths disagree by more than reassociation"
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


def _is_support(mj_model: Any, geom_id: int) -> bool:
    """Whether this geom is the block's support surface."""
    return mj_model.geom(geom_id).name.startswith(("table", "floor", "ground"))


def _is_robot(mj_model: Any, geom_id: int) -> bool:
    """Whether this geom belongs to the borrowed scene's robot."""
    body = mj_model.body(mj_model.geom_bodyid[geom_id]).name
    return body.startswith(("xarm6", "pusher"))


def test_object_mjx_model_strips_the_robot_but_keeps_the_table(
    cfg: Dict[str, Any],
) -> None:
    """The arm goes; the support and gravity stay.

    The arm because nothing here sets `ctrl`, so it would stand frozen at
    its replan pose as a phantom obstacle where the real arm is about to
    push. The support stays -- with gravity, which is the same physical
    choice -- so the object block predicts the world the execution model
    grades it in, including the block falling off the table edge. See
    `oim.runtime.mjcf.SUPPORT_GEOM_NAMES` for the re-measurement that
    retired the friction double-count this used to guard against.
    """
    task, _, _, _ = _build(cfg)
    mj_model = object_mjx_model(task, cfg["world3d"])

    assert not np.all(mj_model.opt.gravity == 0.0)
    for geom_id in range(mj_model.ngeom):
        if _is_robot(mj_model, geom_id):
            assert mj_model.geom_contype[geom_id] == 0
            assert mj_model.geom_conaffinity[geom_id] == 0
        elif _is_support(mj_model, geom_id):
            assert mj_model.geom_contype[geom_id] != 0


def test_dropping_the_support_drops_gravity_with_it(
    cfg: Dict[str, Any],
) -> None:
    """`keep_support=False` must zero gravity, or the block free-falls.

    The two are one choice: a block with a vertical DoF and no table under
    it starts falling at step 0, so a model built to have no support has to
    have no gravity either.
    """
    task, _, _, _ = _build(cfg)
    mj_model = object_mjx_model(task, cfg["world3d"], keep_support=False)

    assert np.all(mj_model.opt.gravity == 0.0)
    for geom_id in range(mj_model.ngeom):
        if _is_support(mj_model, geom_id):
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


def test_cone_friction_owns_the_support_friction(cfg: Dict[str, Any]) -> None:
    """Or the block's support friction is applied twice.

    `"cone"`/`"wrench"` apply the whole support friction themselves,
    coupled, through `MJXObjectRollout._friction_wrench`, so every other
    source of it has to be off. There are two, and which one a scene uses
    varies: `frictionloss` per DoF, or the block<->table contact's own
    Coulomb friction (what the tabletop scenes, `shelf_gap` here among
    them, switched to). Checked together rather than naming one, so this
    keeps meaning the same thing whichever mechanism the scene carries.
    """
    task, _, _, _ = _build(cfg)
    box = object_mjx_model(task, cfg["world3d"], friction="box")
    cone = object_mjx_model(task, cfg["world3d"], friction="cone")
    dofs = [box.joint(j).dofadr[0] for j in ("T_x", "T_y", "T_z")]

    # `box` leaves the scene's own mechanism alone, whichever it is.
    per_dof = np.all(box.dof_frictionloss[dofs] > 0.0)
    contact = box.npair > 0 and np.all(np.asarray(box.pair_dim) > 1)
    assert per_dof or contact, "box mode left the block with no friction"

    # `cone` must switch both off.
    assert np.all(cone.dof_frictionloss[dofs] == 0.0)
    assert np.all(np.asarray(cone.pair_dim) == 1)


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


def test_warp_object_rollout_keeps_the_block_on_the_table() -> None:
    """Warp's contact buffers must be big enough, or it drops contacts.

    Warp sizes its narrowphase and constraint arrays up front and does not
    raise when they are too small. It prints to stderr *from its C runtime*
    -- beneath `contextlib.redirect_stderr`, so an in-process capture sees
    nothing -- discards the contacts that did not fit, and integrates on.
    With the table contacts gone the block has nothing holding it down and
    leaves the scene: worst |xy| measured 26 m against 0.96 m healthy.

    icra_sign is the model that asked for the most (`njmax` 96, against the
    T scenes' 72, from its seven glyph obstacles), so it is the one pinned.
    The assertion is on the resulting motion rather than on the warning,
    because the warning is exactly what a Python-level check cannot see.

    Bounds are generous on purpose: the healthy figure is ~1 m and the
    failure is tens of metres, so this separates them without pinning
    Warp's own numbers, which are not MJX-JAX's and are not meant to be.
    """
    scene, robot, horizon, samples = "icra_sign", "point", 12, 32
    cfg_point = load_config(robot)
    task, _, _, x0 = build_object_only(
        scene, robot, cfg_point, horizon=4, samples=3, seed=0
    )
    rollout = build_object_rollout(
        "mujoco", task, robot, cfg_point["world3d"], impl="warp"
    )
    assert rollout.impl == "warp"

    limit = np.asarray(task.object_model.wrench_limit)
    wrenches = jnp.asarray(
        np.random.default_rng(0).uniform(-1.0, 1.0, (samples, horizon, 3))
        * (2.425 * limit)
    )
    z_adr = rollout.mj_model.joint("T_zs").qposadr[0]

    def one(seq: jnp.ndarray) -> jnp.ndarray:
        data = rollout.init(jnp.asarray(x0))

        def body(carry, w):
            carry = rollout.step(carry, w)
            return carry, jnp.concatenate(
                [rollout.pose(carry), carry.qpos[z_adr][None]]
            )

        return jax.lax.scan(body, data, seq)[1]

    traj = np.asarray(jax.jit(jax.vmap(one))(wrenches))
    assert np.isfinite(traj).all()
    assert np.abs(traj[:, :, :2]).max() < 3.0, "block left the table"
    assert traj[:, :, 3].max() < 0.5, "block launched off the table"

