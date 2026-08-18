"""Predicting the object's motion with MJX instead of the limit surface.

`oim.algs.admm.ObjectSubproblem` plans the object block against an injected
`ObjectRollout`. The default is `AnalyticObjectRollout` -- the paper's
quasi-static eq. 5, a closed form with no state beyond the pose. This module
is the other implementation: the same block, the same sampler, the same
costs and the same consensus math, rolling the object out through
`mjx.step` on the scene it is actually going to be graded in.

WHY THIS IS MJX AND NOT `oim.worlds.object_only.plant.MujocoPlant`. That
plant is CPU MuJoCo: it mutates a `mujoco.MjData` in place, one wrench at a
time. The object block's rollout runs inside `jax.lax.scan`, under `vmap`
over `num_samples` and under `jit`, so it needs a *traceable, batchable*
step. MJX is that; `mujoco.mj_step` is not, and no wrapper makes it so.
The two are the same physics engine and not the same numbers -- different
solver defaults, and MJX's own contact model -- so a run that predicts with
MJX and executes with `MujocoPlant` still has a nonzero model error. It is
a much smaller one, and it no longer contains the limit surface's
quasi-static assumption.

WHAT THIS COSTS, AND WHICH KNOB CHANGES IT. Not the sample count. An
object-block solve is latency-bound, not throughput-bound: `mjx.step` is
cheap and wide but must be issued once per horizon step, sequentially, and
the GPU is nowhere near saturated at these batch widths. Measured on
shelf_gap/xarm6:

    samples    64      128     512
    ms/step   165.4   156.4   155.6      <- flat; 64 is no cheaper than 512

What it *is* linear in is sequential depth, `H * (iterations + 1)` per
`optimize` call -- `iterations` batched sample rollouts plus the one
nominal rollout at the end:

    H    iters   depth   ms/step   ms/depth
     8     4        40      36.3     0.908
    16     4        80      69.3     0.866
    32     4       160     142.0     0.888
    32     2        96      86.3     0.899
    32     1        64      59.5     0.930

So ~0.89 ms per sequential step, and the knobs that matter are `--horizon`
and `--iterations` (and, in the 3D world, `n_admm`, which multiplies the
whole thing again). Lowering `--object-samples` buys nothing and costs
search quality -- the batch is free, so spend it.

One consequence worth naming: the single nominal rollout is 1 of
`iterations + 1`, so it is 20% of the cost at `iterations=4` and 50% at
`iterations=1` -- which is what the 3D world runs, `n_admm` times per
control step. Batching it away is the largest remaining win here, and is
not simply a matter of routing it through `vmap`; see
`ObjectSubproblem._rollout`'s callers.

WHAT IS TAKEN OUT OF THE SCENE, and why it matches the CPU plant exactly:
the borrowed scene's arm and its support surface both come out of collision
via `oim.runtime.mjcf.disable_collisions`, and gravity is switched off. The
arm because the object block has no robot in it and a parked arm would be
an obstacle the analytic model cannot see; the support because the block's
support friction is already the `frictionloss` on its own three joints, and
resting it on the table counts that friction twice. Both constants are
shared with the plant rather than restated, so the prediction model and the
execution model cannot silently drift apart.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from oim.alg_base import quiet_mjx_cast_overflow
from oim.algs.admm import ObjectRollout
from oim.runtime.mjcf import (
    ROBOT_BODY_PREFIXES,
    SUPPORT_GEOM_NAMES,
    disable_collisions,
    point_start_qpos,
    xarm6_start_qpos,
)
from oim.tasks.pusht import PushT

# Below this the block counts as at rest under `friction="cone"`, so
# friction opposes the *applied wrench* rather than the twist. In newtons:
# `||c * v||` is the wrench that would produce the twist `v`
# quasi-statically, so this is 0.01 N against a cone of 7.848 N -- about
# 1.3 mm/s. The same constant, for the same reason, as
# `oim.worlds.object_only.plant._REST_WRENCH`.
REST_WRENCH = 1e-2

FRICTION_MODELS = ("box", "cone", "wrench")

# Solver effort for a *prediction* rollout, against the 20/20 the scene
# ships for execution. Measured on shelf_gap/point, H=15, 64 samples: the
# final pose is bit-identical from 20/20 down to 4/4 and only diverges at
# 2/2, while the batched rollout goes 15.80 ms -> 10.85 ms and a single one
# 28.98 ms -> 16.74 ms. So this is the largest free speedup available here,
# and 4 is the knee rather than a guess:
#
#     iters/ls   batch of 64   single    final pose
#     20/20         15.80 ms   28.98 ms  [0.20354232 0.21599205 1.1187236]
#     10/10         12.65 ms   17.10 ms  [0.20354232 0.21599205 1.1187236]
#      4/4          10.85 ms   16.74 ms  [0.20354232 0.21599205 1.1187236]
#      2/2           9.96 ms   12.93 ms  [0.21530728 0.21871057 1.0310434]
#
# 2/2 is where the answer moves, so 4 is the last honest setting.
#
# The execution plant is deliberately left alone: it is what grades the
# run, so it keeps the scene's own settings and a cheaper prediction shows
# up honestly in `pred_pos_err` rather than being hidden by matching error
# on both sides.
PREDICT_SOLVER_ITERATIONS = 4


def object_mjx_model(
    task: PushT,
    cfg: Dict[str, Any],
    *,
    substeps: int = 1,
    friction: str = "box",
    solver_iterations: int = PREDICT_SOLVER_ITERATIONS,
    keep_robot: bool = False,
    keep_support: bool = False,
) -> mujoco.MjModel:
    """A stripped CPU model of `task`'s scene, for the object block to plan in.

    Returned as a `mujoco.MjModel` rather than an `mjx.Model` so the caller
    can still build an `mjx.Data` from it and inspect it with the normal
    MuJoCo introspection; `MJXObjectRollout` does the `mjx.put_model`.

    Args:
        task: The task whose scene and block to simulate. Its own
            `mj_model` is deepcopied, never edited.
        cfg: The config's `world3d` block, for `planning_dt`.
        substeps: Physics steps per planning step. The model's timestep
            becomes `planning_dt / substeps`. 1 keeps the planner's own
            coarse integration, which is what the analytic backend uses and
            so the honest default for a comparison; raise it to separate
            "MuJoCo disagrees with eq. 5" from "0.05 s is too coarse a step
            for MuJoCo".
        friction: `"box"` leaves MuJoCo's own three independent per-DoF
            `frictionloss` elements alone. `"cone"` and `"wrench"` zero them
            and apply a coupled law through `qfrc_applied` instead -- the
            same three laws, implemented to match
            `oim.worlds.object_only.plant.MujocoPlant` branch for branch, so
            that `--friction` means one thing whether it is shaping the
            prediction or the execution.
        solver_iterations: Solver and line-search iterations for the
            prediction model, replacing the scene's own. See
            `PREDICT_SOLVER_ITERATIONS` for why the default is 4 and what
            it costs to raise or lower it.
        keep_robot: Leave the borrowed scene's arm in collision.
        keep_support: Leave the block resting on the table.

    Returns:
        The stripped model.

    Raises:
        ValueError: If `substeps` or `solver_iterations` is below 1, or
            `friction` is unknown.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be at least 1, got {substeps}")
    if solver_iterations < 1:
        raise ValueError(
            f"solver_iterations must be at least 1, got {solver_iterations}"
        )
    if friction not in FRICTION_MODELS:
        raise ValueError(
            f"unknown friction {friction!r}; expected one of {FRICTION_MODELS}"
        )
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = float(cfg["planning_dt"]) / substeps
    mj_model.opt.iterations = solver_iterations
    mj_model.opt.ls_iterations = solver_iterations
    if not keep_robot:
        disable_collisions(mj_model, ROBOT_BODY_PREFIXES)
    if not keep_support:
        disable_collisions(mj_model, SUPPORT_GEOM_NAMES, geom=True)
    # Nothing in the block's own dynamics needs it -- its planar friction is
    # `frictionloss`, a constant rather than a normal-force product, and with
    # the support excluded above nothing drives its vertical DoF where it
    # has one -- while leaving it on would drop the borrowed
    # scene's unactuated arm through the (now non-colliding) table across a
    # horizon that has no robot in it.
    mj_model.opt.gravity[:] = 0.0
    if friction != "box":
        # Or it would be counted twice: once per DoF by the solver, and
        # once, coupled, by `MJXObjectRollout._friction_wrench`.
        dofs = _block_dof_adr(mj_model)
        mj_model.dof_frictionloss[dofs] = 0.0
    return mj_model


def _block_dof_adr(mj_model: mujoco.MjModel) -> np.ndarray:
    """DoF addresses of the block's two slides and its hinge."""
    return np.array(
        [mj_model.joint(j).dofadr[0] for j in ("T_x", "T_y", "T_z")],
        dtype=int,
    )


class MJXObjectRollout(ObjectRollout):
    """The object block's dynamics as `mjx.step`, driven by its own wrench.

    The consensus wrench goes to `qfrc_applied` on the block's two slides
    and hinge and is held there for the whole planning step, which is the
    zero-order hold `object_spline: zero` already assumes. For two
    world-axis slides and a hinge about z that generalized force *is* the
    world-frame planar wrench of eq. 5, so nothing has to be converted.

    A lone rollout through this backend costs more than a whole batch
    (14.19 ms against 10.87 ms for 64 at H=15) -- a single `mjx.step`
    misses the tiled batched-solver path `vmap` lowers onto. Entering the
    nominal through `vmap` with a batch of one was tried and reverted: it
    is worth ~5% of a control step, and it does not reliably reproduce the
    direct call. Measured 0.0 difference in one process and 2.98e-8 (one
    float32 ulp) in another, because `vmap` is free to reassociate. A
    numerically unstable path in the ADMM core is not worth 5%.

    Unlike the analytic backend this carries velocity along the horizon:
    the block accelerates into a push and coasts out of one, which is
    precisely the term eq. 5 drops. `init` starts every rollout from rest
    -- see its docstring for why that is a real approximation and not a
    detail.
    """

    def __init__(
        self,
        task: PushT,
        mj_model: mujoco.MjModel,
        robot: str,
        *,
        substeps: int = 1,
        friction: str = "box",
    ) -> None:
        """Put the stripped model on device and cache the block's addresses.

        Args:
            task: The task, for the block's qpos/DoF addresses and the
                friction-cone limit.
            mj_model: An `object_mjx_model` result. Built by the caller
                rather than here so a run can inspect or further edit it,
                and so the `substeps`/`friction` it was built with are
                visible at the call site.
            robot: Embodiment, selecting the start pose for the bodies this
                rollout does not drive.
            substeps: Physics steps per planning step. Must match what
                `mj_model` was built with -- that call set the timestep,
                this one sets how many are taken.
            friction: Must match what `mj_model` was built with.
        """
        self.task = task
        self.substeps = substeps
        self.friction = friction
        self.mj_model = mj_model

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = (
            xarm6_start_qpos(mj_model)
            if robot == "xarm6"
            else point_start_qpos(mj_model)
        )
        mujoco.mj_forward(mj_model, mj_data)

        self.model = mjx.put_model(mj_model)
        # The scene at rest with everything but the block already placed:
        # `init` overwrites only the block's own qpos, so the obstacles this
        # rollout has to route around keep whatever pose the scene gives
        # them.
        self._base = mjx.put_data(mj_model, mj_data)
        self._qpos_adr = jnp.asarray(
            np.asarray(task.block_qpos_indices), dtype=int
        )
        self._dof_adr = jnp.asarray(_block_dof_adr(mj_model), dtype=int)
        self._limit = jnp.asarray(task.object_model.wrench_limit, dtype=float)

    def init(self, obj_state: jax.Array) -> mjx.Data:
        """Place the block at `obj_state`, at rest, in the stripped scene.

        AT REST is an approximation, and the one this backend is most
        exposed to. The object's true velocity at replan time is known --
        it is `state.qvel[task.block_dofs]` on the robot side -- but the
        object block is handed only a pose (`object_state_from_robot`), so
        a horizon that should begin mid-slide begins from a standstill and
        under-predicts the first step or two. At the shipped 20 Hz replan
        rate that is one planning step of momentum, which is small beside
        the term this backend exists to add; it is nonetheless the first
        thing to fix if the MJX object plan lags the execution.
        """
        qpos = self._base.qpos.at[self._qpos_adr].set(obj_state)
        return self._base.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(self._base.qvel),
            qfrc_applied=jnp.zeros_like(self._base.qfrc_applied),
        )

    def pose(self, carry: mjx.Data) -> jax.Array:
        """The block's SE(2) pose.

        `qpos` *is* the world pose here: the block's joints are two slides
        along world x/y and a hinge about world z, anchored at the body's
        declared position -- the same invariant `PushT._block_pose` reads
        back on the planning side.
        """
        return carry.qpos[self._qpos_adr]

    def _friction_wrench(self, carry: mjx.Data, w: jax.Array) -> jax.Array:
        """Coupled ellipsoidal Coulomb friction, opposing motion.

        The traced twin of
        `oim.worlds.object_only.plant.MujocoPlant._friction_wrench`, branch
        for branch, written with `jnp.where` because both sides run under
        `vmap`. Sliding: the associated flow rule puts the twist normal to
        the limit surface, so the friction wrench opposing `v` is
        `-c**2 v / ||c v||`, which satisfies `||w_f / c|| = 1` exactly. At
        rest: friction opposes the applied wrench and saturates at the cone,
        `-w / max(s, 1)`, whose net is eq. 5's own excess term `w (1 - 1/s)`.
        """
        s = jnp.linalg.norm(w / self._limit)
        at_rest = -w / jnp.maximum(s, 1.0)
        if self.friction == "wrench":
            # Eq. 5's own force balance, applied at every instant regardless
            # of motion. Kept because it marks the boundary: ported into a
            # second-order integrator it diverges, which is what says no
            # friction law fixes inertia from the simulator's side.
            return at_rest
        v = carry.qvel[self._dof_adr]
        speed = jnp.linalg.norm(self._limit * v)
        sliding = -(self._limit**2) * v / jnp.maximum(speed, 1e-12)
        return jnp.where(speed > REST_WRENCH, sliding, at_rest)

    def _substep(self, data: mjx.Data, w: jax.Array) -> mjx.Data:
        """One `mjx.step` with `w` held on the block's DoFs."""
        applied = w
        if self.friction != "box":
            # Recomputed per substep, not per planning step: the friction
            # direction follows the twist, which changes as the block
            # accelerates. Held constant it would keep pushing along the
            # velocity the step started with.
            applied = w + self._friction_wrench(data, w)
        qfrc = data.qfrc_applied.at[self._dof_adr].set(applied)
        # See `quiet_mjx_cast_overflow`: MJX's box-box narrowphase emits a
        # float64-into-float32 overflow warning while this traces. Shared
        # with the robot block's `MJXRollout.step` rather than restated, so
        # the two MJX call sites cannot disagree about it.
        with quiet_mjx_cast_overflow():
            return mjx.step(self.model, data.replace(qfrc_applied=qfrc))

    def step(self, carry: mjx.Data, w: jax.Array) -> mjx.Data:
        """Hold `w` on the block's DoFs for one planning step."""
        # The default is one substep, where a `scan` is a scan over a single
        # element: all of its machinery (a carry pytree round-trip through
        # the loop primitive, per element of `mjx.Data`) and none of its
        # benefit. Called H times per rollout inside another scan, so the
        # overhead is not once.
        if self.substeps == 1:
            return self._substep(carry, w)

        def body(data: mjx.Data, _: Any) -> tuple:
            return self._substep(data, w), None

        out, _ = jax.lax.scan(body, carry, None, length=self.substeps)
        return out


def build_object_rollout(
    kind: str,
    task: PushT,
    robot: str,
    cfg: Dict[str, Any],
    *,
    substeps: int = 1,
    friction: str = "box",
    solver_iterations: int = PREDICT_SOLVER_ITERATIONS,
) -> Optional[ObjectRollout]:
    """An object-block dynamics backend by name.

    Args:
        kind: `"analytic"` for the paper's eq. 5, or `"mujoco"` for MJX.
        task: The task supplying the object model and the scene.
        robot: Embodiment, selecting the scene variant.
        cfg: The config's `world3d` block.
        substeps: Physics steps per planning step; MJX only.
        friction: `"box"`, `"cone"` or `"wrench"`. See `object_mjx_model`.
        solver_iterations: Solver/line-search effort for the prediction
            model. See `PREDICT_SOLVER_ITERATIONS`.

    Returns:
        `None` for `"analytic"` -- the callers all pass this straight to a
        `rollout=` parameter that already defaults to the analytic backend,
        so returning it explicitly would be a second place for the default
        to be decided. An `MJXObjectRollout` for `"mujoco"`.

    Raises:
        ValueError: If `kind` is not a known backend.
    """
    if kind == "analytic":
        return None
    if kind == "mujoco":
        mj_model = object_mjx_model(
            task,
            cfg,
            substeps=substeps,
            friction=friction,
            solver_iterations=solver_iterations,
        )
        return MJXObjectRollout(
            task, mj_model, robot, substeps=substeps, friction=friction
        )
    raise ValueError(
        f"unknown object dynamics '{kind}' (expected analytic or mujoco)"
    )
