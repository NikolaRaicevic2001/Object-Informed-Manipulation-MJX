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

WHAT IS TAKEN OUT OF THE SCENE. The borrowed scene's arm comes out of
collision via `oim.runtime.mjcf.disable_collisions`, because the object
block has no robot in it: nothing here ever sets `ctrl`, so the arm would
sit frozen at its replan configuration for the whole horizon -- a phantom
obstacle exactly where the real arm is about to move in and push. The
support surface STAYS, and gravity with it, so the block rests on the table
in prediction the way it does in execution.

That is a change (2026-08-19). The support used to be excluded and gravity
zeroed, on the grounds that resting the block on the table double-counts
its support friction -- already modelled as the `frictionloss` on its own
three joints. Re-measured, that no longer happens anywhere. Force needed to
break the block loose, 2 s ramp, support kept (+gravity) vs excluded (g=0):

    scene                              kept      excluded   mu*m*g
    open_table  (T_zs, condim=1 pair)  7.90 N    7.90 N     7.848 N
    clutter     (T_zs locked)          7.90 N    7.90 N     7.848 N

The tabletop family fixes it at the source -- the block has a vertical DoF
and a frictionless `condim="1"` block<->table `<pair>`, so the support
contact carries no tangential force. The locked-vertical-DoF scenes
(`pusht_clutter`) never touch their support at
all: their block hovers 10 mm and 0.2 mm above the floor plane
respectively, with zero block/support contacts at the start pose and no DoF
that could ever create one. So the exclusion was a no-op for friction in
every scene, while costing the one thing it was hiding: with the table gone
and gravity off, the object block could not predict the block falling off
the edge -- which the execution model, which has always run support and
gravity, is grading it on.

The arm does not sag under the restored gravity: `xarm6.xml` carries
`gravcomp="1"` on link1..link6 precisely because its velocity servos at
`ctrl=0` only damp and have no proportional term. The `xarm6_stick` body
(0.05 kg) is uncompensated, but that is equally true of the execution
model, so it is a shared property rather than a new prediction/execution
gap.

THE SUPPORT HAS TO BE SOLVED, NOT JUST DECLARED (2026-08-19). Keeping the
table means the block now stands on a real `condim="3"` contact, and a
frictional contact is a constraint the solver has to converge. At the
budget this module used to set -- `PREDICT_SOLVER_ITERATIONS` = 4, against
the 20/20 every other model in the pipeline runs -- it does not converge,
and the unconverged solve injects a normal impulse that launches the block
off the table on the first step of any real push. Measured on
open_table/point, 24 steps, 64 random wrench sequences at full action
authority, worst |x| or |y| reached:

    Newton/ls     4/4      8/8    10/10    12/12    16/16    20/20
    worst |xy|  8.2e7 m   170 m    128 m   1.45 m   1.47 m   1.47 m

Both budgets have to rise together: 12/4 and 20/4 both stall at 904 m and
4/20 is worse still at 8318 m, so it is neither the Newton loop nor the
line search alone. 12 is the knee and 20 is the value taken.

That is the Newton budget only. The two knees are NOT the same, and the
line search's is lower -- see `PREDICT_LS_ITERATIONS`, where 16 beats the
scene's own 20 on stability *and* on time. So the prediction does not
simply inherit the scene's solver settings; it takes the pair that was
measured on the block's own contact.

The old default was not wrong when it was measured. It was measured against
a `condim="1"` block<->table pair, which contributes no friction rows at
all; the 2026-08-19 switch to real table friction added eight of them and
moved the knee from below 4 to 12.

WHAT IS LEFT, AND WHAT BUYS IT DOWN. Converged, the block stays on the
table, but the horizon is still integrated at 0.05 s while execution runs
at 0.002 s, and box-box contact points jump ~4 mm per step at that stride.
Each jump is a small transient the block rides up on. Error against the
`MujocoPlant` the run is actually graded by (open_table/point, 24 steps, 5
random wrench sequences, mean final |dxy|; `zmax` is how far the block ever
rises off the table):

    iters/substeps   4/1        20/1      20/2      20/3
    mean |dxy|       1.07e7 m   0.231 m   0.033 m   0.038 m
    zmax             1.19e7 mm   87.0 mm   18.5 mm   18.8 mm
    ms/horizon step     1.08       4.16      6.24      9.31

So `--object-substeps 2` is where the integration error stops dominating:
it takes the prediction gap down 7x and the hop down 4.7x for 1.5x the
time, and 3 buys nothing beyond it. That is now the default. (Those
timings predate `PREDICT_LS_ITERATIONS`, which takes another 13% off every
column without moving the errors.) Substeps and
solver iterations are not interchangeable, and substeps cannot stand in for
iterations -- worst |xy| reached over the same sequences at 4 iterations is
4.9e7 m at 1 substep, 5.1e6 at 2, 5.0 at 5 and 1.6e5 at 10, which is not a
convergent sequence but an unconverged contact behaving differently at each
stride.

WHAT WAS TRIED AND DOES NOT WORK. Recorded so it is not retried: contact
`solref` timeconst (0.002 through 0.5, no effect -- `solimp` d0 = 0.999
saturates the impedance), `impratio` (1 through 100, identical to 4 decimal
places), `noslip_iterations`, disabling warm start, and both integrators.
The elliptic friction cone is the one that looked most promising and is the
most firmly ruled out: on CPU it is exactly the isotropic limit surface
eq. 5 assumes (see `oim.worlds.object_only.plant`, which records what the
default pyramidal cone costs in its place), but under MJX it returns NaN on
open_table and icra_sign alike, at 3.7x and 14x the time. It is not an
option until MJX's elliptic solver is.

A `T_zs` joint limit at z = 0, and clamping the vertical velocity to be
non-positive after each step, both *look* like the right statement of "a
planar wrench cannot lift the block" and both corrupt the friction badly:
they delete the contact's restoring velocity, the block settles into
penetration, the normal force climbs and travel falls to 20% of the
executed value. The hop is an integration artifact, so only
integration resolution removes it.
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

# Solver effort for a *prediction* rollout. The same 20/20 the scene ships
# and the robot block already runs at: the object block is the only model in
# the pipeline that ever set its own, and setting it lower is what made the
# object plan fly off the table once the tabletop scenes got real contact
# friction. See the module docstring for the divergence table and for why
# the previous value of 4 was correct for the `condim="1"` pair it was
# measured against and wrong for the `condim="3"` one that replaced it.
#
# Kept as a named constant rather than read from `mj_model.opt` so the
# prediction budget stays visible and overridable at the call site.
#
# The execution plant is deliberately left alone: it is what grades the run,
# so it keeps the scene's own settings and any prediction shortfall shows up
# honestly in `pred_pos_err` rather than being hidden by matching error on
# both sides.
PREDICT_SOLVER_ITERATIONS = 20

# MJX physics steps per planning step, for the same reason: at 0.05 s the
# block's box-box contact points jump ~4 mm a step and the block rides up on
# each jump. 2 takes the gap against the execution plant from 0.263 m to
# 0.034 m and the hop from 87 mm to 19 mm for 1.7x the time; 5 buys nothing
# beyond it. See the module docstring. Raising this cannot substitute for
# `PREDICT_SOLVER_ITERATIONS` -- at 4 iterations the rollout diverges at
# every stride -- so the two are set together and not traded off.
PREDICT_SUBSTEPS = 2

# Line-search iterations, separately from the Newton loop above, because the
# two do not have the same knee. Measured on all ten tabletop models,
# `substeps` = 2, 320 random wrench sequences each, at `iterations` = 20 --
# worst |xy| reached in m, worst lift off the table in mm, median ms per
# horizon step:
#
#     ls           20        16        12        10
#     bad rows      1         0         2         3
#     worst |xy|   1.53      0.96      1.46      1.53
#     worst lift    421 mm    101 mm    406 mm    416 mm
#     ms/hstep    5.3-8.8   4.6-8.1   4.0-7.3   3.6-6.7
#
# 16 is the only value clean on every row, and it is 8-17% cheaper than 20
# (13% mean) -- so this is not a speed-for-accuracy trade: matching the
# scene's declared 20 here was strictly worse on both counts. The failures
# are not gradual. A "bad row" is the block leaving the table by 0.4 m,
# which is why the count matters more than any average.
#
# It also tightens the rollout's own conditioning, which is the third
# number this pair moves. `MJXObjectRollout` is entered directly for the
# nominal and through `vmap` for the samples; `vmap` may reassociate, so
# how far those two drift apart measures how much the rollout amplifies a
# one-ulp perturbation. At H = 24 on shelf_gap/xarm6, max |direct -
# vmapped| as a fraction of the trajectory:
#
#     iters/ls    20/20     20/16
#     substeps 1   26.8%     23.7%
#     substeps 2    9.3%      1.6%   <- shipped
#
# So the shipped pair is 16x better conditioned than what this module used
# before, not merely faster. `tests/test_object_rollout.py::
# test_single_rollout_is_not_routed_through_vmap` carries the full table
# and the reason its own H = 4 reading looks like the opposite.
PREDICT_LS_ITERATIONS = 16

# Contact-buffer capacity for the Warp backend, which sizes its narrowphase
# and constraint arrays up front instead of deriving them from the model.
# The defaults are far too small for this scene family and Warp does not
# raise -- it prints "narrowphase overflow - please increase nconmax to 5 or
# naconmax to 312" to stderr, silently DROPS the contacts that did not fit,
# and integrates on. The block then has nothing holding it on the table and
# leaves the scene: worst |xy| went to 26 m, which reads exactly like the
# unconverged-solver bug and is a completely different cause.
#
# TWO caps, and both are needed: `naconmax` sizes the contact arrays and
# `njmax` the constraint rows they expand into. Setting only the first still
# overflowed on icra_sign ("nefc overflow - please increase njmax to 96").
#
# THESE ARE BATCH ARENAS, NOT PER-ROLLOUT LIMITS. Warp shares one allocation
# across every parallel rollout, so the requirement scales with the object
# block's sample count. That is why they are computed per-sample and
# multiplied by the batch (`warp_arenas`) rather than fixed: a constant is
# only ever right for the batch it was measured at.
#
# Measured peak demand, worst tabletop model, divided by the batch:
# ~19 contacts and ~2 constraint rows per rollout. The per-sample figures
# below are ~3x that, and the floors cover a batch of 1.
#
# Getting it wrong is silent. Warp does not raise: it prints "narrowphase
# overflow - please increase nconmax to N or naconmax to M" from its C
# runtime, DROPS the contacts that did not fit, and integrates on -- the
# block then has nothing holding it on the table. A fixed 1024 here was
# measured against a 64-sample probe and overflowed at ~4800 in a real
# 256-sample run. `PushT.make_data` had already learned this for the robot
# block; see its comment.
#
# Do not trust an in-process check for these: the warning comes from C,
# underneath `contextlib.redirect_stderr`, so a Python-level capture reports
# success while contacts are being dropped. Read the process's own stderr.
#
# (`nconmax` also exists and also works; it is deliberately not set, being
# deprecated in mujoco-mjx >= 3.5 in favour of `naconmax`.)
WARP_NACON_PER_SAMPLE = 64
WARP_NJMAX_PER_SAMPLE = 8
WARP_NACON_FLOOR = 4096
WARP_NJMAX_FLOOR = 512


def warp_arenas(num_samples: int) -> Dict[str, int]:
    """Warp contact/constraint arena sizes for a batch of `num_samples`.

    Args:
        num_samples: Parallel rollouts the arenas must cover at once.

    Returns:
        `naconmax`/`njmax` keyword arguments for `mjx.put_data`.
    """
    n = max(int(num_samples), 1)
    return {
        "naconmax": max(WARP_NACON_FLOOR, WARP_NACON_PER_SAMPLE * n),
        "njmax": max(WARP_NJMAX_FLOOR, WARP_NJMAX_PER_SAMPLE * n),
    }


def object_mjx_model(
    task: PushT,
    cfg: Dict[str, Any],
    *,
    substeps: int = PREDICT_SUBSTEPS,
    friction: str = "box",
    solver_iterations: int = PREDICT_SOLVER_ITERATIONS,
    ls_iterations: int = PREDICT_LS_ITERATIONS,
    keep_robot: bool = False,
    keep_support: bool = True,
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
            becomes `planning_dt / substeps`. Defaults to
            `PREDICT_SUBSTEPS`, which is where the integration error against
            the execution plant stops dominating; 1 gives the planner the
            same coarse integration the analytic backend gets, which is the
            like-for-like setting when the question is how good eq. 5 is.
        friction: `"box"` leaves MuJoCo's own three independent per-DoF
            `frictionloss` elements alone. `"cone"` and `"wrench"` zero them
            and apply a coupled law through `qfrc_applied` instead -- the
            same three laws, implemented to match
            `oim.worlds.object_only.plant.MujocoPlant` branch for branch, so
            that `--friction` means one thing whether it is shaping the
            prediction or the execution.
        solver_iterations: Solver and line-search iterations for the
            prediction model, replacing the scene's own. The default
            reinstates them. Lowering it does not trade accuracy for speed
            below 12: the block's frictional support contact goes
            unconverged and the rollout diverges outright. See
            `PREDICT_SOLVER_ITERATIONS`.
        ls_iterations: Line-search iterations, set separately because the
            two knees differ -- and because the scene's own 20 is measurably
            worse here than the default. See `PREDICT_LS_ITERATIONS`.
        keep_robot: Leave the borrowed scene's arm in collision. Off by
            default -- nothing here sets `ctrl`, so the arm is frozen at
            its replan configuration and would be a phantom obstacle
            standing where the real arm is about to push.
        keep_support: Leave the block resting on the table. **On** by
            default, and gravity is kept with it, so the prediction model
            matches the execution model the run is graded by. See the
            module docstring for the friction measurement that made the
            old exclusion unnecessary. Turning this off zeroes gravity
            too: without a support the block's vertical DoF has nothing to
            rest on and the horizon would open by dropping it.

    Returns:
        The stripped model.

    Raises:
        ValueError: If `substeps`, `solver_iterations` or `ls_iterations`
            is below 1, or `friction` is unknown.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be at least 1, got {substeps}")
    if solver_iterations < 1:
        raise ValueError(
            f"solver_iterations must be at least 1, got {solver_iterations}"
        )
    if ls_iterations < 1:
        raise ValueError(
            f"ls_iterations must be at least 1, got {ls_iterations}"
        )
    if friction not in FRICTION_MODELS:
        raise ValueError(
            f"unknown friction {friction!r}; expected one of {FRICTION_MODELS}"
        )
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = float(cfg["planning_dt"]) / substeps
    mj_model.opt.iterations = solver_iterations
    mj_model.opt.ls_iterations = ls_iterations
    if not keep_robot:
        disable_collisions(mj_model, ROBOT_BODY_PREFIXES)
    if not keep_support:
        # Gravity goes with the support, always: they are one physical
        # choice, not two. With the table gone, a block with a vertical
        # DoF free-falls from step 0 and the horizon is meaningless; with
        # the table kept, gravity is what holds the block against it and
        # what makes it fall off the edge, which is the behaviour the
        # execution model grades. See the module docstring.
        disable_collisions(mj_model, SUPPORT_GEOM_NAMES, geom=True)
        mj_model.opt.gravity[:] = 0.0
    if friction != "box":
        # `"cone"`/`"wrench"` apply the whole support friction themselves,
        # coupled, through `MJXObjectRollout._friction_wrench`. Everything
        # else that could also apply it has to be switched off, or the
        # block is held twice:
        #   - `frictionloss`, one constant bound per DoF (the scenes that
        #     still use it), and
        #   - the block<->table contact's own Coulomb friction, which the
        #     tabletop scenes switched to in 2026-08-19 and which would
        #     otherwise add mu*N on top of the injected law.
        # Dropping the pairs to condim=1 leaves them carrying the normal
        # force only, which is what still holds the block on the table.
        dofs = _block_dof_adr(mj_model)
        mj_model.dof_frictionloss[dofs] = 0.0
        for pair in range(mj_model.npair):
            mj_model.pair_dim[pair] = 1
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
        substeps: int = PREDICT_SUBSTEPS,
        friction: str = "box",
        impl: str = "jax",
        num_samples: int = 1,
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
                this one sets how many are taken. The two default to the
                same `PREDICT_SUBSTEPS` so an unspecified pair cannot
                silently disagree.
            friction: Must match what `mj_model` was built with.
            impl: MJX backend, `"jax"` or `"warp"`. The robot block has
                taken this from `--warp` since it was added; the object
                block used to be hard-wired to `"jax"`, so one flag
                described two different pipelines. Measured 3.6-4.6x
                faster on every tabletop model at no cost in accuracy --
                see `build_object_rollout` for the table.
            num_samples: Parallel rollouts this will be `vmap`ped over.
                Warp only, and it must not be under-stated: its contact
                arenas are shared across the batch, so too small a value
                silently drops contacts. See `warp_arenas`.
        """
        self.task = task
        self.substeps = substeps
        self.friction = friction
        self.impl = impl
        self.mj_model = mj_model

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = (
            xarm6_start_qpos(mj_model)
            if robot == "xarm6"
            else point_start_qpos(mj_model)
        )
        mujoco.mj_forward(mj_model, mj_data)

        self.model = mjx.put_model(mj_model, impl=impl)
        # The scene at rest with everything but the block already placed:
        # `init` overwrites only the block's own qpos, so the obstacles this
        # rollout has to route around keep whatever pose the scene gives
        # them.
        # Warp needs its contact capacity stated; the JAX backend derives
        # its own and rejects the arguments. See `WARP_NACONMAX`.
        put_kwargs = warp_arenas(num_samples) if impl == "warp" else {}
        self._base = mjx.put_data(mj_model, mj_data, impl=impl, **put_kwargs)
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
    substeps: int = PREDICT_SUBSTEPS,
    friction: str = "box",
    solver_iterations: int = PREDICT_SOLVER_ITERATIONS,
    ls_iterations: int = PREDICT_LS_ITERATIONS,
    impl: str = "jax",
    num_samples: int = 1,
) -> Optional[ObjectRollout]:
    """An object-block dynamics backend by name.

    Args:
        kind: `"analytic"` for the paper's eq. 5, or `"mujoco"` for MJX.
        task: The task supplying the object model and the scene.
        robot: Embodiment, selecting the scene variant.
        cfg: The config's `world3d` block.
        substeps: Physics steps per planning step; MJX only. See
            `PREDICT_SUBSTEPS`.
        friction: `"box"`, `"cone"` or `"wrench"`. See `object_mjx_model`.
        solver_iterations: Newton effort for the prediction model. See
            `PREDICT_SOLVER_ITERATIONS`.
        ls_iterations: Line-search effort. See `PREDICT_LS_ITERATIONS`.
        impl: MJX backend, `"jax"` or `"warp"`; MJX only. The 3D world
            passes whatever `--warp` selected, so the object block and the
            robot block run the same pipeline -- they did not before.

            Warp is 3.6-4.6x faster and, measured against the `MujocoPlant`
            that grades the run (16 random wrench sequences per model, all
            ten tabletop models, mean final error):

                            position           heading
                jax     0.0503 +- 0.039 m      1.169 rad
                warp    0.0487 +- 0.041 m      0.672 rad

            -- a tie on position (the 3% gap is a tenth of the seed spread)
            and better on heading, at a quarter of the time. It is NOT
            bit-identical: a different backend is a different rollout, the
            same way MJX is not CPU MuJoCo, so switching moves results
            without making them worse.
        num_samples: The object block's sample count, which Warp's contact
            arenas have to cover all at once. Pass the real one; see
            `warp_arenas` for what under-stating it costs.

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
            ls_iterations=ls_iterations,
        )
        return MJXObjectRollout(
            task,
            mj_model,
            robot,
            substeps=substeps,
            friction=friction,
            impl=impl,
            num_samples=num_samples,
        )
    raise ValueError(
        f"unknown object dynamics '{kind}' (expected analytic or mujoco)"
    )
