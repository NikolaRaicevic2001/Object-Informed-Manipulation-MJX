"""ADMM-coordinated object/robot consensus planning.

Object-informed MPPI: a hierarchical object-level and robot-level
sampling-based sub-optimizer reach consensus on a shared variable z_t
(e.g. a contact wrench) via ADMM. See the paper "Object-Informed Model
Predictive Path Integral Control for Non-Prehensile Robot Manipulation"
(Algorithm 4).

Both sub-optimizers are pluggable: any `oim.alg_base.SamplingBasedController`
(MPPI, CBO, CEM, ...) can be injected as either, since this module only
calls `sample_knots`/`update_params`.
"""

from abc import ABC, abstractmethod
from functools import partial
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from oim.alg_base import (
    SamplingBasedController,
    Trajectory,
    quiet_mjx_cast_overflow,
)
from oim.objects.planar_pushing import wrap_angle
from oim.task_base import ConsensusTask

#: Accepted values of `ADMM.lagged_consensus`; see its docstring.
_LAGGED_CONSENSUS = ("off", "robot", "both")


class ConsensusSpace(ABC):
    """A consensus variable space for ADMM, shared between two subproblems.

    Owns all math that touches z, so the object block's proposed value A^o
    and the robot block's realized value A^r provably use the same units,
    frame and definitions.
    """

    dim: int

    @abstractmethod
    def normalize(self, v: jax.Array) -> jax.Array:
        """Map a consensus-space tangent to dimensionless units.

        This is what makes `rho`/`eps_r`/`eps_s` scale-free.
        """

    def difference(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Return a (-) b: the tangent vector from b to a.

        Plain subtraction, and both shipped spaces are vector spaces, so
        nothing overrides it today. Kept as the single seam a manifold
        consensus variable would need: every ADMM subtraction routes
        through here, so one override reaches the penalty, both residuals,
        the dual update and the z-update at once.
        """
        return a - b

    def increment(self, base: jax.Array, tangent: jax.Array) -> jax.Array:
        """Return base (+) tangent, the inverse of `difference`.

        Projection Pi_Z of eq. 14 -- identity for a vector space.
        """
        return base + tangent

    def shift(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift.

        Zero-fills the vacated tail, right when zero is a meaningful
        consensus value; overridden where it is not.
        """
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)

    def penalty_cost(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array, rho: jax.Array
    ) -> jax.Array:
        """(rho/2) * ||(actual (-) z) + dual||^2, normalized (eq. 25-26).

        `rho` may be a scalar or a per-dimension vector (the paper's
        anisotropic P = diag(rho_f, rho_f, rho_tau)).
        """
        diff = self.normalize(self.difference(actual, z) + dual)
        return 0.5 * jnp.sum(rho * diff**2, axis=-1)

    def residual_norm(self, v: jax.Array) -> jax.Array:
        """RMS of an already-tangent residual, in normalized units.

        Root-mean-square, not the plain 2-norm. `normalize` divides by the
        consensus scale, which frees the value of the object's
        friction-cone limit -- but the primal caller hands this both
        blocks' residuals concatenated over the whole horizon, a
        `(2H, dim)` array, so a plain norm still grows like
        `sqrt(2*H*dim)`. `eps_r`/`eps_s` then tighten silently with the
        horizon: H 24 -> 32 makes the same per-channel disagreement read
        1.15x larger, and since the two gate `_admm_iteration`'s early
        exit, a horizon change quietly changes how many rounds run.

        Dividing by `sqrt(size)` makes the value read as "typical
        disagreement per channel, as a fraction of the consensus scale",
        which is horizon-free. `eps_r`/`eps_s` are on this scale: divide
        an old value by `sqrt(2*H*dim)` to carry it over.
        """
        return jnp.linalg.norm(self.normalize(v)) / jnp.sqrt(
            jnp.asarray(v.size, dtype=float)
        )

    def z_update(
        self,
        a_o: jax.Array,
        a_r: jax.Array,
        dual_o: jax.Array,
        dual_r: jax.Array,
        base: jax.Array,
        object_weight: jax.Array = 0.5,
    ) -> jax.Array:
        """Apply paper eq. 27 with N = 2, taken about `base`.

            z <- base (+) w_o*[(a_o (-) base) + y_o]
                        (+) w_r*[(a_r (-) base) + y_r],   w_r = 1 - w_o

        At w_o = 0.5 this is eq. 27 exactly -- the plain average
        0.5*(a_o+y_o+a_r+y_r) on a vector space. Taking it about `base`
        is needed only when Z is a manifold, where the four terms must be
        lifted to a common tangent space first.

        w_o != 0.5 tilts the agreed value toward one block's proposal.
        It is then no longer the exact z-minimizer of the augmented
        Lagrangian (that is the average, or the rho-weighted average when
        the two blocks carry different penalties), so the convergence
        proof does not carry over; see `ADMM.__init__`.
        """
        w_o = jnp.asarray(object_weight)
        tangent = w_o * (self.difference(a_o, base) + dual_o) + (
            1.0 - w_o
        ) * (self.difference(a_r, base) + dual_r)
        return self.increment(base, tangent)

    @abstractmethod
    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual step y <- y + (A x (-) z), paper eq. 28."""


class WrenchConsensus(ConsensusSpace):
    """Planar contact wrench consensus z_t = [f_x, f_y, tau]; arrays (H, 3).

    z is the wrench the robot applies to the object, world frame about the
    object's pose origin, in N and N.m. A^o is the object planner's own
    decision (eq. 23); A^r is what the robot's rollout actually imparts,
    read from the simulator's contact forces.

    `scale` normalizes by the friction-cone limit (mu*m*g for forces,
    r*mu*m*g for torque) -- the inverse of the limit-surface compliance D
    -- so the residual reads as a fraction of the maximum transmissible
    wrench. Without it the penalty (forces ~10N, squared) dwarfs the task
    costs (~1) and the robot optimizes wrench matching over reaching the
    object.
    """

    dim = 3

    def __init__(self, max_dual: float, scale: jax.Array = None) -> None:
        """Set the anti-windup clip and the per-dimension normalization.

        Args:
        max_dual: Dual anti-windup clip, same units as z.
        scale: Per-dimension characteristic magnitude of z. Defaults
            to ones (no normalization).
        """
        self.max_dual = max_dual
        self.scale = jnp.ones(self.dim) if scale is None else jnp.asarray(scale)

    def normalize(self, v: jax.Array) -> jax.Array:
        """Divide by the per-dimension characteristic magnitude."""
        return v / self.scale

    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual + (actual - z), clipped to [-max_dual, max_dual]."""
        return jnp.clip(
            dual + self.difference(actual, z), -self.max_dual, self.max_dual
        )


class ContactPointConsensus(ConsensusSpace):
    """Point-contact consensus z_t = [p_x, p_y, lambda]; arrays (H, 3).

    The alternative to `WrenchConsensus`: the blocks agree on *where* to
    touch the object and *how hard*, not on the net wrench. p is a point in
    the object's **body** frame, lambda >= 0 the normal-force magnitude
    along the inward boundary normal there. The wrench follows through the
    contact Jacobian, w = J_c^T f, recomputed at the object's current pose
    every rollout step -- so one fixed z describes a contact that stays on
    the same material point as the object rotates, which no fixed wrench
    can express.

    Stronger than a wrench agreement, deliberately: many contact points
    produce the same net wrench, so under `WrenchConsensus` the blocks can
    agree while still disagreeing about where the push comes from. Here
    they cannot, which is the point -- but it also makes the primal
    residual larger and not comparable to a wrench run's.

    Still a plain vector space (no angular coordinate), so `difference`,
    `increment` and the penalty are the base class's.
    """

    dim = 3

    def __init__(self, max_dual: jax.Array, scale: jax.Array = None) -> None:
        """Set the anti-windup clip and the per-dimension normalization.

        Args:
        max_dual: Dual anti-windup clip, per dimension or scalar.
        scale: Characteristic magnitude, e.g. `(r_body, r_body, f_max)`
            so a normalized residual of 1 means "one body radius of
            contact-point disagreement, or the full normal force".
            Defaults to ones.
        """
        self.max_dual = jnp.asarray(max_dual)
        self.scale = jnp.ones(self.dim) if scale is None else jnp.asarray(scale)

    def normalize(self, v: jax.Array) -> jax.Array:
        """Divide by the per-dimension characteristic magnitude."""
        return v / self.scale

    def shift(self, seq: jax.Array) -> jax.Array:
        """Shift by one and repeat the last value.

        Zero-fill would put the vacated tail at the object's own origin --
        a specific point that is
        *inside* the footprint, where the boundary normal is undefined.
        """
        return jnp.concatenate([seq[1:], seq[-1:]], axis=0)

    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual + (actual - z), clipped to [-max_dual, max_dual]."""
        return jnp.clip(
            dual + self.difference(actual, z), -self.max_dual, self.max_dual
        )


class ObjectPoseConsensus(ConsensusSpace):
    """Object pose consensus z_t = [x, y, yaw]; arrays (H, 3).

    The blocks agree on *where the object ends up*, not on what pushes it
    there. The object block still decides and rolls out a wrench -- A^o is
    the pose eq. 5 produces from it -- and the robot block reads the pose
    straight out of its own rollout, so A^r needs no force estimator at
    all. That is the practical draw: the wrench the robot imparts is
    inferred (from a twist inversion or from contact forces, both noisy),
    while the object's pose is simply observed.

    Weaker than either force-level agreement, deliberately: many wrenches
    reach the same pose, so agreeing here leaves the blocks free to
    disagree about the push entirely. Its residual is therefore not
    comparable to a wrench run's.

    **The one space on this manifold.** yaw lives on a circle, so
    `difference` and `increment` are overridden to wrap it -- the seam the
    base class documents. Every ADMM subtraction routes through those two,
    so the penalty, both residuals, the dual update and the z-update all
    become angle-aware at once. Without it a yaw pair straddling +-pi
    reads as a 2pi disagreement the blocks can never resolve, the dual
    winds up against a bound it should never have reached, and both
    objectives are dominated by an error that is not there.
    """

    dim = 3

    def __init__(self, max_dual: jax.Array, scale: jax.Array = None) -> None:
        """Set the dual anti-windup clip and the per-dimension scale.

        Args:
            max_dual: Dual anti-windup clip, per dimension or scalar.
            scale: Characteristic magnitude, e.g. `(r_body, r_body, 1.0)`,
                which makes the three channels commensurate: a yaw error
                of 1 rad sweeps the footprint's edge through one body
                radius, so a normalized residual of 1 means the same
                physical displacement in every channel. Defaults to ones.
        """
        self.max_dual = jnp.asarray(max_dual)
        self.scale = jnp.ones(self.dim) if scale is None else jnp.asarray(scale)

    def normalize(self, v: jax.Array) -> jax.Array:
        """Divide by the per-dimension characteristic magnitude."""
        return v / self.scale

    def difference(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Subtract, wrapping the yaw channel into [-pi, pi)."""
        delta = a - b
        return delta.at[..., 2].set(wrap_angle(delta[..., 2]))

    def increment(self, base: jax.Array, tangent: jax.Array) -> jax.Array:
        """Add, re-wrapping yaw so z stays a valid pose."""
        out = base + tangent
        return out.at[..., 2].set(wrap_angle(out[..., 2]))

    def shift(self, seq: jax.Array) -> jax.Array:
        """Shift by one and repeat the last value.

        Zero-fill would put the vacated tail at the world origin at zero
        heading -- a specific pose, and never the one the object is
        heading for.
        """
        return jnp.concatenate([seq[1:], seq[-1:]], axis=0)

    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual + (actual (-) z), wrapped in yaw, clipped to +-max_dual."""
        return jnp.clip(
            dual + self.difference(actual, z), -self.max_dual, self.max_dual
        )


def make_object_shim(task: ConsensusTask, dt: float) -> Any:
    """Build a lightweight duck-typed task for the object-level optimizer.

    The object subproblem samples in the object block's own action space,
    not robot actuator space, so a `SamplingBasedController` built for it
    needs its own task-like object to construct against. `ObjectSubproblem`
    only reads `.model.nu`/`.dt`/`.u_min`/`.u_max` off it; all real cost
    and dynamics come from the real task's `object_*` methods.
    """
    nu = task.object_action_dim
    u_min, u_max = task.object_action_bounds()
    return SimpleNamespace(
        model=SimpleNamespace(nu=nu),
        dt=dt,
        u_min=u_min,
        u_max=u_max,
    )


def shift_object_actions(task: ConsensusTask, seq: jax.Array) -> jax.Array:
    """Receding-horizon shift of the object block's decision sequence.

    Zero-fills the vacated tail when the decision is the consensus value
    itself; holds the last value instead for a structured action space,
    where zero need not be feasible (a zero contact point is the object's
    own origin, not on its boundary).

    Module-level so the standalone object-level driver
    (`oim.worlds.object_only.build`) warm-starts identically to ADMM's.
    """
    if task.initial_object_action() is None:
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)
    return jnp.concatenate([seq[1:], seq[-1:]], axis=0)


def _finite_or(
    value: jax.Array, fallback: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """`value` where it is finite, `fallback` elsewhere; and whether it was.

    The mirror of `oim.algs.mppi.MPPI.update_params`' cost guard, which
    this layer lacked. A NaN reaching A^o or A^r is ABSORBING without it:
    `z` and both duals are carried in `ADMMParams` across control steps and
    never reset, so one bad rollout makes `penalty_cost` NaN for every
    sample from then on, MPPI's own guard then sees no finite cost and
    holds its mean, and the arm is frozen for the rest of the run --
    observed as `primal=nan dual=nan` with the pose bit-identical for
    hundreds of steps.

    Substituting the fallback costs that ROUND its consensus update and
    nothing more; the blocks re-solve from a clean `z` on the next one.

    Returns:
        The sanitized array, and a scalar bool: True if every entry was
        already finite (in which case the array is returned unchanged, so
        a healthy round is bit-identical).
    """
    ok = jnp.all(jnp.isfinite(value))
    return jnp.where(ok, value, fallback), ok


@dataclass
class ADMMTrajectory(Trajectory):
    """Trajectory with the realized consensus value A^r(U^r)_t at each step."""

    consensus_values: jax.Array


class RobotRollout(ABC):
    """How the robot block advances its state by one planning step.

    The only place the robot subproblem is tied to a particular simulator
    -- swapping this is what lets the same `ADMM` class drive both an MJX
    scene and the object-only world. Implementations must be pure
    functions of (model, state, control): they run inside `jax.lax.scan`
    under `vmap` and `jit`.
    """

    @abstractmethod
    def step(self, model: Any, state: Any, control: jax.Array) -> Any:
        """Advance the robot-side state by one planning timestep."""


class MJXRollout(RobotRollout):
    """The default backend: `substeps` `mjx.step`s of a MuJoCo MJX model.

    The object block's counterpart is `oim.runtime.object_mjx`, which
    builds a whole model at `planning_dt / substeps`. This one cannot:
    the model is handed to `step` per rollout (domain randomization gives
    each sample its own), so the timestep is scaled on the traced model
    instead. Same result, and it follows a randomized model.
    """

    def __init__(self, substeps: int = 1) -> None:
        """Set how finely contact integrates inside one planning step.

        Args:
            substeps: Physics steps per planning step. The model's
                timestep is divided by this, so one `step` still advances
                exactly `planning_dt` and the horizon keeps its length in
                both steps and seconds -- only the contact integration
                inside each planning step gets finer. Costs `substeps`x
                in the robot rollout, which is most of a solve.

        Raises:
            ValueError: If `substeps` is below 1.
        """
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")
        self.substeps = int(substeps)

    def step(
        self, model: mjx.Model, state: mjx.Data, control: jax.Array
    ) -> mjx.Data:
        """Set the control and advance one planning step."""
        with quiet_mjx_cast_overflow():
            state = state.replace(ctrl=control)
            # One substep is the shipped default and the pre-existing
            # behaviour: step the model exactly as handed over, with no
            # `replace` on it and no loop primitive. Same reasoning as
            # `MJXObjectRollout.step` -- this runs H times per rollout
            # inside another scan, so the overhead is not once.
            if self.substeps == 1:
                return mjx.step(model, state)
            model = model.replace(
                opt=model.opt.replace(
                    timestep=model.opt.timestep / self.substeps
                )
            )

            def body(data: mjx.Data, _: Any) -> Tuple[mjx.Data, None]:
                return mjx.step(model, data), None

            out, _ = jax.lax.scan(body, state, None, length=self.substeps)
            return out


class ObjectRollout(ABC):
    """How the object block advances its own state by one planning step.

    The object-side counterpart to `RobotRollout`: swapping this swaps the
    object dynamics while the sampler, costs, projection, warm start and
    consensus math stay identical. The carry is opaque -- an analytic
    backend carries just the SE(2) pose, an MJX-backed one a whole
    `mjx.Data` so velocity persists along the horizon; `pose` projects
    the carry back to x^o.

    Implementations must be pure functions (traced under `vmap`/`jit`),
    which is why the CPU `oim.worlds.object_only.plant.MujocoPlant`
    (mutates `mujoco.MjData` in place) cannot be used here.
    """

    @abstractmethod
    def init(self, obj_state: jax.Array) -> Any:
        """Build the carry for a horizon starting at pose `obj_state`."""

    @abstractmethod
    def pose(self, carry: Any) -> jax.Array:
        """The object's SE(2) configuration x^o held in `carry`."""

    @abstractmethod
    def step(self, carry: Any, w: jax.Array) -> Any:
        """Advance the carry by one planning step under consensus decision w."""


class AnalyticObjectRollout(ObjectRollout):
    """The default backend: `task.object_dynamics`, the paper's eq. 5.

    Quasi-static, so the pose is the state and `init`/`pose` are the
    identity.
    """

    def __init__(self, task: ConsensusTask) -> None:
        """Keep the task; the analytic rollout needs nothing else."""
        self.task = task

    def init(self, obj_state: jax.Array) -> jax.Array:
        """The pose is the whole state; there is nothing to build."""
        return obj_state

    def pose(self, carry: jax.Array) -> jax.Array:
        """The carry is already x^o."""
        return carry

    def step(self, carry: jax.Array, w: jax.Array) -> jax.Array:
        """One forward-Euler step of the limit surface."""
        return self.task.object_dynamics(carry, w)


class ObjectSubproblem:
    """Object-level ADMM subproblem: pluggable dynamics, pluggable optimizer.

    Wraps any `SamplingBasedController` (built against a `make_object_shim`
    task) to sample/reweight consensus-space decisions w^o_t. Costs come
    from `task.object_running_cost`/`object_terminal_cost`; dynamics come
    from an injected `ObjectRollout`, defaulting to the task's own
    closed-form `object_dynamics`.
    """

    def __init__(
        self,
        task: ConsensusTask,
        optimizer: SamplingBasedController,
        consensus: ConsensusSpace,
        proximal_weight: float,
        rollout: Optional[ObjectRollout] = None,
        lagged: bool = False,
    ) -> None:
        """Wire the object block to its task, optimizer and penalty.

        Args:
        task: The real task, providing object-level dynamics/costs.
        optimizer: A `SamplingBasedController` built against
            `make_object_shim(task, ...)`.
        consensus: The consensus space used for the ADMM penalty.
        proximal_weight: Weight (gamma) on the proximal term (eq. 24).
        rollout: How to advance the object one step. Defaults to
            `AnalyticObjectRollout`; pass
            `oim.runtime.object_mjx.MJXObjectRollout` to plan against
            the simulator instead.
        lagged: Read A^o and the plan off the *incoming* mean, carried as
            one extra row of the sample batch, instead of re-rolling the
            outgoing mean on its own afterwards. Saves a whole
            single-trajectory rollout per call at the cost of a one-round
            lag; see `ADMM.__init__`'s `lagged_consensus`. Off by
            default, which is also what the object-only world runs.
        """
        self.task = task
        self.optimizer = optimizer
        self.consensus = consensus
        self.proximal_weight = proximal_weight
        self.lagged = lagged
        self.rollout = AnalyticObjectRollout(task) if rollout is None else (
            rollout
        )

    def _rollout(
        self, obj_state0: jax.Array, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Scan `self.rollout` over an (H, action_dim) sequence.

        The action -> wrench map runs inside the scan since a contact
        wrench depends on the object's current pose, not just the action.

        Returns:
            States x^o_1..x^o_H; wrenches w^o_0..w^o_{H-1} that produced
            them; and extracted consensus values A^o_0..A^o_{H-1} (equal
            to the wrenches when the consensus variable is the wrench, and
            to the actions themselves when it is the contact point).
        """

        def step(
            carry: Any, action: jax.Array
        ) -> Tuple[Any, Tuple[jax.Array, jax.Array, jax.Array]]:
            obj_state = self.rollout.pose(carry)
            w = self.task.object_action_to_consensus(obj_state, action)
            carry = self.rollout.step(carry, w)
            new_state = self.rollout.pose(carry)
            # A^o read after the step, matching the robot block reading A^r
            # after `rollout.step` -- index t is the value at t+1 on both.
            a_o = self.task.object_consensus(new_state, w, action)
            return carry, (new_state, w, a_o)

        carry0 = self.rollout.init(obj_state0)
        _, (states, ws, a_o) = jax.lax.scan(step, carry0, actions)
        return states, ws, a_o

    def optimize(
        self,
        obj_state0: jax.Array,
        params: Any,
        z: jax.Array,
        dual_o: jax.Array,
        rho: jax.Array,
        prev_knots: jax.Array,
        rng: jax.Array,
        weight_scale: jax.Array = 1.0,
    ) -> Tuple[Any, jax.Array, jax.Array, jax.Array]:
        """Run `optimizer.iterations` MPPI-style passes against a fixed target.

        Args:
            obj_state0: The object's current configuration x^o_0.
            params: The object optimizer's current policy parameters.
            z: The consensus target.
            dual_o: The object block's dual.
            rho: The penalty weight.
            prev_knots: Previous ADMM iteration's knots (proximal term).
            rng: Random key.
            weight_scale: Goal-tracking ramp, shared with the robot block.

        Returns:
            Updated params; the object's proposed decision W^o
            (params.mean); its nominal state reference; and the last
            iteration's sampled trajectories for visualization.
        """
        opt = self.optimizer

        def _scan_body(params: Any, rng_i: jax.Array) -> Tuple[Any, jax.Array]:
            sample_rng = rng_i
            # The task may own the proposal distribution (e.g. contact
            # points constrained to the object's boundary); `num_samples`
            # is read defensively since not every controller exposes one
            # (Evosax keeps its population size inside the ES state).
            num_samples = getattr(opt, "num_samples", None)
            custom = (
                None
                if num_samples is None
                else self.task.sample_object_actions(
                    params.mean, sample_rng, num_samples, obj_state0
                )
            )
            if custom is None:
                knots, params = opt.sample_knots(params)
            else:
                knots = custom
            knots = jnp.clip(knots, opt.task.u_min, opt.task.u_max)
            knots = self.task.project_object_action(knots, obj_state0)
            if self.lagged:
                # One extra row carrying this pass's INCOMING nominal, so
                # A^o comes out of a batch that is being dispatched
                # anyway. The block is latency-bound, not throughput-bound
                # -- measured 30.42 ms at 256 rows against 30.54 at 257,
                # while the separate nominal rollout below costs 21.4 --
                # so the row is free and the rollout is not. Sliced off
                # again before any cost or update sees it.
                knots = jnp.concatenate(
                    [
                        knots,
                        self.task.project_object_action(
                            params.mean, obj_state0
                        )[None],
                    ],
                    axis=0,
                )

            states, ws, a_o = jax.vmap(self._rollout, in_axes=(None, 0))(
                obj_state0, knots
            )
            nominal_pack = None
            if self.lagged:
                nominal_pack = (states[-1], a_o[-1])
                knots, states, ws, a_o = (
                    knots[:-1],
                    states[:-1],
                    ws[:-1],
                    a_o[:-1],
                )
            # J_o: dt-weighted running cost + terminal cost.
            running = self.task.dt * jax.vmap(
                jax.vmap(self.task.object_running_cost, in_axes=(0, 0, None)),
                in_axes=(0, 0, None),
            )(states[:, :-1], ws[:, :-1], weight_scale)
            terminal = jax.vmap(
                self.task.object_terminal_cost, in_axes=(0, None)
            )(states[:, -1], weight_scale)

            # Proximal term (gamma/2)||U^o - U^{o,(l)}||^2, paper eq. 24.
            proximal = (
                0.5
                * self.proximal_weight
                * jnp.sum((knots - prev_knots) ** 2, axis=(-2, -1))
            )
            # Rate penalty: sequence-level, anchored to the value the
            # previous solve already intended for this step, so it charges
            # for changing course across control steps as well as within
            # the horizon.
            #
            # Charged on the block's own *decision*, which
            # `object_rate_values` selects -- not blindly on the wrench.
            # Identical under `consensus="wrench"`, where decision, wrench
            # and A^o are all the same array. Under `"contact_point"` they
            # are not, and feeding the wrench here made
            # `PushT.object_rate_cost` normalize newtons and newton-metres
            # by [r_body, r_body, f_max]: a 5 N change in f_x scored 18422
            # against goal costs of order 20, so the only affordable plan
            # was a wrench that never changes -- w = 0, below breakaway,
            # the object held still whatever the temperature or the force
            # ceiling.
            w_prev = self.task.object_action_to_consensus(
                obj_state0, prev_knots[0]
            )
            a_prev = self.task.object_consensus(
                obj_state0, w_prev, prev_knots[0]
            )
            rate = jax.vmap(self.task.object_rate_cost, in_axes=(0, None))(
                self.task.object_rate_values(ws, a_o),
                self.task.object_rate_values(w_prev, a_prev),
            )
            terminal = terminal + proximal + rate

            # ADMM penalty (rho/2)||A^o(U^o)_t - z_t + y^o_t||^2, eq. 25.
            penalty = self.consensus.penalty_cost(a_o, z, dual_o, rho)
            costs = jnp.concatenate([running, terminal[:, None]], axis=1)
            costs = costs + penalty

            rollouts = Trajectory(
                controls=knots,
                knots=knots,
                costs=costs,
                trace_sites=jnp.zeros((knots.shape[0], 1, 3)),
            )
            params = opt.update_params(params, rollouts)
            # Re-project the aggregated nominal, not just the samples: an
            # optimizer update averages samples, and averaging feasible
            # points need not be feasible (measured up to 1.5mm for the T
            # footprint's contact points either side of a corner).
            params = params.replace(
                mean=self.task.project_object_action(params.mean, obj_state0)
            )
            return params, (states, nominal_pack)

        rngs = jax.random.split(rng, opt.iterations)
        params, (all_states, nominal_packs) = jax.lax.scan(
            _scan_body, params, rngs
        )
        # Last iteration's sample population, for visualization -- free,
        # already computed above for `object_running_cost`.
        object_samples = all_states[-1]

        if self.lagged:
            # Pass 0's extra row: A^o and the plan of the mean this call
            # was ENTERED with, not the one it leaves with. `[0]` and not
            # `[-1]`, because only the first pass's row is the mean the
            # ADMM round started from.
            ref_states, a_obj = jax.tree.map(lambda v: v[0], nominal_packs)
        else:
            # A^o for the nominal (eq. 24), recovered by rolling the
            # nominal actions out -- necessary for a non-trivial action
            # parameterization or a pose consensus variable.
            nominal = self.task.project_object_action(params.mean, obj_state0)
            ref_states, _, a_obj = self._rollout(obj_state0, nominal)
        return params, a_obj, ref_states, object_samples

    def nominal_plan(self, obj_state0: jax.Array, params: Any) -> jax.Array:
        """The object trajectory this block currently intends, x^o_1..x^o_H.

        Recomputed from the stored nominal so a caller can ask without
        re-solving. Cheap under the default backend: H closed-form
        dynamics steps, no sampling.
        """
        nominal = self.task.project_object_action(params.mean, obj_state0)
        states, _, _ = self._rollout(obj_state0, nominal)
        return states


class RobotSubproblem:
    """Robot-level ADMM subproblem: real MJX contact, pluggable optimizer.

    Reads `.model`/`.randomized_axes`/`.risk_strategy`/`.interp_func`/
    `.num_randomizations`/`.ctrl_steps`/`.iterations` off the injected
    optimizer generically (never off `self`), so domain randomization and
    risk strategies keep working regardless of which `SamplingBasedController`
    subclass is plugged in.
    """

    def __init__(
        self,
        task: ConsensusTask,
        optimizer: SamplingBasedController,
        consensus: ConsensusSpace,
        proximal_weight: float,
        rollout: Optional[RobotRollout] = None,
        lagged: bool = False,
    ) -> None:
        """Wire the robot block to its task, optimizer and penalty.

        Args:
        task: The real task, providing the MJX model, robot-level
            cost, and the A^r extraction map.
        optimizer: A `SamplingBasedController` built against `task`.
        consensus: The consensus space; the ADMM penalty is added via
            `consensus.penalty_cost`, shared with the object block.
        proximal_weight: Weight (gamma) on the proximal term (eq. 25).
        rollout: How to advance the robot state one step. Defaults to
            `MJXRollout`.
        lagged: Return A^r for the *incoming* mean, carried as one extra
            row of the sample batch, instead of leaving the caller to
            re-simulate the outgoing mean through
            `nominal_realized_consensus`. Saves the single largest line
            in an ADMM solve at the cost of a one-round lag; see
            `ADMM.__init__`'s `lagged_consensus`.
        """
        self.task = task
        self.optimizer = optimizer
        self.consensus = consensus
        self.proximal_weight = proximal_weight
        self.lagged = lagged
        self.rollout = rollout or MJXRollout()

    @partial(
        jax.vmap,
        in_axes=(None, None, None, 0, 0, None, None, None, None, None),
    )
    def _eval_rollouts_one(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
        z: jax.Array,
        dual_r: jax.Array,
        rho: jax.Array,
        obj_ref: jax.Array,
        prev_knots: jax.Array,
    ) -> Tuple[mjx.Data, ADMMTrajectory]:
        """Roll out one control sequence, scored against the ADMM penalty.

        Also returns the realized consensus value at each step.
        """
        # Read once at the horizon start: `mjx.Data.time` advances along
        # the rollout, so a per-step read would over-weight step H.
        weight_scale = getattr(self.task, "time_ramp", lambda _t: 1.0)(
            state.time
        )

        def _scan_fn(
            x: mjx.Data,
            inputs: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array, jax.Array]]:
            u, z_t, dual_t, ref_t = inputs
            x = self.rollout.step(model, x, u)
            # Which point of the object block's plan to aim at, re-picked
            # from the object's pose at THIS step -- the base class
            # answers `obj_ref[-1]` (the plan endpoint, fixed for the
            # whole horizon), a pursuit override slides it forward along
            # the plan as the rollout advances. Either way it is the
            # task's business, not this layer's.
            local_goal = self.task.local_goal_from_plan(
                obj_ref, self.task.object_state_from_robot(x)
            )
            # J_r: the task's own cost, dt-weighted.
            cost = self.optimizer.dt * self.task.robot_running_cost(
                x, u, ref_t, local_goal, weight_scale
            )
            # A^r: the wrench the robot's motion actually imparts on the
            # object, read from the simulator (eq. 23).
            consensus_val = self.task.realized_consensus(x)
            # ADMM penalty (rho/2)||A^r(U^r)_t - z_t + y^r_t||^2 (eq. 25).
            cost = cost + self.consensus.penalty_cost(
                consensus_val, z_t, dual_t, rho
            )
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, consensus_val, sites)

        final_state, (states, costs, consensus_vals, trace_sites) = (
            jax.lax.scan(_scan_fn, state, (controls, z, dual_r, obj_ref))
        )

        # Proximal term (gamma/2)||U^r - U^{r,(l)}||^2, paper eq. 25.
        proximal = (
            0.5 * self.proximal_weight * jnp.sum((knots - prev_knots) ** 2)
        )
        final_cost = (
            self.task.robot_terminal_cost(
                final_state,
                self.task.local_goal_from_plan(
                    obj_ref, self.task.object_state_from_robot(final_state)
                ),
                weight_scale,
            )
            + proximal
        )
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, ADMMTrajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
            consensus_values=consensus_vals,
        )

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
        z: jax.Array,
        dual_r: jax.Array,
        rho: jax.Array,
        obj_ref: jax.Array,
        prev_knots: jax.Array,
    ) -> ADMMTrajectory:
        """Like `SamplingBasedController.rollout_with_randomizations`.

        z/dual_r/rho/obj_ref (the fixed target every sample is scored
        against) and the proximal anchor are threaded through too.
        """
        opt = self.optimizer
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(opt.num_randomizations), state
        )
        if opt.num_randomizations > 1:
            subrngs = jax.random.split(rng, opt.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        tq = jnp.linspace(tk[0], tk[-1], opt.ctrl_steps)
        controls = opt.interp_func(tq, tk, knots)

        _, rollouts = jax.vmap(
            self._eval_rollouts_one,
            in_axes=(
                opt.randomized_axes,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )(
            opt.model,
            states,
            controls,
            knots,
            z,
            dual_r,
            rho,
            obj_ref,
            prev_knots,
        )

        costs = opt.risk_strategy.combine_costs(rollouts.costs)
        return rollouts.replace(
            costs=costs,
            controls=rollouts.controls[0],
            knots=rollouts.knots[0],
            trace_sites=rollouts.trace_sites[0],
            consensus_values=rollouts.consensus_values[0],
        )

    def optimize(
        self,
        state: mjx.Data,
        params: Any,
        z: jax.Array,
        dual_r: jax.Array,
        rho: jax.Array,
        obj_ref: jax.Array,
        prev_knots: jax.Array,
        rng: jax.Array,
    ) -> Tuple[Any, ADMMTrajectory, Optional[jax.Array]]:
        """Run `optimizer.iterations` passes against a fixed target.

        Returns:
            `(params, rollouts, a_rob)`. `a_rob` is A^r for the mean this
            call was *entered* with when `lagged`, and `None` otherwise --
            the caller then reads A^r off the outgoing mean itself with
            `nominal_realized_consensus`.
        """
        opt = self.optimizer
        tk = params.tk

        def _scan_body(
            params: Any, rng_i: jax.Array
        ) -> Tuple[Any, Tuple[ADMMTrajectory, Optional[jax.Array]]]:
            knots, params = opt.sample_knots(params)
            dr_rng = rng_i
            knots = jnp.clip(knots, self.task.u_min, self.task.u_max)
            if self.lagged:
                # One extra row carrying this pass's INCOMING mean. The
                # batch is nearly free (measured 84.45 ms at 256 rows
                # against 84.34 at 257) while the separate nominal
                # rollout it replaces costs 61.4 -- same horizon, same
                # substeps, batch of one. Appended AFTER the clip and
                # unclipped, matching what `nominal_realized_consensus`
                # rolls out; the mean is a weighted average of already
                # clipped samples, so it is in bounds regardless.
                knots = jnp.concatenate([knots, params.mean[None]], axis=0)
            rollouts = self.rollout_with_randomizations(
                state, tk, knots, dr_rng, z, dual_r, rho, obj_ref, prev_knots
            )
            nominal_consensus = None
            if self.lagged:
                nominal_consensus = rollouts.consensus_values[-1]
                # Off again before the update, so the optimizer reweights
                # exactly the `num_samples` rollouts it would have seen.
                rollouts = jax.tree.map(lambda v: v[:-1], rollouts)
            params = opt.update_params(params, rollouts)
            return params, (rollouts, nominal_consensus)

        rngs = jax.random.split(rng, opt.iterations)
        params, (rollouts, nominal) = jax.lax.scan(_scan_body, params, rngs)
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        # `[0]`, not `[-1]`: only the first pass's extra row is the mean
        # the ADMM round started from. At `iterations = 1`, which both
        # ADMM blocks run, they are the same row.
        a_rob = nominal[0] if self.lagged else None
        return params, rollouts_final, a_rob

    def nominal_realized_consensus(
        self, state: mjx.Data, params: Any
    ) -> jax.Array:
        """Re-simulate `params.mean` alone to read the realized consensus.

        The wrapped optimizer's update blends all samples into a new mean,
        so there's no single winning rollout to read A^r off directly.
        """
        opt = self.optimizer
        tk = params.tk
        tq = jnp.linspace(tk[0], tk[-1], opt.ctrl_steps)
        controls = opt.interp_func(tq, tk, params.mean[None, ...])[0]

        def _scan_fn(x: mjx.Data, u: jax.Array) -> Tuple[mjx.Data, jax.Array]:
            x = self.rollout.step(self.task.model, x, u)
            return x, self.task.realized_consensus(x)

        _, consensus_vals = jax.lax.scan(_scan_fn, state, controls)
        return consensus_vals

    def nominal_plan(
        self, state: mjx.Data, params: Any
    ) -> Tuple[jax.Array, jax.Array]:
        """What this block's nominal controls would produce.

        The robot block's counterpart to `ObjectSubproblem.nominal_plan`:
        both trajectories are of the same object, so their disagreement is
        the consensus residual made spatial. Costs one real rollout (H
        simulator steps), a fraction of the `num_samples` already run.

        Returns:
            `(obj_states, trace_sites)`: object states (H, object_state_dim)
            and the task's trace-site position at each step (H, 3).
        """
        opt = self.optimizer
        tk = params.tk
        tq = jnp.linspace(tk[0], tk[-1], opt.ctrl_steps)
        controls = opt.interp_func(tq, tk, params.mean[None, ...])[0]

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[jax.Array, jax.Array]]:
            x = self.rollout.step(self.task.model, x, u)
            return x, (
                self.task.object_state_from_robot(x),
                self.task.get_trace_sites(x)[0],
            )

        _, (obj_states, trace_sites) = jax.lax.scan(_scan_fn, state, controls)
        return obj_states, trace_sites


@dataclass
class ADMMParams:
    """Warm-started policy parameters for the ADMM controller.

    Attributes:
        robot_params: The robot optimizer's own params (e.g. `MPPIParams`).
        object_params: The object optimizer's own params.
        z: The consensus variable, (H, dim).
        gamma_o: The object block's scaled dual variable, (H, dim).
        gamma_r: The robot block's scaled dual variable, (H, dim).
        rho: The current ADMM penalty weight (adapted online).
        primal_residual: Last iteration's primal residual. Logging only.
        dual_residual: Last iteration's dual residual. Logging only.
        object_samples: The object block's last sampled trajectories,
            (num_samples, H, object_state_dim). Logging only.
        a_obj: A^o, the object block's extracted consensus value, (H, dim).
            Carried so the ADMM penalty is reconstructible from a run file.
        a_rob: A^r as the nominal robot plan realized it, (H, dim) -- the
            planned value, so both blocks' penalties reflect what they
            planned (the runners log the executed A^r separately).
        nonfinite_rounds: Non-finite events this control step had to
            repair: one per ADMM round whose consensus update was skipped
            because a block returned a non-finite A (see `_finite_or`),
            plus one if the warm start itself arrived poisoned. 0 on a
            healthy step. Logged per step so a swallowed event leaves a
            trace: without it the guard turns a loud `primal=nan` into a
            silent one, and the failure it protects against becomes
            undiagnosable.
        rng: PRNG key, split per iteration for the two blocks.
    """

    robot_params: Any
    object_params: Any
    z: jax.Array
    gamma_o: jax.Array
    gamma_r: jax.Array
    rho: jax.Array
    primal_residual: jax.Array
    dual_residual: jax.Array
    object_samples: jax.Array
    a_obj: jax.Array
    a_rob: jax.Array
    nonfinite_rounds: jax.Array
    rng: jax.Array

    @property
    def tk(self) -> jax.Array:
        """Knot times, delegated to the robot params."""
        return self.robot_params.tk

    @property
    def mean(self) -> jax.Array:
        """Mean control knots, delegated to the robot params."""
        return self.robot_params.mean


@dataclass
class _ADMMCarry:
    """Internal `jax.lax.while_loop` carry for one real control step."""

    it: jax.Array
    object_params: Any
    robot_params: Any
    z: jax.Array
    gamma_o: jax.Array
    gamma_r: jax.Array
    rho: jax.Array
    primal_res: jax.Array
    dual_res: jax.Array
    object_samples: jax.Array
    rng: jax.Array
    a_obj: jax.Array
    a_rob: jax.Array
    nonfinite: jax.Array


class ADMM(SamplingBasedController):
    """ADMM-coordinated object/robot consensus planning (Algorithm 4).

    Coordinates an object-level and a robot-level sampling-based
    sub-optimizer to reach consensus on a task-defined consensus variable
    every real control step.

    The ADMM iteration (two full sub-optimizations plus consensus math per
    round, with an early exit) doesn't fit the generic sample/rollout/
    update template, so `optimize()` is overridden entirely;
    `sample_knots`/`update_params` raise `NotImplementedError`.
    """

    def __init__(
        self,
        task: ConsensusTask,
        robot_optimizer: SamplingBasedController,
        object_optimizer: SamplingBasedController,
        consensus: ConsensusSpace,
        n_admm: int,
        eps_r: float,
        eps_s: float,
        proximal_weight: float = 0.0,
        rho_init: float = 1.0,
        rho_adapt: bool = False,
        rho_bound_factor: float = 8.0,
        consensus_object_weight: float = 0.5,
        rollout: Optional[RobotRollout] = None,
        object_rollout: Optional[ObjectRollout] = None,
        lagged_consensus: str = "off",
        debug_print: bool = False,
    ) -> None:
        """Build the ADMM controller from two pre-built sub-optimizers.

        Args:
            task: A task implementing both `Task` and `ConsensusTask`.
            robot_optimizer: Built against `task`; its
                `plan_horizon`/`num_knots`/`spline_type`/etc. become this
                controller's own, and its `ctrl_steps` must equal
                `object_optimizer.num_knots` (the consensus horizon H).
            object_optimizer: Built against `make_object_shim(task, ...)`;
                its `num_knots` sets H.
            consensus: The consensus space (e.g. `WrenchConsensus`).
            n_admm: Max ADMM iterations per real step.
            eps_r: Primal residual tolerance for early exit.
            eps_s: Dual residual tolerance for early exit.
            proximal_weight: Weight (gamma) on the proximal term (eq. 24-25).
            rho_init: The ADMM penalty weight, fixed unless `rho_adapt`.
            rho_adapt: Residual-balancing rule (Algorithm 4 step 7):
                doubles `rho` when the primal residual dominates, halves
                it when the dual does. Off by default -- the rule is
                multiplicative and `rho` persists across control steps, so
                a residual imbalance that never resolves (an infeasible
                target) compounds every iteration rather than settling.
            rho_bound_factor: When `rho_adapt` is on, how far the rule may
                move `rho` from `rho_init` (multiplicative, either way).
            consensus_object_weight: w_o in `ConsensusSpace.z_update`, the
                object block's share of the agreed value. 0.5 is eq. 27
                (the plain average). Above 0.5, z sits nearer the object
                block's proposal, so the robot's penalty chases a plan
                built in 3 DOF rather than a compromise with its own
                5-DOF-sampled realization -- at the cost of z no longer
                being the exact z-minimizer, so this is a heuristic and
                not the paper's update.
            rollout: How the robot block advances its state one step.
                Defaults to `MJXRollout`; pass
                `oim.worlds.object_only` for the object-only world.
            object_rollout: How the object block advances its state one
                step. Defaults to `AnalyticObjectRollout` (eq. 5); pass
                `oim.runtime.object_mjx.MJXObjectRollout` to make both
                blocks predict with MJX instead, which also makes the
                object block's cost no longer free per sample.
            lagged_consensus: Where to read A from. `"off"` (the default,
                and the paper's Algorithm 4) re-simulates each block's
                *outgoing* mean on its own to get A. `"robot"` and
                `"both"` instead carry the block's *incoming* mean as one
                extra row of the sample batch and read A off that.

                WHY IT IS WORTH A SWITCH. Those two re-simulations are
                one trajectory each, but a solve is bound by sequential
                simulator steps and not by batch width -- the robot block
                measured 84.45 ms at 256 rollouts against 84.34 at 257,
                while its lone nominal rollout, same horizon and same
                substeps, costs 61.4. Together the two are 37% of a
                solve (330 ms of 894 on xarm6/open_table). Removing the
                robot's alone measured 871 -> 660 ms per control step,
                1.15 -> 1.52 Hz.

                WHAT IT CHANGES, AND WHY IT IS NOT FREE. The extra row
                has to be dispatched with the batch, so it can only carry
                the mean the round was entered with -- the outgoing mean
                does not exist yet. z and the duals at round l are then
                built from A(x_{l-1}) rather than A(x_l): the block
                updates and the consensus are the same count, offset by
                one round. Under `"both"` the robot block additionally
                tracks the object plan from x^o_{l-1}, which is the
                tighter coupling of the two and the reason `"robot"`
                exists separately. Neither is Algorithm 4, so this is an
                opt-in switch and not a silent optimization, and a run
                under it is not comparable to one without.

                One further caveat: widening the batch by a row can move
                the other rollouts at float32-ulp level, since MJX's
                batched solver tiles across the batch. The sampled knots
                themselves are bit-identical -- `sample_knots` sees the
                same rng and the same `num_samples`, and the extra row is
                sliced off before any cost or optimizer update reads it.
            debug_print: Print residuals/penalty weight every ADMM
                iteration. Off by default: it's a host callback inside the
                compiled loop (costs a device sync that inflates the
                recorded `compute_time`) and floods the per-step summary.
        """
        if n_admm < 1:
            raise ValueError("n_admm must be at least 1")
        if not 0.0 <= consensus_object_weight <= 1.0:
            raise ValueError(
                "consensus_object_weight must be in [0, 1], got "
                f"{consensus_object_weight}"
            )
        # YAML 1.1 reads a bare `off` as the boolean False, so a config
        # written the obvious way arrives here as `False` rather than
        # `"off"`. The shipped configs quote it; this accepts the
        # unquoted form too, since False has no other meaning here.
        if lagged_consensus is False:
            lagged_consensus = "off"
        if lagged_consensus not in _LAGGED_CONSENSUS:
            raise ValueError(
                "lagged_consensus must be one of "
                f"{sorted(_LAGGED_CONSENSUS)}, got {lagged_consensus!r}"
            )
        if robot_optimizer.ctrl_steps != object_optimizer.num_knots:
            raise ValueError(
                "robot_optimizer.ctrl_steps must equal object_optimizer."
                f"num_knots (the consensus horizon H), got "
                f"{robot_optimizer.ctrl_steps} != {object_optimizer.num_knots}"
            )

        self.task = task
        self.robot_optimizer = robot_optimizer
        self.object_optimizer = object_optimizer
        self.consensus = consensus
        self.n_admm = n_admm
        self.eps_r = eps_r
        self.eps_s = eps_s
        self.rho_init = rho_init
        self.rho_adapt = rho_adapt
        # Per-dimension, so an anisotropic P = diag(rho_f, rho_f, rho_tau)
        # keeps its ratio rather than collapsing under one scalar band.
        self.rho_min = jnp.asarray(rho_init) / rho_bound_factor
        self.rho_max = jnp.asarray(rho_init) * rho_bound_factor
        self.consensus_object_weight = consensus_object_weight
        self.lagged_consensus = lagged_consensus
        self.debug_print = debug_print

        self.object_subproblem = ObjectSubproblem(
            task,
            object_optimizer,
            consensus,
            proximal_weight,
            rollout=object_rollout,
            lagged=lagged_consensus == "both",
        )
        self.robot_subproblem = RobotSubproblem(
            task,
            robot_optimizer,
            consensus,
            proximal_weight,
            rollout=rollout,
            lagged=lagged_consensus in ("robot", "both"),
        )

        # Alias off robot_optimizer rather than calling
        # SamplingBasedController.__init__, which would rebuild a
        # redundant domain-randomization ensemble. Makes ADMM a drop-in
        # for run_interactive.
        self.model = robot_optimizer.model
        self.randomized_axes = robot_optimizer.randomized_axes
        self.risk_strategy = robot_optimizer.risk_strategy
        self.num_randomizations = robot_optimizer.num_randomizations
        self.dt = robot_optimizer.dt
        self.plan_horizon = robot_optimizer.plan_horizon
        self.ctrl_steps = robot_optimizer.ctrl_steps
        self.spline_type = robot_optimizer.spline_type
        self.num_knots = robot_optimizer.num_knots
        self.interp_func = robot_optimizer.interp_func
        self.iterations = robot_optimizer.iterations

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> ADMMParams:
        """Initialize the policy parameters."""
        object_params = self.object_optimizer.init_params(seed=seed)
        robot_params = self.robot_optimizer.init_params(
            initial_knots=initial_knots, seed=seed
        )
        h = self.object_optimizer.num_knots
        dim = self.consensus.dim

        # Let the task seed its own action space if zeros are degenerate
        # there (e.g. a contact point at the object's origin).
        seed_action = self.task.initial_object_action()
        if seed_action is not None:
            object_params = object_params.replace(
                mean=jnp.broadcast_to(
                    jnp.asarray(seed_action), (h, self.task.object_action_dim)
                )
            )
        return ADMMParams(
            robot_params=robot_params,
            object_params=object_params,
            z=jnp.zeros((h, dim)),
            gamma_o=jnp.zeros((h, dim)),
            gamma_r=jnp.zeros((h, dim)),
            rho=jnp.asarray(self.rho_init, dtype=jnp.float32),
            # Large-but-finite sentinel for maximal initial exploration --
            # must stay finite: an actual inf would nan every sampled knot.
            primal_residual=jnp.asarray(100.0, dtype=jnp.float32),
            dual_residual=jnp.asarray(100.0, dtype=jnp.float32),
            # Placeholder, overwritten by `_admm_iteration`'s first call.
            object_samples=jnp.zeros((1, 1, 1), dtype=jnp.float32),
            a_obj=jnp.zeros((h, dim)),
            a_rob=jnp.zeros((h, dim)),
            nonfinite_rounds=jnp.asarray(0, dtype=jnp.int32),
            rng=jax.random.key(seed),
        )

    def _shift(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift: seq[t] <- seq[t+1], last slot <- 0."""
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)

    def _shift_object(self, seq: jax.Array) -> jax.Array:
        """Delegate to `shift_object_actions`.

        The standalone object-only driver then warm-starts exactly as
        ADMM does.
        """
        return shift_object_actions(self.task, seq)

    def _admm_iteration(
        self, carry: _ADMMCarry, obj_state0: jax.Array, state: mjx.Data
    ) -> Tuple[_ADMMCarry, ADMMTrajectory]:
        """One ADMM iteration: object update -> robot update -> consensus."""
        rng, obj_rng, rob_rng = jax.random.split(carry.rng, 3)
        prev_object_knots = carry.object_params.mean
        prev_robot_knots = carry.robot_params.mean

        # Goal-tracking ramp, read once at the horizon start; the robot
        # block reads the identical value from the same `state.time`
        # inside `_eval_rollouts_one`.
        weight_scale = getattr(self.task, "time_ramp", lambda _t: 1.0)(
            state.time
        )
        # Near-goal fade on the consensus penalty: once the object is one
        # correction from the goal, each block optimizes its own objective
        # instead of negotiating a wrench. Read once from `obj_state0` and
        # handed to both blocks -- one shared `rho` is what lets
        # `consensus_object_weight` be the only asymmetry in `z_update`.
        fade = getattr(self.task, "shaping_fade", lambda _p: 1.0)(obj_state0)
        penalty_rho = carry.rho * fade
        object_params, a_obj, obj_ref, object_samples = (
            self.object_subproblem.optimize(
                obj_state0,
                carry.object_params,
                carry.z,
                carry.gamma_o,
                penalty_rho,
                prev_object_knots,
                obj_rng,
                weight_scale,
            )
        )
        robot_params, rollouts, a_rob = self.robot_subproblem.optimize(
            state,
            carry.robot_params,
            carry.z,
            carry.gamma_r,
            penalty_rho,
            obj_ref,
            prev_robot_knots,
            rob_rng,
        )
        if a_rob is None:
            # Algorithm 4: A^r off the mean this round produced, which
            # needs its own rollout. Under `lagged_consensus` the block
            # has already returned A^r for the mean it started from, out
            # of a batch that was being dispatched anyway.
            a_rob = self.robot_subproblem.nominal_realized_consensus(
                state, robot_params
            )

        # A non-finite A poisons z, both duals and both residuals, all of
        # which persist across control steps -- so it is caught here, at
        # the one place both blocks' values are in hand. Falling back to
        # the incoming z means "this block proposed no change this round",
        # which the z-update and the dual step both handle already.
        a_obj, obj_ok = _finite_or(a_obj, carry.z)
        a_rob, rob_ok = _finite_or(a_rob, carry.z)
        blocks_ok = obj_ok & rob_ok

        z_new = self.consensus.z_update(
            a_obj,
            a_rob,
            carry.gamma_o,
            carry.gamma_r,
            carry.z,
            self.consensus_object_weight,
        )
        z_new = jnp.where(blocks_ok, z_new, carry.z)
        # Duals move with the same fade: inside the fade radius nothing
        # charges for a disagreement, so a full-step dual would integrate
        # against nothing and release the banked amount once the object
        # drifts back out. Interpolating toward the stepped value (rather
        # than scaling the increment) keeps `dual_update`'s clip intact.
        gamma_o = carry.gamma_o + fade * (
            self.consensus.dual_update(a_obj, z_new, carry.gamma_o)
            - carry.gamma_o
        )
        gamma_r = carry.gamma_r + fade * (
            self.consensus.dual_update(a_rob, z_new, carry.gamma_r)
            - carry.gamma_r
        )
        # Duals integrate disagreement, so a round with no trustworthy
        # disagreement must not move them either.
        gamma_o = jnp.where(blocks_ok, gamma_o, carry.gamma_o)
        gamma_r = jnp.where(blocks_ok, gamma_r, carry.gamma_r)

        # Residuals, normalized so eps_r/eps_s are scale-free:
        #   primal r = [A^o (-) z ; A^r (-) z]   dual d = rho*(z^{l+1} (-) z^l)
        primal_res = self.consensus.residual_norm(
            jnp.concatenate(
                [
                    self.consensus.difference(a_obj, z_new),
                    self.consensus.difference(a_rob, z_new),
                ]
            )
        )
        dual_res = self.consensus.residual_norm(
            carry.rho * self.consensus.difference(z_new, carry.z)
        )
        # `_cond` reads these, and NaN compares False against every
        # tolerance -- so an unguarded NaN would silently end the round
        # loop early as "converged". Carry the last trustworthy value.
        primal_res = jnp.where(blocks_ok, primal_res, carry.primal_res)
        dual_res = jnp.where(blocks_ok, dual_res, carry.dual_res)

        # Algorithm 4 step 7: adaptive penalty, off by default and bounded
        # when on (see __init__).
        if self.rho_adapt:
            rho = jnp.where(
                primal_res > 10.0 * dual_res,
                carry.rho * 2.0,
                jnp.where(
                    dual_res > 10.0 * primal_res, carry.rho / 2.0, carry.rho
                ),
            )
            rho = jnp.clip(rho, self.rho_min, self.rho_max)
        else:
            rho = carry.rho

        if self.debug_print:
            jax.debug.print(
                "ADMM it={it} primal={p:.4f} dual={d:.4f} rho={r}",
                it=carry.it,
                p=primal_res,
                d=dual_res,
                r=rho,
            )

        new_carry = carry.replace(
            it=carry.it + 1,
            object_params=object_params,
            robot_params=robot_params,
            z=z_new,
            gamma_o=gamma_o,
            gamma_r=gamma_r,
            rho=rho,
            primal_res=primal_res,
            dual_res=dual_res,
            object_samples=object_samples,
            rng=rng,
            a_obj=a_obj,
            a_rob=a_rob,
            nonfinite=carry.nonfinite + jnp.where(blocks_ok, 0, 1),
        )
        return new_carry, rollouts

    def optimize(
        self, state: mjx.Data, params: ADMMParams
    ) -> Tuple[ADMMParams, ADMMTrajectory]:
        """Perform up to `n_admm` ADMM iterations to update the policy."""
        # Warm-start: shift the object mean and the consensus/dual
        # variables by one real control step. z shifts through the
        # consensus space (which knows the right vacated-tail value); the
        # duals are tangent vectors either way, so zero is right for them.
        object_params = params.object_params.replace(
            mean=self._shift_object(params.object_params.mean)
        )
        z = self.consensus.shift(params.z)
        gamma_o = self._shift(params.gamma_o)
        gamma_r = self._shift(params.gamma_r)
        # The other half of `_finite_or`'s job. That guard stops a bad
        # block from poisoning z; this stops an ALREADY poisoned z from
        # outliving the step that produced it. Both are needed, because
        # the guard's fallback is `carry.z` -- which is no help when z is
        # itself the NaN, and z persists across control steps, so without
        # this the run stays frozen until the receding-horizon shift
        # happens to flush the bad index out.
        #
        # Zero, elementwise: it is exactly right for the duals (tangent
        # vectors, and what `init_params` uses) and for z it means "no
        # agreed value yet", which the first round's `z_update` overwrites
        # outright with the two blocks' own proposals. A healthy warm start
        # is untouched -- `where` returns the same array.
        z = jnp.where(jnp.isfinite(z), z, 0.0)
        gamma_o = jnp.where(jnp.isfinite(gamma_o), gamma_o, 0.0)
        gamma_r = jnp.where(jnp.isfinite(gamma_r), gamma_r, 0.0)
        # `rho` is carried too, and `_admm_iteration` multiplies the
        # penalty by it, so a non-finite one is equally absorbing.
        rho = jnp.where(jnp.isfinite(params.rho), params.rho, self.rho_init)
        # Counted, not just repaired. A warm start that needed cleaning is
        # the same class of event as a skipped round -- something upstream
        # produced a non-finite value -- and silently fixing it would hide
        # exactly what the counter exists to surface.
        warm_start_repaired = ~(
            jnp.all(jnp.isfinite(params.z))
            & jnp.all(jnp.isfinite(params.gamma_o))
            & jnp.all(jnp.isfinite(params.gamma_r))
            & jnp.all(jnp.isfinite(params.rho))
            & jnp.all(jnp.isfinite(params.primal_residual))
        )

        # Warm-start the robot mean/knot-times the same way the generic
        # SamplingBasedController.optimize() does.
        tk = params.robot_params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        clamped_tk = jnp.clip(new_tk, tk[0], tk[-1])
        new_mean = self.robot_optimizer.interp_func(
            clamped_tk, tk, params.robot_params.mean[None, ...]
        )[0]
        robot_params = params.robot_params.replace(tk=new_tk, mean=new_mean)

        obj_state0 = self.task.object_state_from_robot(state)
        rng, admm_rng = jax.random.split(params.rng)

        init_carry = _ADMMCarry(
            it=jnp.asarray(0, dtype=jnp.int32),
            object_params=object_params,
            robot_params=robot_params,
            z=z,
            gamma_o=gamma_o,
            gamma_r=gamma_r,
            rho=rho,
            primal_res=jnp.where(
                jnp.isfinite(params.primal_residual),
                params.primal_residual,
                jnp.asarray(100.0, dtype=jnp.float32),
            ),
            dual_res=jnp.asarray(jnp.inf, dtype=jnp.float32),
            object_samples=params.object_samples,
            rng=admm_rng,
            # Shape/dtype seeds only: the first round overwrites both with
            # its own A. Warm-started z rather than zeros, since zero is
            # not a neutral consensus value for a pose.
            a_obj=z,
            a_rob=z,
            # Per control step, not cumulative: the runners log it every
            # step, so a sum over the log gives the run total anyway while
            # a per-step value also says WHEN.
            nonfinite=jnp.where(warm_start_repaired, 1, 0).astype(jnp.int32),
        )

        # Run one ADMM iteration unconditionally (n_admm >= 1), which also
        # gives correctly-shaped rollouts to seed the while_loop carry.
        carry1, rollouts1 = self._admm_iteration(init_carry, obj_state0, state)

        CarryAndRollouts = Tuple[_ADMMCarry, ADMMTrajectory]

        def _cond(carry_and_rollouts: CarryAndRollouts) -> jax.Array:
            carry, _ = carry_and_rollouts
            unconverged = (carry.primal_res > self.eps_r) | (
                carry.dual_res > self.eps_s
            )
            return (carry.it < self.n_admm) & unconverged

        def _body(carry_and_rollouts: CarryAndRollouts) -> CarryAndRollouts:
            carry, _ = carry_and_rollouts
            return self._admm_iteration(carry, obj_state0, state)

        final_carry, final_rollouts = jax.lax.while_loop(
            _cond, _body, (carry1, rollouts1)
        )

        new_params = ADMMParams(
            robot_params=final_carry.robot_params,
            object_params=final_carry.object_params,
            z=final_carry.z,
            gamma_o=final_carry.gamma_o,
            gamma_r=final_carry.gamma_r,
            rho=final_carry.rho,
            primal_residual=final_carry.primal_res,
            dual_residual=final_carry.dual_res,
            object_samples=final_carry.object_samples,
            a_obj=final_carry.a_obj,
            a_rob=final_carry.a_rob,
            nonfinite_rounds=final_carry.nonfinite,
            rng=final_carry.rng,
        )
        return new_params, final_rollouts

    def nominal_plans(
        self, state: mjx.Data, params: ADMMParams
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Both blocks' predicted object trajectories, for visualization.

        The consensus is an agreement about a wrench, easier to read as
        motion than as a number: these are the two object trajectories
        that wrench is negotiated over. Where they coincide the blocks
        agree; where they diverge is what the primal residual measures.

        Deliberately not folded into `optimize`'s return, so a caller that
        never asks pays nothing extra.

        Returns:
            `(object_plan, robot_plan, robot_trace)`: what the object
            block intends and what the robot block's controls would
            produce for the object, each (H, object_state_dim); and the
            end-effector's own physical position along that rollout,
            (H, 3).
        """
        obj_state0 = self.task.object_state_from_robot(state)
        object_plan = self.object_subproblem.nominal_plan(
            obj_state0, params.object_params
        )
        robot_plan, robot_trace = self.robot_subproblem.nominal_plan(
            state, params.robot_params
        )
        return object_plan, robot_plan, robot_trace

    def local_goal(self, state: mjx.Data, params: ADMMParams) -> jax.Array:
        """The point of the object block's plan the robot aims at.

        Exactly the value `RobotSubproblem._eval_rollouts_one` hands the
        task as `local_goal`, resolved through the same
        `local_goal_from_plan` -- the plan endpoint x^{o*}_H by default,
        a pursuit carrot where the task overrides it. What the task then
        tracks is still the task's own business (`PushT.tracking_goal`
        ignores this with `local_goal=False`, and snaps back to the
        global goal near the goal even with it on).

        Cheap: H steps of the injected object backend, no sampling, no
        robot rollout. Kept separate from `nominal_plans` so a caller that
        wants only the endpoint doesn't pay for the robot rollout too.
        """
        obj_state0 = self.task.object_state_from_robot(state)
        plan = self.object_subproblem.nominal_plan(
            obj_state0, params.object_params
        )
        return self.task.local_goal_from_plan(plan, obj_state0)

    def nominal_trace(self, state: mjx.Data, params: ADMMParams) -> jax.Array:
        """The robot block's chosen end-effector path, (H, 3).

        Overridden not to change the answer -- `ADMMParams.mean` already
        delegates to the robot block's -- but to reuse the rollout
        `nominal_plan` already does rather than paying for a second one.
        """
        _, robot_trace = self.robot_subproblem.nominal_plan(
            state, params.robot_params
        )
        return robot_trace

    def sample_knots(self, params: ADMMParams) -> Tuple[jax.Array, ADMMParams]:
        """Not used -- `ADMM.optimize()` overrides the generic template."""
        raise NotImplementedError(
            "ADMM overrides optimize() directly; sample_knots/update_params "
            "are not used. See ADMM.optimize()."
        )

    def update_params(
        self, params: ADMMParams, rollouts: ADMMTrajectory
    ) -> ADMMParams:
        """Not used -- `ADMM.optimize()` overrides the generic template."""
        raise NotImplementedError(
            "ADMM overrides optimize() directly; sample_knots/update_params "
            "are not used. See ADMM.optimize()."
        )
