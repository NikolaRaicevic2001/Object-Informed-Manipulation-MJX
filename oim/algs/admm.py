"""ADMM-coordinated object/robot consensus planning.

Implements the object-informed MPPI formulation, where a hierarchical
object-level and robot-level sampling-based sub-optimizer reach consensus on
a shared consensus variable z_t (e.g. a contact wrench) via ADMM
(Alternating Direction Method of Multipliers). See the paper "Object-Informed
Model Predictive Path Integral Control for Non-Prehensile Robot Manipulation"
(Algorithm 4) for the underlying math.

Both the object- and robot-level sub-optimizers are pluggable: any
`oim.alg_base.SamplingBasedController` (MPPI, CBO, CEM, ...) can be
injected as either, since this module only ever calls the two methods every
such controller implements (`sample_knots`/`update_params`).
"""

from abc import ABC, abstractmethod
from functools import partial
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from oim.alg_base import SamplingBasedController, Trajectory
from oim.objects.planar_pushing import wrap_angle
from oim.task_base import ConsensusTask


class ConsensusSpace(ABC):
    """A consensus variable space for ADMM, shared between two subproblems.

    The consensus variable z_t must be *the same physical quantity, in the
    same units and frame*, on both blocks -- the object block's proposed
    value A^o and the robot block's realized value A^r. This class owns all
    the math that touches z, so both blocks provably use identical
    definitions rather than each hand-rolling their own copy.
    """

    dim: int

    @abstractmethod
    def normalize(self, v: jax.Array) -> jax.Array:
        """Map a consensus-space *tangent* value to dimensionless units.

        Used for both the penalty and the residual norms, so that `rho`,
        `eps_r` and `eps_s` are scale-free and comparable against the task
        costs regardless of the physical units z happens to carry.
        """

    def difference(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Return a (-) b: the tangent vector from b to a.

        Plain subtraction when Z is a vector space, which is the paper's
        case (Z = R^{p^o}, the wrench). A consensus space on a manifold --
        `PoseConsensus`, where Z = SE(2) -- overrides this. Every place the
        ADMM iteration subtracts two consensus values routes through here,
        so one override reaches the penalty, both residuals, the dual
        update and the z-update at once, rather than each re-deriving it.
        """
        return a - b

    def increment(self, base: jax.Array, tangent: jax.Array) -> jax.Array:
        """Return base (+) tangent: back onto Z from the tangent space.

        The inverse of `difference`, and the projection Pi_Z of eq. 14 in
        the paper's notation: for a vector space it is plain addition and
        Pi_Z is the identity, exactly as the paper states.
        """
        return base + tangent

    def shift(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift of a consensus-valued sequence.

        Zero-filling the vacated tail is right when zero is a meaningful
        consensus value (no wrench is a valid plan). A space where it is
        not -- the pose (0, 0, 0) is the world origin, not "no pose" --
        overrides this.
        """
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)

    def penalty_cost(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array, rho: jax.Array
    ) -> jax.Array:
        """(rho/2) * ||(actual (-) z) + dual||^2, in normalized units.

        Collapses only the consensus dimension; the caller sums over the
        horizon and samples. Shared by both blocks (paper eq. 25-26).

        `actual (-) z` is a tangent vector and `dual` is already one, so
        the sum stays in the tangent space and is not re-projected.

        `rho` may be a scalar (paper's Algorithm 4) or a per-dimension
        vector broadcastable against the consensus dim -- the paper's
        anisotropic P = diag(rho_f, rho_f, rho_tau). Weighting each squared
        term before summing is what makes the vector case differ from the
        scalar one; for a scalar it reduces to the same value either order.
        """
        diff = self.normalize(self.difference(actual, z) + dual)
        return 0.5 * jnp.sum(rho * diff**2, axis=-1)

    def residual_norm(self, v: jax.Array) -> jax.Array:
        """Norm of a residual, in the same normalized units as the penalty.

        `v` is already a tangent vector -- the caller forms it with
        `difference` -- so no wrapping happens here.
        """
        return jnp.linalg.norm(self.normalize(v))

    def z_update(
        self,
        a_o: jax.Array,
        a_r: jax.Array,
        dual_o: jax.Array,
        dual_r: jax.Array,
        base: jax.Array,
    ) -> jax.Array:
        """Paper eq. 27 with N = 2, taken about `base`.

            z <- base (+) 0.5*[(a_o (-) base) + y_o + (a_r (-) base) + y_r]

        For a vector space this is *identically* the paper's plain average
        0.5*(a_o + y_o + a_r + y_r) -- the base point cancels -- so
        introducing it changes nothing there. It is needed only when Z is a
        manifold, where the four terms cannot be averaged directly and must
        be lifted to a common tangent space first.

        Args:
            a_o: The object block's extracted value A^o.
            a_r: The robot block's extracted value A^r.
            dual_o: The object block's scaled dual (a tangent vector).
            dual_r: The robot block's scaled dual.
            base: The point to linearize about; the previous z.
        """
        tangent = 0.5 * (
            self.difference(a_o, base)
            + dual_o
            + self.difference(a_r, base)
            + dual_r
        )
        return self.increment(base, tangent)

    @abstractmethod
    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual step y <- y + (A x (-) z), paper eq. 28."""


class WrenchConsensus(ConsensusSpace):
    """Planar contact wrench consensus z_t = [f_x, f_y, tau]; arrays (H, 3).

    z is the wrench the robot applies *to the object*, expressed in the world
    frame about the object's pose origin, in Newtons and Newton-metres. Both
    blocks must report it that way:

    * A^o: the object planner's proposed wrench, read directly off its own
      decision variable (paper eq. 23).
    * A^r: the wrench the robot's rolled-out motion actually imparts on the
      object, read from the simulator's contact forces (paper eq. 23).

    `scale` sets the characteristic magnitude used to normalize the penalty
    and residuals. The natural choice is the friction-cone limit
    (mu*m*g for forces, r*mu*m*g for the torque), which is exactly the
    inverse of the limit-surface compliance D -- i.e. the normalized residual
    measures disagreement as a *fraction of the maximum transmissible
    wrench*. Without this the penalty (forces ~10 N, squared -> ~10^2)
    dwarfs the task costs (~1) and the robot ends up optimizing wrench
    matching to the exclusion of actually reaching the object.
    """

    dim = 3

    def __init__(self, max_dual: float, scale: jax.Array = None) -> None:
        """Set the dual anti-windup clip and the normalization scale.

        Args:
            max_dual: Maximum magnitude of the scaled dual variables, in the
                same physical units as z (anti-windup; an extension beyond
                the paper, which leaves the duals unbounded).
            scale: Per-dimension characteristic magnitude of z. Defaults to
                ones (no normalization).
        """
        self.max_dual = max_dual
        self.scale = jnp.ones(self.dim) if scale is None else jnp.asarray(scale)

    def normalize(self, v: jax.Array) -> jax.Array:
        """Divide through by the per-dimension characteristic magnitude."""
        return v / self.scale

    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual + (actual - z), clipped to [-max_dual, max_dual]."""
        return jnp.clip(
            dual + self.difference(actual, z), -self.max_dual, self.max_dual
        )


class PoseConsensus(ConsensusSpace):
    """SE(2) object-pose consensus z_t = [x, y, theta]; arrays (H, 3).

    The alternative to `WrenchConsensus`: the two blocks negotiate *where
    the object should be* over the horizon rather than *what wrench should
    act on it*. Both extraction maps then report the same directly-observed
    state:

    * A^o: the pose trajectory the object block's wrench sequence induces
      through the limit surface (paper eq. 5), which is that linear
      relation integrated -- so A^o is **affine** in U^o, and the paper's
      assumption that A_i is a linear extraction map holds exactly here
      rather than by the triviality of a selection matrix.
    * A^r: the object's pose along the robot block's rollout, read straight
      out of the simulator state. No twist inversion, no clip, and no
      per-embodiment estimator -- which matters because an articulated arm
      cannot read the contact wrench off a single pair of DOFs at all and
      is forced onto model inversion, whose amplification of `qvel` noise
      by D^-1 is measurably the larger half of the two blocks'
      disagreement.

    Unlike the wrench space, SE(2) is *not* a vector space, so the paper's
    remark that Pi_Z reduces to the identity does not hold: differences
    wrap theta into (-pi, pi], the duals live in the tangent space (they
    are twists, not poses), and eq. 27's average is taken about a base
    point. The wrap is not cosmetic -- the tabletop goal is theta = pi,
    sitting exactly on the branch cut, so an unwrapped subtraction would
    report a 2*pi disagreement precisely at the goal.
    """

    dim = 3

    def __init__(self, max_dual: jax.Array, scale: jax.Array = None) -> None:
        """Set the dual anti-windup clip and the normalization scale.

        Args:
            max_dual: Maximum magnitude of the scaled duals, per dimension
                or as a scalar, in the same units as z (metres, radians).
            scale: Per-dimension characteristic magnitude of a pose
                difference. The natural choice is
                `(r_body, r_body, 1.0)` for the object's bounding radius,
                so a normalized residual of 1 means "the two blocks
                disagree by one body radius, or by one radian". Defaults to
                ones (no normalization).
        """
        self.max_dual = jnp.asarray(max_dual)
        self.scale = jnp.ones(self.dim) if scale is None else jnp.asarray(scale)

    def normalize(self, v: jax.Array) -> jax.Array:
        """Divide through by the per-dimension characteristic magnitude."""
        return v / self.scale

    def difference(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Return a (-) b with the heading wrapped to (-pi, pi]."""
        d = a - b
        return d.at[..., 2].set(wrap_angle(d[..., 2]))

    def increment(self, base: jax.Array, tangent: jax.Array) -> jax.Array:
        """Return base (+) tangent, re-wrapping the heading. This is Pi_Z."""
        out = base + tangent
        return out.at[..., 2].set(wrap_angle(out[..., 2]))

    def shift(self, seq: jax.Array) -> jax.Array:
        """Shift by one and repeat the last pose.

        Zero-filling would put the vacated tail at the *world origin*,
        which is a specific pose rather than the absence of one, and would
        pull the last step of every warm-started horizon toward it.
        """
        return jnp.concatenate([seq[1:], seq[-1:]], axis=0)

    def dual_update(
        self, actual: jax.Array, z: jax.Array, dual: jax.Array
    ) -> jax.Array:
        """Dual + (actual (-) z), clipped to [-max_dual, max_dual]."""
        return jnp.clip(
            dual + self.difference(actual, z), -self.max_dual, self.max_dual
        )


def make_object_shim(task: ConsensusTask, dt: float) -> Any:
    """Build a lightweight duck-typed task for the object-level optimizer.

    The object subproblem samples in the object block's own action space
    (dimension `task.object_action_dim`, which defaults to
    `task.consensus_dim`), not robot actuator space, so a
    `SamplingBasedController` (e.g. `MPPI`) built for the object side needs
    its own small task-like object to construct against. `ObjectSubproblem`
    never calls anything on this shim besides `.model.nu`/`.dt`/`.u_min`/
    `.u_max`; all real cost and dynamics come from the real task's
    `object_*` methods.

    Args:
        task: The real task, used to read `object_action_dim` and bounds.
        dt: The planning timestep for the object-level optimizer.

    Returns:
        A duck-typed object exposing `.model.nu`, `.dt`, `.u_min`, `.u_max`.
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

    Zero-filling the vacated tail is right when the decision *is* the
    consensus value (no wrench is a valid plan), but wrong for a structured
    action space, where the zero vector need not be a feasible action at
    all -- a zero contact point is the object's own origin, which is not on
    its boundary. There the last value is held instead, which is always
    feasible since it came from the previous solution.

    Module-level rather than a method so the standalone object-level driver
    (`oim.worlds.object_only.build`) warm-starts *identically* to the ADMM
    one. It is the
    only part of a control step the object block does not own, so a second
    copy of it is the one place the two could silently diverge.

    Args:
        task: The task, asked whether zero is a feasible action.
        seq: The decision sequence, (H, action_dim).

    Returns:
        The shifted sequence, same shape.
    """
    if task.initial_object_action() is None:
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)
    return jnp.concatenate([seq[1:], seq[-1:]], axis=0)


@dataclass
class ADMMTrajectory(Trajectory):
    """Trajectory with the realized consensus value A^r(U^r)_t at each step."""

    consensus_values: jax.Array


class RobotRollout(ABC):
    """How the robot block advances its state by one planning step.

    This is the *only* place the robot subproblem is tied to a particular
    simulator. Everything else in this module -- the ADMM loop, the
    consensus/dual/penalty math, the object subproblem -- is already
    backend-agnostic, so swapping this one object is what lets the same
    `ADMM` class drive both an MJX scene and the analytic 2D world in
    `oim.worlds.sim2d`.

    Implementations must be pure functions of (model, state, control): they
    run inside `jax.lax.scan` under `vmap` and `jit`.
    """

    @abstractmethod
    def step(self, model: Any, state: Any, control: jax.Array) -> Any:
        """Advance the robot-side state by one planning timestep.

        Args:
            model: The (possibly domain-randomized) model for this rollout.
            state: The current robot-side state.
            control: The control action u^r_t.

        Returns:
            The next state, of the same pytree structure.
        """


class MJXRollout(RobotRollout):
    """The default backend: one `mjx.step` of a MuJoCo MJX model."""

    def step(
        self, model: mjx.Model, state: mjx.Data, control: jax.Array
    ) -> mjx.Data:
        """Set the control and step the MJX model."""
        return mjx.step(model, state.replace(ctrl=control))


class ObjectSubproblem:
    """Object-level ADMM subproblem: closed-form dynamics, pluggable optimizer.

    Wraps any `SamplingBasedController` (built against a `make_object_shim`
    task) to sample/reweight consensus-space decisions w^o_t. Rollout and
    cost use `task.object_dynamics`/`object_running_cost`/`object_terminal_cost`
    directly, since there's no MJX model on the object side.
    """

    def __init__(
        self,
        task: ConsensusTask,
        optimizer: SamplingBasedController,
        consensus: ConsensusSpace,
        proximal_weight: float,
    ) -> None:
        """Pair the task with its injected sampling optimizer.

        Args:
            task: The real task, providing object-level dynamics/costs.
            optimizer: Any `SamplingBasedController` built against a
                `make_object_shim(task, ...)` task.
            consensus: The consensus space used for the ADMM penalty.
            proximal_weight: Weight (gamma) on the proximal term (eq. 24)
                that anchors this ADMM iteration's update to the previous one.
        """
        self.task = task
        self.optimizer = optimizer
        self.consensus = consensus
        self.proximal_weight = proximal_weight

    def _rollout(
        self, obj_state0: jax.Array, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Scan the closed-form dynamics over an (H, action_dim) sequence.

        The action -> wrench map runs *inside* the scan, since a contact
        wrench depends on the object's current pose, not just on the action.
        With the default (identity-style) map this is exactly the old
        `actions * object_action_scale()`, evaluated one step at a time.

        Returns:
            The object states x^o_1..x^o_H; the wrenches w^o_0..w^o_{H-1}
            that produced them (the effort term in `object_running_cost`
            needs the wrench itself, whatever the consensus variable is);
            and the extracted consensus values A^o_0..A^o_{H-1}.

            The last two coincide when the consensus variable *is* the
            wrench (`ConsensusTask.object_consensus`'s default, paper
            eq. 24) and differ when it is the pose.
        """

        def step(
            obj_state: jax.Array, action: jax.Array
        ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
            w = self.task.object_action_to_consensus(obj_state, action)
            new_state = self.task.object_dynamics(obj_state, w)
            # A^o is read after the step, matching the robot block, which
            # reads A^r after `rollout.step` -- so index t is the value at
            # t+1 on both sides and the two are directly comparable.
            a_o = self.task.object_consensus(new_state, w)
            return new_state, (new_state, w, a_o)

        _, (states, ws, a_o) = jax.lax.scan(step, obj_state0, actions)
        return states, ws, a_o

    def optimize(
        self,
        obj_state0: jax.Array,
        params: Any,
        z: jax.Array,
        dual_o: jax.Array,
        rho: jax.Array,
        prev_knots: jax.Array,
        noise_scale: jax.Array,
        rng: jax.Array,
    ) -> Tuple[Any, jax.Array, jax.Array, jax.Array]:
        """Run `optimizer.iterations` MPPI-style passes against a fixed target.

        Args:
            obj_state0: The object's current configuration x^o_0.
            params: The object optimizer's current policy parameters.
            z: The current consensus variable, (H, dim).
            dual_o: The object block's scaled dual variable, (H, dim).
            rho: The current ADMM penalty weight.
            prev_knots: The previous ADMM iteration's knots, for the
                proximal term (eq. 24).
            noise_scale: Extra residual-scaled exploration noise (Alg. 4
                step 8, generalized -- see module docstring).
            rng: Random key for the extra noise and optimizer iterations.

        Returns:
            Updated params, the object's proposed decision W^o = params.mean
            (in physical units), its own nominal state reference under
            those decisions, and the last iteration's sampled object
            trajectories, (num_samples, H, object_state_dim), for
            visualization.
        """
        opt = self.optimizer

        def _scan_body(params: Any, rng_i: jax.Array) -> Tuple[Any, jax.Array]:
            noise_rng, sample_rng = jax.random.split(rng_i)
            # The task may own the proposal distribution (e.g. contact points
            # that must stay on the object's boundary); otherwise the
            # injected optimizer samples as usual. `num_samples` is read
            # defensively: not every SamplingBasedController exposes one
            # (Evosax carries its population size inside the ES state), and
            # those are exactly the ones that must keep their own sampler.
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
            noise = noise_scale * jax.random.normal(noise_rng, knots.shape)
            knots = jnp.clip(knots + noise, opt.task.u_min, opt.task.u_max)
            knots = self.task.project_object_action(knots, obj_state0)

            states, ws, a_o = jax.vmap(self._rollout, in_axes=(None, 0))(
                obj_state0, knots
            )
            # J_o: dt-weighted running cost + terminal cost, matching the
            # robot block's convention so the two blocks stay balanced.
            running = self.task.dt * jax.vmap(
                jax.vmap(self.task.object_running_cost)
            )(states[:, :-1], ws[:, :-1])
            terminal = jax.vmap(self.task.object_terminal_cost)(states[:, -1])

            # Proximal term (gamma/2)||U^o - U^{o,(l)}||^2, paper eq. 24.
            proximal = (
                0.5
                * self.proximal_weight
                * jnp.sum((knots - prev_knots) ** 2, axis=(-2, -1))
            )
            # Rate penalty: a sequence-level term, so it is folded in here
            # beside the proximal one rather than into `running`, which is
            # scored per step. Anchored to the wrench the previous solve
            # already intended for this step (`prev_knots[0]`), so it
            # charges for changing course across control steps as well as
            # within the horizon -- and needs no extra plumbing to do it,
            # which is why both worlds get it identically.
            w_prev = self.task.object_action_to_consensus(
                obj_state0, prev_knots[0]
            )
            rate = jax.vmap(self.task.object_rate_cost, in_axes=(0, None))(
                ws, w_prev
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
            # Re-project the *aggregated* nominal, not just the samples. An
            # optimizer update is generally a weighted average of samples,
            # and averaging feasible points need not be feasible: means of
            # contact points either side of a corner land inside the
            # object (measured: up to 1.5 mm for the T footprint). Leaving
            # that unprojected would let an infeasible action anchor the
            # proximal term, seed the next round's sampling, and carry into
            # the warm start. No-op for an unconstrained action space.
            params = params.replace(
                mean=self.task.project_object_action(params.mean, obj_state0)
            )
            return params, states

        rngs = jax.random.split(rng, opt.iterations)
        params, all_states = jax.lax.scan(_scan_body, params, rngs)
        # The last iteration's sample population, for visualization -- the
        # candidate object trajectories this block considered before
        # settling on `params.mean`. Free: `states` is already computed for
        # `object_running_cost` above, this just keeps the last iteration's
        # instead of discarding it.
        object_samples = all_states[-1]

        # A^o for the nominal (paper eq. 24), recovered by rolling the
        # nominal actions out -- necessary for any non-trivial action
        # parameterization, and for a pose consensus variable in every case.
        nominal = self.task.project_object_action(params.mean, obj_state0)
        ref_states, _, a_obj = self._rollout(obj_state0, nominal)
        return params, a_obj, ref_states, object_samples

    def nominal_plan(self, obj_state0: jax.Array, params: Any) -> jax.Array:
        """The object trajectory this block currently intends, x^o_1..x^o_H.

        The same rollout `optimize` ends with, recomputed from the stored
        nominal so a caller can ask for it without re-solving. Cheap: H
        closed-form dynamics steps, no sampling and no simulator.

        Args:
            obj_state0: The object's current configuration x^o_0.
            params: The object optimizer's policy parameters.

        Returns:
            Object states of shape (H, object_state_dim).
        """
        nominal = self.task.project_object_action(params.mean, obj_state0)
        states, _, _ = self._rollout(obj_state0, nominal)
        return states


class RobotSubproblem:
    """Robot-level ADMM subproblem: real MJX contact, pluggable optimizer.

    Reads `.model`/`.randomized_axes`/`.risk_strategy`/`.interp_func`/
    `.num_randomizations`/`.ctrl_steps`/`.iterations` off the *injected*
    optimizer generically (never off `self`), so domain randomization and
    risk strategies keep working no matter which `SamplingBasedController`
    subclass is plugged in.
    """

    def __init__(
        self,
        task: ConsensusTask,
        optimizer: SamplingBasedController,
        consensus: ConsensusSpace,
        proximal_weight: float,
        rollout: Optional[RobotRollout] = None,
    ) -> None:
        """Pair the task with its injected sampling optimizer.

        Args:
            task: The real task (also a `Task`), providing the MJX model,
                the robot-level task cost, and the A^r extraction map.
            optimizer: Any `SamplingBasedController` built against `task`.
            consensus: The consensus space. The ADMM penalty is added here,
                by the same `consensus.penalty_cost` the object block uses,
                so both blocks are guaranteed to score the consensus
                variable identically.
            proximal_weight: Weight (gamma) on the proximal term (eq. 25).
            rollout: How to advance the robot state one step. Defaults to
                `MJXRollout`, i.e. a MuJoCo MJX scene.
        """
        self.task = task
        self.optimizer = optimizer
        self.consensus = consensus
        self.proximal_weight = proximal_weight
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
        # x^{o*}_H: where the object block's own plan ends. Free to compute
        # -- `obj_ref` is already here in full, broadcast across samples,
        # and the scan below only slices it per step. Handed to the task on
        # every call rather than behind a flag of this layer's own: what a
        # "goal" means is the task's business, and the ADMM layer has no
        # opinion beyond supplying the plan it already produced.
        local_goal = obj_ref[-1]

        def _scan_fn(
            x: mjx.Data,
            inputs: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array, jax.Array]]:
            u, z_t, dual_t, ref_t = inputs
            x = self.rollout.step(model, x, u)
            # J_r: the task's own cost, dt-weighted per oim convention.
            cost = self.optimizer.dt * self.task.robot_running_cost(
                x, u, ref_t, local_goal
            )
            # A^r: the wrench the robot's motion actually imparts on the
            # object, read from the simulator (paper eq. 23).
            consensus_val = self.task.realized_consensus(x)
            # ADMM penalty (rho/2)||A^r(U^r)_t - z_t + y^r_t||^2 (eq. 25).
            # Added here, not inside the task, so it is literally the same
            # function the object block uses.
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
            self.task.robot_terminal_cost(final_state, local_goal) + proximal
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

        z/dual_r/rho/obj_ref are shared (the fixed target every sample is
        scored against), and the proximal anchor is threaded through too.
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
        noise_scale: jax.Array,
        rng: jax.Array,
    ) -> Tuple[Any, ADMMTrajectory]:
        """Run `optimizer.iterations` passes against a fixed target."""
        opt = self.optimizer
        tk = params.tk

        def _scan_body(
            params: Any, rng_i: jax.Array
        ) -> Tuple[Any, ADMMTrajectory]:
            knots, params = opt.sample_knots(params)
            noise_rng, dr_rng = jax.random.split(rng_i)
            noise = noise_scale * jax.random.normal(noise_rng, knots.shape)
            knots = jnp.clip(knots + noise, self.task.u_min, self.task.u_max)
            rollouts = self.rollout_with_randomizations(
                state, tk, knots, dr_rng, z, dual_r, rho, obj_ref, prev_knots
            )
            params = opt.update_params(params, rollouts)
            return params, rollouts

        rngs = jax.random.split(rng, opt.iterations)
        params, rollouts = jax.lax.scan(_scan_body, params, rngs)
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final

    def nominal_realized_consensus(
        self, state: mjx.Data, params: Any
    ) -> jax.Array:
        """Re-simulate `params.mean` alone to read the realized consensus.

        The wrapped optimizer's update blends all samples into a new mean,
        so there's no single winning rollout already computed to read A^r
        off of. Uses `self.task.model` directly (not a domain-randomization
        ensemble), since there's a single state here.
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

        The robot block's counterpart to `ObjectSubproblem.nominal_plan`,
        and the reason the pair is worth looking at: both are trajectories
        of the *same* object, so their disagreement is the consensus
        residual made spatial rather than scalar. The trace-site output is
        for visualizing the robot's *physical* path (e.g. the end-effector),
        which is a different thing from what it does to the object.

        Unlike the object block's, this one costs a real rollout -- H steps
        of the simulator -- but only one, against the `num_samples` the
        block already ran, so it is a fraction of a percent of a control
        step. The trace sites are read off the same rollout, at no
        additional cost.

        Args:
            state: The robot-side state to roll out from.
            params: The robot optimizer's policy parameters.

        Returns:
            `(obj_states, trace_sites)`: object states, shape
            (H, object_state_dim), and the task's trace site position at
            each step, shape (H, 3).
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
        primal_residual: The last ADMM iteration's primal residual, carried
            across real control steps to seed exploration-noise annealing.
        dual_residual: The last ADMM iteration's dual residual. Not used by
            any control-flow logic (only `primal_residual` feeds the noise
            anneal and the adaptive-`rho` rule reads both from `_ADMMCarry`
            directly) -- carried here purely so callers can log/plot it.
        object_samples: The object block's last iteration's sampled
            trajectories, (num_samples, H, object_state_dim). Not used by
            any control-flow logic either -- carried here purely so
            callers can visualize it, the same way `primal_residual` is.
        rng: PRNG key for the ADMM-level exploration noise.
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
    rng: jax.Array

    @property
    def tk(self) -> jax.Array:
        """Knot times, delegated to the robot params (queried externally)."""
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
    a_obj_ema: jax.Array
    a_rob_ema: jax.Array


class ADMM(SamplingBasedController):
    """ADMM-coordinated object/robot consensus planning (Algorithm 4).

    Coordinates an object-level and a robot-level sampling-based
    sub-optimizer (any `SamplingBasedController`, e.g. `MPPI` or `CBO`) to
    reach consensus on a task-defined consensus variable (e.g. a contact
    wrench) every real control step.

    Because the ADMM iteration is fundamentally different from the generic
    sample/rollout/update template used by every other algorithm (it runs two
    full sub-optimizations plus consensus math per iteration, with an early
    exit), `optimize()` is overridden entirely rather than fitting the
    `sample_knots`/`update_params` template; those two methods are not used
    and raise `NotImplementedError`.
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
        noise_min: float = 0.0,
        noise_kappa: float = 0.0,
        noise_max: Optional[float] = None,
        rollout: Optional[RobotRollout] = None,
        debug_print: bool = False,
        consensus_alpha: float = 1.0,
    ) -> None:
        """Build the ADMM controller from two pre-built sub-optimizers.

        Args:
            task: A task implementing both `Task` and `ConsensusTask`.
            robot_optimizer: A `SamplingBasedController` built against
                `task`, used for the robot-level subproblem. Its
                `plan_horizon`/`num_knots`/`spline_type`/etc. become this
                controller's own (so `run_interactive` can drive it
                directly); its `ctrl_steps` must equal
                `object_optimizer.num_knots` (the consensus horizon H).
            object_optimizer: A `SamplingBasedController` built against
                `make_object_shim(task, ...)`, used for the object-level
                subproblem. Its `num_knots` sets the consensus horizon H.
            consensus: The consensus space (e.g. `WrenchConsensus`).
            n_admm: The maximum number of ADMM iterations per real step.
            eps_r: Primal residual tolerance for early exit.
            eps_s: Dual residual tolerance for early exit.
            proximal_weight: Weight (gamma) on the proximal term (eq. 24-25)
                that anchors each ADMM iteration to the previous one.
            rho_init: The ADMM penalty weight. Held fixed at this value
                unless `rho_adapt` is set.
            rho_adapt: Whether to apply the residual-balancing rule
                (Algorithm 4 step 7), which doubles `rho` when the primal
                residual dominates and halves it when the dual does.

                Off, so the configured `rho` is the `rho` the run uses.
                The rule is multiplicative and `rho` persists across real
                control steps, so a residual imbalance that does not
                resolve -- which is exactly what a block whose target is
                infeasible produces -- compounds every iteration: measured
                drifting 10 -> 5 -> 2.5 within six control steps, with no
                configuration anywhere naming 2.5. Balancing residuals is
                only meaningful when both can in fact be driven down; when
                one block cannot reach the consensus set at all, the rule
                reads a structural gap as a tuning error and quietly
                changes the algorithm out from under the config file.
            rho_bound_factor: When `rho_adapt` is on, how far the rule may
                move `rho` from `rho_init`, as a multiplicative factor
                either way. Ignored when `rho_adapt` is off.
            noise_min: Minimum extra exploration-noise scale.
            noise_kappa: Scale of extra exploration noise relative to the
                primal residual (Algorithm 4 step 8, generalized -- see the
                module docstring for why this differs from the paper).
            noise_max: Maximum extra exploration-noise scale. Since the
                residual has no natural upper bound, an uncapped
                `noise_kappa * residual` can create a runaway feedback loop
                (more noise -> more disagreement -> larger residual -> more
                noise). Defaults to `noise_min`, i.e. annealing is inert
                (a constant noise floor) unless explicitly raised.
            rollout: How the robot block advances its state one step.
                Defaults to `MJXRollout` (a MuJoCo MJX scene); pass
                `oim.worlds.sim2d.Analytic2DRollout` to drive the 2D world with
                this same controller.
            debug_print: Whether to print the residuals and penalty weight
                every ADMM iteration. Off by default, for two reasons: it
                is a host callback inside the compiled loop, so it costs a
                device synchronization per iteration, and that sync falls
                inside the `compute_time` the runners record -- depressing
                the planning rate they report; and at `n_admm` lines per
                control step it buries the closed loop's own per-step
                summary, which carries the same residuals alongside the
                goal errors. Turn it on to debug the consensus iteration
                itself.
            consensus_alpha: EMA weight on A^o/A^r *across ADMM rounds*
                (1.0 = raw, matching the paper). Each round's A^o/A^r is a
                single noisy resampling estimate (one MPPI pass over a
                freshly-sampled batch), not a converged proposal, so the
                disagreement it feeds into z/the residuals is dominated by
                resampling variance between rounds rather than by the
                blocks actually disagreeing. Smoothing across rounds (not
                within one rollout's own horizon -- that axis carries a
                different, much smaller noise source) targets that
                variance directly.
        """
        if n_admm < 1:
            raise ValueError("n_admm must be at least 1")
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
        # keeps its ratio: bounding a vector rho by one scalar band would
        # let the components collapse onto each other.
        self.rho_min = jnp.asarray(rho_init) / rho_bound_factor
        self.rho_max = jnp.asarray(rho_init) * rho_bound_factor
        self.noise_min = noise_min
        self.noise_kappa = noise_kappa
        self.noise_max = noise_min if noise_max is None else noise_max
        self.debug_print = debug_print
        self.consensus_alpha = consensus_alpha

        self.object_subproblem = ObjectSubproblem(
            task, object_optimizer, consensus, proximal_weight
        )
        self.robot_subproblem = RobotSubproblem(
            task, robot_optimizer, consensus, proximal_weight, rollout=rollout
        )

        # Alias bookkeeping attributes off robot_optimizer, rather than
        # calling `SamplingBasedController.__init__`, which would rebuild a
        # redundant (and potentially mismatched) domain-randomization
        # ensemble. This is what makes `ADMM` a drop-in for `run_interactive`.
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
            # must stay finite since it's multiplied into the noise scale
            # (an actual inf would immediately turn every sampled knot nan).
            primal_residual=jnp.asarray(100.0, dtype=jnp.float32),
            dual_residual=jnp.asarray(100.0, dtype=jnp.float32),
            # Placeholder: never read as input (only ever overwritten by
            # `_admm_iteration`'s real computation), so its shape here
            # doesn't need to match -- unlike the other fields above, which
            # seed real ADMM math on the first call.
            object_samples=jnp.zeros((1, 1, 1), dtype=jnp.float32),
            rng=jax.random.key(seed),
        )

    def _shift(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift: seq[t] <- seq[t+1], last slot <- 0."""
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)

    def _shift_object(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift for the object block's decision.

        Delegates to `shift_object_actions`, which
        `oim.worlds.object_only.build` also
        calls, so the standalone object-level driver warm-starts exactly as
        the ADMM one does.
        """
        return shift_object_actions(self.task, seq)

    def _admm_iteration(
        self, carry: _ADMMCarry, obj_state0: jax.Array, state: mjx.Data
    ) -> Tuple[_ADMMCarry, ADMMTrajectory]:
        """One ADMM iteration: object update -> robot update -> consensus."""
        rng, obj_rng, rob_rng = jax.random.split(carry.rng, 3)
        noise_scale = jnp.clip(
            self.noise_kappa * carry.primal_res, self.noise_min, self.noise_max
        )
        prev_object_knots = carry.object_params.mean
        prev_robot_knots = carry.robot_params.mean

        object_params, a_obj, obj_ref, object_samples = (
            self.object_subproblem.optimize(
                obj_state0,
                carry.object_params,
                carry.z,
                carry.gamma_o,
                carry.rho,
                prev_object_knots,
                noise_scale,
                obj_rng,
            )
        )
        robot_params, rollouts = self.robot_subproblem.optimize(
            state,
            carry.robot_params,
            carry.z,
            carry.gamma_r,
            carry.rho,
            obj_ref,
            prev_robot_knots,
            noise_scale,
            rob_rng,
        )
        a_rob = self.robot_subproblem.nominal_realized_consensus(
            state, robot_params
        )

        # EMA across ADMM rounds (not within one rollout's horizon -- see
        # `consensus_alpha`'s docstring): each round's raw A^o/A^r is a
        # single noisy resampling estimate, so consensus is computed
        # against a smoothed A^o/A^r instead of the raw one. Written as an
        # increment along the tangent from the *raw* value so it is exact
        # on a manifold too; for a vector space it is algebraically the
        # plain `alpha*raw + (1-alpha)*prev`, and at alpha = 1.0 (the
        # shipped default) it is the raw value either way.
        blend = 1.0 - self.consensus_alpha
        a_obj_ema = self.consensus.increment(
            a_obj, blend * self.consensus.difference(carry.a_obj_ema, a_obj)
        )
        a_rob_ema = self.consensus.increment(
            a_rob, blend * self.consensus.difference(carry.a_rob_ema, a_rob)
        )

        z_new = self.consensus.z_update(
            a_obj_ema, a_rob_ema, carry.gamma_o, carry.gamma_r, carry.z
        )
        gamma_o = self.consensus.dual_update(a_obj_ema, z_new, carry.gamma_o)
        gamma_r = self.consensus.dual_update(a_rob_ema, z_new, carry.gamma_r)

        # Residuals, in the same normalized units as the penalty so that
        # eps_r/eps_s are scale-free.
        #   primal r = [A^o (-) z ; A^r (-) z]   (both blocks stacked)
        #   dual   d = rho * (z^{l+1} (-) z^{l})
        primal_res = self.consensus.residual_norm(
            jnp.concatenate(
                [
                    self.consensus.difference(a_obj_ema, z_new),
                    self.consensus.difference(a_rob_ema, z_new),
                ]
            )
        )
        # Weighted inside the norm, not multiplied after: identical to the
        # old `rho * residual_norm(...)` for a scalar rho (a scalar factors
        # out of a norm either way), but stays a scalar -- needed by
        # `_cond` below -- when rho is a per-dimension vector.
        dual_res = self.consensus.residual_norm(
            carry.rho * self.consensus.difference(z_new, carry.z)
        )

        # Algorithm 4 step 7: adaptive penalty, off by default and bounded
        # when on. See `__init__` for why the unbounded rule is a trap here.
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
            a_obj_ema=a_obj_ema,
            a_rob_ema=a_rob_ema,
        )
        return new_carry, rollouts

    def optimize(
        self, state: mjx.Data, params: ADMMParams
    ) -> Tuple[ADMMParams, ADMMTrajectory]:
        """Perform up to `n_admm` ADMM iterations to update the policy."""
        # Warm-start: shift the object mean and the consensus/dual variables
        # by one real control step.
        object_params = params.object_params.replace(
            mean=self._shift_object(params.object_params.mean)
        )
        # z shifts through the consensus space, which knows what belongs in
        # the vacated tail (zero for a wrench, the last pose for a pose).
        # The duals are tangent vectors in either case, so zero is right.
        z = self.consensus.shift(params.z)
        gamma_o = self._shift(params.gamma_o)
        gamma_r = self._shift(params.gamma_r)

        # Warm-start the robot mean/knot-times the same way the generic
        # `SamplingBasedController.optimize()` does, since `run_interactive`
        # queries `params.tk`/`params.mean` directly via `interp_func`.
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
            rho=params.rho,
            primal_res=params.primal_residual,
            dual_res=jnp.asarray(jnp.inf, dtype=jnp.float32),
            # Never read by `_admm_iteration` (only overwritten by it), so
            # carrying forward whatever shape the previous control step
            # left here is fine -- the first real call below replaces it.
            object_samples=params.object_samples,
            rng=admm_rng,
            # Seeded from the warm-started z each real control step: the
            # EMA is scoped to one step's ADMM rounds, the same as
            # `dual_res`'s inf-init above. Seeding from z rather than from
            # zeros because zero is not a neutral consensus value in every
            # space -- for a pose it is the world origin. Unused at
            # `consensus_alpha = 1.0`, where the first round's raw A
            # replaces it outright.
            a_obj_ema=z,
            a_rob_ema=z,
        )

        # Run one ADMM iteration unconditionally (n_admm >= 1), which also
        # gives us correctly-shaped rollouts to seed the while_loop carry.
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
            rng=final_carry.rng,
        )
        return new_params, final_rollouts

    def nominal_plans(
        self, state: mjx.Data, params: ADMMParams
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Both blocks' predicted object trajectories, for visualization.

        The consensus is an agreement about a *wrench*, which is hard to
        read as a number but easy to read as motion: these are the two
        object trajectories that wrench is being negotiated over. Where they
        coincide the blocks agree; where they diverge is exactly what the
        primal residual is measuring.

        Deliberately not folded into `optimize`'s return. Both plans are
        recomputed here from the stored nominals, so a caller that never
        asks pays nothing at all -- no extra tracing, no extra rollout, no
        change to the ADMM loop. A caller that does ask pays one object
        rollout (closed-form, negligible) and one simulator rollout, against
        the `num_samples * n_admm` the step already ran.

        Args:
            state: The robot-side state the plans start from.
            params: Policy parameters as returned by `optimize`.

        Returns:
            `(object_plan, robot_plan, robot_trace)`: what the object block
            intends and what the robot block's controls would produce for
            the object, each (H, object_state_dim); and the robot/
            end-effector's own physical position along that same rollout,
            (H, 3) -- a different thing from `robot_plan`, which is in
            object-pose space, not robot space.
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
        """The object block's horizon endpoint x^{o*}_H, for drawing it.

        Exactly the value `RobotSubproblem._eval_rollouts_one` hands the
        task as `local_goal`: both are the last entry of the object block's
        nominal rollout under `params.object_params.mean`, so a marker
        driven by this shows the pose the robot block was actually scored
        against, not an approximation of it.

        Cheap enough to call every control step unconditionally -- H
        closed-form limit-surface steps, no sampling and no simulator. Kept
        separate from `nominal_plans` so a caller that wants only the
        endpoint does not also pay for the robot block's rollout.

        Args:
            state: The robot-side state the plan starts from.
            params: Policy parameters as returned by `optimize`.

        Returns:
            The object's planned SE(2) pose at the end of the horizon, (3,).
        """
        obj_state0 = self.task.object_state_from_robot(state)
        return self.object_subproblem.nominal_plan(
            obj_state0, params.object_params
        )[-1]

    def nominal_trace(self, state: mjx.Data, params: ADMMParams) -> jax.Array:
        """The robot block's chosen end-effector path, (H, 3).

        Overridden not to change the answer -- the base class would roll out
        the same `params.mean`, since `ADMMParams.mean` delegates to the
        robot block's -- but to reuse the rollout `nominal_plan` already
        does, rather than paying for a second one.

        Args:
            state: The state the plan starts from.
            params: The ADMM parameters `optimize` just returned.

        Returns:
            The trace site's world positions, (H, 3).
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
