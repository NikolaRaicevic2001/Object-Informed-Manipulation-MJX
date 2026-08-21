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


class ConsensusSpace(ABC):
    """A consensus variable space for ADMM, shared between two subproblems.

    Owns all math that touches z, so the object block's proposed value A^o
    and the robot block's realized value A^r provably use the same units,
    frame and definitions.
    """

    dim: int

    @abstractmethod
    def normalize(self, v: jax.Array) -> jax.Array:
        """Map a consensus-space tangent value to dimensionless units, so
        `rho`/`eps_r`/`eps_s` are scale-free."""

    def difference(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Return a (-) b: the tangent vector from b to a.

        Plain subtraction for a vector space (the paper's case). Overridden
        for a manifold (`PoseConsensus`); every ADMM subtraction routes
        through here so one override reaches the penalty, both residuals,
        the dual update and the z-update at once.
        """
        return a - b

    def increment(self, base: jax.Array, tangent: jax.Array) -> jax.Array:
        """Return base (+) tangent: the inverse of `difference`, i.e.
        projection Pi_Z of eq. 14 -- identity for a vector space."""
        return base + tangent

    def shift(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift. Zero-fills the vacated tail, right when
        zero is a meaningful consensus value; overridden where it is not."""
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
        """Norm of an already-tangent residual, in normalized units."""
        return jnp.linalg.norm(self.normalize(v))

    def z_update(
        self,
        a_o: jax.Array,
        a_r: jax.Array,
        dual_o: jax.Array,
        dual_r: jax.Array,
        base: jax.Array,
    ) -> jax.Array:
        """Paper eq. 27 with N = 2, taken about `base`:

            z <- base (+) 0.5*[(a_o (-) base) + y_o + (a_r (-) base) + y_r]

        Identical to the plain average 0.5*(a_o+y_o+a_r+y_r) on a vector
        space; needed only when Z is a manifold, where the four terms must
        be lifted to a common tangent space first.
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
        """Args:
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


class PoseConsensus(ConsensusSpace):
    """SE(2) object-pose consensus z_t = [x, y, theta]; arrays (H, 3).

    The alternative to `WrenchConsensus`: the two blocks negotiate where
    the object should be over the horizon rather than what wrench acts on
    it. A^o is the pose trajectory the object block's wrench sequence
    induces through the limit surface (paper eq. 5, integrated -- affine
    in U^o); A^r is the object's pose along the robot rollout, read
    straight from the simulator state, with no twist inversion or clip.

    SE(2) is not a vector space, so Pi_Z is not the identity: differences
    wrap theta into (-pi, pi], the duals live in the tangent space (they
    are twists, not poses), and eq. 27's average is taken about a base
    point. Not cosmetic: the tabletop goal is theta = pi, on the branch
    cut, so an unwrapped subtraction reports 2*pi disagreement at the goal.
    """

    dim = 3

    def __init__(self, max_dual: jax.Array, scale: jax.Array = None) -> None:
        """Args:
            max_dual: Dual anti-windup clip, per dimension or scalar.
            scale: Characteristic pose-difference magnitude, e.g.
                `(r_body, r_body, 1.0)` so a normalized residual of 1
                means "one body radius, or one radian" of disagreement.
                Defaults to ones.
        """
        self.max_dual = jnp.asarray(max_dual)
        self.scale = jnp.ones(self.dim) if scale is None else jnp.asarray(scale)

    def normalize(self, v: jax.Array) -> jax.Array:
        """Divide by the per-dimension characteristic magnitude."""
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
        """Shift by one and repeat the last pose (zero-fill would put the
        vacated tail at the world origin, a specific pose, not "no pose")."""
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


@dataclass
class ADMMTrajectory(Trajectory):
    """Trajectory with the realized consensus value A^r(U^r)_t at each step."""

    consensus_values: jax.Array


class RobotRollout(ABC):
    """How the robot block advances its state by one planning step.

    The only place the robot subproblem is tied to a particular simulator
    -- swapping this is what lets the same `ADMM` class drive both an MJX
    scene and the analytic 2D world. Implementations must be pure
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
        """Args:
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
    ) -> None:
        """Args:
            task: The real task, providing object-level dynamics/costs.
            optimizer: A `SamplingBasedController` built against
                `make_object_shim(task, ...)`.
            consensus: The consensus space used for the ADMM penalty.
            proximal_weight: Weight (gamma) on the proximal term (eq. 24).
            rollout: How to advance the object one step. Defaults to
                `AnalyticObjectRollout`; pass
                `oim.runtime.object_mjx.MJXObjectRollout` to plan against
                the simulator instead.
        """
        self.task = task
        self.optimizer = optimizer
        self.consensus = consensus
        self.proximal_weight = proximal_weight
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
            to the wrenches when the consensus variable is the wrench,
            differing when it's the pose).
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
            a_o = self.task.object_consensus(new_state, w)
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
        noise_scale: jax.Array,
        rng: jax.Array,
        weight_scale: jax.Array = 1.0,
    ) -> Tuple[Any, jax.Array, jax.Array, jax.Array]:
        """Run `optimizer.iterations` MPPI-style passes against a fixed target.

        Args:
            obj_state0: The object's current configuration x^o_0.
            params: The object optimizer's current policy parameters.
            z, dual_o, rho: The consensus target, dual and penalty weight.
            prev_knots: Previous ADMM iteration's knots (proximal term).
            noise_scale: Extra residual-scaled exploration noise.
            rng: Random key.
            weight_scale: Goal-tracking ramp, shared with the robot block.

        Returns:
            Updated params; the object's proposed decision W^o
            (params.mean); its nominal state reference; and the last
            iteration's sampled trajectories for visualization.
        """
        opt = self.optimizer

        def _scan_body(params: Any, rng_i: jax.Array) -> Tuple[Any, jax.Array]:
            noise_rng, sample_rng = jax.random.split(rng_i)
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
            noise = noise_scale * jax.random.normal(noise_rng, knots.shape)
            knots = jnp.clip(knots + noise, opt.task.u_min, opt.task.u_max)
            knots = self.task.project_object_action(knots, obj_state0)

            states, ws, a_o = jax.vmap(self._rollout, in_axes=(None, 0))(
                obj_state0, knots
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
            # Rate penalty: sequence-level, anchored to the wrench the
            # previous solve already intended for this step, so it charges
            # for changing course across control steps as well as within
            # the horizon.
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
            # Re-project the aggregated nominal, not just the samples: an
            # optimizer update averages samples, and averaging feasible
            # points need not be feasible (measured up to 1.5mm for the T
            # footprint's contact points either side of a corner).
            params = params.replace(
                mean=self.task.project_object_action(params.mean, obj_state0)
            )
            return params, states

        rngs = jax.random.split(rng, opt.iterations)
        params, all_states = jax.lax.scan(_scan_body, params, rngs)
        # Last iteration's sample population, for visualization -- free,
        # already computed above for `object_running_cost`.
        object_samples = all_states[-1]

        # A^o for the nominal (eq. 24), recovered by rolling the nominal
        # actions out -- necessary for a non-trivial action parameterization
        # or a pose consensus variable.
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
    ) -> None:
        """Args:
            task: The real task, providing the MJX model, robot-level
                cost, and the A^r extraction map.
            optimizer: A `SamplingBasedController` built against `task`.
            consensus: The consensus space; the ADMM penalty is added via
                `consensus.penalty_cost`, shared with the object block.
            proximal_weight: Weight (gamma) on the proximal term (eq. 25).
            rollout: How to advance the robot state one step. Defaults to
                `MJXRollout`.
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
        """Like `SamplingBasedController.rollout_with_randomizations`;
        z/dual_r/rho/obj_ref (the fixed target every sample is scored
        against) and the proximal anchor are threaded through too."""
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
        primal_residual: Last iteration's primal residual, carried across
            real control steps to seed exploration-noise annealing.
        dual_residual: Last iteration's dual residual. Logging only.
        object_samples: The object block's last sampled trajectories,
            (num_samples, H, object_state_dim). Logging only.
        a_obj: A^o, the object block's extracted consensus value, (H, dim).
            Carried so the ADMM penalty is reconstructible from a run file.
        a_rob: A^r as the nominal robot plan realized it, (H, dim) -- the
            planned value, so both blocks' penalties reflect what they
            planned (the runners log the executed A^r separately).
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
    a_obj: jax.Array
    a_rob: jax.Array
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
    a_obj_ema: jax.Array
    a_rob_ema: jax.Array


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
        noise_min: float = 0.0,
        noise_kappa: float = 0.0,
        noise_max: Optional[float] = None,
        rollout: Optional[RobotRollout] = None,
        object_rollout: Optional[ObjectRollout] = None,
        debug_print: bool = False,
        consensus_alpha: float = 1.0,
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
            eps_r, eps_s: Primal/dual residual tolerances for early exit.
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
            noise_min, noise_max: Exploration-noise scale bounds.
                `noise_max` defaults to `noise_min` (inert) since an
                uncapped `noise_kappa * residual` can runaway (more noise
                -> more disagreement -> larger residual -> more noise).
            noise_kappa: Extra exploration noise relative to the primal
                residual (Algorithm 4 step 8, generalized).
            rollout: How the robot block advances its state one step.
                Defaults to `MJXRollout`; pass
                `oim.worlds.sim2d.Analytic2DRollout` for the 2D world.
            object_rollout: How the object block advances its state one
                step. Defaults to `AnalyticObjectRollout` (eq. 5); pass
                `oim.runtime.object_mjx.MJXObjectRollout` to make both
                blocks predict with MJX instead, which also makes the
                object block's cost no longer free per sample.
            debug_print: Print residuals/penalty weight every ADMM
                iteration. Off by default: it's a host callback inside the
                compiled loop (costs a device sync that inflates the
                recorded `compute_time`) and floods the per-step summary.
            consensus_alpha: EMA weight on A^o/A^r across ADMM rounds (1.0
                = raw, matching the paper). Each round's A is one noisy
                resampling estimate, not a converged proposal, so
                smoothing targets that resampling variance directly.
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
        # keeps its ratio rather than collapsing under one scalar band.
        self.rho_min = jnp.asarray(rho_init) / rho_bound_factor
        self.rho_max = jnp.asarray(rho_init) * rho_bound_factor
        self.noise_min = noise_min
        self.noise_kappa = noise_kappa
        self.noise_max = noise_min if noise_max is None else noise_max
        self.debug_print = debug_print
        self.consensus_alpha = consensus_alpha

        self.object_subproblem = ObjectSubproblem(
            task,
            object_optimizer,
            consensus,
            proximal_weight,
            rollout=object_rollout,
        )
        self.robot_subproblem = RobotSubproblem(
            task, robot_optimizer, consensus, proximal_weight, rollout=rollout
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
            rng=jax.random.key(seed),
        )

    def _shift(self, seq: jax.Array) -> jax.Array:
        """Receding-horizon shift: seq[t] <- seq[t+1], last slot <- 0."""
        return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)

    def _shift_object(self, seq: jax.Array) -> jax.Array:
        """Delegates to `shift_object_actions`, so the standalone
        object-only driver warm-starts exactly as ADMM does."""
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

        # Goal-tracking ramp, read once at the horizon start; the robot
        # block reads the identical value from the same `state.time`
        # inside `_eval_rollouts_one`.
        weight_scale = getattr(self.task, "time_ramp", lambda _t: 1.0)(
            state.time
        )
        # Near-goal fade on the consensus penalty: once the object is one
        # correction from the goal, each block optimizes its own objective
        # instead of negotiating a wrench. Read once from `obj_state0` and
        # handed to both blocks -- `z_update`'s plain average only holds
        # while both penalties carry equal weight.
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
                noise_scale,
                obj_rng,
                weight_scale,
            )
        )
        robot_params, rollouts = self.robot_subproblem.optimize(
            state,
            carry.robot_params,
            carry.z,
            carry.gamma_r,
            penalty_rho,
            obj_ref,
            prev_robot_knots,
            noise_scale,
            rob_rng,
        )
        a_rob = self.robot_subproblem.nominal_realized_consensus(
            state, robot_params
        )

        # EMA across ADMM rounds (not within one horizon): each round's
        # raw A^o/A^r is a single noisy resampling estimate, smoothed
        # before consensus. At alpha = 1.0 (shipped default) this is the
        # raw value.
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
        # Duals move with the same fade: inside the fade radius nothing
        # charges for a disagreement, so a full-step dual would integrate
        # against nothing and release the banked amount once the object
        # drifts back out. Interpolating toward the stepped value (rather
        # than scaling the increment) keeps `dual_update`'s clip intact.
        gamma_o = carry.gamma_o + fade * (
            self.consensus.dual_update(a_obj_ema, z_new, carry.gamma_o)
            - carry.gamma_o
        )
        gamma_r = carry.gamma_r + fade * (
            self.consensus.dual_update(a_rob_ema, z_new, carry.gamma_r)
            - carry.gamma_r
        )

        # Residuals, normalized so eps_r/eps_s are scale-free:
        #   primal r = [A^o (-) z ; A^r (-) z]   dual d = rho*(z^{l+1} (-) z^l)
        primal_res = self.consensus.residual_norm(
            jnp.concatenate(
                [
                    self.consensus.difference(a_obj_ema, z_new),
                    self.consensus.difference(a_rob_ema, z_new),
                ]
            )
        )
        dual_res = self.consensus.residual_norm(
            carry.rho * self.consensus.difference(z_new, carry.z)
        )

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
            a_obj_ema=a_obj_ema,
            a_rob_ema=a_rob_ema,
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
            rho=params.rho,
            primal_res=params.primal_residual,
            dual_res=jnp.asarray(jnp.inf, dtype=jnp.float32),
            object_samples=params.object_samples,
            rng=admm_rng,
            # Seeded from the warm-started z (not zeros, since zero is not
            # a neutral consensus value for a pose). Unused at
            # consensus_alpha = 1.0, where the first round's raw A
            # replaces it outright.
            a_obj_ema=z,
            a_rob_ema=z,
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
            a_obj=final_carry.a_obj_ema,
            a_rob=final_carry.a_rob_ema,
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
        """The point of the object block's plan the robot aims at, for
        drawing it.

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
