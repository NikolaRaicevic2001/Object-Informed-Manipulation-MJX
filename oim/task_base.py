from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Sequence[str] | None = None,
        impl: str = "jax",
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.
            impl: The backend implementation for rollouts ("jax" for standard
                  MJX or "warp" for MjWarp).

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model, impl=impl)

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def make_data(self, **kwargs) -> mjx.Data:
        """Create a new state consistent with this task.

        By default, this just creates a new `mjx.Data` instance from the model.
        Specific tasks can override this method to set parameters that must be
        adjusted per task, e.g., nconmax and naconmax.

        TODO(vincekurtz): figure out a smarter place to set naconmax and njmax.
        N.B. when performing parallel rollouts with MjWarp, naconmax and
        njmax need to be set high enough to support constraint solving across
        *all* rollouts. This means that these parameters scale with the number
        of parallel rollouts/samples, as well as the complexity of the task.

        Args:
            **kwargs: Additional keyword arguments to pass to `mjx.make_data`.

        Returns:
            A new `mjx.Data` instance for this task.
        """
        return mjx.make_data(self.mj_model, impl=self.model.impl, **kwargs)


class ConsensusTask(ABC):
    """Mixin for tasks that support ADMM object/robot consensus.

    A task combines this with `Task` (e.g. `class MyTask(Task, ConsensusTask)`)
    to expose the object-level closed-form subproblem, the robot-level
    ADMM-penalized cost, and the consensus variable that ties them together,
    so that `oim.algs.admm.ADMM` can coordinate the two. Tasks that don't
    need ADMM are unaffected by this mixin's existence.

    The consensus variable z_t is generic: it might be a contact wrench, an
    object pose, or something else entirely. `consensus_dim` declares its
    dimension, and `realized_consensus` extracts it from the robot's rollout
    state.
    """

    @property
    @abstractmethod
    def consensus_dim(self) -> int:
        """The dimension of the consensus variable z_t."""

    @abstractmethod
    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Closed-form object-side dynamics x^o_{t+1} = f^o(x^o_t, w_t).

        Args:
            obj_state: The object's configuration x^o_t.
            w: The object-level consensus decision w^o_t.

        Returns:
            The next object configuration x^o_{t+1}.
        """

    @abstractmethod
    def object_running_cost(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """The object-level running cost ℓ_o(x^o_t).

        Includes any effort regularization on the consensus decision w^o_t.
        """

    @abstractmethod
    def object_terminal_cost(self, obj_state: jax.Array) -> jax.Array:
        """The object-level terminal cost ℓ_f(x^o_H)."""

    def object_rate_cost(
        self, wrenches: jax.Array, w_prev: Optional[jax.Array] = None
    ) -> jax.Array:
        """Penalty on how fast the object decision changes along a rollout.

        A sequence-level term, so it cannot live in `object_running_cost`,
        which sees one step at a time. Defaults to zero, i.e. the paper's
        cost exactly; override to charge for reversing the wrench.

        Args:
            wrenches: This rollout's wrenches, (H, consensus_dim).
            w_prev: The wrench the previous solve intended for the first
                step, or None to score only within-horizon changes.

        Returns:
            A scalar penalty for the whole sequence.
        """
        return jnp.zeros(())

    def object_action_scale(self) -> jax.Array:
        """Per-dimension scale for the object optimizer's sampled knots.

        Applied before they're treated as the physical consensus decision
        w^o_t. Defaults to no scaling; override when the sampler's natural
        units differ substantially across dimensions (e.g. force vs torque).
        """
        return jnp.ones(self.object_action_dim)

    # Object action parameterization (optional). By default the object
    # block samples w^o_t directly. Overriding these four hooks lets a
    # task decide something richer instead -- e.g. `oim.worlds.sim2d`'s contact
    # action [p_x, p_y, f_n, f_t], mapped to a wrench via w = J_c^T f so
    # every proposal is inside the friction cone by construction.

    @property
    def object_action_dim(self) -> int:
        """Dimension of the object block's decision variable.

        Defaults to `consensus_dim` (the block decides the consensus value
        itself). Override when the block decides something that *maps* to
        the consensus value, in which case this is the dimension the object
        optimizer samples in.
        """
        return self.consensus_dim

    def object_action_bounds(self) -> Tuple[jax.Array, jax.Array]:
        """Box bounds (lo, hi) on the object block's decision variable.

        The **unit box**, the convention `object_action_to_consensus`
        assumes: the action is dimensionless and `object_action_scale()`
        carries the physics, so `noise_level` reads as a fraction of the
        largest admissible decision.

        Note this makes the reachable wrench set `object_action_scale()`
        exactly -- if that is below the limit-surface breakaway threshold
        the block cannot move the object at all. See `PushT`'s
        `wrench_sample_fraction`.

        Override only for a structured action space whose components carry
        real units of their own (see `oim.worlds.sim2d`'s contact action).
        """
        ones = jnp.ones(self.object_action_dim)
        return -ones, ones

    def object_action_to_consensus(
        self, obj_state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Map an object-block decision to the consensus value A^o.

        Evaluated *inside* the object rollout, so the map may depend on the
        object's current configuration (a contact wrench depends on where
        the object currently is, not just on the action).

        Args:
            obj_state: The object's configuration x^o_t.
            action: The object block's decision at time t.

        Returns:
            The consensus value w^o_t, shape (consensus_dim,).
        """
        return action * self.object_action_scale()

    def project_object_action(
        self, action: jax.Array, obj_state: Optional[jax.Array] = None
    ) -> jax.Array:
        """Project a sampled object action onto the feasible set.

        Applied to every sample before it is rolled out, so infeasible
        proposals never reach the cost. Defaults to the identity.

        Args:
            action: The sampled action(s) to project.
            obj_state: The object configuration this action would be
                applied from, if the projection needs it (e.g. to gate
                itself by proximity to the goal). None for callers with
                no state to offer; a task whose override needs it should
                treat None as "don't gate, behave as if not overridden".
        """
        return action

    def initial_object_action(self) -> Optional[jax.Array]:
        """Seed value for the object block's decision, or None for zeros.

        Zeros are a fine start when the block decides the consensus value
        directly, but not for a structured action space: a contact point of
        (0, 0) is the object's own origin, which is *inside* the footprint
        rather than on it, and every contact quantity derived from it (the
        surface normal, and hence the sampler's rejection test) is then
        degenerate. Tasks with such an action space should return a valid
        point instead.

        Returns:
            An array of shape `(object_action_dim,)`, broadcast across the
            horizon, or None to keep the optimizer's own zero init.
        """
        return None

    def sample_object_actions(
        self,
        nominal: jax.Array,
        rng: jax.Array,
        num_samples: int,
        obj_state: jax.Array,
    ) -> Optional[jax.Array]:
        """Optionally take over sampling for the object block.

        Return `None` (the default) to let the injected optimizer sample
        with its own `sample_knots`. Return an array of shape
        `(num_samples, H, object_action_dim)` to override it -- needed when
        the action space has geometry a Gaussian cannot respect, such as a
        contact point that must stay on the object's boundary.

        `obj_state` is the object's configuration at the start of the
        horizon, supplied because a geometry-aware proposal generally needs
        it (where it is worth touching the object depends on where the
        object currently is relative to its goal and the obstacles).

        N.B. overriding this bypasses the optimizer's own proposal
        distribution, so it composes with optimizers whose `sample_knots`
        is a stateless function of `params.mean` (MPPI, CEM, predictive
        sampling) but *not* with ones carrying their own particle state
        across iterations (CBO). The optimizer's `update_params` is still
        what turns the scored samples into the next mean.
        """
        return None

    def object_consensus(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """The extraction map A^o, at one step of the object rollout.

        Evaluated *after* `object_dynamics` has been applied, so
        `obj_state` is x^o_{t+1} and `w` is the w^o_t that produced it.
        That matches the robot block, which reads A^r after its own step,
        so index t means the same instant on both sides.

        Defaults to the wrench itself -- the paper's A^o(U^o)_t = w^o_t,
        a selection off the block's own decision variable. Override to
        return `obj_state` for a pose consensus variable, where A^o is
        instead the limit-surface relation integrated, and hence affine in
        U^o rather than a selection.

        Args:
            obj_state: The object configuration after this step.
            w: The wrench applied over this step.

        Returns:
            The consensus value A^o_t, shape (consensus_dim,).
        """
        return w

    def consensus_scale(self) -> jax.Array:
        """Characteristic per-dimension magnitude of the consensus variable.

        Used to normalize the ADMM penalty and residuals so that `rho` and
        the residual tolerances are scale-free and the penalty is comparable
        in magnitude to the task costs. Defaults to ones (no normalization);
        override with e.g. the largest physically transmissible wrench.
        """
        return jnp.ones(self.consensus_dim)

    @abstractmethod
    def robot_running_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
        obj_ref_t: jax.Array,
        local_goal: Optional[jax.Array] = None,
    ) -> jax.Array:
        """The robot-level running cost J_r (paper eq. 17).

        This is the task's own cost only — ℓ_o + ℓ_r + ℓ_c plus any control
        effort term. Do **not** add the ADMM consensus penalty here: the
        ADMM layer adds it via `ConsensusSpace.penalty_cost`, the same
        function the object block uses, so both blocks are guaranteed to
        score the consensus variable identically.

        Args:
            state: The robot's current MJX state x^r_t.
            control: The control action u^r_t.
            obj_ref_t: The object planner's current reference x^{o*}_t.
            local_goal: The object planner's reference at the *end* of the
                horizon, x^{o*}_H -- the same array for every t, since it
                is a property of the plan rather than of this step. Offered
                unconditionally by the ADMM layer (it is one index into a
                sequence already in hand); a task chooses whether its goal
                tracking aims at this or at the global goal. `None` means
                no plan is available -- the direct callers in the tests --
                and must behave as the global goal.

        Returns:
            The scalar robot-level running cost.
        """

    @abstractmethod
    def robot_terminal_cost(
        self, state: mjx.Data, local_goal: Optional[jax.Array] = None
    ) -> jax.Array:
        """The robot-level terminal cost (shared with the object goal).

        Args:
            state: The final robot state x^r_H.
            local_goal: As in `robot_running_cost`. This is the term where
                it matters most: the terminal cost carries the heavier
                `qf_*` weights and, unlike the stage costs, is not
                dt-weighted in the rollout.
        """

    @abstractmethod
    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """The extraction map A^r: robot state -> realized consensus value."""

    @abstractmethod
    def object_state_from_robot(self, state: mjx.Data) -> jax.Array:
        """Extract the object's own configuration from the robot state.

        Reads x^o_t out of the robot's combined MJX state, to seed the
        object subproblem each real step.
        """
