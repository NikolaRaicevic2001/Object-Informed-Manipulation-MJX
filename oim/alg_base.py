from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator, Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from mujoco import mjx

from oim.risk import AverageCost, RiskStrategy
from oim.task_base import Task
from oim.utils.spline import get_interp_func


@contextmanager
def quiet_mjx_cast_overflow() -> Iterator[None]:
    """Silence MJX's own float64-constant-into-float32-array warning.

    MJX's box-box narrowphase writes `jp.finfo(float).max` -- float64's
    1.798e308 -- into a float32 array, twice
    (`mjx/_src/collision_convex.py:717` and `:930`). JAX casts it, numpy
    reports `RuntimeWarning: overflow encountered in cast`, and the
    resulting value is `inf`.

    Benign, and specifically so: those lines exist to *replace* an infinite
    distance with a large sentinel, and the overflow hands back the `inf`
    that was already there. "No separating axis found" stays "no separating
    axis found", and every reduction downstream handles `inf`. The warning
    is about the constant, not the geometry -- it fires unconditionally
    wherever that line is traced, whatever the scene.

    Suppressed with `np.errstate` rather than a `warnings` filter, and
    around each `mjx` call rather than globally, because that is the
    narrowest scope that works: the cast happens while Python traces the
    MJX function, so this covers exactly MJX's own arithmetic and nothing
    of ours. A genuine overflow anywhere else still reports -- which a
    filter on the message could not promise, since our `jnp` calls raise it
    through the same JAX frame.

    The cost of that precision is that it must wrap every entry into MJX,
    and there are five: `MJXRollout.step` and `SamplingBasedController.
    rollout` here, `oim.runtime.object_mjx.MJXObjectRollout._substep`, and
    the `mjx.forward` priming calls in `oim.worlds.sim3d.run` and
    `oim.runtime.viewer`. A new `mjx.*` call site needs this too, or the
    warning comes back for that path alone -- which is exactly how it was
    missed the first time, when the object world gained an MJX backend and
    nothing else did.

    Zero runtime cost: the wrapped call is Python, so it executes during
    tracing only -- afterwards the compiled XLA runs with no context
    manager in sight.
    """
    with np.errstate(over="ignore"):
        yield


@dataclass
class Trajectory:
    """Data class for storing rollout data.

    Throughout, H denotes the number of control steps (given by the times at
    which the control spline is interpolated).

    Attributes:
        controls: Control actions of shape (num_rollouts, H, nu).
        knots: Control spline knots of shape (num_rollouts, num_knots, nu).
        costs: Costs of shape (num_rollouts, H+1).
        trace_sites: Positions of trace sites of shape (num_rollouts, H+1, 3).
    """

    controls: jax.Array
    knots: jax.Array
    costs: jax.Array
    trace_sites: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1


@dataclass
class SamplingParams:
    """Parameters for sampling-based control algorithms.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """

    tk: jax.Array
    mean: jax.Array
    rng: jax.Array


class SamplingBasedController(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
        plan_horizon: float,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # time-related variables
        # NOTE: we always interpret self.task.model as the controller's
        # internal model, not the model used for simulation. dt is the
        # time between spline queries.
        self.plan_horizon = plan_horizon
        self.dt = self.task.dt
        self.ctrl_steps = int(round(self.plan_horizon / self.dt))

        # Spline setup for control interpolation
        self.spline_type = spline_type
        self.num_knots = num_knots
        self.interp_func = get_interp_func(spline_type)

        # Use a single model (no domain randomization) by default
        self.model = task.model
        self.randomized_axes = None

        # Number of optimization iterations
        if iterations < 1:
            raise ValueError("iterations must be greater than 0!")

        self.iterations = iterations

        if self.num_randomizations > 1:
            # Make domain randomized models
            rng = jax.random.key(seed)
            rng, subrng = jax.random.split(rng)
            subrngs = jax.random.split(subrng, num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_model)(subrngs)
            self.model = self.task.model.tree_replace(randomizations)

            # Keep track of which elements of the model have randomization
            self.randomized_axes = jax.tree.map(lambda x: None, self.task.model)
            self.randomized_axes = self.randomized_axes.tree_replace(
                {key: 0 for key in randomizations.keys()}
            )

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )

        # Clamp query times to the old spline's domain to avoid extrapolation,
        # which can produce wildly wrong values for linear/cubic splines.
        clamped_tk = jnp.clip(new_tk, tk[0], tk[-1])

        new_mean = self.interp_func(clamped_tk, tk, params.mean[None, ...])[0] 
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(params: Any, _: Any):
            # Sample random control sequences from spline knots
            knots, params = self.sample_knots(params)
            knots = jnp.clip(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, nu)

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations(
                state, new_tk, knots, dr_rng
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(params, rollouts)

            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        return params, rollouts_final

    def nominal_trace(self, state: mjx.Data, params: Any) -> jax.Array:
        """The trace-site path the current plan would follow, for drawing.

        Every sampling-based controller reduces a population of rollouts to
        one trajectory it intends to execute; this is that trajectory, in
        world coordinates, so `oim.runtime.overlay` can draw it thicker
        than the candidates it was chosen from. Implemented once here
        because the reduction always lands in `params.mean` -- an algorithm
        that means something else by "chosen" overrides this.

        Costs one extra rollout of H steps per control step, against the
        `num_samples` the optimizer already ran, so it is a fraction of a
        percent of a control step. Nothing calls it unless an overlay is on.

        Args:
            state: The state the plan starts from.
            params: The policy parameters `optimize` just returned.

        Returns:
            The first trace site's world positions, (H+1, 3).
        """
        knots = jnp.clip(
            params.mean[None, ...], self.task.u_min, self.task.u_max
        )
        rollouts = self.rollout_with_randomizations(
            state, params.tk, knots, jax.random.key(0)
        )
        return rollouts.trace_sites[0, :, 0, :]

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # compute the control sequence from the knots
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        _, rollouts = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None)
        )(self.model, states, controls, knots)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """
        # Physics steps per planning step. `task.robot_substeps` absent or 1
        # is the pre-existing single coarse `mjx.step`, so nothing changes for
        # a task that does not set it -- which is every task today except a
        # real-table one, whose driver sets it from `world3d.robot_substeps`.
        #
        # Why it exists on this path at all: a contact's response time in
        # MuJoCo is clamped to at least 2*dt, so at a planning_dt of 0.05 the
        # block free-falls g*dt^2 = 24.5 mm before the table catches it, and
        # the rollout over-predicts how far a push travels. ADMM's robot block
        # has had the same knob since 2026-08 (`oim.algs.MJXRollout`, wired up
        # in `oim/worlds/sim3d/build.py`); the flat samplers never did, so
        # every flat MPPI run has been rolling out at the coarse step. This
        # closes that gap without changing any existing run.
        #
        # The horizon is untouched: `controls` still has one entry per
        # planning step and each is held across the substeps, so H steps of
        # planning_dt still cover the same span in seconds. Only the contact
        # integration gets finer, at ~substeps x the cost.
        substeps = max(int(getattr(self.task, "robot_substeps", 1) or 1), 1)
        if substeps > 1:
            # Scaled on the traced model rather than by building a second one,
            # for the reason `MJXRollout` gives: domain randomization hands a
            # different model in per sample, and this has to follow it.
            model = model.replace(
                opt=model.opt.replace(timestep=model.opt.timestep / substeps)
            )

        def _step(x: mjx.Data) -> mjx.Data:
            # step model + compute site positions. See
            # `oim.algs.admm.quiet_mjx_cast_overflow` for the wrapper.
            with quiet_mjx_cast_overflow():
                if substeps == 1:
                    return mjx.step(model, x)
                out, _ = jax.lax.scan(
                    lambda s, _: (mjx.step(model, s), None),
                    x,
                    None,
                    length=substeps,
                )
                return out

        # The clock the COSTS see. `freeze_cost_time` pins it to the
        # rollout's own start, so a weight that ramps with elapsed control
        # steps is constant across the horizon instead of compounding
        # inside it -- the same contract ADMM's robot block already has,
        # where `t` is read once by `RobotSubproblem._eval_rollouts_one`.
        # The DYNAMICS always see the advancing clock: only the state
        # handed to the cost functions is rewritten.
        t0 = state.time
        freeze = getattr(self.task, "freeze_cost_time", False)

        def _for_cost(x: mjx.Data) -> mjx.Data:
            return x.replace(time=t0) if freeze else x

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = x.replace(ctrl=u)
            x = _step(x)
            cost = self.dt * self.task.running_cost(_for_cost(x), u)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)

        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = self.task.terminal_cost(_for_cost(final_state))
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_knots: The initial knots of the control spline.
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = jax.random.key(seed)
        mean = (
            initial_knots
            if initial_knots is not None
            else jnp.zeros((self.num_knots, self.task.model.nu))
        )
        assert mean.shape == (self.num_knots, self.task.model.nu), (
            f"Initial knots must have shape (num_knots, nu), got {mean.shape}"
        )
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        return SamplingParams(tk=tk, mean=mean, rng=rng)

    @abstractmethod
    def sample_knots(self, params: Any) -> Tuple[jax.Array, Any]:
        """Sample a set of control spline knots U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            Control spline knots U, size (num rollouts, num_knots).
            Updated parameters (e.g., with a new PRNG key).
        """

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """

    def get_action(self, params: SamplingParams, t: jax.Array) -> jax.Array:
        """Get the control action at a given point along the trajectory.

        Args:
            params: The policy parameters, U ~ π(params).
            t: The current time at which to query the spline. Spline times are
                continually evolving as the simulation progresses, so this
                number should roughly track mj_data.time.

        Returns:
            The control action u(t).
        """
        knots = params.mean[None, ...]  # (1, num_knots, nu)
        tk = params.tk
        u = self.interp_func(t, tk, knots)[0]  # (nu,)
        return u
