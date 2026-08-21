from typing import Any, Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from oim.alg_base import SamplingBasedController, SamplingParams, Trajectory
from oim.risk import RiskStrategy
from oim.task_base import Task


@dataclass
class MPPIParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as `SamplingParams`, plus fields that must vary per control step
    but cannot live as plain `self.` attributes since `optimize` is jitted
    as a bound method (`self` is captured by closure, so a plain attribute
    would be baked in as a constant at first trace).

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, mu = [u_0, ...].
        rng: The pseudo-random number generator key.
        noise_scale: Multiplier on `noise_level`, 1.0 by default. Set
            externally (e.g. a variance-annealing schedule).
        temperature: The softmax temperature used this step. Traced
            state rather than a `self.` attribute so a caller can set it
            per step without re-tracing `optimize`.
        task_jac_inv: Damped-inverse tip Jacobian, (nu, 5), mapping a
            task-space perturbation [dx, dy, dz, d(tilt_x), d(tilt_y)] to
            joint space. Only meaningful when `MPPI.use_task_space_noise`
            is set; recomputed each control step by the caller from the
            real (non-MJX) model via `mujoco.mj_jacSite`, which can't run
            inside a jitted method. Zero when the mechanism is off. Only
            ever applied to the z/tilt bias and z/tilt noise -- x/y noise
            uses `task_noise_map` instead.
        task_bias: Task-space feedback bias, (5,) -- `[0, 0, -alpha*(z -
            z_d), -alpha*tilt_x, -alpha*tilt_y]`, pulling tip height and
            tilt toward their safe references while leaving x/y unbiased.
        task_noise_map: (nu, 2), maps x/y exploration noise to joint space
            through the null space of the tip Jacobian's z/tilt rows,
            instead of `task_jac_inv`'s (generally coupled) inverse. A
            joint velocity built purely from this map produces exactly
            zero z/tilt velocity to first order, so x/y exploration can't
            itself disturb z/tilt. Zero before the first update or when
            the mechanism is off.
    """

    noise_scale: jax.Array
    temperature: jax.Array
    task_jac_inv: jax.Array
    task_bias: jax.Array
    task_noise_map: jax.Array


class MPPI(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: Any,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        stuck_kick_steps: int = 0,
        stuck_kick_scale: float = 0.0,
        task_space_noise: Any = None,
        task_space_alpha: float = 0.0,
        task_space_damping: float = 1e-4,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: Std of Gaussian noise added to sampled controls.
                A scalar applies the same std to every dimension; a
                length-`nu` sequence gives each joint its own, broadcasting
                against `sample_knots`' `(num_samples, num_knots, nu)`
                noise along the trailing axis.
            temperature: The temperature parameter lambda. Higher values take a
                         more even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            stuck_kick_steps: Consecutive no-progress control steps (a
                caller's concern to detect) before perturbing
                `MPPIParams.mean` to escape a stuck local optimum. 0 disables.
            stuck_kick_scale: Std of the perturbation applied on a kick.
            task_space_noise: Length-5 std [sigma_x, sigma_y, sigma_z,
                sigma_tilt_x, sigma_tilt_y] for exploration noise sampled
                in task space (tip linear velocity in x/y/z, plus two
                small-angle tilt-rate proxies) and mapped to joint
                velocities through a damped-inverse Jacobian, instead of
                `noise_level`'s per-joint scalars. None (default) disables
                this entirely. xarm6/PushT only: needs `task.tip_site_id`
                and `task.robot_dof_adr`, and a caller that populates
                `MPPIParams.task_jac_inv`/`task_bias` each step (currently
                only the flat-baseline closed loop, not ADMM).
            task_space_alpha: Feedback gain (1/s) pulling tip height
                toward `task.tip_target_z` and tip tilt toward vertical,
                `d/dt(error) = -alpha * error`. Only read by the caller
                that builds `task_bias`; irrelevant when
                `task_space_noise` is None.
            task_space_damping: Damping term (Levenberg-Marquardt style,
                added to `J^T J` before inverting) in the tip Jacobian's
                pseudo-inverse, for numerical stability near kinematic
                singularities. Only read by the same caller.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        # jnp.asarray so a per-joint list becomes a real (nu,) array here.
        self.noise_level = jnp.asarray(noise_level)
        self.num_samples = num_samples
        self.temperature = temperature
        # Read by the closed loop in plain Python between jitted `optimize`
        # calls -- see `MPPIParams.mean`.
        self.stuck_kick_steps = stuck_kick_steps
        self.stuck_kick_scale = stuck_kick_scale
        # Static (never traced) -- read as a plain Python bool inside
        # `sample_knots` to pick between the two noise mechanisms.
        self.use_task_space_noise = task_space_noise is not None
        self.task_noise_level = (
            jnp.asarray(task_space_noise)
            if task_space_noise is not None
            else jnp.zeros(5)
        )
        self.task_space_alpha = task_space_alpha
        self.task_space_damping = task_space_damping

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return MPPIParams(
            tk=_params.tk,
            mean=_params.mean,
            rng=_params.rng,
            noise_scale=jnp.asarray(1.0),
            temperature=jnp.asarray(self.temperature),
            task_jac_inv=jnp.zeros((self.task.model.nu, 5)),
            task_bias=jnp.zeros(5),
            task_noise_map=jnp.zeros((self.task.model.nu, 2)),
        )

    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence.

        Two mutually exclusive noise mechanisms, chosen by
        `self.use_task_space_noise` -- fixed at construction, so this
        branches at trace time, not per-call.
        """
        rng, sample_rng = jax.random.split(params.rng)
        if self.use_task_space_noise:
            # x/y noise routed through the null space of the z/tilt rows
            # (task_noise_map) rather than task_jac_inv's general inverse,
            # which generically couples every task direction and would
            # make x/y noise itself a source of z/tilt disturbance.
            xy_rng, zt_rng = jax.random.split(sample_rng)
            xy_noise = jax.random.normal(
                xy_rng, (self.num_samples, self.num_knots, 2)
            )
            zt_noise = jax.random.normal(
                zt_rng, (self.num_samples, self.num_knots, 5)
            )
            # x/y (indices 0, 1) forced to zero: their exploration comes
            # entirely from xy_noise/task_noise_map below.
            zt_scale = self.task_noise_level.at[:2].set(0.0)
            zt_perturb = zt_scale * params.noise_scale * zt_noise + params.task_bias
            zt_qdot = jnp.einsum("ij,ktj->kti", params.task_jac_inv, zt_perturb)
            xy_scaled = self.task_noise_level[:2] * params.noise_scale * xy_noise
            xy_qdot = jnp.einsum("ij,ktj->kti", params.task_noise_map, xy_scaled)
            perturb = zt_qdot + xy_qdot
        else:
            noise = jax.random.normal(
                sample_rng,
                (
                    self.num_samples,
                    self.num_knots,
                    self.task.model.nu,
                ),
            )
            perturb = self.noise_level * params.noise_scale * noise
        controls = params.mean + perturb
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average.

        A fixed-temperature softmax over `-costs / params.temperature`.

        Non-finite costs are neutralized first. One NaN rollout otherwise
        poisons the entire batch -- the `max` shift propagates it to every
        weight -- and the NaN then lives in `mean` for the rest of the
        run, so a single bad sample ends the episode. `inf` is no better:
        if every sample is `inf` the shift is `-inf - -inf`, NaN again.
        """
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # Non-finite means "worse than anything usable", so those samples
        # are moved just past the worst finite cost and weighted out. A
        # no-op whenever every cost is finite: `where` returns the
        # original array unchanged, so a healthy step is bit-identical.
        finite = jnp.isfinite(costs)
        any_finite = jnp.any(finite)
        worst = jnp.max(jnp.where(finite, costs, -jnp.inf))
        costs = jnp.where(
            finite, costs, jnp.where(any_finite, worst, 0.0) + 1.0
        )
        temp = params.temperature
        shifted = -costs / temp
        shifted = shifted - jnp.max(shifted)
        exp_costs = jnp.exp(shifted)
        weights = exp_costs / jnp.sum(exp_costs)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        # No usable sample at all: every rollout was non-finite, so the
        # weights above are the uniform average of garbage knots. Stand
        # still on the previous nominal instead.
        mean = jnp.where(any_finite, mean, params.mean)
        return params.replace(mean=mean)
