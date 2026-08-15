from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from oim.alg_base import SamplingBasedController, SamplingParams, Trajectory
from oim.risk import RiskStrategy
from oim.task_base import Task


@dataclass
class MPPIParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        noise_scale: Multiplier on `noise_level`, 1.0 by default. Lives here
            (not as a `self.` attribute) because `optimize` is jitted as a
            bound method (`jax.jit(ctrl.optimize)`) with `self` captured by
            closure -- a plain Python attribute would be baked in as a
            constant at first trace and silently ignored on later mutation.
            Only a field of the traced params pytree can safely vary
            control-step to control-step, the same reason `mean`/`rng` are
            here rather than on `self`. Set externally (e.g. `_run_plain`'s
            variance-annealing schedule); `sample_knots` never changes it
            itself.
        temperature: The softmax temperature actually used this step. Same
            reason as `noise_scale` for living here rather than on `self`
            -- when `eta_frac_high > 0` (see `MPPI.__init__`), `update_params`
            adapts this every step, so it has to be traced state, not a
            constant baked in at first jit.
    """

    noise_scale: jax.Array
    temperature: jax.Array


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
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        noise_anneal_dist: float = 0.0,
        noise_anneal_min: float = 1.0,
        stuck_kick_steps: int = 0,
        stuck_kick_scale: float = 0.0,
        eta_frac_high: float = 0.0,
        eta_frac_low: float = 0.0,
        temp_sharpen: float = 0.9,
        temp_widen: float = 1.2,
        temp_min: float = 0.05,
        temp_max: float = 5.0,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            noise_anneal_dist: Distance (task-defined units, e.g. position
                error) over which a caller may ramp `MPPIParams.noise_scale`
                down to `noise_anneal_min` -- read by the caller driving the
                closed loop (e.g. `_run_plain`), not by this class itself,
                which never changes `noise_scale`. 0 = the caller should
                treat annealing as disabled. Stored here only so config
                drives it the same way `noise_level`/`temperature` do.
                Currently inert (config keeps this at 0) -- tried and
                rejected, see Tasks.md.
            noise_anneal_min: Floor `noise_scale` may be annealed to.
            stuck_kick_steps: Consecutive no-progress control steps (a
                caller's concern to detect and count, not this class's)
                before it should perturb `MPPIParams.mean` to escape a
                stuck local optimum. 0 = disabled. Currently active (150
                in the shipped config) -- validated, kept.
            stuck_kick_scale: Std of the perturbation applied on a kick.
            eta_frac_high: Adaptive temperature, mirroring `mppi_torch`'s
                (the reference IsaacGym baseline) `beta` update, adapted to
                this codebase's much smaller `num_samples` -- the
                reference's absolute eta bounds (5/10) were tuned for
                K~1200-2000 and don't transfer, so this uses fractions of
                `num_samples` instead. Every step, `update_params` computes
                the effective sample size eta = sum(softmax numerator)
                (bounded in [1, num_samples]): if
                eta > eta_frac_high * num_samples, too many samples carry
                near-equal weight (not decisive), temperature is multiplied
                by `temp_sharpen` (<1). If eta < eta_frac_low * num_samples,
                too few dominate (over-committed to one/few samples, maybe
                just noise), temperature is multiplied by `temp_widen`
                (>1). 0 (either bound) = adaptive temperature disabled,
                `update_params` behaves exactly as before (fixed
                `temperature`). Currently disabled (0/0 in the shipped
                config) -- tried and rejected, made stalls worse, see
                Tasks.md. Code kept in case a differently-tuned version is
                worth revisiting.
            eta_frac_low: See `eta_frac_high`.
            temp_sharpen: Multiplier applied when eta is too high.
            temp_widen: Multiplier applied when eta is too low.
            temp_min: Floor `MPPIParams.temperature` may be adapted to.
            temp_max: Ceiling `MPPIParams.temperature` may be adapted to.
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
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        # Read by `_run_plain`'s closed loop, in plain Python between jitted
        # `optimize` calls -- never inside a jitted method on this class.
        # See `MPPIParams.noise_scale`'s docstring for why that split matters.
        self.noise_anneal_dist = noise_anneal_dist
        self.noise_anneal_min = noise_anneal_min
        self.stuck_kick_steps = stuck_kick_steps
        self.stuck_kick_scale = stuck_kick_scale
        # Plain `self.` attributes are fine here (unlike `MPPIParams.
        # temperature`, which these govern): they never change within a
        # run, only read inside `update_params`, same as `noise_level`.
        self.eta_frac_high = eta_frac_high
        self.eta_frac_low = eta_frac_low
        self.temp_sharpen = temp_sharpen
        self.temp_widen = temp_widen
        self.temp_min = temp_min
        self.temp_max = temp_max

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
        )

    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * params.noise_scale * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average.

        Softmax over `-costs / params.temperature`, decomposed by hand
        (rather than calling `jax.nn.softmax`) so the effective sample
        size `eta` -- how many of the `num_samples` rollouts carry
        non-negligible weight, in [1, num_samples] -- is available to
        adapt `params.temperature` itself when `eta_frac_high > 0` (see
        `__init__`). At the defaults (`eta_frac_high = eta_frac_low =
        0.0`) `params.temperature` never moves and this computes exactly
        the same `weights`/`mean` as the fixed-temperature softmax it
        replaces -- same stabilization `jax.nn.softmax` does internally
        (subtract the max before exponentiating), just inlined.
        """
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        temp = params.temperature
        shifted = -costs / temp
        shifted = shifted - jnp.max(shifted)
        exp_costs = jnp.exp(shifted)
        eta = jnp.sum(exp_costs)
        weights = exp_costs / eta
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)

        # eta too high: too many samples carry near-equal weight, the
        # update isn't decisive -- sharpen. eta too low: too few dominate,
        # possibly just noise -- widen. See `__init__` for why these are
        # fractions of num_samples rather than the reference's absolute
        # bounds (5/10), which were tuned for K~1200-2000.
        eta_high = self.eta_frac_high * self.num_samples
        eta_low = self.eta_frac_low * self.num_samples
        new_temp = jnp.where(
            eta > eta_high,
            temp * self.temp_sharpen,
            jnp.where(eta < eta_low, temp * self.temp_widen, temp),
        )
        new_temp = jnp.clip(new_temp, self.temp_min, self.temp_max)
        temperature = jnp.where(self.eta_frac_high > 0.0, new_temp, temp)
        return params.replace(mean=mean, temperature=temperature)
