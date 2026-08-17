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
        task_jac_inv: Damped-inverse tip Jacobian, (nu, 5), mapping a task-
            space perturbation [dx, dy, dz, d(tilt_x), d(tilt_y)] to a
            joint-space one. Only meaningful when `MPPI.use_task_space_noise`
            is set; a caller (`oim.worlds.sim3d.run._run_plain`) recomputes
            it from the real current configuration every control step, the
            same reason `noise_scale`/`temperature` live here rather than on
            `self` -- it depends on state that changes step to step, using
            `mujoco.mj_jacSite` on the real (non-MJX) model, which cannot run
            inside a jitted method. Zeros (a structural no-op) when the
            mechanism is off, or before the first update. Only ever applied
            to a vector with zero x/y components (the z/tilt bias, and any
            z/tilt noise) -- x/y noise uses `task_noise_map` instead (see
            its docstring for why the two need different maps).
        task_bias: Task-space feedback bias, (5,) -- `[0, 0, -alpha*(z -
            z_d), -alpha*tilt_x, -alpha*tilt_y]`, pulling tip height and
            tilt back toward their safe references while leaving x/y
            unbiased (still purely MPPI-optimized through `mean`). See
            `task_jac_inv`; same caller, same reason it lives here.
        task_noise_map: (nu, 2), maps x/y exploration noise to joint space
            through the NULL SPACE of the tip Jacobian's z/tilt rows,
            instead of `task_jac_inv`'s general (and generally coupled --
            see `MPPI.__init__`'s `task_space_noise`) inverse. A joint
            velocity built purely from this map is guaranteed, by
            construction, to produce exactly zero z/tilt velocity (to
            first order, at the configuration it was computed from) --
            not merely small, not merely damped by a feedback law, exactly
            zero, so x/y exploration cannot itself be a source of z/tilt
            disturbance the way it was before this. Recomputed every
            control step alongside `task_jac_inv`, same reason and same
            caller. Zeros before the first update or when the mechanism
            is off.
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
        task_space_noise: Any = None,
        task_space_alpha: float = 0.0,
        task_space_damping: float = 1e-4,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
                A scalar applies the same std to every control dimension
                (the default, and what every config used before this).
                A length-`nu` sequence gives each joint its own std
                instead -- `sample_knots`'s `self.noise_level * noise`
                broadcasts a `(nu,)` array against noise of shape
                `(num_samples, num_knots, nu)` with no other code change
                needed, since it aligns against the trailing axis.
                Motivated by a per-joint Jacobian check on xarm6 (see
                Tasks.md): joint 1 alone moves the tip in the xy-plane
                with zero tilt cost, while every other joint costs the
                same tilt per unit of noise regardless of whether it is
                buying useful z-reach (joints 2/3) or xy-motion joint 1
                already provides for free (joints 4/5) -- so weighting
                exploration noise unevenly across joints, not just
                picking one scalar for all of them, is the direct
                implication of that finding.
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
            task_space_noise: Length-5 std [sigma_x, sigma_y, sigma_z,
                sigma_tilt_x, sigma_tilt_y] for exploration noise sampled
                in *task space* (the tip's linear velocity in x/y/z, plus
                two small-angle tilt-rate proxies) and mapped to joint
                velocities through a damped-inverse Jacobian, instead of
                `noise_level`'s per-joint scalars. None (default) disables
                this entirely -- `sample_knots` takes the exact `noise_level`
                path it always has, and every existing caller/config is
                unaffected. xarm6/PushT only: needs `task.tip_site_id` and
                `task.robot_dof_adr`, and a caller that populates
                `MPPIParams.task_jac_inv`/`task_bias` each step (currently
                only `oim.worlds.sim3d.run._run_plain`, i.e. the flat
                baseline, not ADMM -- see that function). Motivated by
                the same per-joint Jacobian finding `noise_level` above
                already responds to, but principled rather than hand-set:
                z/tilt get a fixed reference to explore *around*
                (`task_space_alpha` below) instead of a smaller but still
                unconstrained joint-space noise scale. See Tasks.md for
                the full derivation.
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
        # jnp.asarray so a per-joint list (see docstring above) becomes
        # a real (nu,) array here rather than staying a Python list that
        # only happens to work by luck in sample_knots' multiplication.
        self.noise_level = jnp.asarray(noise_level)
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
        # Static (never traced) -- read as a plain Python bool inside
        # `sample_knots` to pick between the two noise mechanisms, the
        # same closure-baked-constant reasoning as `noise_level` above.
        # `task_noise_level` is (5,) regardless of `nu`; `task_jac_inv`'s
        # shape is (nu, 5) so the mechanism is inert-by-construction
        # (never populated with anything but the `init_params` zeros) for
        # any controller nothing ever updates it for -- see
        # `MPPIParams.task_jac_inv`.
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
        `self.use_task_space_noise` -- a plain Python bool, fixed at
        construction, so this branches at trace time (one of the two
        bodies is simply never compiled), not per-call.
        """
        rng, sample_rng = jax.random.split(params.rng)
        if self.use_task_space_noise:
            # Two independent draws, two different maps -- see
            # MPPIParams' `task_noise_map` docstring for why x/y cannot
            # share `task_jac_inv` with z/tilt: that general inverse
            # generically couples every task direction through whichever
            # joints happen to move more than one of them at once (real
            # kinematic coupling, not a code artifact -- see Tasks.md),
            # so x/y noise routed through it was itself a source of
            # z/tilt disturbance. `task_noise_map` instead routes x/y
            # noise through the null space of the z/tilt rows alone,
            # which cannot move z/tilt at all by construction.
            xy_rng, zt_rng = jax.random.split(sample_rng)
            xy_noise = jax.random.normal(
                xy_rng, (self.num_samples, self.num_knots, 2)
            )
            zt_noise = jax.random.normal(
                zt_rng, (self.num_samples, self.num_knots, 5)
            )
            # x/y (indices 0, 1) forced to zero here -- their exploration
            # comes entirely from xy_noise/task_noise_map below, not this
            # path. z/tilt (indices 2-4) keep their own noise (usually 0,
            # see MPPI.__init__) plus the deterministic feedback bias,
            # exactly as before this change.
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
