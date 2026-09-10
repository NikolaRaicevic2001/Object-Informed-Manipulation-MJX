"""Frozen-state CBF/CLF projections of robot control tapes, before rollout.

The analytical mode corrects the floor first, then the slider ceiling in
the null space of vertical tip velocity. The QP mode solves both CBFs and
a soft tilt CLF together. Neither mode runs inside the physics scan.
"""

from dataclasses import dataclass as config_dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from mujoco import mjx

from oim.objects.sdf import rotate


@config_dataclass(frozen=True)
class ProjectionConfig:
    """Projection settings; heights and distances are in world metres.

    ``z_min`` and ``z_near`` default to offsets from the slider mid-height
    in ``configure_projection``. No config block means no projection.
    """

    mode: str = "off"
    z_min: float = 0.0
    z_near: float = 0.01
    z_far: float = 0.10
    distance_near: float = 0.02
    distance_far: float = 0.15
    footprint_margin: float = 0.01
    cbf_alpha: float = 2.0
    tilt_rate: float = 2.0
    tilt_weight: float = 10.0
    # QPax 0.1.4 floors internal slacks/duals at sqrt(float32 epsilon).
    # A tighter KKT tolerance stalls even at an accurate primal solution.
    # Independently check the physical constraints at feasibility_tol.
    solver_tol: float = 1e-3
    feasibility_tol: float = 1e-5
    max_iter: int = 30

    def __post_init__(self) -> None:
        """Reject malformed constraints before tracing the controller."""
        if self.mode not in {"off", "analytical", "qpax"}:
            raise ValueError("projection.mode must be off, analytical or qpax")
        values = [v for v in vars(self).values() if isinstance(v, (int, float))]
        if not np.all(np.isfinite(values)):
            raise ValueError("projection settings must be finite")
        if not self.z_min < self.z_near <= self.z_far:
            raise ValueError("projection requires z_min < z_near <= z_far")
        if not 0 <= self.distance_near < self.distance_far:
            raise ValueError(
                "projection requires 0 <= distance_near < distance_far"
            )
        if self.footprint_margin < 0:
            raise ValueError("projection.footprint_margin must be nonnegative")
        if (
            min(
                self.cbf_alpha,
                self.tilt_rate,
                self.tilt_weight,
                self.solver_tol,
                self.feasibility_tol,
            )
            <= 0
        ):
            raise ValueError(
                "projection gains, weights and tolerances must be positive"
            )
        if self.max_iter < 1 or int(self.max_iter) != self.max_iter:
            raise ValueError("projection.max_iter must be a positive integer")


@dataclass
class ProjectionConstraints:
    """Linear constraints frozen at the observed state for one MPC solve.

    CBFs use ``cbf_a @ u + cbf_b >= 0``; the CLF uses
    ``tilt_a @ u + tilt_b <= slack``. Rows are floor, slider ceiling.
    """

    cbf_a: jax.Array
    cbf_b: jax.Array
    tilt_a: jax.Array
    tilt_b: jax.Array
    u_min: jax.Array
    u_max: jax.Array


@dataclass
class ProjectionDiagnostics:
    """Per-command diagnostics, with the same leading axes as the tape."""

    valid: jax.Array
    converged: jax.Array
    hard_violation: jax.Array
    correction_norm: jax.Array
    tilt_slack: jax.Array
    iterations: jax.Array


def height_ceiling(
    distance: jax.Array, config: ProjectionConfig
) -> tuple[jax.Array, jax.Array]:
    """Return a smooth distance-dependent ceiling and its derivative."""
    width = config.distance_far - config.distance_near
    t = jnp.clip((distance - config.distance_near) / width, 0.0, 1.0)
    height = config.z_near + (config.z_far - config.z_near) * t**2 * (3 - 2 * t)
    slope = (config.z_far - config.z_near) * 6 * t * (1 - t) / width
    return height, slope


def make_constraints(
    position: jax.Array,
    axis: jax.Array,
    jac_position: jax.Array,
    jac_rotation: jax.Array,
    distance: jax.Array,
    distance_gradient: jax.Array,
    distance_drift: jax.Array,
    u_min: jax.Array,
    u_max: jax.Array,
    config: ProjectionConfig,
) -> ProjectionConstraints:
    """Construct CBF/CLF rows from tip kinematics and footprint distance.

    ``distance_drift`` is distance rate due to the slider's motion alone.
    The stick site's local z axis points down in the upright xArm pose.
    """
    ceiling, slope = height_ceiling(distance, config)
    floor_a = jac_position[2]
    ceiling_a = slope * (distance_gradient @ jac_position[:2]) - floor_a
    cbf_b = jnp.stack(
        (
            config.cbf_alpha * (position[2] - config.z_min),
            config.cbf_alpha * (ceiling - position[2]) + slope * distance_drift,
        )
    )
    target = jnp.array([0.0, 0.0, -1.0])
    return ProjectionConstraints(
        cbf_a=jnp.stack((floor_a, ceiling_a)),
        cbf_b=cbf_b,
        tilt_a=jnp.cross(target, axis) @ jac_rotation,
        tilt_b=config.tilt_rate * (1.0 - target @ axis),
        u_min=u_min,
        u_max=u_max,
    )


def _bounded_correction(
    controls: jax.Array,
    row: jax.Array,
    bias: jax.Array,
    direction: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
) -> jax.Array:
    """Correct a half-space along a ray without leaving the velocity box."""
    denom = row @ direction
    amount = jnp.maximum(-(controls @ row + bias), 0.0) / jnp.maximum(
        denom, 1e-12
    )
    nonzero = jnp.abs(direction) > 1e-10
    safe_direction = jnp.where(nonzero, direction, 1.0)
    room = (
        jnp.where(direction > 0, hi - controls, lo - controls) / safe_direction
    )
    capacity = jnp.maximum(
        jnp.min(jnp.where(nonzero, room, jnp.inf), axis=-1), 0.0
    )
    amount = jnp.where(denom > 1e-12, jnp.minimum(amount, capacity), 0.0)
    return controls + amount[..., None] * direction


def analytical_project(
    controls: jax.Array, constraints: ProjectionConstraints
) -> jax.Array:
    """Apply two analytical CBF corrections; the slider cannot change dz.

    The second direction is the ceiling normal projected into null(J_z).
    It changes horizontal tip motion and may change orientation; version 1
    does not constrain tilt. Bounds can prevent a complete correction, so
    callers must inspect the feasibility diagnostics from ``project``.
    """
    c = constraints
    out = jnp.clip(controls, c.u_min, c.u_max)
    floor, slider = c.cbf_a
    out = _bounded_correction(out, floor, c.cbf_b[0], floor, c.u_min, c.u_max)
    direction = slider - floor * (floor @ slider) / jnp.maximum(
        floor @ floor, 1e-12
    )
    return _bounded_correction(
        out, slider, c.cbf_b[1], direction, c.u_min, c.u_max
    )


class ControlProjector:
    """Batch projector with an optional QPax dependency."""

    def __init__(self, config: ProjectionConfig) -> None:
        """Load QPax only when its mode is selected."""
        self.config = config
        self.solve_qp = None
        if config.mode == "qpax":
            try:
                from qpax import solve_qp  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "QPax projection requires: uv sync --extra projection"
                ) from exc
            self.solve_qp = solve_qp

    def _qp_project(
        self, controls: jax.Array, c: ProjectionConstraints
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Batch standard QPs with shared matrices and CLF-only slack."""
        n = controls.shape[-1]
        q_mat = jnp.diag(
            jnp.concatenate((jnp.ones(n), jnp.array([self.config.tilt_weight])))
        )
        cbf = jnp.concatenate((-c.cbf_a, jnp.zeros((2, 1))), axis=1)
        clf = jnp.concatenate((c.tilt_a, jnp.array([-1.0])))[None]
        box = jnp.concatenate((jnp.eye(n), jnp.zeros((n, 1))), axis=1)
        slack_row = jnp.concatenate((jnp.zeros(n), jnp.array([-1.0])))[None]
        g = jnp.concatenate((cbf, clf, box, -box, slack_row))
        h = jnp.concatenate(
            (c.cbf_b, -c.tilt_b[None], c.u_max, -c.u_min, jnp.zeros(1))
        )
        a, b = jnp.zeros((0, n + 1)), jnp.zeros(0)

        def solve(u: jax.Array) -> tuple:
            linear = jnp.concatenate((-u, jnp.zeros(1)))
            x, _, _, _, converged, iterations = self.solve_qp(
                q_mat,
                linear,
                a,
                b,
                g,
                h,
                solver_tol=self.config.solver_tol,
                max_iter=self.config.max_iter,
            )
            # Solve the soft CLF alone exactly. If no hard constraint is
            # active, this is also the full QP optimum. The interior-point
            # stopping tolerance otherwise introduces a substantial drift
            # even for a stationary, upright arm with inactive barriers.
            a_clf = c.tilt_a
            amount = self.config.tilt_weight * jnp.maximum(
                a_clf @ u + c.tilt_b, 0.0
            ) / (1.0 + self.config.tilt_weight * (a_clf @ a_clf))
            free_u = u - amount * a_clf
            free_slack = jnp.maximum(a_clf @ free_u + c.tilt_b, 0.0)
            free_valid = (
                jnp.all(c.cbf_a @ free_u + c.cbf_b >= 0.0)
                & jnp.all(free_u >= c.u_min)
                & jnp.all(free_u <= c.u_max)
            )
            return (
                jnp.where(free_valid, free_u, x[:n]),
                jnp.where(free_valid, free_slack, x[n]),
                jnp.where(free_valid, True, converged),
                jnp.where(free_valid, 0, iterations),
            )

        out, slack, converged, iterations = jax.vmap(solve)(
            controls.reshape(-1, n)
        )
        shape = controls.shape[:-1]
        return (
            out.reshape(controls.shape),
            slack.reshape(shape),
            converged.reshape(shape),
            iterations.reshape(shape),
        )

    def project(
        self, controls: jax.Array, constraints: ProjectionConstraints
    ) -> tuple[jax.Array, ProjectionDiagnostics]:
        """Project an arbitrary control batch and report unresolved constraints.

        Failed QPs use finite bounded analytical controls for simulation,
        but remain invalid. The rollout dispatcher rejects their tapes.
        """
        c = constraints
        shape = controls.shape[:-1]
        finite = jnp.all(jnp.isfinite(controls), axis=-1)
        nominal = jnp.where(jnp.isfinite(controls), controls, 0.0)
        out = analytical_project(nominal, c)
        converged = jnp.ones(shape, dtype=bool)
        iterations = jnp.zeros(shape, dtype=jnp.int32)
        slack = jnp.zeros(shape)
        if self.config.mode == "off":
            return controls, ProjectionDiagnostics(
                valid=finite,
                converged=jnp.ones(shape, dtype=bool),
                hard_violation=jnp.zeros(shape),
                correction_norm=jnp.zeros(shape),
                tilt_slack=slack,
                iterations=iterations,
            )
        if self.config.mode == "qpax":
            candidate, slack, converged, iterations = self._qp_project(
                nominal, c
            )
            converged = (
                (converged > 0)
                & jnp.all(jnp.isfinite(candidate), axis=-1)
                & jnp.isfinite(slack)
            )
            out = jnp.where(converged[..., None], candidate, out)
        violation = jnp.maximum(
            0.0, jnp.max(-(out @ c.cbf_a.T + c.cbf_b), axis=-1)
        )
        violation = jnp.maximum(
            violation,
            jnp.max(jnp.maximum(c.u_min - out, out - c.u_max), axis=-1),
        )
        valid = (
            finite
            & converged
            & jnp.isfinite(violation)
            & (violation <= self.config.feasibility_tol)
        )
        if self.config.mode == "qpax":
            valid = valid & (slack >= -self.config.feasibility_tol)
            valid = valid & (
                out @ c.tilt_a + c.tilt_b - slack <= self.config.feasibility_tol
            )
        return out, ProjectionDiagnostics(
            valid=valid,
            converged=converged,
            hard_violation=violation,
            correction_norm=jnp.linalg.norm(out - controls, axis=-1),
            tilt_slack=slack,
            iterations=iterations,
        )


class RobotControlProjector(ControlProjector):
    """Prepare xArm constraints from qpos once, outside all rollout scans."""

    def __init__(self, task: Any, config: ProjectionConfig) -> None:
        """Cache a JAX kinematics model, also usable with Warp rollouts."""
        super().__init__(config)
        self.task = task
        self.model = mjx.put_model(task.mj_model, impl="jax")
        self.data = mjx.make_data(self.model)
        self.site_id = task.tip_site_id
        self.body_id = int(task.mj_model.site_bodyid[self.site_id])
        self.dofs = jnp.asarray(task.robot_dof_adr)

    def prepare(self, state: mjx.Data) -> ProjectionConstraints:
        """Compute fresh kinematics from observations with stale site arrays."""
        data = self.data.replace(
            qpos=state.qpos,
            qvel=state.qvel,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
        )
        data = mjx.com_pos(self.model, mjx.kinematics(self.model, data))
        position = data.site_xpos[self.site_id]
        jacp, jacr = mjx.jac(self.model, data, position, self.body_id)
        pose = self.task.object_state_from_robot(data)
        local = rotate(-pose[2], position[:2] - pose[:2])
        distance, grad_local = self.task.object_model.footprint.sdf_and_grad(
            local
        )
        gradient = rotate(pose[2], grad_local)
        # The task resolves its planar translation/yaw DOFs by joint name.
        pose_rate = data.qvel[self.task.block_dofs]
        relative = position[:2] - pose[:2]
        surface_velocity = pose_rate[:2] + pose_rate[2] * jnp.array(
            [-relative[1], relative[0]]
        )
        return make_constraints(
            position,
            data.site_xmat[self.site_id, :, 2],
            jacp[self.dofs].T,
            jacr[self.dofs].T,
            distance - self.config.footprint_margin,
            gradient,
            -gradient @ surface_velocity,
            self.task.u_min,
            self.task.u_max,
            self.config,
        )


def configure_projection(task: Any, settings: dict | None) -> None:
    """Attach an opt-in xArm projector in both real/simulation builders."""
    if not settings or settings.get("mode", "off") == "off":
        return
    if getattr(task, "robot", None) != "xarm6":
        raise ValueError("control_projection currently supports only xarm6")
    if task.model.nu != len(task.robot_dof_adr):
        raise ValueError(
            "projection requires one velocity command per robot DOF"
        )
    if not np.all(np.isfinite(np.concatenate((task.u_min, task.u_max)))):
        raise ValueError("projection requires finite robot velocity limits")
    config = ProjectionConfig(
        **{
            "z_min": task.tip_target_z - 0.01,
            "z_near": task.tip_target_z,
            "z_far": task.tip_target_z + 0.10,
            **settings,
        }
    )
    task.control_projector = RobotControlProjector(task, config)


def prepare_projection(
    task: Any, state: mjx.Data
) -> ProjectionConstraints | None:
    """Prepare shared data once per solve, with a structural no-op default."""
    projector = getattr(task, "control_projector", None)
    return None if projector is None else projector.prepare(state)


def project_controls(
    task: Any,
    state: mjx.Data,
    controls: jax.Array,
    constraints: ProjectionConstraints | None = None,
) -> tuple[jax.Array, ProjectionDiagnostics | None]:
    """Project a tape before rollout dispatch, or return it unchanged."""
    projector = getattr(task, "control_projector", None)
    if projector is None:
        return controls, None
    if constraints is None:
        constraints = projector.prepare(state)
    return projector.project(controls, constraints)


def reject_invalid_tapes(
    costs: jax.Array, diagnostics: ProjectionDiagnostics | None
) -> jax.Array:
    """Reject tapes containing any unresolved projection."""
    if diagnostics is None:
        return costs
    valid = jnp.all(diagnostics.valid, axis=-1)
    return costs.at[..., -1].set(jnp.where(valid, costs[..., -1], jnp.inf))
