"""Analytic 2D contact physics, as a drop-in robot rollout backend for ADMM.

This is the 2D counterpart of an MJX scene: a point/disc robot pushing a
rigid planar object across a support surface, with single-point contact
resolved in closed form against the object's signed-distance field. It
implements `oim.algs.admm.RobotRollout`, so the *same* `ADMM` controller,
`ConsensusSpace`, and `ObjectSubproblem` that drive the MuJoCo tasks drive
this world too -- the only thing that changes is how one robot step is
computed.

The point of having it is debuggability. An MJX rollout that misbehaves
could be failing in the contact solver, the constraint allocation, the
integrator, the actuator model, or the ADMM math; here the physics is a
few dozen lines of arithmetic you can read, so anything that goes wrong is
attributable to the algorithm.

Everything is `jax.numpy` and side-effect free, so it jits and vmaps like
the MJX path -- but it also runs eagerly, and under `jax.disable_jit()`
every intermediate is a concrete array you can print or breakpoint on.
"""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from oim.algs.admm import RobotRollout
from oim.objects.contact import contact_force_to_com_wrench
from oim.objects.planar_pushing import wrap_angle
from oim.objects.sdf import Shape, rotate


@dataclass
class Sim2DState:
    """The 2D world's state, playing the role `mjx.Data` plays for MJX.

    Attributes:
        object_pose: The object's SE(2) pose [x, y, theta].
        robot_pos: The robot's world-frame position [x, y].
        wrench: The world-frame CoM wrench the robot applied over the last
            step, [f_x, f_y, tau]. Carried on the state because that is
            where the robot block's extraction map A^r reads it from --
            the analytic analogue of MJX's contact forces.
        time: Simulation time, used by the controllers' spline warm-start.
    """

    object_pose: jax.Array
    robot_pos: jax.Array
    wrench: jax.Array
    time: jax.Array

    @classmethod
    def create(
        cls,
        object_pose: Sequence[float] = (0.0, 0.0, 0.0),
        robot_pos: Sequence[float] = (0.0, 0.0),
    ) -> "Sim2DState":
        """Build a zero-time state with no contact history."""
        return cls(
            object_pose=jnp.asarray(object_pose, dtype=float),
            robot_pos=jnp.asarray(robot_pos, dtype=float),
            wrench=jnp.zeros(3),
            time=jnp.asarray(0.0),
        )


@dataclass
class Sim2DModel:
    """Physical parameters of the 2D world -- this layer's `mjx.Model`.

    Fills exactly the role `mjx.Model` fills on the MuJoCo side, and for the
    same reason: `SamplingBasedController` reads `task.model.nu` to size its
    control samples and then hands `task.model` straight to the rollout as
    the thing being stepped, so both must be the same object. The array
    fields are a pytree so domain randomization can `vmap` over a stack of
    these; `nu` is static because it sets array shapes.

    Attributes:
        limit_surface_d: Limit-surface compliance D, shape (3,), so that
            xdot^o = D w. Its inverse is the friction-cone limit.
        mu_c: Coulomb friction coefficient between robot and object.
        f_max: Cap on the normal contact force.
        robot_radius: Radius of the disc robot (0 for a point robot).
        nu: Control dimension (2: a planar velocity command).
    """

    limit_surface_d: jax.Array
    mu_c: jax.Array
    f_max: jax.Array
    robot_radius: jax.Array
    nu: int = field(pytree_node=False, default=2)

    @classmethod
    def create(
        cls,
        limit_surface_d: jax.Array,
        mu_c: float = 0.5,
        f_max: float = 4.0,
        robot_radius: float = 0.012,
        nu: int = 2,
    ) -> "Sim2DModel":
        """Build a model from scalars."""
        return cls(
            limit_surface_d=jnp.asarray(limit_surface_d, dtype=float),
            mu_c=jnp.asarray(mu_c, dtype=float),
            f_max=jnp.asarray(f_max, dtype=float),
            robot_radius=jnp.asarray(robot_radius, dtype=float),
            nu=nu,
        )


def _push_point_out(point: jax.Array, obstacles: Sequence[Shape]) -> jax.Array:
    """Project a point out of any obstacle it has entered."""
    for obs in obstacles:
        d, grad = obs.sdf_and_grad(point)
        point = jnp.where(d < 0.0, point - d * grad, point)
    return point


def _push_object_out(
    shape: Shape,
    pose: jax.Array,
    obstacles: Sequence[Shape],
    boundary: jax.Array,
    iterations: int,
) -> jax.Array:
    """Translate the object out of obstacle penetration, worst vertex first.

    Position-only correction (no rotation), applied a fixed number of times
    rather than to convergence so the loop unrolls under `jit`.
    """
    for _ in range(iterations):
        for obs in obstacles:
            verts = pose[:2] + rotate(pose[2], boundary)
            d, grad = obs.sdf_and_grad(verts)
            worst = jnp.argmin(d)
            d_w, grad_w = d[worst], grad[worst]
            pose = pose.at[:2].add(
                jnp.where(d_w < 0.0, -d_w * grad_w, jnp.zeros(2))
            )
    return pose


def resolve_contact(
    shape: Shape,
    model: Sim2DModel,
    pose: jax.Array,
    robot_pos: jax.Array,
    robot_vel: jax.Array,
    dt: float,
) -> jax.Array:
    """Closed-form single-point contact, returning the world CoM wrench.

    The normal force is the penetration recovered over one timestep through
    the limit-surface compliance -- i.e. exactly the force needed to move
    the object by the amount the robot has overlapped it -- capped at
    `f_max`. The tangential force is Coulomb friction opposing the sliding
    direction, and is zero when there is no tangential motion.

    Args:
        shape: The object's footprint, in its body frame.
        model: Physical parameters.
        pose: The object's SE(2) pose.
        robot_pos: The robot's world position.
        robot_vel: The robot's commanded world velocity.
        dt: The timestep over which the penetration is recovered.

    Returns:
        The world-frame CoM wrench [f_x, f_y, tau], zero when out of contact.
    """
    q = rotate(-pose[2], robot_pos - pose[:2])  # robot in object body frame
    d, grad = shape.sdf_and_grad(q)
    gap = d - model.robot_radius
    in_contact = gap <= 0.0

    n_body = -grad  # inward: the direction a pusher presses
    t_body = jnp.stack([-n_body[1], n_body[0]])
    n_world = rotate(pose[2], n_body)
    t_world = rotate(pose[2], t_body)

    # Penetration -> normal force through the limit-surface compliance.
    penetration = jnp.clip(-gap, 0.0, None)
    f_n = jnp.clip(
        penetration / (dt * model.limit_surface_d[0]), 0.0, model.f_max
    )
    v_t = jnp.dot(robot_vel, t_world)
    sliding = jnp.abs(v_t) > 1e-4
    f_t = jnp.where(sliding, -model.mu_c * f_n * jnp.sign(v_t), 0.0)
    f_n = jnp.where(in_contact, f_n, 0.0)
    f_t = jnp.where(in_contact, f_t, 0.0)

    # Contact point: the robot's position projected onto the boundary.
    contact_body = q - gap * grad
    p_world = pose[:2] + rotate(pose[2], contact_body)
    f_world = f_n * n_world + f_t * t_world
    wrench = contact_force_to_com_wrench(pose, p_world, f_world)
    return jnp.where(in_contact, wrench, jnp.zeros(3))


class Analytic2DRollout(RobotRollout):
    """Robot rollout backend: analytic planar contact instead of `mjx.step`.

    Geometry (object footprint, obstacles) and the integration settings are
    held as ordinary Python attributes, so they are static under `jit`;
    only `Sim2DModel` and `Sim2DState` are traced.
    """

    def __init__(
        self,
        shape: Shape,
        obstacles: Optional[Sequence[Shape]] = None,
        dt: float = 0.05,
        n_substeps: int = 4,
        contact_step_margin: float = 0.003,
        max_contact_step: float = 0.008,
        obstacle_margin: float = 0.015,
        pushout_iters: int = 4,
        boundary_samples_per_edge: int = 4,
        max_speed: float = 0.6,
    ) -> None:
        """Configure the 2D world's geometry and integration.

        Args:
            shape: The object's footprint in its body frame.
            obstacles: Static obstacles. None means an empty scene.
            dt: The planning/execution timestep.
            n_substeps: Contact substeps per timestep. More substeps make
                the contact less likely to tunnel through the object.
            contact_step_margin: Slack left between the robot and whatever
                it is approaching when a substep is shortened.
            max_contact_step: Cap on how far the robot may move in one
                substep once it is already in contact.
            obstacle_margin: Clearance used by the object push-out.
            pushout_iters: Fixed obstacle push-out iterations per substep.
            boundary_samples_per_edge: Footprint sampling density for the
                object's obstacle push-out.
            max_speed: Cap on the robot's commanded speed magnitude.
        """
        self.shape = shape
        self.obstacles = list(obstacles or [])
        self.dt = dt
        self.n_substeps = n_substeps
        self.contact_step_margin = contact_step_margin
        self.max_contact_step = max_contact_step
        self.obstacle_margin = obstacle_margin
        self.pushout_iters = pushout_iters
        self.max_speed = max_speed
        self.boundary = shape.sample_boundary(boundary_samples_per_edge)

    def _safe_displacement(
        self,
        model: Sim2DModel,
        pose: jax.Array,
        robot_pos: jax.Array,
        disp: jax.Array,
    ) -> jax.Array:
        """Shorten a displacement so the robot cannot tunnel through geometry.

        Without this, a fast robot can cross the whole object in one
        substep and register no contact at all -- the analytic analogue of
        MuJoCo's own penetration limits.
        """
        q = rotate(-pose[2], robot_pos - pose[:2])
        d_obj = self.shape.sdf(q[None, :])[0] - model.robot_radius
        safe = jnp.where(
            d_obj > 0.0, d_obj + self.contact_step_margin, self.max_contact_step
        )
        for obs in self.obstacles:
            d_obs = obs.sdf(robot_pos[None, :])[0] - model.robot_radius
            safe = jnp.minimum(
                safe,
                jnp.where(
                    d_obs > 0.0,
                    d_obs + self.contact_step_margin,
                    self.max_contact_step,
                ),
            )
        norm = jnp.linalg.norm(disp)
        scale = jnp.where(
            norm > 1e-12, jnp.clip(safe / (norm + 1e-12), 0.0, 1.0), 1.0
        )
        return disp * scale

    def _substep(
        self,
        model: Sim2DModel,
        state: Sim2DState,
        vel: jax.Array,
        sub_dt: float,
    ) -> Tuple[Sim2DState, jax.Array]:
        """One collision-safe substep. Returns the new state and the wrench."""
        pose, robot = state.object_pose, state.robot_pos

        disp = self._safe_displacement(model, pose, robot, sub_dt * vel)
        robot_free = _push_point_out(robot + disp, self.obstacles)

        wrench = resolve_contact(
            self.shape, model, pose, robot_free, vel, sub_dt
        )

        # Object responds through the limit surface, then is de-penetrated.
        new_pose = pose + sub_dt * model.limit_surface_d * wrench
        new_pose = new_pose.at[2].set(wrap_angle(new_pose[2]))
        if self.obstacles:
            new_pose = _push_object_out(
                self.shape,
                new_pose,
                self.obstacles,
                self.boundary,
                self.pushout_iters,
            )

        # The object may have moved into the robot; push the robot back out.
        q = rotate(-new_pose[2], robot_free - new_pose[:2])
        d, grad = self.shape.sdf_and_grad(q)
        gap = d - model.robot_radius
        corrected = new_pose[:2] + rotate(new_pose[2], q - gap * grad)
        new_robot = jnp.where(gap < 0.0, corrected, robot_free)
        new_robot = _push_point_out(new_robot, self.obstacles)

        return (
            state.replace(object_pose=new_pose, robot_pos=new_robot),
            wrench,
        )

    def step(
        self, model: Sim2DModel, state: Sim2DState, control: jax.Array
    ) -> Sim2DState:
        """Advance the 2D world by one planning timestep.

        Args:
            model: Physical parameters.
            state: The current state.
            control: Commanded robot velocity [vx, vy].

        Returns:
            The next state, carrying the mean wrench applied over the step
            in `state.wrench` for the robot block's A^r to read.
        """
        # Saturate on the velocity *magnitude*, here in the plant rather than
        # in the sampler's box bounds. A per-axis box would let a diagonal
        # command reach sqrt(2) times the nominal top speed, which is a
        # capability the robot does not physically have -- and the planner
        # will happily exploit any capability its rollouts appear to grant.
        speed = jnp.linalg.norm(control)
        control = control * jnp.minimum(
            1.0, self.max_speed / jnp.maximum(speed, 1e-12)
        )
        sub_dt = self.dt / self.n_substeps

        def _body(
            carry: Tuple[Sim2DState, jax.Array], _: jax.Array
        ) -> Tuple[Tuple[Sim2DState, jax.Array], None]:
            st, acc = carry
            st, w = self._substep(model, st, control, sub_dt)
            return (st, acc + w), None

        (state, wrench_sum), _ = jax.lax.scan(
            _body, (state, jnp.zeros(3)), jnp.arange(self.n_substeps)
        )
        return state.replace(
            wrench=wrench_sum / self.n_substeps,
            time=state.time + self.dt,
        )
