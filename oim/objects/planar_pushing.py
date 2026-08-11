"""Analytic object model for quasi-static planar pushing.

Implements the object-level subproblem of the object-informed MPPI
formulation: closed-form SE(2) limit-surface dynamics (paper eq. 4-5) plus
the object-level stage costs (paper eq. 18-19). This is deliberately
independent of MuJoCo -- a task supplies the goal, footprint, and obstacle
field, and gets dynamics and costs back, rather than re-deriving the
geometry in every task file.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from oim.objects.sdf import ObstacleField, Polygon, rotate


def wrap_angle(a: jax.Array) -> jax.Array:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def se2_distance_sq(
    pose: jax.Array, goal: jax.Array, w_pos: float, w_theta: float
) -> jax.Array:
    """Weighted squared SE(2) distance d^2(x, g), paper eq. 19.

    Args:
        pose: SE(2) pose(s) [x, y, theta], shape (..., 3).
        goal: Goal pose [x, y, theta], shape (3,).
        w_pos: Weight on the translational error.
        w_theta: Weight on the (wrapped) rotational error.

    Returns:
        The weighted squared distance, shape (...,).
    """
    diff_pos = pose[..., :2] - goal[:2]
    diff_theta = wrap_angle(pose[..., 2] - goal[2])
    return w_pos * jnp.sum(diff_pos**2, axis=-1) + w_theta * diff_theta**2


class PlanarPushingObject:
    """Quasi-static planar object driven by a contact wrench.

    Dynamics are the limit-surface relation (paper eq. 4-5):

        x^o_{t+1} = x^o_t + dt * D w^o_t,
        D = diag(mu*m*g, mu*m*g, c*r*mu*m*g)^{-1},

    where w^o = [f_x, f_y, tau] is the contact wrench applied to the object.
    D is isotropic in translation and the torque is a scalar, so in 2D the
    relation is identical whether the wrench is expressed in the world frame
    or the object body frame -- the implementation uses the world frame
    throughout, matching what the simulator reports.
    """

    def __init__(
        self,
        dt: float,
        goal: jax.Array,
        footprint: Polygon,
        obstacles: Optional[ObstacleField] = None,
        mu: float = 0.4,
        mass: float = 2.0,
        gravity: float = 9.81,
        pressure_coeff: float = 1.0,
        limit_surface_radius: float = 0.06,
        w_pos: float = 40.0,
        w_theta: float = 10.0,
        wf_pos: float = 500.0,
        wf_theta: float = 150.0,
        w_effort: float = 0.01,
        w_obstacle: float = 60000.0,
        obstacle_margin: float = 0.015,
        boundary_samples_per_edge: int = 4,
        wrench_sample_fraction: float = 0.5,
    ) -> None:
        """Configure the object's physics, goal, geometry, and cost weights.

        Args:
            dt: Planning timestep for the forward-Euler integration.
            goal: Goal pose [x, y, theta].
            footprint: The object's outline in its body frame, used for
                obstacle clearance checks.
            obstacles: Static obstacles to avoid. None means no obstacles.
            mu: Friction coefficient between object and support surface.
            mass: Object mass.
            gravity: Gravitational acceleration.
            pressure_coeff: The limit-surface pressure coefficient `c`.
            limit_surface_radius: The characteristic radius `r`.
            w_pos: Running-cost weight on translational goal error.
            w_theta: Running-cost weight on rotational goal error.
            wf_pos: Terminal-cost weight on translational goal error.
            wf_theta: Terminal-cost weight on rotational goal error.
            w_effort: Weight on the squared wrench (control effort).
            w_obstacle: Weight on the obstacle clearance hinge.
            obstacle_margin: Clearance below which the hinge activates.
            boundary_samples_per_edge: Footprint boundary sampling density.
            wrench_sample_fraction: A unit sample from the object optimizer
                maps to this fraction of the friction-cone limit. Sets
                `action_scale`.
        """
        self.dt = dt
        self.goal = jnp.asarray(goal)
        self.footprint = footprint
        self.obstacles = obstacles or ObstacleField([])

        # Limit-surface compliance D and its inverse, the friction-cone
        # limit (the largest wrench the support surface can transmit).
        f_limit = mu * mass * gravity
        tau_limit = pressure_coeff * limit_surface_radius * f_limit
        self.wrench_limit = jnp.array([f_limit, f_limit, tau_limit])
        self.D = 1.0 / self.wrench_limit

        # A unit sample from the object optimizer -> physical wrench.
        self.action_scale = wrench_sample_fraction * self.wrench_limit

        self.w_pos, self.w_theta = w_pos, w_theta
        self.wf_pos, self.wf_theta = wf_pos, wf_theta
        self.w_effort = w_effort
        self.w_obstacle, self.obstacle_margin = w_obstacle, obstacle_margin
        self.boundary_samples = footprint.sample_boundary(
            boundary_samples_per_edge
        )

    @property
    def dim(self) -> int:
        """Dimension of the wrench / consensus variable."""
        return 3

    def step(self, pose: jax.Array, wrench: jax.Array) -> jax.Array:
        """One forward-Euler step of the limit-surface dynamics (eq. 5).

        A limit surface's own definition is the boundary between sticking
        (wrenches inside it produce zero relative motion) and slipping
        (wrenches at or beyond it do) -- but eq. 5's plain proportional
        formula extends across that boundary with no such cutoff, so a
        wrench well inside wrench_limit still predicts a small nonzero
        step. Diagnosed (2026-08-09/10) as the root cause of the
        near-goal stall: as position error shrinks, the object's optimal
        wrench under the smooth model shrinks with it, continuously,
        with nothing to stop it from settling below the real breakaway
        force -- confirmed against real runs, where the negotiated
        consensus wrench falls under the ~7.85N/0.47N*m threshold well
        before the goal, and the robot's realized wrench is then
        genuinely near-zero on 96-98% of steps in that regime, not just
        smaller. The sticking region is restored here: a wrench whose
        magnitude, normalized by wrench_limit component-wise, is inside
        the unit ball is treated as producing no motion at all, matching
        both MuJoCo's frictionloss and the real physics wrench_limit is
        already meant to describe.
        """
        normalized_mag = jnp.linalg.norm(wrench / self.wrench_limit)
        effective_wrench = jnp.where(
            normalized_mag < 1.0, jnp.zeros_like(wrench), wrench
        )
        new_pose = pose + self.dt * self.D * effective_wrench
        return new_pose.at[2].set(wrap_angle(new_pose[2]))

    def world_boundary(self, pose: jax.Array) -> jax.Array:
        """The footprint boundary samples transformed into the world frame."""
        return pose[:2] + rotate(pose[2], self.boundary_samples)

    def running_cost(self, pose: jax.Array, wrench: jax.Array) -> jax.Array:
        """Object stage cost: goal tracking + clearance + effort (eq. 18)."""
        cost = se2_distance_sq(pose, self.goal, self.w_pos, self.w_theta)
        cost += self.obstacles.hinge_cost(
            self.world_boundary(pose), self.w_obstacle, self.obstacle_margin
        )
        cost += self.w_effort * jnp.sum(wrench**2)
        return cost

    def terminal_cost(self, pose: jax.Array) -> jax.Array:
        """Object terminal cost: heavier goal tracking only (ell_f)."""
        return se2_distance_sq(pose, self.goal, self.wf_pos, self.wf_theta)


def t_shape_footprint(
    crossbar_half: Sequence[float] = (0.090, 0.015),
    stem_half: Sequence[float] = (0.015, 0.060),
    crossbar_y: float = 0.030,
    stem_y: float = -0.045,
) -> Polygon:
    """Build the outline of a capital-T footprint from its two box halves.

    Defaults match the `pusht_clutter` MJCF's `block_crossbar`/`block_stem`
    geoms, so the analytic footprint and the simulated geometry agree.

    Args:
        crossbar_half: Half-extents (x, y) of the crossbar box.
        stem_half: Half-extents (x, y) of the stem box.
        crossbar_y: Crossbar center offset along y.
        stem_y: Stem center offset along y.

    Returns:
        The T outline as a `Polygon`, wound counter-clockwise.
    """
    cx, cy = crossbar_half
    sx, sy = stem_half
    top, bot = crossbar_y + cy, crossbar_y - cy
    stem_bot = stem_y - sy
    return Polygon(
        jnp.array(
            [
                [-cx, top],
                [cx, top],
                [cx, bot],
                [sx, bot],
                [sx, stem_bot],
                [-sx, stem_bot],
                [-sx, bot],
                [-cx, bot],
            ]
        )
    )


def c_shape_footprint(
    half_width: float = 0.0350,
    half_height: float = 0.0515,
    half_stroke: float = 0.010,
) -> Polygon:
    """Build the outline of a block capital-C from its stroke dimensions.

    The C is a full-height spine down the left with a full-width bar at the
    top and another at the bottom. This is the exact union outline of the
    three boxes making up the `block` body in
    `models/xarm6_pusht_tabletop/icra_sign.xml`, so the analytic footprint
    and the simulated geometry agree there the same way `t_shape_footprint`
    agrees with the T scenes.

    Defaults match that MJCF: 0.070 x 0.103 m, the cap height and stroke of
    the IsaacGym repo's own `assets/urdf/alphabets` glyphs. Those are
    smoothly curved; this is a block-letter stand-in with the same bounding
    box, stroke width and single concavity. See `icra_sign.xml` for why a
    mesh is not used, and why the C rather than another letter.

    Args:
        half_width: Half the letter's overall width (x).
        half_height: Half the letter's overall height (y).
        half_stroke: Half the thickness of the spine and each bar.

    Returns:
        The C outline as a `Polygon`, wound counter-clockwise.
    """
    x, y, t = half_width, half_height, half_stroke
    # Inner faces: where the two bars end, and the spine's right face.
    top_bar_bot = y - 2.0 * t
    bot_bar_top = -y + 2.0 * t
    spine_inner = -x + 2.0 * t
    return Polygon(
        jnp.array(
            [
                [-x, -y],
                [x, -y],
                # Up the bottom bar's right end, then left along its top
                # face as far as the spine.
                [x, bot_bar_top],
                [spine_inner, bot_bar_top],
                # Up the spine's right face and out along the top bar.
                [spine_inner, top_bar_bot],
                [x, top_bar_bot],
                [x, y],
                [-x, y],
            ]
        )
    )
