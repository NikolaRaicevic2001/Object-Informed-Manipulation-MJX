"""The 2D push-T task: same ADMM formulation, analytic physics.

`PushT2D` is the 2D twin of `oim.tasks.pusht.PushT`. It implements the same
`ConsensusTask` contract against the analytic world in
`oim.worlds.sim2d.engine`,
and reuses `oim.objects.PlanarPushingObject` for the object block verbatim
-- so the object-level subproblem is *literally the same code* in 2D and in
MJX, and only the robot block's physics differs.

The object block samples the consensus wrench directly, matching the MJX
tasks: *where* the robot touches the object is the robot block's business,
and the object planner only needs to say what motion it wants. Setting
`contact_actions=True` switches it to deciding a contact action
`[p_x, p_y, f_n, f_t]` and deriving the wrench as `w = J_c^T f` instead,
which constrains every proposal to the friction cone but also makes the
object block choose a contact point it has no reason to care about. Both
are available so the two can be compared.
"""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from oim.objects import (
    CONTACT_ACTION_DIM,
    ObstacleField,
    PlanarPushingObject,
    Polygon,
    Shape,
    contact_action_to_wrench,
    estimate_contact_point,
    pack_contact_action,
    project_contact_action,
    sample_contact_actions,
    se2_distance_sq,
    unpack_contact_action,
)
from oim.task_base import ConsensusTask
from oim.worlds.sim2d.engine import Analytic2DRollout, Sim2DModel, Sim2DState


class PushT2D(ConsensusTask):
    """Push a planar object to an SE(2) goal with a velocity-controlled robot.

    Deliberately *not* a `Task`: there is no MuJoCo model here. It exposes
    the handful of attributes a `SamplingBasedController` actually reads off
    a task (`model.nu`, `dt`, `u_min`, `u_max`) plus the `ConsensusTask`
    interface, which is all `ADMM` ever touches.
    """

    def __init__(
        self,
        footprint: Polygon,
        goal: Sequence[float],
        obstacles: Optional[Sequence[Shape]] = None,
        dt: float = 0.05,
        mu: float = 0.4,
        mass: float = 2.0,
        limit_surface_radius: float = 0.06,
        mu_c: float = 0.5,
        f_max: float = 4.0,
        robot_radius: float = 0.012,
        robot_max_speed: float = 0.6,
        n_substeps: int = 4,
        contact_actions: bool = False,
        sigma_p: float = 0.012,
        sigma_fn: float = 0.7,
        sigma_ft: float = 0.3,
        tau_n: float = 0.7,
        relocate_contact: bool = True,
        n_p_est: int = 48,
        r_est: int = 4,
        h_p_est: int = 5,
        w_ee: float = 20.0,
        r0: float = 0.05,
        r_r: float = 0.05,
        q_pos: float = 40.0,
        q_theta: float = 10.0,
        qf_pos: float = 500.0,
        qf_theta: float = 150.0,
        w_obstacle_robot: float = 60000.0,
        obstacle_margin: float = 0.015,
        obstacle_decay: float = 0.02,
        local_goal: bool = False,
    ) -> None:
        """Configure geometry, physics, and cost weights.

        Args:
            footprint: The object's outline in its body frame.
            goal: Goal pose [x, y, theta].
            obstacles: Static obstacles. None means an empty scene.
            dt: Planning and execution timestep.
            mu: Object/support friction coefficient.
            mass: Object mass.
            limit_surface_radius: Limit-surface characteristic radius.
            mu_c: Robot/object Coulomb friction coefficient.
            f_max: Cap on the normal contact force.
            robot_radius: Radius of the disc robot.
            robot_max_speed: Cap on the robot's speed. Bounds the sampler
                per-axis and, more importantly, saturates the *magnitude*
                in the plant, so a diagonal command cannot reach sqrt(2)
                times this value.
            n_substeps: Contact substeps per timestep.
            contact_actions: Whether the object block decides a contact
                action `[p, f_n, f_t]` and derives the wrench from it, or
                (default) samples the wrench directly as the MJX tasks do.
                Direct wrench is the default because *where* the robot
                touches the object is the robot block's concern, not the
                object planner's -- the object block only needs to say what
                motion it wants. Set True to A/B the two.
            sigma_p: Contact-point sampling std-dev.
            sigma_fn: Normal-force sampling std-dev.
            sigma_ft: Tangential-force sampling std-dev.
            tau_n: Normal-alignment threshold for contact-point rejection.
            relocate_contact: Whether to re-choose the contact point by a
                global search over the boundary each control step. Off
                means the contact can only drift along the face it is
                already on, which cannot route the object around clutter.
            n_p_est: Candidates per contact-point search iteration.
            r_est: Contact-point search iterations.
            h_p_est: Lookahead steps used to score a candidate contact.
            w_ee: Weight pulling the robot toward the object.
            r0: Radius inside which the approach term goes slack.
            r_r: Control-effort weight.
            q_pos: Running weight on object translational goal error.
            q_theta: Running weight on object rotational goal error.
            qf_pos: Terminal weight on translational goal error.
            qf_theta: Terminal weight on rotational goal error.
            w_obstacle_robot: Weight on the robot's own obstacle clearance.
            obstacle_margin: Clearance below which the ROBOT hinge activates.
            obstacle_decay: e-folding length of the object proximity cost.
            local_goal: Whether the robot block's goal tracking aims at the
                object block's horizon endpoint x^{o*}_H instead of the
                global goal. Mirrors `PushT`'s own flag -- see it for the
                reasoning; kept here so the analytic world can reproduce
                the same formulation change, which is the whole point of
                having it (`README_ADMM.md` §10).
        """
        self.goal = jnp.asarray(goal, dtype=float)
        self.footprint = footprint
        self.obstacle_field = ObstacleField(list(obstacles or []))
        self.dt = dt
        self.contact_actions = contact_actions
        self.mu_c = mu_c
        self.f_max = f_max
        self.sigma_p = sigma_p
        self.sigma_fn = sigma_fn
        self.sigma_ft = sigma_ft
        self.tau_n = tau_n
        self.relocate_contact = relocate_contact
        self.n_p_est, self.r_est, self.h_p_est = n_p_est, r_est, h_p_est
        # Candidate seeds spanning the whole boundary, so the contact-point
        # search can jump between faces rather than only refine locally.
        self._boundary_candidates = footprint.sample_boundary(6)
        self.w_ee, self.r0, self.r_r = w_ee, r0, r_r
        # Same as PushT's _ell_r (paper eq. 21): not exposed as constructor
        # args there either, so kept hardcoded here too for parity.
        self.w_align, self.gamma0 = 5.0, jnp.cos(jnp.pi / 6)
        self.q_pos, self.q_theta = q_pos, q_theta
        self.qf_pos, self.qf_theta = qf_pos, qf_theta
        self.w_obstacle_robot = w_obstacle_robot
        self.obstacle_margin = obstacle_margin
        self.use_local_goal = local_goal

        # The object block: the same analytic model the MJX task uses.
        self.object_model = PlanarPushingObject(
            dt=dt,
            goal=self.goal,
            footprint=footprint,
            obstacles=self.obstacle_field,
            mu=mu,
            mass=mass,
            limit_surface_radius=limit_surface_radius,
            obstacle_decay=obstacle_decay,
        )

        # `model` plays the exact role `mjx.Model` plays for a MuJoCo task:
        # the controllers read `.nu` off it and the rollout backend steps
        # it. `sim_model` is kept as an alias for readability at call sites
        # that mean "the physics", not "the thing the controller sizes to".
        self.model = Sim2DModel.create(
            limit_surface_d=self.object_model.D,
            mu_c=mu_c,
            f_max=f_max,
            robot_radius=robot_radius,
            nu=2,
        )
        self.sim_model = self.model
        self.u_min = -robot_max_speed * jnp.ones(2)
        self.u_max = robot_max_speed * jnp.ones(2)
        self.rollout = Analytic2DRollout(
            shape=footprint,
            obstacles=self.obstacle_field.shapes,
            dt=dt,
            n_substeps=n_substeps,
            obstacle_margin=obstacle_margin,
            max_speed=robot_max_speed,
        )

    # ------------------------------------------------------------------
    # Task-side plumbing the controllers expect
    # ------------------------------------------------------------------

    def make_data(
        self,
        object_pose: Sequence[float] = (0.0, 0.0, 0.0),
        robot_pos: Sequence[float] = (0.0, 0.0),
    ) -> Sim2DState:
        """Create an initial state, mirroring `Task.make_data`."""
        return Sim2DState.create(object_pose=object_pose, robot_pos=robot_pos)

    def get_trace_sites(self, state: Sim2DState) -> jax.Array:
        """The robot position, lifted to 3D so trace plumbing matches MJX."""
        return jnp.concatenate([state.robot_pos, jnp.zeros(1)])[None, :]

    def domain_randomize_model(self, rng: jax.Array) -> dict:
        """No randomization by default."""
        return {}

    def domain_randomize_data(self, data: Sim2DState, rng: jax.Array) -> dict:
        """No randomization by default."""
        return {}

    # ------------------------------------------------------------------
    # ConsensusTask: object block
    # ------------------------------------------------------------------

    @property
    def consensus_dim(self) -> int:
        """The consensus variable is the planar wrench [f_x, f_y, tau]."""
        return 3

    @property
    def object_action_dim(self) -> int:
        """4 for a contact action, else 3 (the wrench itself)."""
        return CONTACT_ACTION_DIM if self.contact_actions else 3

    def consensus_scale(self) -> jax.Array:
        """Friction-cone limit, normalizing the ADMM penalty and residuals."""
        return self.object_model.wrench_limit

    def object_action_scale(self) -> jax.Array:
        """Unit scaling for contact actions; wrench scaling otherwise.

        With the contact parameterization the action carries physical units
        already (metres and newtons), and `project_contact_action` bounds
        it, so there is nothing to rescale.
        """
        if self.contact_actions:
            return jnp.ones(CONTACT_ACTION_DIM)
        return self.object_model.action_scale

    def object_action_bounds(self) -> Tuple[jax.Array, jax.Array]:
        """Box bounds; the friction cone itself is handled by projection.

        Direct-wrench mode defers to the base class (+/- consensus_scale).
        Contact-action mode needs its own box, since the action isn't the
        wrench itself.
        """
        if not self.contact_actions:
            return super().object_action_bounds()
        reach = self.footprint.bounding_radius
        lo = jnp.array([-reach, -reach, 0.0, -self.mu_c * self.f_max])
        hi = jnp.array([reach, reach, self.f_max, self.mu_c * self.f_max])
        return lo, hi

    def initial_object_action(self) -> Optional[jax.Array]:
        """A valid starting contact: a real boundary point, not the origin.

        Taken from `sample_boundary` rather than by projecting some interior
        seed, because the footprint's origin lies on its medial axis where
        projection provably stalls (see `Shape.project_to_boundary`). The
        specific face chosen barely matters -- the rejection sampler and the
        MPPI update migrate the contact along the boundary from here -- it
        only has to be *on* the surface so the normal is well defined.
        """
        if not self.contact_actions:
            return None
        samples = self.footprint.sample_boundary(4)
        start = samples[jnp.argmin(samples[:, 1])]  # lowest point in body y
        return jnp.array([start[0], start[1], 0.25 * self.f_max, 0.0])

    def object_action_to_consensus(
        self, obj_state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Derive the wrench: w = J_c^T f for a contact action."""
        if self.contact_actions:
            return contact_action_to_wrench(self.footprint, obj_state, action)
        return action * self.object_action_scale()

    def project_object_action(
        self, action: jax.Array, obj_state: Optional[jax.Array] = None
    ) -> jax.Array:
        """Keep contact on the boundary and force inside the friction cone.

        obj_state is unused here -- this projection is unconditional,
        unlike PushT's own goal-proximity-gated override; accepted only
        to match the (task_base.py) interface's now-optional second arg.
        """
        if not self.contact_actions:
            return action
        return project_contact_action(
            self.footprint, action, self.mu_c, self.f_max
        )

    def _object_goal_cost(self, poses: jax.Array) -> jax.Array:
        """Score a batch of object poses: goal tracking + obstacle clearance.

        The zero wrench drops the effort term, leaving exactly the part of
        the object stage cost that depends on *where the object ends up* --
        which is what ranks candidate contact points.
        """
        return jax.vmap(
            lambda p: self.object_model.running_cost(p, jnp.zeros(3))
        )(poses)

    def sample_object_actions(
        self,
        nominal: jax.Array,
        rng: jax.Array,
        num_samples: int,
        obj_state: jax.Array,
    ) -> Optional[jax.Array]:
        """Boundary-aware contact sampling; None to defer to the optimizer."""
        if not self.contact_actions:
            return None
        if self.relocate_contact:
            # Re-seed the whole horizon's contact point by a global search
            # over the boundary before sampling locally around it. Without
            # this the rejection filter pins the contact to whichever face
            # it started on, so the object block can never decide to push
            # from somewhere else -- which is what routing around an
            # obstacle requires.
            rng, est_rng = jax.random.split(rng)
            p0 = estimate_contact_point(
                shape=self.footprint,
                pose=obj_state,
                goal_cost_fn=self._object_goal_cost,
                rng=est_rng,
                candidates=self._boundary_candidates,
                f_n=0.5 * self.f_max,
                num_samples=self.n_p_est,
                iterations=self.r_est,
                horizon=self.h_p_est,
            )
            _, f_n_nom, f_t_nom = unpack_contact_action(nominal)
            nominal = pack_contact_action(
                jnp.broadcast_to(p0, (nominal.shape[0], 2)), f_n_nom, f_t_nom
            )
        return sample_contact_actions(
            shape=self.footprint,
            nominal=nominal,
            rng=rng,
            num_samples=num_samples,
            sigma_p=self.sigma_p,
            sigma_fn=self.sigma_fn,
            sigma_ft=self.sigma_ft,
            mu_c=self.mu_c,
            f_max=self.f_max,
            tau_n=self.tau_n,
        )

    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Quasi-static limit-surface dynamics, shared with the MJX task."""
        return self.object_model.step(obj_state, w)

    def object_running_cost(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """Goal tracking + obstacle clearance + effort, shared with MJX."""
        return self.object_model.running_cost(obj_state, w)

    def object_terminal_cost(self, obj_state: jax.Array) -> jax.Array:
        """Heavier goal tracking, shared with MJX."""
        return self.object_model.terminal_cost(obj_state)

    # ------------------------------------------------------------------
    # ConsensusTask: robot block
    # ------------------------------------------------------------------

    def object_state_from_robot(self, state: Sim2DState) -> jax.Array:
        """The object pose lives directly on the 2D state."""
        return state.object_pose

    def realized_consensus(self, state: Sim2DState) -> jax.Array:
        """A^r: the wrench the contact solver actually produced.

        The analytic counterpart of reading MJX's contact forces -- and
        unlike the MJX side, it needs no twist-inversion workaround, since
        the wrench is computed explicitly by `resolve_contact`.
        """
        return state.wrench

    def tracking_goal(
        self, pose: jax.Array, local_goal: Optional[jax.Array]
    ) -> jax.Array:
        """What goal tracking aims at; see `PushT.tracking_goal`.

        `pose` is accepted and ignored. That version uses it to snap back
        to the global goal inside the shaping-fade radius, and there is no
        fade here (see `robot_running_cost`) -- giving the disc a snap
        radius would mean inventing a distance this task has no other use
        for, so the local goal holds all the way in. The argument stays in
        the signature because this is the public interface both tasks
        offer: `oim.runtime.logs.local_goal_marker` calls it positionally
        on whichever task it is handed, and an arity that varied by task
        would surface as a TypeError inside a viewer loop.
        """
        del pose
        if self.use_local_goal and local_goal is not None:
            return local_goal
        return self.goal

    def robot_running_cost(
        self,
        state: Sim2DState,
        control: jax.Array,
        obj_ref_t: jax.Array,
        local_goal: Optional[jax.Array] = None,
        weight_scale: jax.Array = 1.0,
    ) -> jax.Array:
        """J_r = effort + approach + align + clearance + goal + coupling.

        Mirrors `PushT.robot_running_cost`, minus the terms that only mean
        something for a 3D end-effector (tilt, tip height) -- there is no
        orientation or height for a point/disc robot to shape. The ADMM
        consensus penalty is *not* added here -- the ADMM layer adds it.

        There is no `shaping_fade` here to hold back on the global goal
        (the disc has no posture to fade), so `ell_o` is the only term
        `local_goal` reaches.
        """
        pose = state.object_pose
        robot = state.robot_pos

        effort = self.r_r * jnp.sum(control**2)
        d_ee = jnp.sum((robot - pose[:2]) ** 2)
        approach = self.w_ee * jnp.clip(d_ee - self.r0**2, 0.0, None)

        to_object = pose[:2] - robot
        to_ref = obj_ref_t[:2] - pose[:2]
        cos_angle = jnp.sum(to_object * to_ref) / (
            jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
        )
        align = self.w_align * jnp.clip(self.gamma0 - cos_angle, 0.0, None)

        clearance = self.obstacle_field.hinge_cost(
            robot[None, :], self.w_obstacle_robot, self.obstacle_margin
        )
        target = self.tracking_goal(pose, local_goal)
        ell_o = se2_distance_sq(pose, target, self.q_pos, self.q_theta)
        ell_c = se2_distance_sq(pose, obj_ref_t, self.q_pos, self.q_theta)
        return effort + approach + align + clearance + ell_o + ell_c

    def robot_terminal_cost(
        self,
        state: Sim2DState,
        local_goal: Optional[jax.Array] = None,
        weight_scale: jax.Array = 1.0,
    ) -> jax.Array:
        """Heavier goal tracking, matching the object block's terminal cost."""
        pose = state.object_pose
        target = self.tracking_goal(pose, local_goal)
        return weight_scale * se2_distance_sq(
            pose, target, self.qf_pos, self.qf_theta
        )
