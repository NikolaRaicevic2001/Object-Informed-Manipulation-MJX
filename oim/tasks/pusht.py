from typing import Dict, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from oim import ROOT
from oim.objects import PlanarPushingObject, se2_distance_sq
from oim.task_base import ConsensusTask, Task
from oim.utils.scenes import SCENES

# Cost weights, in one place because several of them must be *identical* on
# the two ADMM blocks. `q_*`/`qf_*` are read by both `robot_running_cost`
# (via `ell_o`/`ell_c`) and `PlanarPushingObject`'s own goal tracking: the
# blocks negotiate a wrench toward a shared objective, so a run where they
# differ is one where the two halves are pulling toward different targets.
# They used to be written out twice -- here and as `PlanarPushingObject`'s
# defaults -- and agreed only by coincidence.
#
# `oim/configs/{robot}.yaml`'s `costs:` block overrides any subset of this;
# anything it omits keeps the value below, so a task constructed directly
# (the tests, a notebook) behaves exactly as it always did.
DEFAULT_COSTS = {
    # Shared by both blocks.
    "q_pos": 40.0,  # running goal tracking, translation
    "q_theta": 10.0,  # running goal tracking, rotation
    "qf_pos": 500.0,  # terminal goal tracking, translation
    "qf_theta": 150.0,  # terminal goal tracking, rotation
    # Object block only.
    "w_effort": 0.01,  # squared wrench
    "w_obstacle": 60000.0,  # clearance hinge on the object's footprint
    "obstacle_margin": 0.015,  # clearance below which that hinge activates
    # Robot block only (paper eq. 20-22).
    "r_r": 0.05,  # squared control effort
    "w_ee": 40.0,  # approach: pull the tip toward the object
    "r0": 0.02,  # radius inside which approach goes slack
    "w_align": 15.0,  # stay behind the object relative to the reference
    "gamma0_deg": 15.0,  # alignment cone half-angle
    "w_tilt": 30.0,  # keep the stick pointing down (3D only)
    "w_tip_z": 8.0,  # keep the tip at the block's mid-height (3D only)
    # Fade align/tilt/tip_z as ||p - p_g|| → 0 (0 = disabled). Approach
    # is never faded. See `shaping_fade`.
    "shaping_fade_dist": 0.0,
}


def resolve_costs(costs: Optional[Dict[str, float]]) -> Dict[str, float]:
    """`DEFAULT_COSTS` with `costs` applied over it, rejecting typos.

    Args:
        costs: Overrides for any subset of `DEFAULT_COSTS`, or None.

    Returns:
        The full weight mapping.

    Raises:
        ValueError: If `costs` names a weight `DEFAULT_COSTS` has not.
            Ignoring it would leave the run using defaults while its run
            file advertised the tuning that was asked for.
    """
    unknown = sorted(set(costs or {}) - set(DEFAULT_COSTS))
    if unknown:
        raise ValueError(
            f"unknown cost weight(s) {unknown}; "
            f"known: {sorted(DEFAULT_COSTS)}"
        )
    return {**DEFAULT_COSTS, **(costs or {})}


class PushT(Task, ConsensusTask):
    """Push a T-shaped block to a desired pose, optionally through clutter.

    With `clutter=False` (default), loads the plain `models/pusht` scene and
    supports ordinary sampling-based MPC (`running_cost`/`terminal_cost`).

    With `clutter=True`, loads `models/pusht_clutter` (static obstacles, and
    a model whose joint friction is tuned to match the analytic limit-surface
    object model) and additionally implements `ConsensusTask`, so it can be
    driven by `oim.algs.admm.ADMM`. The object-level subproblem is
    delegated to `oim.objects.PlanarPushingObject`.

    `robot` selects the embodiment: `"point"` (default) is the original
    free 2-DOF point-mass pusher; `"xarm6"` swaps that for a real 6-DoF
    UFACTORY xArm6 with a rigid pushing-stick end-effector. Only meaningful
    with `clutter=True` -- there is no non-cluttered xArm6 scene. The two
    embodiments share every method below except the handful that read the
    "pusher position" or realize the contact wrench, which branch on
    `self.robot`; everything about the object side (goal, obstacles,
    limit-surface dynamics/costs) is exactly the same physics regardless of
    which robot is pushing.

    `env` selects which scene to load, by name, from the
    `oim.utils.scenes.SCENES` registry -- `PushT` itself holds no
    scene-specific data or branching; it asks the registry for one
    `SceneSpec` (MJCF path per embodiment, goal, obstacles, footprint, and
    the xArm6 base placement) and wraps cost functions/ADMM plumbing around
    whatever it's handed. Adding an environment is a new `SCENES` entry
    plus its own MJCF, never a change here.
    """

    def __init__(
        self,
        impl: str = "jax",
        clutter: bool = False,
        planning_dt: Optional[float] = None,
        robot: Literal["point", "xarm6"] = "point",
        consensus_source: Literal["twist", "contact"] = "twist",
        env: str = "clutter",
        goal: Optional[Sequence[float]] = None,
        costs: Optional[Dict[str, float]] = None,
        realized_wrench_clip: Optional[Sequence[float]] = None,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            impl: The backend implementation for rollouts ("jax" or "warp").
            clutter: Whether to load `env`'s scene (with obstacles) and
                enable the ADMM `ConsensusTask` methods.
            planning_dt: If given, overrides the model's simulation timestep.
                Used to run the planner at a coarser rate than execution.
            robot: Which embodiment pushes the block, `"point"` (default,
                the original free 2-DOF pusher) or `"xarm6"` (a real 6-DoF
                arm). Ignored (must be `"point"`) when `clutter=False`.
            consensus_source: How the robot block estimates A^r. `"twist"`
                (default) inverts the limit-surface relation, `w = D^-1
                xdot^o`; works on both backends and both embodiments, and is
                continuous through contact breaks. `"contact"` reads the
                simulator's constraint force literally, matching the paper's
                wording, but is only valid for `robot="point"`.
            env: Which scene to load, by name from
                `oim.utils.scenes.SCENES` (only meaningful with
                `clutter=True`). Must support `robot`, or raises.
            goal: Overrides the scene's own goal pose, world-frame SE(2)
                `[x, y, theta]`. Used by `examples/poses/<task>.yaml` to
                run one scene against several goals. The goal marker in
                the MJCF is a mocap body, moved separately by
                `oim.sim3d.build`; this sets what the *costs* aim at.
            costs: Overrides for any subset of `DEFAULT_COSTS`, normally
                the `costs:` block of `oim/configs/{robot}.yaml`. One
                mapping feeds both ADMM blocks, so the shared goal-tracking
                weights cannot drift apart between them. Unknown keys
                raise: a misspelled weight would otherwise be ignored in
                silence and the run file would report a tuning that never
                happened.
            realized_wrench_clip: [f_x, f_y, tau] bound for
                `realized_consensus`'s clip, or `None` (default) to use
                `object_model.wrench_limit` (the friction-cone limit) as
                before -- unaffected for every existing caller. Separate
                from `wrench_limit` on purpose: that value also sets the
                object block's own action bounds and the ADMM dual clip
                (`consensus_scale`), so widening it to stop the robot's
                *estimate* from saturating would silently widen those too.
                `consensus_source="contact"` (point robot) reads
                `qfrc_constraint` literally, which sustains near or above
                the friction-cone limit under real contact, not just
                spiking at onset -- clipping tightly to `wrench_limit`
                there pins the estimate at the ceiling for many consecutive
                steps rather than only during the brief onset transient
                the clip was originally added for (see
                `realized_consensus`'s own docstring).

        Raises:
            ValueError: If `costs` names a weight `DEFAULT_COSTS` has not.
        """
        if robot not in ("point", "xarm6"):
            raise ValueError(f"robot must be 'point' or 'xarm6', got {robot!r}")
        if robot == "xarm6" and not clutter:
            raise ValueError("robot='xarm6' requires clutter=True")
        if consensus_source not in ("twist", "contact"):
            raise ValueError(
                "consensus_source must be 'twist' or 'contact', got "
                f"{consensus_source!r}"
            )
        if consensus_source == "contact" and robot != "point":
            raise ValueError(
                "consensus_source='contact' is only valid for robot='point'; "
                "an articulated arm's contact force appears as J^T f spread "
                "across its joints, not at a single pair of DOFs."
            )

        cost = resolve_costs(costs)
        self.costs = cost
        self.clutter = clutter
        self.robot = robot
        self.consensus_source = consensus_source
        self.env = env
        if not clutter:
            scene_path = "pusht/scene.xml"
        else:
            if env not in SCENES:
                raise ValueError(
                    f"env={env!r} is not in oim.utils.scenes.SCENES "
                    f"(available: {sorted(SCENES)})"
                )
            spec = SCENES[env]
            scene_path = spec.mjcf_scene(robot)
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/" + scene_path)
        if planning_dt is not None:
            mj_model.opt.timestep = planning_dt

        if robot == "xarm6":
            # Ground-mounted base placement, not baked into xarm6.xml itself
            # (that file is a reusable, placement-agnostic robot asset) --
            # same pattern as overriding opt.timestep above: mutate the
            # loaded mj_model before it's handed to mjx. Each scene has its
            # own mount, since the workspace moves between them.
            base_id = mj_model.body("xarm6_link_base").id
            mj_model.body_pos[base_id] = [*spec.xarm6_base_pos, 0.0]
            yaw = jnp.deg2rad(spec.xarm6_base_yaw_deg)
            mj_model.body_quat[base_id] = [
                float(jnp.cos(yaw / 2)),
                0.0,
                0.0,
                float(jnp.sin(yaw / 2)),
            ]
            trace_site = "xarm6_tip"
        else:
            trace_site = "pusher"
        super().__init__(mj_model, trace_sites=[trace_site], impl=impl)

        # Sensor ids (defined identically in all three scenes).
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

        if clutter:
            if robot == "xarm6":
                # Block qpos addresses looked up explicitly, not assumed to
                # be qpos[:3] -- unlike pusht_clutter.xml (block declared
                # before the pusher), the composed xarm6 scene compiles the
                # arm's 5 joints first, so the block's SE(2) pose actually
                # lands at qpos[5:8].
                self.block_qpos_adr = jnp.array(
                    [
                        mj_model.joint("T_x").qposadr[0],
                        mj_model.joint("T_y").qposadr[0],
                        mj_model.joint("T_z").qposadr[0],
                    ]
                )
                self.tip_site_id = mj_model.site("xarm6_tip").id
                self.stick_body_id = mj_model.body("xarm6_stick").id
                self.block_body_id = mj_model.body("block").id
            else:
                pusher_x_dof = mj_model.joint("root_x").dofadr[0]
                pusher_y_dof = mj_model.joint("root_y").dofadr[0]
                self.pusher_dofs = jnp.array([pusher_x_dof, pusher_y_dof])
                # qpos[3:5] is the root_x/root_y slide joints' displacement
                # from the pusher body's own declared XML pos, not its
                # world position -- _pusher_pos needs the latter (see its
                # own fix below), so the body id to read it from is
                # captured here, the same way tip_site_id is for xarm6.
                self.pusher_body_id = mj_model.body("pusher").id

            # The block's own velocity DOFs, used by the default
            # ("twist") consensus extraction. Looked up by joint name so
            # it is correct for both embodiments' qpos/qvel layouts.
            self.block_dofs = jnp.array(
                [
                    mj_model.joint("T_x").dofadr[0],
                    mj_model.joint("T_y").dofadr[0],
                    mj_model.joint("T_z").dofadr[0],
                ]
            )

            # goal/obstacles/footprint come from the scene registry (see
            # oim.utils.scenes) -- the only things that differ between
            # scenes. mu/mass/limit_surface_radius stay fixed (physics of
            # the block/table, not its shape), chosen so the friction-cone
            # limit mu*m*g equals the block joints' `frictionloss` in the
            # MJCF for every scene's geoms.
            # One goal pose feeds both blocks' costs; a pose file overrides
            # it per run. Both must read the same array or the two ADMM
            # blocks would negotiate a wrench toward different targets.
            goal_pose = (
                spec.goal if goal is None else jnp.asarray(goal, dtype=float)
            )
            # The object block's goal weights are the robot block's, from
            # the same mapping -- the two negotiate one wrench, so they
            # have to be aiming at the same thing.
            self.object_model = PlanarPushingObject(
                dt=self.dt,
                goal=goal_pose,
                footprint=spec.footprint(),
                obstacles=spec.obstacles,
                mu=0.4,
                mass=2.0,
                limit_surface_radius=0.06,
                w_pos=cost["q_pos"],
                w_theta=cost["q_theta"],
                wf_pos=cost["qf_pos"],
                wf_theta=cost["qf_theta"],
                w_effort=cost["w_effort"],
                w_obstacle=cost["w_obstacle"],
                obstacle_margin=cost["obstacle_margin"],
            )
            self._realized_wrench_clip = (
                jnp.asarray(realized_wrench_clip, dtype=float)
                if realized_wrench_clip is not None
                else self.object_model.wrench_limit
            )

            # Robot-level cost weights (paper eq. 20).
            self.r_r = cost["r_r"]
            self.w_ee, self.r0 = cost["w_ee"], cost["r0"]
            self.w_align = cost["w_align"]
            self.gamma0 = jnp.cos(jnp.deg2rad(cost["gamma0_deg"]))
            # w_tilt/w_tip_z: not in the paper. w_tilt raised from 5.0 once
            # _tilt's sign bug was fixed (task 11/12) -- at 5.0 the tip
            # still averaged ~35 degrees off vertical -- then 20.0, 50.0
            # (which cost too much task performance) and back to 30.0.
            # None of that worked: measured over five 500-step runs the
            # tilt angle is a random walk that goes *up* on 52-55% of
            # steps, total variation ~8 rad for a net drift of ~1.3, and
            # the mean tilt rank-orders exactly with the final position
            # error across all five scenes. A linear penalty has a
            # constant restoring gradient, which cannot arrest a drift
            # whose source is that psi >= 0 has a reflecting boundary at
            # zero; the weight is not the free parameter here, the
            # functional form is.
            self.w_tilt = cost["w_tilt"]
            self.w_tip_z = cost["w_tip_z"]
            self.shaping_fade_dist = float(cost["shaping_fade_dist"])
            # Target tip height: the block's own resting z, read from the
            # model rather than hardcoded.
            self.tip_target_z = float(mj_model.body("block").pos[2])
            self.q_pos, self.q_theta = cost["q_pos"], cost["q_theta"]
            self.qf_pos, self.qf_theta = cost["qf_pos"], cost["qf_theta"]
            self.goal = goal_pose

    # ------------------------------------------------------------------
    # Plain (non-ADMM) sampling-based MPC interface
    # ------------------------------------------------------------------

    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the block relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the block relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher relative to the block."""
        block_pos = self._block_pose(state)[:2]
        pusher_pos = self._pusher_pos(state)
        if self.robot == "point":
            pusher_pos = pusher_pos + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) for plain (non-ADMM) MPC.

        Reuses `_ell_r`'s approach/align/tilt shaping for both embodiments,
        with `self.goal` standing in for the object planner's reference
        (plain MPC has no object-level plan) -- the same formula
        `robot_running_cost` already uses for ADMM, xarm6 or point, and the
        one README.md documents (paper eq. 21). `robot="point"` used to
        take a separate, simpler sensor-based formula here with no align
        term (`_get_position_err`/`_get_orientation_err`/
        `_close_to_block_err`, still used by tests/test_pusht.py, just no
        longer by this method) -- align is exactly what keeps the pusher
        *behind* the object relative to the goal, and its absence was
        measured letting the pusher park anywhere near the block, including
        the wrong side, without any push-worthy contact ever resulting.

        The obstacle clearance hinge is the same term the ADMM object
        block scores (eq. 18). Without it, flat MPPI only learns about an
        obstacle *after* a rollout wedges the block against it: physics
        blocks progress but nothing marks near-obstacle states as bad in
        advance, so trajectories that skirt an obstacle by 1 mm and by
        5 cm score identically. In rollouts the block cannot penetrate,
        so the hinge only fires inside the margin: a soft clearance
        buffer, not a cliff. Zero on obstacle-free scenes (open_table),
        for both embodiments now that they share this formula.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        ell_o = se2_distance_sq(pose, self.goal, self.q_pos, self.q_theta)
        obj = self.object_model
        obstacle = obj.obstacles.hinge_cost(
            obj.world_boundary(pose),
            obj.w_obstacle,
            obj.obstacle_margin,
        )
        # Same hinge, for the pusher's own position -- the block's
        # world boundary above keeps *it* out of obstacles (including
        # the robot-base circle _tee_scene adds), but nothing kept the
        # pusher itself from being commanded straight through one.
        # Diagnosed after repeated shelf_gap runs kept routing the
        # block around the outside of a shelf rather than through the
        # gap even with the block-side hinge active and the gap
        # widened -- the block's path is obstacle-aware, but the
        # pusher chasing "behind the block" (the align term) had no
        # reason to avoid cutting through the shelf or the robot-base
        # circle to get there.
        pusher_obstacle = obj.obstacles.hinge_cost(
            pusher_pos,
            obj.w_obstacle,
            obj.obstacle_margin,
        )
        ell_r = self._ell_r(state, pose, pusher_pos, self.goal)
        return ell_o + obstacle + pusher_obstacle + ell_r

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T) for plain (non-ADMM) MPC.

        Heavier SE(2) goal tracking (`qf_*`) **plus** the same
        contact-shaping ℓ_r as the stage cost, for both embodiments --
        originally xarm6-only (2026-08-10), extended to point here so
        the same fix applies to both flat-MPPI baselines rather than
        leaving point on the older terminal=running_cost formula.

        Stage costs are multiplied by `dt` in the rollout; the terminal
        term is not. That made the old `terminal = running_cost` the main
        place approach/align/tilt were scored at full weight. Replacing it
        with goal-only ℓ_f let MPPI buy a better predicted pose at the
        horizon by abandoning "stay behind the object" / stick posture —
        and without that geometry the push cannot finish. Measured on
        open_table (xarm6): goal-only terminal with qf=2000 moved final
        error from 0.07 m to 0.98 m.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        ell_f = se2_distance_sq(pose, self.goal, self.qf_pos, self.qf_theta)
        return ell_f + self._ell_r(state, pose, pusher_pos, self.goal)

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        if self.clutter and self.robot == "xarm6":
            # More headroom than the point-mass case: the arm has its own
            # (mostly-excluded) self-contact pairs in addition to the
            # stick/block/obstacle contacts, and a too-small allocation
            # silently drops contacts rather than erroring.
            #
            # With MuJoCo Warp, `naconmax`/`njmax` are *batch* arenas shared
            # by all parallel rollouts, so they must grow with
            # `num_samples`. 2048 was enough for 128 samples; at 512 the
            # broadphase asked for ~3456 and then dropped contacts. 8192
            # covers 512 with margin. `njmax` was unset until a run asked
            # for 67 (`nefc overflow`); 128 leaves headroom.
            return super().make_data(nconmax=256, naconmax=8192, njmax=128)
        if self.clutter:
            # Enough contact slots for the pusher, block, and 3 obstacles;
            # the default is too small and silently drops contacts.
            return super().make_data(nconmax=128, naconmax=1024)
        return super().make_data(nconmax=6000)

    # ------------------------------------------------------------------
    # ConsensusTask (ADMM) interface -- only meaningful when clutter=True
    # ------------------------------------------------------------------

    def _block_pose(self, state: mjx.Data) -> jax.Array:
        if self.robot == "xarm6":
            return state.qpos[self.block_qpos_adr]
        return state.qpos[:3]

    @property
    def block_qpos_indices(self) -> jax.Array:
        """Where the object's SE(2) pose sits in `qpos`, per embodiment.

        The arm's five joints compile first, so the block lands at
        `qpos[5:8]`; the point pusher's scene declares the block first, so
        it is `qpos[:3]`. Read by anything writing a start pose in.
        """
        if self.robot == "xarm6":
            return self.block_qpos_adr
        return jnp.array([0, 1, 2])

    def _pusher_pos(self, state: mjx.Data) -> jax.Array:
        """World-frame (x, y) position of the pusher's contact point."""
        if self.robot == "xarm6":
            return state.site_xpos[self.tip_site_id, :2]
        # Was state.qpos[3:5]: that is the root_x/root_y joints'
        # displacement from the pusher body's own declared XML pos, not
        # its world position -- wrong whenever that pos is nonzero (every
        # point-robot scene, including the original clutter). Confirmed
        # directly: qpos[3:5]=[0,0] at reset while the true world xpos is
        # [0,-0.18] (tabletop scenes) / [-0.05,-0.06] (clutter). Every
        # cost term reading pusher_pos (_ell_r's approach/align,
        # _close_to_block_err) was silently computing against this wrong,
        # constant-offset position for the whole episode, not just at
        # reset -- e.g. approach cost measured 0.0 (already "arrived")
        # against a true value of 1.28 at the very first step of
        # open_table, so MPPI had essentially no real incentive to
        # approach the block at all, on any point-robot scene, ever.
        return state.xpos[self.pusher_body_id, :2]

    @property
    def consensus_dim(self) -> int:
        """The consensus variable is the planar wrench [f_x, f_y, tau]."""
        return 3

    def consensus_scale(self) -> jax.Array:
        """Characteristic wrench magnitude: the friction-cone limit.

        Used by `WrenchConsensus` to normalize the ADMM penalty/residuals.
        """
        return self.object_model.wrench_limit

    def object_action_scale(self) -> jax.Array:
        """Map a unit sample from the object optimizer to a physical wrench."""
        return self.object_model.action_scale

    # Position error (m) below which project_object_action starts
    # gating its snap on. Matches _ANNEAL_REF_POS's old reasoning (now
    # removed with the noise-annealing revert): the diagnosed collapse
    # (consensus wrench falling under the friction threshold) is visible
    # starting well above goal_pos_tol=0.05, so the gate needs the same
    # margin, not just the tolerance itself.
    _PROJECT_GATE_POS = 0.3

    def project_object_action(
        self, action: jax.Array, obj_state: Optional[jax.Array] = None
    ) -> jax.Array:
        """Snap a nonzero action up to the friction breakaway threshold.

        Companion to PlanarPushingObject.step()'s own breakaway deadzone
        (2026-08-10): giving the object dynamics a real sticking region
        fixed what the model *believes* about small wrenches, but not
        what the optimizer *tries* -- MPPI's mean settles near a small
        magnitude close to the goal (running_cost still trades a smaller
        wrench for a smaller effort penalty), and its noise (tuned small,
        for smooth tracking elsewhere) rarely samples far enough past
        that mean to land above threshold. Widening the noise to search
        further did reach past threshold, but made every sample noisier,
        not just the near-goal ones -- confirmed visibly worse sample and
        optimal-trajectory quality at noise_level 1.5 and, to a lesser
        extent, 0.8. This is the alternative: leave sampling exactly as
        it was, and instead reinterpret whatever direction a sample picks
        -- however small -- as a real, threshold-crossing push in that
        same direction, rather than a token one. Physically closer to how
        breakaway friction actually behaves: near threshold, the real
        choice is closer to *whether* and *which way* to push, not a
        continuous dial on *how hard*.

        Gated on obj_state, since an unconditional snap turned out to
        amplify *any* small action, not just goal-tracking-driven ones --
        caught by test_proximal_term_pulls_toward_previous_iterate, whose
        synthetic scenario isolates the ADMM proximal term with the
        object well away from any goal (obj_state0 = origin, ~0.69 from
        the default clutter scene's goal), where a mild proximal pull
        was getting snapped to full strength same as a real push would
        be, washing out the very difference the test measures. Gating on
        proximity to goal confines the snap to the regime it was
        diagnosed in and leaves it off everywhere else, including that
        test's.

        Args:
            action: Raw optimizer-space action(s), any leading batch
                shape, last axis is the wrench dimension (matches
                object_action_scale()).
            obj_state: The object configuration this action would be
                applied from. None (no state offered) behaves as the
                base class's identity default.

        Returns:
            action unchanged if obj_state is None, too far from goal, or
            action is already at/above threshold or exactly zero;
            rescaled to threshold magnitude, same direction, otherwise.
        """
        if obj_state is None:
            return action
        pos_err = jnp.linalg.norm(obj_state[:2] - self.goal[:2])
        # Normalized component-wise by wrench_limit first, matching
        # PlanarPushingObject.step()'s own deadzone check exactly (a raw
        # norm would treat the 7.848 N force limit and 0.471 N*m torque
        # limit as the same scale, incorrectly).
        physical = action * self.object_action_scale()
        normalized = physical / self.object_model.wrench_limit
        normalized_mag = jnp.linalg.norm(normalized, axis=-1, keepdims=True)
        direction = normalized / jnp.clip(normalized_mag, min=1e-8)
        snapped = jnp.where(normalized_mag < 1.0, direction, normalized)
        # normalized_mag == 0 gives a zero direction and would otherwise
        # be snapped to a nonzero push -- "don't push at all" must stay
        # available.
        snapped = jnp.where(normalized_mag < 1e-8, normalized, snapped)
        snapped_physical = snapped * self.object_model.wrench_limit
        gated_physical = jnp.where(
            pos_err < self._PROJECT_GATE_POS, snapped_physical, physical
        )
        return gated_physical / self.object_action_scale()

    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Quasi-static limit-surface dynamics (paper eq. 5)."""
        return self.object_model.step(obj_state, w)

    def object_running_cost(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """Object stage cost: goal tracking + clearance + effort (eq. 18)."""
        return self.object_model.running_cost(obj_state, w)

    def object_terminal_cost(self, obj_state: jax.Array) -> jax.Array:
        """Object terminal cost, heavier goal tracking only."""
        return self.object_model.terminal_cost(obj_state)

    def object_state_from_robot(self, state: mjx.Data) -> jax.Array:
        """Extract the object's SE(2) pose from the combined robot state."""
        return self._block_pose(state)

    def _consensus_from_twist(self, state: mjx.Data) -> jax.Array:
        """A^r via the limit-surface relation `xdot^o = D w^o` (paper eq. 4).

        Inverted to recover the wrench that produced the observed twist.
        Default estimator: backend-agnostic (needs only `qvel`), robot-
        agnostic (no contact enumeration), and continuous (contact forces
        are exactly zero between contacts, so `_consensus_from_contact`
        gives a chattery signal; this doesn't).
        """
        return self.object_model.wrench_limit * state.qvel[self.block_dofs]

    def _consensus_from_contact(self, state: mjx.Data) -> jax.Array:
        """A^r read literally from the simulator's constraint force.

        `qfrc_constraint` at the pusher's DOFs is the force acting on the
        pusher; its negation is the force applied to the object (Newton's
        third law). Point pusher only: relies on the pusher's DOFs being
        exactly the two translational DOFs in contact with the block, which
        doesn't hold for an articulated arm.
        """
        f = -state.qfrc_constraint[self.pusher_dofs]
        r = self._pusher_pos(state) - self._block_pose(state)[:2]
        tau = r[0] * f[1] - r[1] * f[0]
        return jnp.array([f[0], f[1], tau])

    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """A^r: the wrench the robot imparts on the object (paper eq. 23).

        Expressed in the world frame about the block's pose origin, in N and
        N·m -- the same frame, reference point and units the object block's
        A^o uses, so both ADMM blocks report the identical physical quantity.

        Which estimator is used is set by `consensus_source` on the task; see
        `_consensus_from_twist` (default) and `_consensus_from_contact`.

        Clipped to `_realized_wrench_clip` (see `__init__`'s own
        docstring), `consensus_scale()` unless overridden: a rigid-body
        contact solver can report a one-step force or an implied velocity
        far past the friction-cone limit at contact onset (measured up to
        ~16x on this task), which no sustained push can exceed. Left
        unclipped, that outlier drags the consensus average z outside the
        object block's own feasible bound -- which it can never match,
        since its actions are already confined to that bound -- and the
        resulting disagreement persists for several steps after the spike
        itself is gone (task 10). `consensus_source="contact"` sustains
        near this limit under real, ongoing contact though, not just a
        brief onset spike -- clipping it to the same tight bound pins the
        estimate at the ceiling for many consecutive steps, which is a
        different failure mode this override exists to relax.
        """
        raw = (
            self._consensus_from_contact(state)
            if self.consensus_source == "contact"
            else self._consensus_from_twist(state)
        )
        return jnp.clip(
            raw, -self._realized_wrench_clip, self._realized_wrench_clip
        )

    def _tilt(self, state: mjx.Data) -> jax.Array:
        """1 - cos(psi_tilt): the tip's z-axis away from world -z (eq. 22).

        The tip site's z-axis is the stick's pointing direction, so -R_33 is
        exactly the cosine between it and straight down. This returns
        1 - cos(psi): 0 vertical, 1 horizontal, 2 inverted.

        Cosine rather than the angle. `arccos` is *linear* in psi, so its
        restoring gradient is constant -- and measured over five 500-step
        runs the tilt angle is a random walk that rises on 52-55% of steps
        (total variation ~8 rad for a net drift of ~1.3). A constant
        gradient cannot arrest a drift whose source is that psi >= 0 has a
        reflecting boundary at zero, which is why raising `w_tilt` through
        5, 20, 30 and 50 never fixed it: the weight was not the free
        parameter, the functional form was. 1 - cos(psi) ~ psi^2/2 near
        vertical, so it is slack where the tip is already nearly right and
        stiffens as it leaves. It also avoids `arccos`'s unbounded
        derivative at both poles.

        For `robot="point"` this is a constant 2.0: the pusher site is
        unrotated, so its z-axis points *up* and no DOF can change that.
        The offset is identical across samples, so it cancels in the cost
        differences every sampler uses -- control is unaffected, but a
        reported stage cost carries it.
        """
        return 1.0 + state.site_xmat[self.trace_site_ids[0]][2, 2]

    @staticmethod
    def tilt_angle(r_mat: jax.Array) -> jax.Array:
        """psi_tilt in radians, from a tip-site rotation matrix (3, 3).

        The *diagnostic* angle, not the cost: `oim/sim3d/run.py` logs it as
        `tip_tilt` so a run file records tilt in readable units. The cost
        `_tilt` uses is 1 - cos(psi), and `oim.utils.costs` recovers that
        from this angle rather than storing it twice.
        """
        return jnp.arccos(jnp.clip(-r_mat[2, 2], -1.0, 1.0))

    def _tip_height_err(self, state: mjx.Data) -> jax.Array:
        """(z_tip - tip_target_z)^2: not in the paper.

        Keeps the pusher at the block's height for side contact.
        Identically zero for `robot="point"`.
        """
        z_tip = state.site_xpos[self.trace_site_ids[0], 2]
        return (z_tip - self.tip_target_z) ** 2

    def shaping_fade(self, pose: jax.Array) -> jax.Array:
        """Scale in [0, 1] for align/tilt/tip_z from distance to the goal.

        1 when ``||p - p_g|| >= shaping_fade_dist`` (full shaping), 0 at
        the goal. ``shaping_fade_dist <= 0`` disables the fade. Approach
        is never faded — the tip still has to stay on the block to push.
        """
        fade_dist = self.shaping_fade_dist
        pos_err = jnp.linalg.norm(pose[:2] - self.goal[:2])
        return jnp.where(
            fade_dist > 0.0,
            jnp.clip(pos_err / fade_dist, 0.0, 1.0),
            jnp.asarray(1.0, dtype=pos_err.dtype),
        )

    def _ell_r(
        self,
        state: mjx.Data,
        pose: jax.Array,
        pusher_pos: jax.Array,
        obj_ref: jax.Array,
    ) -> jax.Array:
        """Robot stage cost ℓ_r (paper eq. 20-22).

        approach + fade * (align + tilt + tip height). See `shaping_fade`.
        """
        d_ee = jnp.sum((pusher_pos - pose[:2]) ** 2)
        approach = self.w_ee * jnp.clip(d_ee - self.r0**2, 0.0, None)

        to_object = pose[:2] - pusher_pos
        to_ref = obj_ref[:2] - pose[:2]
        cos_angle = jnp.sum(to_object * to_ref) / (
            jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
        )
        align = self.w_align * jnp.clip(self.gamma0 - cos_angle, 0.0, None)

        tilt = self.w_tilt * self._tilt(state)
        tip_height = self.w_tip_z * self._tip_height_err(state)
        fade = self.shaping_fade(pose)
        return approach + fade * (align + tilt + tip_height)

    def robot_running_cost(
        self, state: mjx.Data, control: jax.Array, obj_ref_t: jax.Array
    ) -> jax.Array:
        """Robot stage cost J_r = r_r||u||^2 + ℓ_o + ℓ_r + ℓ_c (paper eq. 17).

        The ADMM consensus penalty is *not* added here -- the ADMM layer adds
        it with the same `ConsensusSpace.penalty_cost` the object block uses.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        ell_o = se2_distance_sq(pose, self.goal, self.q_pos, self.q_theta)
        ell_r = self._ell_r(state, pose, pusher_pos, obj_ref_t)
        ell_c = se2_distance_sq(pose, obj_ref_t, self.q_pos, self.q_theta)
        # Same pusher-vs-obstacle hinge as running_cost's -- see its
        # comment. ADMM's own object block already keeps the block
        # itself clear of obstacles (paper eq. 18); this is the robot
        # block's matching term for the pusher.
        obj = self.object_model
        pusher_obstacle = obj.obstacles.hinge_cost(
            pusher_pos,
            obj.w_obstacle,
            obj.obstacle_margin,
        )
        return (
            self.r_r * jnp.sum(control**2)
            + ell_o
            + ell_r
            + ell_c
            + pusher_obstacle
        )

    def robot_terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Heavier goal tracking, matching the object block's ℓ_f."""
        return se2_distance_sq(
            self._block_pose(state), self.goal, self.qf_pos, self.qf_theta
        )
