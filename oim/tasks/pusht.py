from typing import Any, Dict, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from oim import ROOT
from oim.objects import PlanarPushingObject, se2_distance_sq
from oim.task_base import ConsensusTask, Task
from oim.utils.scenes import SCENES

# Cost weights in one place because several must be *identical* on the two
# ADMM blocks: `q_*`/`qf_*` are read by both `robot_running_cost` and
# `PlanarPushingObject`'s own goal tracking, so a run where they differ is
# one where the two halves aim at different targets.
# `oim/configs/robots/{robot}.yaml`'s `costs:` block overrides any subset.
DEFAULT_COSTS = {
    # Shared by both blocks.
    "q_pos": 40.0,  # running goal tracking, translation
    "q_theta": 10.0,  # running goal tracking, rotation
    "qf_pos": 500.0,  # terminal goal tracking, translation
    "qf_theta": 150.0,  # terminal goal tracking, rotation
    # Object block only.
    "w_effort": 0.01,  # squared wrench
    # Squared step-to-step change in wrench; a scalar or [f_x, f_y, tau].
    "w_rate": 0.0,  # see PlanarPushingObject.rate_cost
    "w_obstacle": 60000.0,  # clearance hinge on the object's footprint
    "obstacle_margin": 0.015,  # clearance below which that hinge activates
    # What one unit of object action is worth, as a fraction of the
    # friction-cone limit (`PlanarPushingObject.wrench_sample_fraction`).
    # `None` keeps the per-embodiment default this used to be hardcoded to,
    # so a `PushT` built with no `costs:` behaves exactly as before; every
    # shipped config now states it outright instead.
    "wrench_fraction": None,
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
    # Pusher-vs-obstacle hinge, scaled relative to w_obstacle and with its
    # own reach. Defaults make both inert (1.0, and the same margin as the
    # block's); only xarm6.yaml overrides them.
    "pusher_obstacle_weight": 1.0,
    "pusher_obstacle_margin": 0.015,
    # Position error below which project_object_action snaps. 0 = off,
    # which is right now that `step` subtracts friction -- see it.
    "project_gate_pos": 0.0,
}


def resolve_costs(costs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

    `robot` selects the embodiment: `"point"` (a free 2-DOF point mass) or
    `"xarm6"` (a 6-DoF arm with a rigid pushing stick), meaningful only with
    `clutter=True`. They share every method except those reading the pusher
    position or realizing the wrench; the object side is identical physics
    either way.

    `env` names a scene in the `oim.utils.scenes.SCENES` registry. `PushT`
    holds no scene-specific data -- it wraps costs and ADMM plumbing around
    one `SceneSpec`, so a new environment is a registry entry plus an MJCF,
    never a change here.
    """

    def __init__(
        self,
        impl: str = "jax",
        clutter: bool = False,
        planning_dt: Optional[float] = None,
        robot: Literal["point", "xarm6"] = "point",
        consensus_source: Literal["twist", "contact"] = "twist",
        consensus_variable: Literal["wrench", "pose"] = "wrench",
        env: str = "clutter",
        goal: Optional[Sequence[float]] = None,
        costs: Optional[Dict[str, Any]] = None,
        realized_wrench_clip: Optional[Sequence[float]] = None,
        local_goal: bool = False,
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
                `oim.worlds.sim3d.build`; this sets what the *costs* aim at.
            costs: Overrides for any subset of `DEFAULT_COSTS`, normally
                the `costs:` block of `oim/configs/robots/{robot}.yaml`. One
                mapping feeds both ADMM blocks, so the shared goal-tracking
                weights cannot drift apart between them. Unknown keys
                raise: a misspelled weight would otherwise be ignored in
                silence and the run file would report a tuning that never
                happened.
            consensus_variable: What the two ADMM blocks agree on.
                `"wrench"` (default) is the paper's own choice, eq. 24:
                A^o is the object block's proposed wrench and A^r is the
                wrench the robot's rollout imparts. `"pose"` makes it the
                object's SE(2) pose trajectory instead, so A^o is eq. 5
                integrated (affine in U^o) and A^r is a state read rather
                than a force estimate -- no twist inversion, no clip, and
                the same quantity for both embodiments. Also drops
                `robot_running_cost`'s `ell_c`, which the ADMM penalty
                then subsumes; see that method.
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
            local_goal: Whether the robot block's *goal tracking* aims at
                the object block's horizon endpoint x^{o*}_H (the "local
                goal") instead of the global goal g. ADMM only -- the flat
                baselines' `running_cost`/`terminal_cost` have no object
                plan to read and are unaffected either way.

                Off by default, so every existing config and recorded run
                keeps its meaning; `--local-goal` / `admm.local_goal:`
                turns it on.

                The two blocks presently pull toward targets that can be
                far apart: the object block routes around obstacles toward
                g over H steps, while the robot block is scored against g
                directly, including the `qf_*` terminal term at full
                weight. Anything the plan does that is not straight at the
                goal -- going around a shelf rather than through it -- the
                robot block is actively penalized for following. Tracking
                x^{o*}_H instead asks it for what the plan asks for, which
                is the reference the coupling term `ell_c` already uses
                pointwise.

                Affects exactly two terms, `robot_running_cost`'s `ell_o`
                and `robot_terminal_cost`, and only outside the
                `shaping_fade_dist` radius -- within it both snap back to
                g so the last few centimetres are closed against the real
                goal rather than against the plan's residual error. See
                `tracking_goal`. The fade *itself* is deliberately never
                retargeted; see `shaping_fade`.

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
        if consensus_variable not in ("wrench", "pose"):
            raise ValueError(
                "consensus_variable must be 'wrench' or 'pose', got "
                f"{consensus_variable!r}"
            )

        cost = resolve_costs(costs)
        self.costs = cost
        self.clutter = clutter
        self.robot = robot
        self.consensus_source = consensus_source
        self.consensus_variable = consensus_variable
        self.use_local_goal = local_goal
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
            mj_model.body_pos[base_id] = [
                *spec.xarm6_base_pos,
                spec.xarm6_base_z,
            ]
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

            # Scene metadata the real-robot driver needs: where the block
            # starts and where the arm homes (the mock starts there; on
            # hardware both are read from the robot), and which TF frame the
            # planner's world is expressed in, so `Ros2Interface` knows
            # whether it has to publish a world -> base transform at all.
            self.start = spec.object_start
            self.world_frame = spec.world_frame
            self.arm_start_deg = spec.xarm6_arm_start_deg

            # Ground-mount placement, surfaced for the real driver's world -> base
            # static TF. The values live in the SCENES registry and are read into
            # the task here; the interface takes them as plain values so the
            # hardware I/O seam stays unaware of the registry. Used only when
            # world_frame != base_frame.
            self.base_pos = spec.xarm6_base_pos
            self.base_yaw_deg = spec.xarm6_base_yaw_deg
            self.base_z = spec.xarm6_base_z

            # goal/obstacles/footprint/physics all come from the scene
            # registry (see oim.utils.scenes). mu/mass/limit_surface_radius
            # default to the modelled T the sim scenes share; a scene whose
            # block is a different physical object overrides them, keeping
            # the friction-cone limit mu*m*g equal to the block joints'
            # `frictionloss` in its own MJCF.
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
                mu=spec.mu,
                mass=spec.mass,
                limit_surface_radius=spec.limit_surface_radius,
                w_pos=cost["q_pos"],
                w_theta=cost["q_theta"],
                wf_pos=cost["qf_pos"],
                wf_theta=cost["qf_theta"],
                w_effort=cost["w_effort"],
                w_rate=cost["w_rate"],
                w_obstacle=cost["w_obstacle"],
                obstacle_margin=cost["obstacle_margin"],
                # KNOWN BUG at 0.5. The action box is the unit cube, so
                # the largest expressible wrench is fraction*sqrt(3) in
                # units of the friction-cone limit -- 0.87, below the
                # breakaway threshold `step` enforces. The object block
                # therefore cannot move the object on any scene but the
                # one below, and MPPI converges to w = 0 (with every
                # rollout frozen, effort is the only term still varying
                # across samples). Measured on shelf_gap+xarm6: every
                # candidate spans exactly 0.0000 m at 0.5, 0.015-0.265 m
                # at 1.0. 2026-08-13: reapplied xarm6-wide now that
                # object_action_bounds is unconditionally the unit box
                # (base class, no PushT override) -- the old "other 4
                # scenes keep the pre-fix budget" rationale for scoping
                # this to open_table no longer holds, since that pre-fix
                # budget (bounds=+/-wrench_limit) is not reachable at all
                # anymore -- every non-open_table xarm6 scene was
                # silently running an unvalidated uniform-half budget.
                # Validated against the unmodified (scoped) default at
                # seeds 5/7/11 across all 5 scenes: xarm6-wide 1.0 hit
                # clean convergence (early episode success) on
                # open_table/shelf_gap/icra_sign in most seeds, vs. the
                # scoped default converging cleanly in only 2/15
                # scene-seed combinations overall (and never on
                # shelf_gap). single_obstacle and ycb_clutter remain
                # weak under both -- open follow-up, not fixed by this.
                wrench_sample_fraction=(
                    (1.0 if robot == "xarm6" else 0.5)
                    if cost["wrench_fraction"] is None
                    else cost["wrench_fraction"]
                ),
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
            # Not in the paper. Retuning w_tilt through 5/20/30/50 never
            # arrested the drift: over five 500-step runs the tilt angle
            # rises on 52-55% of steps (total variation ~8 rad for a net
            # ~1.3), and mean tilt rank-orders with final position error
            # across all five scenes. See `_tilt` -- the functional form,
            # not the weight, was the free parameter.
            self.w_tilt = cost["w_tilt"]
            self.w_tip_z = cost["w_tip_z"]
            self.shaping_fade_dist = float(cost["shaping_fade_dist"])
            self.pusher_obstacle_weight = float(
                cost["pusher_obstacle_weight"]
            )
            self.pusher_obstacle_margin = float(
                cost["pusher_obstacle_margin"]
            )
            self.project_gate_pos = float(cost["project_gate_pos"])
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

        Reuses `_ell_r`'s shaping for both embodiments, with `self.goal`
        standing in for the object planner's reference (plain MPC has no
        object-level plan) -- the same formula `robot_running_cost` uses
        (paper eq. 21). Align matters most: without it the pusher parks
        anywhere near the block, including the wrong side.

        The obstacle hinge is the term the ADMM object block scores
        (eq. 18). Without it a flat baseline only learns about an obstacle
        once a rollout wedges the block against it, so skirting by 1 mm
        and by 5 cm score identically. The block cannot penetrate in
        rollouts, so it fires only inside the margin.
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
        pusher_obstacle = self.pusher_obstacle_weight * obj.obstacles.hinge_cost(
            pusher_pos,
            obj.w_obstacle,
            self.pusher_obstacle_margin,
        )
        ell_r = self._ell_r(state, pose, pusher_pos, self.goal)
        return ell_o + obstacle + pusher_obstacle + ell_r

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T) for plain (non-ADMM) MPC.

        Heavier SE(2) goal tracking (`qf_*`) **plus** the same ℓ_r as the
        stage cost. Stage costs are dt-weighted in the rollout and the
        terminal is not, so this is where the pushing geometry is scored
        at full weight; a goal-only terminal let MPPI buy a better
        predicted pose by abandoning it. Measured on open_table (xarm6):
        0.07 m -> 0.98 m final error.
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
        # NOT qpos[3:5]: that is the slide joints' displacement from the
        # pusher body's declared XML pos, not its world position, and is
        # wrong wherever that pos is nonzero (every point-robot scene).
        return state.xpos[self.pusher_body_id, :2]

    @property
    def consensus_dim(self) -> int:
        """The consensus variable is the planar wrench [f_x, f_y, tau]."""
        return 3

    def consensus_scale(self) -> jax.Array:
        """Characteristic magnitude of the consensus variable.

        For `consensus_variable="wrench"`, the friction-cone limit -- the
        largest wrench the support surface can transmit, so a normalized
        residual of 1 means the blocks disagree by that whole budget.

        For `"pose"`, the object's own bounding radius in translation and
        one radian in heading, so a normalized residual of 1 means they
        disagree by a body radius or by a radian. Both are used to
        normalize the ADMM penalty and residuals, which is what keeps
        `rho`, `eps_r` and `eps_s` scale-free across the two choices.
        """
        if self.consensus_variable == "pose":
            r_body = self.object_model.footprint.bounding_radius
            return jnp.array([r_body, r_body, 1.0])
        return self.object_model.wrench_limit

    def object_consensus(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """A^o: the wrench itself, or the pose it moved the object to.

        The wrench case is the paper's eq. 24, a selection off the object
        block's own decision variable. The pose case is eq. 5 integrated,
        which -- since the limit surface is linear -- makes A^o an affine
        map of U^o rather than a selection, and lets the robot block's A^r
        be a state read rather than a force estimate.
        """
        if self.consensus_variable == "pose":
            return obj_state
        return w

    def object_action_scale(self) -> jax.Array:
        """Map a unit sample from the object optimizer to a physical wrench."""
        return self.object_model.action_scale

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
        if obj_state is None or self.project_gate_pos <= 0.0:
            # Off by default since `PlanarPushingObject.step` began
            # subtracting friction instead of gating on it: the snap exists
            # only because the gated form had no motion smaller than
            # `dt * 1.0`, so near the goal the choice was freeze or
            # overshoot and this picked overshoot. With a continuous map a
            # smaller sampled force gives a smaller step, and the snap is
            # actively harmful -- measured across all five scenes at
            # wrench_fraction 2.0, gate 0.1 reached 4/5 in 461-558 steps,
            # gate 0.0 reached 5/5 in 51-161. Kept, not deleted, because
            # it also documents that failure mode. Early return rather
            # than a `where`: at 0.0 the gate can never fire, and the
            # round trip through `* scale` / `/ scale` below is not free.
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
            pos_err < self.project_gate_pos, snapped_physical, physical
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

    def object_rate_cost(
        self, wrenches: jax.Array, w_prev: Optional[jax.Array] = None
    ) -> jax.Array:
        """Charge for reversing the wrench; see `PlanarPushingObject`."""
        return self.object_model.rate_cost(wrenches, w_prev)

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
        if self.consensus_variable == "pose":
            # A^r is the object's pose along the rollout, read straight out
            # of the state. No estimator and no clip: the clip below exists
            # because a rigid-body solver reports wrench *spikes* at
            # contact onset, and a pose has no such transient -- it is the
            # integral of the motion, not a one-step force.
            return self._block_pose(state)
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

        The *diagnostic* angle, not the cost: `oim/worlds/sim3d/run.py`
        logs it as
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

        Always the *global* goal, even under local-goal tracking. The fade
        means "the task is nearly over, stop shaping posture", which is a
        statement about the global goal; the local goal is only H steps
        ahead and the block is near it by construction, so fading against
        it would read ~0 almost every step and switch off align, tilt and
        tip height for the whole run. xarm6.yaml carries `w_tilt: 100`
        precisely because losing that shaping puts the stick horizontal and
        the arm's forearm into the block.

        Its radius does double duty: `tracking_goal` snaps the local goal
        back to g wherever this reads < 1, so the one number decides both
        when posture shaping stops and when the plan endpoint stops being
        the tracking target.
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

    def tracking_goal(
        self, pose: jax.Array, local_goal: Optional[jax.Array]
    ) -> jax.Array:
        """What the robot block's goal-tracking terms aim at.

        `self.goal` unless local-goal tracking is on *and* a plan was
        offered. Both conditions matter: the flag is the run's choice, and
        `local_goal is None` is a caller with no object plan to read (the
        direct-call tests, and any non-ADMM path), for which the global
        goal is the only defined answer.

        Inside the shaping-fade radius the target snaps back to `self.goal`
        even with the flag on. Local-goal tracking exists so the robot is
        not penalized for following a plan that routes *around* something;
        within `shaping_fade_dist` of the goal there is nothing left to
        route around, and x^{o*}_H is then the one thing between the run
        and its last few centimetres -- the plan endpoint is only H steps
        out and carries the object block's own residual error, so tracking
        it there asks the robot to stop short of g by exactly that
        residual.

        The gate is `shaping_fade` itself, not a second distance test, so
        the radius that means "the task is nearly over" cannot come to mean
        two different things. With `shaping_fade_dist <= 0` the fade is
        identically 1 and the gate is inert, which is what `DEFAULT_COSTS`
        and every config without the knob get.

        One consequence worth knowing: an object block stuck under
        breakaway near the goal plans x^{o*}_H = x^o_0 (hold still), and
        inside the radius this overrides that with g -- the robot keeps
        pushing instead of settling for the stall.

        Resolved in one place because the running and terminal terms must
        aim at the *same* target -- they are the same tracking objective at
        two weights, and splitting them would make the terminal term pull
        the horizon somewhere the stage costs penalize it for going.

        Args:
            pose: Object SE(2) pose the cost is being evaluated at, (3,).
                Read only by the fade gate.
            local_goal: The object block's x^{o*}_H, or None.

        Returns:
            The SE(2) pose to track, (3,).
        """
        if not self.use_local_goal or local_goal is None:
            return self.goal
        # 1 outside the fade radius, < 1 inside it.
        return jnp.where(self.shaping_fade(pose) < 1.0, self.goal, local_goal)

    def robot_running_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
        obj_ref_t: jax.Array,
        local_goal: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Robot stage cost J_r = r_r||u||^2 + ℓ_o + ℓ_r + ℓ_c (paper eq. 17).

        The ADMM consensus penalty is *not* added here -- the ADMM layer adds
        it with the same `ConsensusSpace.penalty_cost` the object block uses.

        With `local_goal` tracking on, `ell_o` aims at the object block's
        horizon endpoint rather than the global goal. `ell_c` is left alone:
        it tracks the plan *pointwise* while `ell_o` now rewards reaching
        its end, which are different requests (pointwise tracking penalizes
        running ahead of schedule; endpoint tracking does not). They do
        overlap more than they used to, so dropping `ell_c` under this flag
        is the natural follow-up -- not done here, because changing both at
        once would leave neither measurable.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        target = self.tracking_goal(pose, local_goal)
        ell_o = se2_distance_sq(pose, target, self.q_pos, self.q_theta)
        ell_r = self._ell_r(state, pose, pusher_pos, obj_ref_t)
        # Same pusher-vs-obstacle hinge as running_cost's -- see its
        # comment. ADMM's own object block already keeps the block
        # itself clear of obstacles (paper eq. 19); this is the robot
        # block's matching term for the pusher.
        obj = self.object_model
        pusher_obstacle = self.pusher_obstacle_weight * obj.obstacles.hinge_cost(
            pusher_pos,
            obj.w_obstacle,
            self.pusher_obstacle_margin,
        )
        cost = (
            self.r_r * jnp.sum(control**2) + ell_o + ell_r + pusher_obstacle
        )
        if self.consensus_variable == "wrench":
            # ell_c, the coupling cost of paper eq. 17: track the object
            # block's own plan.
            #
            # Dropped under a pose consensus variable, because there the
            # ADMM penalty the layer adds -- (rho/2)||A^r_t (-) z_t +
            # y^r_t||^2 with A^r_t = x^o_t -- *is* this term, with the
            # negotiated z_t in place of the unilateral x^{o*}_t and a
            # dual offset. Keeping both would score the same disagreement
            # twice under two different weights, and the penalty is by far
            # the larger of the two (the task cost is dt-weighted in the
            # rollout, the penalty is not).
            cost = cost + se2_distance_sq(
                pose, obj_ref_t, self.q_pos, self.q_theta
            )
        return cost

    def robot_terminal_cost(
        self, state: mjx.Data, local_goal: Optional[jax.Array] = None
    ) -> jax.Array:
        """Heavier goal tracking, matching the object block's ℓ_f.

        The term local-goal tracking changes most: `qf_*` are the heaviest
        weights in the robot block, and the terminal cost is not
        dt-weighted in the rollout while the stage costs are -- so this is
        where the mismatch between "what the plan asks for" and "the global
        goal" was priced highest.
        """
        pose = self._block_pose(state)
        target = self.tracking_goal(pose, local_goal)
        return se2_distance_sq(pose, target, self.qf_pos, self.qf_theta)
