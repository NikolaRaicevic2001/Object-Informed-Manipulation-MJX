from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

from oim import ROOT
from oim.objects import PlanarPushingObject, rotate, se2_distance_sq
from oim.objects.sdf import Box
from oim.task_base import ConsensusTask, Task
from oim.utils.scenes import SCENES

# Worldbody geoms that are scenery rather than obstacles. Mirrors
# `tests/test_scenes.py`'s own `_SCENERY`.
_SCENERY_GEOMS = {"floor", "table"}

# `q_*`/`qf_*` are read by both `robot_running_cost` and
# `PlanarPushingObject`'s own goal tracking, so they must match on both
# ADMM blocks. `oim/configs/robots/{robot}.yaml`'s `costs:` block
# overrides any subset.
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
    # Object-vs-obstacle clearance (see `PlanarPushingObject.obstacle_cost`).
    # "exp": no cutoff, exponential in clearance, falling by 1/e every
    # `obstacle_decay` metres -- gives a gradient at every distance, unlike
    # "hinge" (zero cost/gradient until `obstacle_margin`, then quadratic),
    # which let the object get stuck against an obstacle with no avoidance
    # signal until already within margin. `obstacle_margin` is kept for
    # `_pusher_obstacle_cost`, a separate, still-hinge mechanism for the
    # robot tip's own clearance.
    "w_obstacle": 10.0,
    "obstacle_margin": 0.015,
    "obstacle_decay": 0.02,
    # Object-vs-TABLE-EDGE: a keep-IN region, the mirror of the obstacle
    # field. Not in the paper. The tabletop is read from the scene's own
    # support geom (`_support_region`), so it cannot drift from the MJCF,
    # and a scene without one (`clutter`) simply has no term.
    #
    # Why it exists: nothing else told the planner the table ends. The
    # object block's dynamics know -- the block falls once it clears the
    # edge -- but by then the run is lost, and the cost landscape gave no
    # gradient away from the rim beforehand. Runs pushed the object off
    # the table often enough to be the dominant failure.
    #
    # Quadratic past `support_margin` rather than exponential everywhere:
    # see `PlanarPushingObject.support_cost` for why the edge wants the
    # opposite shape from an obstacle.
    # 0.10 m of warning, not the 0.03 a "just don't fall off" reading
    # suggests. Measured headroom: over every start and goal in all five
    # tabletop scenes' pose files, the LEAST clearance any legitimate pose
    # has from the table edge is 0.2164 m (ycb_clutter), so a margin up to
    # ~0.15 m still never charges a pose a run is trying to reach. 0.10
    # gives the planner two to four control steps of gradient before the
    # rim at the speeds these runs push at, instead of firing once the
    # block is already leaving.
    "w_support": 200.0,
    "support_margin": 0.10,
    # Robot-vs-obstacle CONTACT: w * force^2, so a hard hit costs far more
    # than a graze. Proximity is free -- the robot may reach right past an
    # obstacle to push the object off it; only touching costs.
    "w_robot_contact": 1.0,
    # What one unit of object action is worth, as a fraction of the
    # friction-cone limit (`PlanarPushingObject.wrench_sample_fraction`).
    # `None` keeps the per-embodiment default.
    "wrench_fraction": None,
    # Robot block only (paper eq. 20-22).
    "w_robot_effort": 0.05,  # squared control effort
    "w_approach": 40.0,  # approach: pull the tip toward the object
    "r0": 0.02,  # radius inside which approach goes slack
    "w_align": 15.0,  # stay behind the object relative to the reference
    "gamma0_deg": 15.0,  # alignment cone half-angle
    "w_tilt": 30.0,  # keep the stick pointing down (3D only)
    # Tip height, block mid-height or above: ordinary quadratic.
    "w_z_tip": 8.0,
    # Tip height, below block mid-height (heading toward the table):
    # exponential in centimeters instead -- see `_tip_height_cost`.
    "w_z_tip_exp": 1.0,
    # Kinematic hover-slab barrier straddling the block's true top surface
    # -- see `_contact_z_cost`. Weight for the exponential's floor value
    # at the slab's two outer (1cm-either-side-of-surface) edges; the
    # exponential grows from there toward the surface. 0.0 = inert.
    "w_contact_z_exp": 0.0,
    # Ceiling on `_contact_z_cost`'s exponential [cost units]. Exists so
    # the term stays finite -- an uncapped exp(gap**2) overflows to inf
    # across MPPI's sample population and poisons the softmax into nan.
    # At w_contact_z_exp=200 the raw exponential reaches 200*exp(4)=10920
    # at the surface, so the cap trades off keeping a real gradient near
    # the top face against not over-prohibiting samples that graze it --
    # see Tasks.md for the tuning history.
    "contact_z_cap": 10000.0,
    # Extra multiplier applied to `_contact_z_cost` below the block's top
    # face (tip pressing down into it) relative to the same distance
    # above it. 1.0 = symmetric. >1 makes penetrating the surface
    # strictly worse than hovering the same distance over it.
    "contact_z_below_mult": 3.0,
    # Exponential penalty once xarm6_link3's own world z drops below
    # `joint3_cave_z_threshold` -- same structural pattern as
    # `w_z_tip_exp`, see `_joint3_cave_cost`. On by default: guards a
    # real, observed, unrecoverable failure mode.
    "w_joint3_cave_exp": 1.0,
    # xarm6_link3's world z [m] below which `_joint3_cave_cost` fires.
    # 0.20 is the original value (margin under the observed ~0.25 normal
    # floor, catching the collapse only once already underway); 0.275
    # fires at/above the normal floor so the arm never enters the caved
    # configuration at all.
    "joint3_cave_z_threshold": 0.20,
    # Flat baseline only (`running_cost`/`terminal_cost`, not
    # `robot_running_cost`). Multiplier on q_theta/qf_theta, ramping from
    # 1x at pos_err >= theta_ramp_dist to this value at the goal -- 1.0 =
    # inert. Counters a converged orientation's small gradient being
    # knocked back out by continued position-driven pushing. See
    # `_theta_ramp`.
    "q_theta_ramp": 1.0,
    # Radius the above ramps over. 0 = reuse shaping_fade_dist.
    "theta_ramp_dist": 0.0,
    # Goal-tracking weight grows 1 + q_ramp_per_step * step, capped at
    # q_ramp_max. 0.0 = inert. Time-based, unlike q_theta_ramp above.
    #
    # Two independent mechanisms read these same two keys, on two
    # disjoint call paths: `time_ramp`/`weight_scale` inside
    # `robot_running_cost`/`robot_terminal_cost` (ADMM's robot block,
    # linear in elapsed steps -- point.yaml drives this), and
    # `_q_ramp_mult` inside `running_cost` (the flat baseline,
    # compounding -- xarm6.yaml drives this). Never both at once for a
    # given run (see `oim.experiment._run_3d`'s `is_admm` branch), but a
    # task built for one path and driven through the other would
    # silently pick up the wrong formula's ramp.
    "q_ramp_per_step": 0.0,
    "q_ramp_max": 5.0,
    # Fade approach, align, tilt, and the above-threshold branch of tip
    # height (the last two internally, see `_tip_height_cost`) linearly
    # as ||p - p_g|| -> 0 (0 = disabled). Control effort and the
    # xarm6-only pusher-obstacle hinge fade too, in `running_cost`/
    # `robot_running_cost`. tip_height's below-threshold (exponential)
    # branch, contact_z, joint3_cave, and the point-robot's
    # `_robot_contact_cost` are never faded -- hard safety guarantees, or
    # for the last one, matching the point-robot ADMM track. See
    # `shaping_fade`.
    "shaping_fade_dist": 0.0,
    # EXPERIMENTAL: metres of position error below which
    # `_tip_height_cost`'s below-threshold branch blends from the
    # exponential toward the same quadratic the above-threshold branch
    # uses (unfaded), reaching pure quadratic at pos_err=fade_floor. 0 =
    # disabled (always exponential below the threshold).
    "tip_softening_dist": 0.0,
    # Position error at which `shaping_fade` and `_tip_height_cost`'s
    # blend both hit exactly 0, instead of only at the exact goal (which
    # a real rollout rarely if ever hits exactly). 0 (default) reproduces
    # the original zero-only-at-the-goal behavior.
    "fade_floor": 0.0,
    # Pusher-vs-obstacle hinge, scaled relative to w_obstacle and with
    # its own reach. xarm6 only -- see `_robot_obstacle_cost`; the point
    # robot uses `w_robot_contact` instead. Defaults make both inert.
    "pusher_obstacle_weight": 1.0,
    "pusher_obstacle_margin": 0.015,
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


def _support_region(mj_model: mujoco.MjModel) -> Optional[Box]:
    """The tabletop as a keep-in region, read from the scene's own geom.

    Returns the xy footprint of the first geom in `SUPPORT_GEOM_NAMES` that
    the scene actually has, as a `Box` in the world frame -- so the region
    the planner is told to stay on is the surface the simulator holds the
    block up with, and the two cannot drift apart. `oim.utils.scenes` is not
    consulted: it does not carry the table, and adding it there would be a
    second copy of a number the MJCF already states.

    Returns None when the scene has no support geom. `clutter` is the case:
    its block hovers above the floor plane with no vertical DoF, so there is
    no edge to fall off and no region to keep it in.

    Args:
        mj_model: The scene's compiled model.

    Returns:
        The tabletop rectangle, or None if the scene has no support geom.
    """
    # Deferred: `oim.runtime.mjcf` imports `PushT`, so taking the names at
    # module level would close the cycle. Imported rather than restated so
    # the keep-in region and the geoms the object model strips out of a
    # prediction stay one list.
    from oim.runtime.mjcf import SUPPORT_GEOM_NAMES

    for name in SUPPORT_GEOM_NAMES:
        try:
            geom = mj_model.geom(name)
        except KeyError:
            continue
        if geom.type[0] != mujoco.mjtGeom.mjGEOM_BOX:
            # A plane has no edge to fall off, which is the whole point of
            # the term; anything else is a shape this helper has not been
            # taught to read, and silently guessing its extent would be
            # worse than leaving the term off.
            continue
        return Box(
            center=(float(geom.pos[0]), float(geom.pos[1])),
            half_extents=(float(geom.size[0]), float(geom.size[1])),
        )
    return None


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
        planning_iterations: Optional[int] = None,
        planning_ls_iterations: Optional[int] = None,
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
            planning_iterations: If given, overrides the model's solver
                iteration count for planning rollouts. Independent of
                `execution_model`'s own `exec_iterations` override -- the
                two models are separate `MjModel` instances.
            planning_ls_iterations: Same, for the solver's line-search
                iteration count.
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
                weights cannot drift apart between them. Unknown keys raise.
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
                `object_model.wrench_limit` (the friction-cone limit).
                Separate from `wrench_limit` on purpose: that value also
                sets the object block's own action bounds and the ADMM
                dual clip (`consensus_scale`), so widening it to stop the
                robot's *estimate* from saturating would silently widen
                those too. `consensus_source="contact"` (point robot)
                reads `qfrc_constraint` literally, which sustains near or
                above the friction-cone limit under real contact, not
                just spiking at onset -- see `realized_consensus`.
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
        if planning_iterations is not None:
            mj_model.opt.iterations = planning_iterations
        if planning_ls_iterations is not None:
            mj_model.opt.ls_iterations = planning_ls_iterations

        if robot == "xarm6":
            # Ground-mounted base placement, not baked into xarm6.xml
            # itself (a reusable, placement-agnostic robot asset) --
            # mutate the loaded mj_model before it's handed to mjx.
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
                # Block qpos addresses looked up explicitly, not assumed
                # to be qpos[:3]: unlike pusht_clutter.xml (block declared
                # before the pusher), the composed xarm6 scene compiles
                # the arm's 5 joints first, so the block's SE(2) pose
                # lands at qpos[5:8], with its vertical DoF after.
                self.block_qpos_adr = jnp.array(
                    [
                        mj_model.joint("T_x").qposadr[0],
                        mj_model.joint("T_y").qposadr[0],
                        mj_model.joint("T_z").qposadr[0],
                    ]
                )
                self.tip_site_id = mj_model.site("xarm6_tip").id
                # The 5 robot joints' DOF addresses, by name -- for the
                # task-space noise mechanism's tip Jacobian columns,
                # restricted to the robot (not the block's 3 DOFs after).
                self.robot_dof_adr = np.array(
                    [
                        mj_model.joint(f"xarm6_joint{i}").dofadr[0]
                        for i in range(1, 6)
                    ]
                )
                self.stick_body_id = mj_model.body("xarm6_stick").id
                # link3's own world z, for `_joint3_cave_cost`.
                self.joint3_link_id = mj_model.body("xarm6_link3").id
                # Every geom belonging to the stick, for
                # `_contact_normal_force_z`.
                self.stick_geoms = jnp.array(
                    sorted(
                        g
                        for g in range(mj_model.ngeom)
                        if mj_model.geom_bodyid[g] == self.stick_body_id
                    ),
                    dtype=jnp.int32,
                )
                self._stick_geoms_set = set(np.asarray(self.stick_geoms).tolist())
            else:
                pusher_x_dof = mj_model.joint("root_x").dofadr[0]
                pusher_y_dof = mj_model.joint("root_y").dofadr[0]
                self.pusher_dofs = jnp.array([pusher_x_dof, pusher_y_dof])
                # root_x/root_y qpos entries are the pusher's displacement
                # from its declared XML pos, not its world position --
                # _pusher_pos needs the latter, so capture the body id.
                self.pusher_body_id = mj_model.body("pusher").id
                # _contact_normal_force_z is xarm6-specific; an empty
                # `stick_geoms` makes `jnp.isin` against it always False,
                # a structural no-op for the point robot. `block_geoms`
                # is collected below for both embodiments, since the
                # object-vs-obstacle contact cost needs it either way.
                self.stick_geoms = jnp.array([], dtype=jnp.int32)
                self._stick_geoms_set: set = set()

            # The pushed object's geoms, and the geoms that stand for
            # obstacles -- both embodiments, for `_object_obstacle_force`.
            # A set, not one id: the block is more than one geom in every
            # scene (crossbar + stem for a T, three strokes for the C).
            self.block_body_id = mj_model.body("block").id
            self.block_geoms = jnp.array(
                sorted(
                    g
                    for g in range(mj_model.ngeom)
                    if mj_model.geom_bodyid[g] == self.block_body_id
                ),
                dtype=jnp.int32,
            )
            self._block_geoms_set = set(np.asarray(self.block_geoms).tolist())
            # Obstacles are exactly the worldbody geoms that are not
            # scenery: the pushed object and the goal marker each live in
            # their own body, so nothing else hangs off the world.
            self.obstacle_geoms = jnp.array(
                sorted(
                    g
                    for g in range(mj_model.ngeom)
                    if mj_model.geom_bodyid[g] == 0
                    and mj_model.geom(g).name not in _SCENERY_GEOMS
                ),
                dtype=jnp.int32,
            )
            self._obstacle_geoms_set = set(
                np.asarray(self.obstacle_geoms).tolist()
            )
            # Everything that is the robot: not worldbody, not the block,
            # and not a mocap body (the `goal`/`local_goal` ghosts are
            # mocap). Pusher for `point`, every arm link plus the stick
            # for `xarm6`.
            self.robot_geoms = jnp.array(
                sorted(
                    g
                    for g in range(mj_model.ngeom)
                    if mj_model.geom_bodyid[g] not in (0, self.block_body_id)
                    and mj_model.body_mocapid[mj_model.geom_bodyid[g]] < 0
                ),
                dtype=jnp.int32,
            )
            self._robot_geoms_set = set(
                np.asarray(self.robot_geoms).tolist()
            )

            # The block's own velocity DOFs, used by the default
            # ("twist") consensus extraction. By joint name so it is
            # correct for both embodiments' qpos/qvel layouts.
            self.block_dofs = jnp.array(
                [
                    mj_model.joint("T_x").dofadr[0],
                    mj_model.joint("T_y").dofadr[0],
                    mj_model.joint("T_z").dofadr[0],
                ]
            )

            # Scene metadata the real-robot driver needs: where the block
            # starts and where the arm homes, and which TF frame the
            # planner's world is expressed in.
            self.start = spec.object_start
            self.world_frame = spec.world_frame
            self.arm_start_deg = spec.xarm6_arm_start_deg

            # Ground-mount placement, surfaced for the real driver's
            # world -> base static TF. Used only when world_frame !=
            # base_frame.
            self.base_pos = spec.xarm6_base_pos
            self.base_yaw_deg = spec.xarm6_base_yaw_deg
            self.base_z = spec.xarm6_base_z

            # goal/obstacles/footprint/physics all come from the scene
            # registry (see oim.utils.scenes). One goal pose feeds both
            # blocks' costs; a pose file overrides it per run.
            goal_pose = (
                spec.goal if goal is None else jnp.asarray(goal, dtype=float)
            )
            self.object_model = PlanarPushingObject(
                dt=self.dt,
                goal=goal_pose,
                footprint=spec.footprint(),
                obstacles=spec.obstacles_for(robot),
                mu=spec.mu,
                mass=spec.mass,
                limit_surface_radius=spec.limit_surface_radius,
                w_pos=cost["q_pos"],
                w_theta=cost["q_theta"],
                wf_pos=cost["qf_pos"],
                wf_theta=cost["qf_theta"],
                w_effort=cost["w_effort"],
                w_rate=cost["w_rate"],
                obstacle_cost="exp",
                w_obstacle=cost["w_obstacle"],
                obstacle_margin=cost["obstacle_margin"],
                obstacle_decay=cost["obstacle_decay"],
                support=_support_region(mj_model),
                w_support=cost["w_support"],
                support_margin=cost["support_margin"],
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
            self.w_robot_effort = cost["w_robot_effort"]
            self.w_approach, self.r0 = cost["w_approach"], cost["r0"]
            self.w_align = cost["w_align"]
            self.gamma0 = jnp.cos(jnp.deg2rad(cost["gamma0_deg"]))
            # Not in the paper. Zero for the point pusher: its site
            # cannot rotate, so `_tilt` is a constant 2.0, cancelling in
            # every sampler's cost differences -- forced here rather than
            # in config, so no point config can reintroduce it.
            self.w_tilt = 0.0 if robot == "point" else cost["w_tilt"]
            # Likewise zero for the point pusher: no z DOF, tip sits
            # exactly at `tip_target_z` in every point scene, so
            # `_tip_height_cost` is identically 0.
            point_tip = robot == "point"
            self.w_z_tip = 0.0 if point_tip else cost["w_z_tip"]
            self.w_z_tip_exp = 0.0 if point_tip else cost["w_z_tip_exp"]
            self.w_contact_z_exp = float(cost["w_contact_z_exp"])
            # `.get`: run files/configs predating these two keys decompose
            # and replay unchanged, at the old symmetric-slab/1000-cap
            # behaviour.
            self.contact_z_cap = float(cost.get("contact_z_cap", 1000.0))
            self.contact_z_below_mult = float(
                cost.get("contact_z_below_mult", 1.0)
            )
            self.w_robot_contact = float(cost["w_robot_contact"])
            self.pusher_obstacle_weight = float(
                cost["pusher_obstacle_weight"]
            )
            self.pusher_obstacle_margin = float(
                cost["pusher_obstacle_margin"]
            )
            self.q_theta_ramp = float(cost["q_theta_ramp"])
            self.shaping_fade_dist = float(cost["shaping_fade_dist"])
            self.tip_softening_dist = float(cost["tip_softening_dist"])
            self.fade_floor = float(cost["fade_floor"])
            # Target tip height: the block's own resting z, read from the
            # model rather than hardcoded.
            self.tip_target_z = float(mj_model.body("block").pos[2])
            # Where `_tip_height_cost`'s quadratic branch is centered --
            # defaults to `tip_target_z` but settable independently. The
            # exponential's own trigger boundary stays at `tip_target_z`
            # either way.
            self.tip_quadratic_target_z = self.tip_target_z
            # Half the block's own vertical extent, read from its geoms
            # (max over crossbar/stem/strokes) -- `tip_target_z +
            # block_half_height` is the block's true top surface height,
            # used by `_contact_z_cost`'s hover-slab test and
            # `_top_contact_gate`.
            #
            # Collision geoms only (`contype != 0`): icra_sign's block
            # body also carries `c_visual`, a visual-only mesh (contype=0)
            # drawing the C's true outline over the boxes actually
            # collided. `geom_size` on a mesh geom is not a (half_x,
            # half_y, half_z) triple the way it is on a box, so including
            # it in this max picked up the wrong value for that scene.
            self.block_half_height = float(
                max(
                    mj_model.geom_size[int(g)][2]
                    for g in np.asarray(self.block_geoms)
                    if mj_model.geom_contype[int(g)] != 0
                )
            )
            # `_joint3_cave_cost`'s two knobs, set ONLY for the arm. The
            # point robot has no link3 to cave, so the term is undefined
            # rather than merely inert there, and these attributes are
            # left absent -- everything that reads them tests for the arm
            # or for the attribute first.
            #
            # Zeroing `w_joint3_cave_exp` instead would NOT work: the
            # decomposition multiplies it by exp(gap_cm**2) against a
            # placeholder z3=0, which overflows, and 0.0*inf is nan.
            if robot == "xarm6":
                self.w_joint3_cave_exp = float(cost["w_joint3_cave_exp"])
                self.joint3_cave_z_threshold = float(
                    cost["joint3_cave_z_threshold"]
                )
            self.q_pos, self.q_theta = cost["q_pos"], cost["q_theta"]
            self.qf_pos, self.qf_theta = cost["qf_pos"], cost["qf_theta"]
            self.theta_ramp_dist = (
                float(cost["theta_ramp_dist"]) or self.shaping_fade_dist
            )
            self.q_ramp_per_step = float(cost["q_ramp_per_step"])
            self.q_ramp_max = max(1.0, float(cost["q_ramp_max"]))
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
        """The running cost l(x_t, u_t) for plain (non-ADMM) MPC.

        Reuses `_ell_r`'s shaping for both embodiments, with `self.goal`
        standing in for the object planner's reference (plain MPC has no
        object-level plan) -- the same formula `robot_running_cost` uses
        (paper eq. 21). Align matters most: without it the pusher parks
        anywhere near the block, including the wrong side.

        The obstacle hinge is the term the ADMM object block scores
        (eq. 18): without it a flat baseline only learns about an
        obstacle once a rollout wedges the block against it. The
        table-edge keep-in region rides with it for the same reason --
        both blocks have to price leaving the table, or the flat
        baseline pushes the object off it while ADMM does not, and the
        comparison stops being about the planner.

        `q_pos`/`q_theta` are both scaled by `_q_ramp_mult` -- see that
        method for why this is the flat baseline's own ramp, separate
        from `robot_running_cost`'s `time_ramp`/`weight_scale`.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        q_ramp = self._q_ramp_mult(state)
        q_theta = self.q_theta * self._theta_ramp(pose) * q_ramp
        ell_o = se2_distance_sq(pose, self.goal, self.q_pos * q_ramp, q_theta)
        obj = self.object_model
        obstacle = obj.obstacle_cost(pose) + obj.support_cost(pose)
        ell_r = self._ell_r(state, pose, pusher_pos, self.goal)
        # Faded (linearly, like align) -- recomputed here rather than
        # exposed from `_ell_r`, since that method is also
        # `terminal_cost`'s, which has no control to fade.
        effort = self.shaping_fade(pose) * self.w_robot_effort * jnp.sum(control**2)
        # Robot-vs-obstacle avoidance, robot-conditional -- see
        # `_robot_obstacle_cost`. Never faded: a collision near the goal
        # is as wrong as one anywhere else.
        robot_obstacle = self._robot_obstacle_cost(state, pusher_pos)
        return ell_o + obstacle + ell_r + effort + robot_obstacle

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost l_T(x_T) for plain (non-ADMM) MPC.

        Heavier SE(2) goal tracking (`qf_*`) plus the same l_r as the
        stage cost. Stage costs are dt-weighted in the rollout and the
        terminal is not, so this is where the pushing geometry is scored
        at full weight; a goal-only terminal let MPPI buy a better
        predicted pose by abandoning it.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        qf_theta = self.qf_theta * self._theta_ramp(pose)
        ell_f = se2_distance_sq(pose, self.goal, self.qf_pos, qf_theta)
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
        """Create a new state object with extra constraints allocated.

        Sizes are hand-tuned per scene/embodiment: too small silently
        drops contacts rather than erroring, particularly under MuJoCo
        Warp, where `naconmax`/`njmax` are batch arenas shared across all
        parallel rollouts and must grow with `num_samples`.
        """
        if self.clutter and self.robot == "xarm6":
            return super().make_data(nconmax=256, naconmax=8192, njmax=256)
        if self.clutter:
            return super().make_data(nconmax=128, naconmax=8192, njmax=256)
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
        it is `qpos[:3]`.
        """
        if self.robot == "xarm6":
            return self.block_qpos_adr
        return jnp.array([0, 1, 2])

    def _pusher_pos(self, state: mjx.Data) -> jax.Array:
        """World-frame (x, y) position of the pusher's contact point."""
        if self.robot == "xarm6":
            return state.site_xpos[self.tip_site_id, :2]
        # NOT qpos[3:5]: that is the slide joints' displacement from the
        # pusher body's declared XML pos, not its world position.
        return state.xpos[self.pusher_body_id, :2]

    @property
    def consensus_dim(self) -> int:
        """The consensus variable is the planar wrench [f_x, f_y, tau]."""
        return 3

    def consensus_scale(self) -> jax.Array:
        """Characteristic magnitude of the consensus variable.

        For `consensus_variable="wrench"`, the friction-cone limit -- the
        largest wrench the support surface can transmit. For `"pose"`,
        the object's own bounding radius in translation and one radian in
        heading. Both normalize the ADMM penalty and residuals, keeping
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

    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Quasi-static limit-surface dynamics (paper eq. 5)."""
        return self.object_model.step(obj_state, w)

    def object_running_cost(
        self,
        obj_state: jax.Array,
        w: jax.Array,
        weight_scale: jax.Array = 1.0,
    ) -> jax.Array:
        """Object stage cost: goal tracking + obstacle clearance."""
        return self.object_model.running_cost(obj_state, w, weight_scale)

    def object_terminal_cost(
        self, obj_state: jax.Array, weight_scale: jax.Array = 1.0
    ) -> jax.Array:
        """Object terminal cost, heavier goal tracking only."""
        return self.object_model.terminal_cost(obj_state, weight_scale)

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
        N.m -- the same frame, reference point and units the object block's
        A^o uses, so both ADMM blocks report the identical physical quantity.

        Which estimator is used is set by `consensus_source` on the task; see
        `_consensus_from_twist` (default) and `_consensus_from_contact`.

        Clipped to `_realized_wrench_clip` (see `__init__`), defaulting to
        `consensus_scale()`: a rigid-body contact solver can report a
        one-step force or implied velocity far past the friction-cone
        limit at contact onset, which no sustained push can exceed. Left
        unclipped, that outlier drags the consensus average outside the
        object block's own feasible bound, which it can never match, and
        the disagreement persists for several steps after the spike is
        gone. `consensus_source="contact"` sustains near this limit under
        real, ongoing contact though, not just a brief onset spike --
        clipping it to the same tight bound is a different failure mode
        this override exists to relax.
        """
        if self.consensus_variable == "pose":
            # A^r is the object's pose along the rollout, read straight
            # out of the state. No estimator and no clip: the clip below
            # exists because a rigid-body solver reports wrench *spikes*
            # at contact onset, and a pose has no such transient.
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

        Cosine rather than the angle: `arccos` is linear in psi, so its
        restoring gradient is constant and cannot arrest a drift whose
        source is that psi >= 0 has a reflecting boundary at zero.
        1 - cos(psi) ~ psi^2/2 near vertical, so it is slack where the tip
        is already nearly right and stiffens as it leaves; it also avoids
        `arccos`'s unbounded derivative at both poles.

        For `robot="point"` this is a constant 2.0: the pusher site is
        unrotated, so its z-axis points up and no DOF can change that --
        cancels in every sampler's cost differences.
        """
        return 1.0 + state.site_xmat[self.trace_site_ids[0]][2, 2]

    @staticmethod
    def tilt_angle(r_mat: jax.Array) -> jax.Array:
        """psi_tilt in radians, from a tip-site rotation matrix (3, 3).

        The diagnostic angle, not the cost: `oim/worlds/sim3d/run.py`
        logs it as `tip_tilt` so a run file records tilt in readable
        units. The cost `_tilt` uses is 1 - cos(psi); `oim.utils.costs`
        recovers that from this angle rather than storing it twice.
        """
        return jnp.arccos(jnp.clip(-r_mat[2, 2], -1.0, 1.0))

    def _tip_height_cost(
        self, state: mjx.Data, pos_err: jax.Array
    ) -> jax.Array:
        """Piecewise cost on tip height: not in the paper.

        Keeps the pusher at the block's mid-height (`tip_target_z`,
        "t/2") for side contact. At or above mid-height, an ordinary
        quadratic (`w_z_tip`). Below mid-height -- the tip descending
        toward the table -- an exponential in centimeters instead
        (`w_z_tip_exp`): a real table strike is dangerous on hardware,
        not just costly, so the penalty should blow up approaching it
        rather than stay quadratic. Identically at `tip_target_z` (so
        both branches agree) for `robot="point"`, whose tip never leaves
        it.

        The quadratic branch centers on `tip_quadratic_target_z`, not
        necessarily `tip_target_z` -- the exponential's own trigger
        boundary is always `tip_target_z`, unaffected either way, so this
        cannot move the safety guarantee, only where the "resting" pull
        above it aims.

        EXPERIMENTAL: below `tip_target_z`, the exponential itself blends
        toward `quad_ref` (see below) as `pos_err` shrinks from
        `tip_softening_dist` down to `fade_floor` (0 = disabled, always
        exponential below the threshold). At full blend the piecewise
        function collapses into one smooth quadratic bowl, but only once
        the object is essentially at the goal. This does not reopen the
        safety guarantee: `tip_target_z` never moves, and for a large gap
        the exponential term is already astronomical, so any partial
        blend weight on it keeps the total astronomical too.

        The true above-threshold branch also fades, the same
        `shaping_fade_dist`/`fade_floor` radius align/approach/tilt use,
        computed locally here from `pos_err` (`_ell_r` cannot pass `fade`
        through directly since this needs a second, independent radius
        for the softening blend above too). Deliberately kept separate
        from `quad_ref` (the softening blend target): if this fade leaked
        into the blend target, full blend near the goal would let the tip
        rest at `tip_target_z` (mid-height, still "below the table"
        relative to the block) at zero cost.
        """
        z_tip = state.site_xpos[self.trace_site_ids[0], 2]
        # Danger boundary and trigger: always tip_target_z (t/2), never
        # tip_quadratic_target_z.
        gap_cm = 100.0 * (self.tip_target_z - z_tip)  # > 0 below mid-height
        exp_below = self.w_z_tip_exp * jnp.exp(gap_cm**2)
        # The softening blend target -- always the plain, unfaded
        # quadratic. See the safety note above.
        quad_ref = self.w_z_tip * (z_tip - self.tip_quadratic_target_z) ** 2
        abs_pos_err = jnp.abs(pos_err)
        soften = jnp.where(
            self.tip_softening_dist > 0.0,
            jnp.clip(
                (abs_pos_err - self.fade_floor)
                / (self.tip_softening_dist - self.fade_floor),
                0.0,
                1.0,
            ),
            jnp.asarray(1.0),
        )
        below = soften * exp_below + (1.0 - soften) * quad_ref

        # The true above-threshold cost: quad_ref, faded (linearly, like
        # align/approach/tilt) from full weight at shaping_fade_dist down
        # to 0 at fade_floor.
        fade = jnp.where(
            self.shaping_fade_dist > 0.0,
            jnp.clip(
                (abs_pos_err - self.fade_floor)
                / (self.shaping_fade_dist - self.fade_floor),
                0.0,
                1.0,
            ),
            jnp.asarray(1.0),
        )
        above = fade * quad_ref

        return jnp.where(z_tip >= self.tip_target_z, above, below)

    def _contact_normal_force_z(self, state: mjx.Data) -> jax.Array:
        """World-frame z-component of the pusher-block contact's pure
        NORMAL force (friction excluded), summed over every matching
        contact -- not in the paper.

        Targets top-riding directly rather than through
        `_tip_height_cost`'s height proxy: a side push has a
        near-horizontal contact normal (z-component ~0) regardless of
        how hard it pushes, while a top-surface push's normal points
        mostly vertical.

        xarm6 only: `self.stick_geoms` is empty for the point robot, so
        `jnp.isin` against it is always False and this returns 0.0 there
        without a separate robot-type branch.
        """
        geom1, geom2, dist, frame, efc_addr, efc_force = self._contact_arrays(
            state
        )
        matches = self._contact_matches(
            geom1, geom2, dist, efc_addr, self.stick_geoms, self.block_geoms
        )
        addr = jnp.clip(efc_addr, 0, efc_force.shape[0] - 1)
        f_normal = efc_force[addr]
        normal_z = frame[:, 0, 2]
        return jnp.sum(jnp.where(matches, f_normal * normal_z, 0.0))

    def _contact_arrays(self, state: mjx.Data) -> Tuple[jax.Array, ...]:
        """`(geom1, geom2, dist, frame, efc_address, efc_force)`, per backend.

        JAX and Warp use different `Data._impl` layouts, so this branches
        on `self.model.impl` -- static, fixed at trace time. Shared by both
        contact readers so the two layouts cannot drift apart.
        """
        if self.model.impl == mjx.Impl.WARP:
            c = state._impl
            return (
                c.contact__geom[:, 0],
                c.contact__geom[:, 1],
                c.contact__dist,
                c.contact__frame,
                c.contact__efc_address[:, 0],
                c.efc__force,
            )
        c = state._impl.contact
        return (
            c.geom1,
            c.geom2,
            c.dist,
            c.frame.reshape(c.frame.shape[0], 3, 3),
            c.efc_address,
            state._impl.efc_force,
        )

    @staticmethod
    def _contact_matches(
        geom1: jax.Array,
        geom2: jax.Array,
        dist: jax.Array,
        efc_addr: jax.Array,
        set_a: jax.Array,
        set_b: jax.Array,
    ) -> jax.Array:
        """Which contact slots are a live `set_a`-vs-`set_b` pair, either way.

        `dist < 0` and `efc_addr >= 0` reject stale slots: a fixed-size
        contact array can hold an inactive geom pair whose efc_address
        points at no real constraint row. An empty geom set makes this
        uniformly False, so callers no-op without an embodiment branch.
        """
        pair = (jnp.isin(geom1, set_a) & jnp.isin(geom2, set_b)) | (
            jnp.isin(geom2, set_a) & jnp.isin(geom1, set_b)
        )
        return pair & (dist < 0.0) & (efc_addr >= 0)

    def _robot_obstacle_force(self, state: mjx.Data) -> jax.Array:
        """Total normal force between any robot geom and any obstacle.

        Friction excluded (efc row 0 is the normal). Proximity is free by
        design -- the robot may reach past an obstacle to push the object
        off it -- so only real contact registers here.

        Args:
            state: The rollout state to read contacts from.

        Returns:
            The summed normal force, a non-negative scalar.
        """
        geom1, geom2, dist, _, efc_addr, efc_force = self._contact_arrays(state)
        matches = self._contact_matches(
            geom1, geom2, dist, efc_addr, self.robot_geoms, self.obstacle_geoms
        )
        addr = jnp.clip(efc_addr, 0, efc_force.shape[0] - 1)
        return jnp.sum(jnp.where(matches, efc_force[addr], 0.0))

    def _robot_contact_cost(self, state: mjx.Data) -> jax.Array:
        """`w_robot_contact * force^2` on robot-obstacle normal force.

        Quadratic, not linear: a hard hit should be disproportionately
        worse than a graze, so twice the force costs four times as much.
        Inert at weight 0; early return so a run that opts out skips the
        contact scan.
        """
        if self.w_robot_contact == 0.0:
            return jnp.zeros(())
        return self.w_robot_contact * self._robot_obstacle_force(state) ** 2

    def _robot_obstacle_force_mujoco(self, mj_data: mujoco.MjData) -> float:
        """`_robot_obstacle_force` on the execution model, for logging.

        `oim.runtime.logs.log_step` sees a plain `MjData`, not an
        `mjx.Data`. Execution fidelity, so far larger than the planning
        figure the cost weights.

        Args:
            mj_data: The execution model's state at this step.

        Returns:
            The summed normal force in newtons, 0.0 with no such contact.
        """
        result = np.zeros(6)
        total = 0.0
        for c in range(mj_data.ncon):
            con = mj_data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            matches = (
                g1 in self._robot_geoms_set
                and g2 in self._obstacle_geoms_set
            ) or (
                g2 in self._robot_geoms_set
                and g1 in self._obstacle_geoms_set
            )
            if not matches:
                continue
            mujoco.mj_contactForce(self.mj_model, mj_data, c, result)
            total += result[0]
        return total

    def _contact_normal_force_z_mujoco(self, mj_data: mujoco.MjData) -> float:
        """Same quantity as `_contact_normal_force_z`, for logging/plotting.

        `oim.runtime.logs.log_step` runs against the execution model's
        plain `mujoco.MjData`, not an `mjx.Data` -- there is no planning-
        model forward pass at each real executed step to read `_impl`
        off of, only the physical one. Uses `mujoco.mj_contactForce`
        directly, no JAX/jit needed here.

        Reads at execution fidelity (fine timestep, many solver
        iterations), real Newtons -- not literally the number
        `_contact_z_cost` weighted during optimization, which always
        happens at planning fidelity. `oim.utils.costs.cost_series`
        applies the same weight/formula to this larger number for the
        diagnostics figure regardless, so the plotted `contact_z` bar
        reads as "would this be huge at execution scale", not a replay of
        the optimizer's own internal value.
        """
        result = np.zeros(6)
        total = 0.0
        for c in range(mj_data.ncon):
            con = mj_data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            matches = (
                g1 in self._stick_geoms_set and g2 in self._block_geoms_set
            ) or (
                g2 in self._stick_geoms_set and g1 in self._block_geoms_set
            )
            if not matches:
                continue
            mujoco.mj_contactForce(self.mj_model, mj_data, c, result)
            frame = con.frame.reshape(3, 3)
            total += result[0] * frame[0, 2]
        return total

    def _object_obstacle_force_mujoco(self, mj_data: mujoco.MjData) -> float:
        """Same quantity as `_object_obstacle_force`, for logging/plotting.

        The CPU counterpart, and for the same reason
        `_contact_normal_force_z_mujoco` is one -- `log_step` has no
        planning-model `mjx.Data` to read at an executed step.

        `result[0]` is the contact's normal component in its own frame,
        so this is the same friction-excluded normal force the cost
        weights, summed over every block/obstacle contact.

        Reads at execution fidelity -- real newtons, far larger than the
        planning-model figure the optimizer actually weights. Right for a
        human asking "how hard was the block really pressed into that
        obstacle", but not a replay of the optimizer's own number.

        Args:
            mj_data: The execution model's state at this step.

        Returns:
            The summed normal force in newtons, 0.0 with no such contact.
        """
        result = np.zeros(6)
        total = 0.0
        for c in range(mj_data.ncon):
            con = mj_data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            matches = (
                g1 in self._block_geoms_set
                and g2 in self._obstacle_geoms_set
            ) or (
                g2 in self._block_geoms_set
                and g1 in self._obstacle_geoms_set
            )
            if not matches:
                continue
            mujoco.mj_contactForce(self.mj_model, mj_data, c, result)
            total += result[0]
        return total

    def _top_contact_gate(self, state: mjx.Data, pose: jax.Array) -> jax.Array:
        """1 when the tip is resting on/near the block's top surface, 0
        otherwise -- not in the paper.

        Purely kinematic, and deliberately independent of both
        `w_contact_z_exp` and `_tip_height_cost`'s own mirrored-exponential
        experiment: it reads tip/block geometry only, never either
        mechanism's penalty, so `_ell_r`'s use of this (suppressing
        `align` during contact -- see that method's docstring) works
        identically whichever of the two top-riding experiments is
        active, or neither.

        Same footprint gate `_contact_z_cost` uses (xy inside the block's
        real T-shaped footprint, rotated into its frame), but a wider
        z-band above the true top surface -- 5cm, not that method's 1cm
        -- since this just needs to ask "is the tip plausibly still on
        top or just barely off it," not draw a tight penalty boundary.
        Also unlike that method, this does not look below the surface:
        there is nothing to suppress `align` for if the tip is already
        at/under `tip_target_z`, doing the side-push the rest of `_ell_r`
        wants it to.

        xarm6 only -- the point robot's tip never leaves `tip_target_z`,
        so it can never be in this band.
        """
        if self.robot != "xarm6":
            return jnp.asarray(0.0)
        tip_xyz = state.site_xpos[self.tip_site_id]
        top_z = self.tip_target_z + self.block_half_height
        local_xy = rotate(-pose[2], tip_xyz[:2] - pose[:2])
        inside_footprint = self.object_model.footprint.sdf(local_xy) <= 0.0
        near_top = (tip_xyz[2] >= top_z - 0.005) & (tip_xyz[2] <= top_z + 0.05)
        return jnp.where(inside_footprint & near_top, 1.0, 0.0)

    def _contact_z_cost(self, state: mjx.Data, pose: jax.Array) -> jax.Array:
        """Kinematic top-riding barrier.

        Reads no contact/force state -- only the tip site's own position
        and the block's SE(2) pose, both exact and identical at planning
        and execution fidelity. (A previous version penalized
        `_contact_normal_force_z`, the pusher-block contact's own normal
        force, but that solver quantity is unreliable at planning
        fidelity near contact onset.)

        Fires inside a 2cm slab straddling the block's true top surface
        (`tip_target_z + block_half_height`): 1cm into the block (below
        the surface) as well as 1cm above it -- and only when the tip's
        (x, y), rotated into the block's own frame, falls inside its
        actual T-shaped footprint (`self.object_model.footprint`, not an
        approximate circle). Exponential in the remaining clearance,
        symmetric around the surface: maximal exactly on it, still
        substantial a full 1cm either side, zero the instant either gate
        fails.

        The lower edge sits 1cm below the surface rather than only above
        it, so the tip dipping a couple mm below `top_z` -- while still
        plainly resting on top, just sunk in a little from contact
        compliance -- doesn't read as outside the slab and score exactly
        0. A hard cutoff at each 1cm boundary, not a fade: a keep-out
        zone for the hover approach that leads to top-riding, not a
        shaping term that should relax near the goal.

        Capped at `contact_z_cap`: nothing physically bounds how far a
        sampled rollout's tip position can be from the block during
        MPPI's exploration, so an uncapped `exp(gap**2)` reliably
        overflows to `inf` across the sample population and poisons
        MPPI's softmax weighting into `nan`.

        xarm6 only -- point robot's tip never leaves `tip_target_z`, so
        it can never enter the slab; returns 0 there without a separate
        branch mattering numerically, but early-returns anyway to skip
        the footprint/rotation work.
        """
        if self.robot != "xarm6":
            return jnp.asarray(0.0)
        tip_xyz = state.site_xpos[self.tip_site_id]
        top_z = self.tip_target_z + self.block_half_height
        dz_cm = 100.0 * (tip_xyz[2] - top_z)  # 0 at the surface, +/- either side

        local_xy = rotate(-pose[2], tip_xyz[:2] - pose[:2])
        inside_footprint = self.object_model.footprint.sdf(local_xy) <= 0.0
        in_slab = inside_footprint & (dz_cm >= -1.0) & (dz_cm <= 1.0)

        # 1 at the surface, 0 at +/-1cm -- symmetric, so dipping just below
        # the surface is exactly as costly as hovering the same distance
        # above it, not a free escape.
        gap = 1.0 - jnp.clip(jnp.abs(dz_cm), 0.0, 1.0)
        raw = self.w_contact_z_exp * jnp.exp((2.0 * gap) ** 2)
        # Below the surface (tip pressing into the top face) is strictly
        # worse than the same distance above it -- see
        # `contact_z_below_mult`. Discontinuous by that factor exactly at
        # dz == 0, invisible in practice since at default weights both
        # sides are already at `contact_z_cap` there.
        raw = jnp.where(dz_cm < 0.0, raw * self.contact_z_below_mult, raw)
        return jnp.where(in_slab, jnp.clip(raw, a_max=self.contact_z_cap), 0.0)

    def _joint3_cave_cost(self, state: mjx.Data) -> jax.Array:
        """Exponential penalty once xarm6_link3's own world z drops
        below `joint3_cave_z_threshold` -- not in the paper. Same
        structural pattern as `_tip_height_cost`'s below-mid-height
        branch: zero above the threshold, exponential in centimeters
        below it.

        Targets the elbow (joint 3, third from the base) winding all the
        way into a caved, concave configuration near a stuck object, with
        joint 5 rotating to compensate. link3's own world z tracks this
        cleanly: >=0.25m for normal operation, <0.20m once caved.
        `joint3_cave_z_threshold` (config-tunable, see `DEFAULT_COSTS`)
        sets where the exponential starts.

        No physical link3 for `robot="point"`; returns 0 there rather
        than reading a body id -- or the two weights -- that do not exist
        on that model. A point task is not given `w_joint3_cave_exp` or
        `joint3_cave_z_threshold` at all, so this guard is what keeps the
        attribute reads below unreachable there.
        """
        if self.robot != "xarm6":
            return jnp.asarray(0.0)
        z3 = state.xpos[self.joint3_link_id, 2]
        gap_cm = 100.0 * (self.joint3_cave_z_threshold - z3)
        return jnp.where(
            z3 >= self.joint3_cave_z_threshold,
            0.0,
            self.w_joint3_cave_exp * jnp.exp(gap_cm**2),
        )

    def _pusher_obstacle_cost(self, pusher_pos: jax.Array) -> jax.Array:
        """Pusher-vs-obstacle clearance hinge -- xarm6 only.

        Same hinge the object-side clearance term uses in "hinge" mode
        (`PlanarPushingObject.obstacle_cost`), applied to the pusher's
        own position instead of the block's footprint boundary: the
        object-side hinge alone keeps the *block* out of obstacles but
        doesn't stop the pusher itself cutting through one en route to
        "behind the object" -- `align` chases that position with no
        obstacle awareness of its own. Point robot uses
        `_robot_contact_cost` instead -- see `_robot_obstacle_cost`.
        """
        obj = self.object_model
        return self.pusher_obstacle_weight * obj.obstacles.hinge_cost(
            pusher_pos, obj.w_obstacle, self.pusher_obstacle_margin
        )

    def _robot_obstacle_cost(
        self, state: mjx.Data, pusher_pos: jax.Array
    ) -> jax.Array:
        """Robot-vs-obstacle avoidance term, robot-conditional.

        xarm6: `_pusher_obstacle_cost`, the tested geometric hinge on the
        pusher's own position (`pusher_obstacle_weight`/`_margin`).
        Point: `_robot_contact_cost`, the contact-FORCE-based mechanism
        the point-robot ADMM track uses (`w_robot_contact`) -- kept as
        that track's own tuning, not reimplemented against xarm6 without
        validating it there first. Neither is faded by `shaping_fade` --
        see that method.
        """
        if self.robot == "xarm6":
            return self._pusher_obstacle_cost(pusher_pos)
        return self._robot_contact_cost(state)

    def shaping_fade(self, pose: jax.Array) -> jax.Array:
        """Scale in [0, 1] on the near-goal-irrelevant terms.

        1 when ``||p - p_g|| >= shaping_fade_dist`` (full shaping), 0 at
        ``fade_floor`` -- not at the exact goal: reaching exactly
        ``pos_err == 0`` essentially never happens in a real rollout, so
        a fade that only hits zero there is always at least a little
        active in practice. Linear from 1 at ``shaping_fade_dist`` down
        to exactly 0 at ``fade_floor``, clipped flat at 0 for anything
        closer. ``fade_floor = 0`` (the default) reproduces the original
        single-point behavior exactly. ``shaping_fade_dist <= 0``
        disables the whole mechanism (the scale is identically 1).

        Faded (all shape the tip's route, which stops mattering once the
        object is one short correction from the goal): ``approach``,
        ``align``, ``tilt``, tip_height's above-threshold branch (all
        inside `_ell_r`, the last internally -- see `_tip_height_cost`),
        and ``effort`` (in `running_cost`, not here).

        Not faded: the object's clearance hinge (a goal near an obstacle
        is where driving the *block* into it stays wrong); tip_height's
        below-threshold (exponential) branch, ``contact_z``,
        ``joint3_cave`` (hard safety guarantees); the robot-obstacle
        term, xarm6's pusher hinge and the point robot's
        `_robot_contact_cost` alike; ``ell_o``/``ell_c``/the ADMM penalty.

        Also read by the ADMM layer: `ADMM._admm_iteration` scales the
        consensus penalty (`rho` and the duals' step) by this same radius
        for both blocks at once, so inside it the two blocks stop
        negotiating a shared wrench and each optimizes its own objective.
        `robot_running_cost`'s own object-obstacle term is a separate,
        ADMM-only addition and is never faded either.

        Always the global goal, even under local-goal tracking: the fade
        means "the task is nearly over, stop shaping posture", a
        statement about the global goal, while the local goal is only H
        steps ahead and near it by construction.

        Its radius does double duty: `tracking_goal` snaps the local goal
        back to g wherever this reads < 1, so the one number decides both
        when posture shaping stops and when the plan endpoint stops being
        the tracking target.
        """
        fade_dist = self.shaping_fade_dist
        floor = self.fade_floor
        pos_err = jnp.linalg.norm(pose[:2] - self.goal[:2])
        return jnp.where(
            fade_dist > 0.0,
            jnp.clip((pos_err - floor) / (fade_dist - floor), 0.0, 1.0),
            jnp.asarray(1.0, dtype=pos_err.dtype),
        )

    def _theta_ramp(self, pose: jax.Array) -> jax.Array:
        """Multiplier on q_theta/qf_theta, ramping up as position converges.

        1.0 at ``||p - p_g|| >= theta_ramp_dist``, ``q_theta_ramp`` at the
        goal. Inert (returns 1.0) if ``q_theta_ramp <= 1.0`` or
        ``theta_ramp_dist <= 0``. Its own radius, deliberately not
        `shaping_fade_dist` -- see Tasks.md if the two ever need
        reconciling.

        Flat baseline only. A converged orientation has near-zero cost
        gradient at its own weight, so it does little to resist being
        knocked back out by a much larger position-error gradient in the
        same rollout cost; this keeps its effective weight from
        collapsing as its own error does.
        """
        fade_dist = self.theta_ramp_dist
        pos_err = jnp.linalg.norm(pose[:2] - self.goal[:2])
        closeness = jnp.where(
            fade_dist > 0.0,
            1.0 - jnp.clip(pos_err / fade_dist, 0.0, 1.0),
            jnp.asarray(0.0, dtype=pos_err.dtype),
        )
        return jnp.where(
            self.q_theta_ramp > 1.0,
            1.0 + (self.q_theta_ramp - 1.0) * closeness,
            jnp.asarray(1.0, dtype=pos_err.dtype),
        )

    def _q_ramp_mult(self, state: mjx.Data) -> jax.Array:
        """Multiplier on q_pos/q_theta, compounding with real elapsed time.

        Flat baseline only (`running_cost`) -- see `q_ramp_per_step`'s own
        `DEFAULT_COSTS` comment for why this does not also apply inside
        `robot_running_cost`/`robot_terminal_cost`, which read the same
        two config keys through a different, non-compounding mechanism
        (`time_ramp`/`weight_scale`) for the ADMM track.

        ``min((1 + q_ramp_per_step) ** steps, q_ramp_max)``, where
        ``steps = state.time / self.dt`` -- `state.time` is always the
        real simulator clock, so this needs no controller-level state.
        Inside a rollout's own horizon, `state.time` keeps advancing past
        the real current step, which is correct: a rollout imagining
        itself further into a still-stuck future should see a
        correspondingly further-ramped cost.

        Inert (returns 1.0) if ``q_ramp_per_step <= 0`` or
        ``q_ramp_max <= 1.0``.
        """
        steps = state.time / self.dt
        grown = (1.0 + self.q_ramp_per_step) ** steps
        return jnp.where(
            (self.q_ramp_per_step > 0.0) & (self.q_ramp_max > 1.0),
            jnp.minimum(grown, self.q_ramp_max),
            jnp.asarray(1.0, dtype=grown.dtype),
        )

    def time_ramp(self, t: jax.Array) -> jax.Array:
        """Multiplier on goal tracking, growing with elapsed control steps.

        ``1 + q_ramp_per_step * (t / dt)``, capped at ``q_ramp_max``;
        inert at 0. Time-based, unlike `_theta_ramp`/`shaping_fade` which
        key on distance: other terms keep their weights while goal
        tracking pulls away from them.

        Constant over a horizon: `t` is read once at the rollout start by
        `RobotSubproblem._eval_rollouts_one`. Reading `state.time` per
        step would weight step H above step 0 and tilt plans toward their
        own tail.

        Args:
            t: Simulation time at the start of the horizon, in seconds.

        Returns:
            A scalar multiplier in ``[1, q_ramp_max]``.
        """
        steps = t / self.dt
        return jnp.clip(
            1.0 + self.q_ramp_per_step * steps, 1.0, self.q_ramp_max
        )

    def _ell_r(
        self,
        state: mjx.Data,
        pose: jax.Array,
        pusher_pos: jax.Array,
        obj_ref: jax.Array,
    ) -> jax.Array:
        """Robot stage cost l_r (paper eq. 20-22).

        fade * (approach + align + tilt) + tip height (its own
        above-threshold branch faded the same way, internally -- see
        `_tip_height_cost`) + contact_z + joint3_cave. See `shaping_fade`.

        Approach, align, and tilt all fade, linearly, all reaching
        exactly 1 at shaping_fade_dist and exactly 0 at fade_floor:
        quadratic shaping costs relax near the goal, since the task is
        essentially done and holding posture that tightly stops
        mattering. tip_height's below-threshold branch, contact_z, and
        joint3_cave stay unfaded: staying off the table and out of a
        caved configuration should never go slack, even near the goal.
        Control effort is faded the same way too, but in `running_cost`,
        not here -- see that method.

        `align` is additionally suppressed to 0 while `_top_contact_gate`
        reads 1 -- diagnosed from real single_obstacle failures: with the
        tip resting on the block's top surface, `align`'s pull toward a
        new approach angle and `_tip_height_cost`/`contact_z`'s pull
        downward-and-off fire at the same time, and since the tip is
        still in contact, the sideways half of that combined motion drags
        the block across the table via top-surface friction instead of
        cleanly disengaging first. Cutting `align` during contact leaves
        only the vertical escape pressure unopposed, so the tip comes off
        top-first, then `align` resumes once clear to drive the actual
        reorientation. Independent of `w_contact_z_exp` and of which
        top-riding experiment is active (see `_top_contact_gate`) -- it
        only reads geometry, not either mechanism's own penalty.
        """
        top_contact = self._top_contact_gate(state, pose)
        d_ee = jnp.sum((pusher_pos - pose[:2]) ** 2)
        approach = self.w_approach * jnp.clip(d_ee - self.r0**2, 0.0, None)

        to_object = pose[:2] - pusher_pos
        to_ref = obj_ref[:2] - pose[:2]
        cos_angle = jnp.sum(to_object * to_ref) / (
            jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
        )
        align = (
            self.w_align
            * jnp.clip(self.gamma0 - cos_angle, 0.0, None)
            * (1.0 - top_contact)
        )

        tilt = self.w_tilt * self._tilt(state)
        # Always against the true global goal, same convention as
        # `shaping_fade` -- see `_tip_height_cost` for why this feeds its
        # blend, not `obj_ref` (which under local-goal tracking is only
        # the plan's own endpoint).
        pos_err = jnp.linalg.norm(pose[:2] - self.goal[:2])
        tip_height = self._tip_height_cost(state, pos_err)
        contact_z = self._contact_z_cost(state, pose)
        joint3_cave = self._joint3_cave_cost(state)
        fade = self.shaping_fade(pose)
        return (
            fade * (approach + align + tilt) + tip_height
            + contact_z + joint3_cave
        )

    def tracking_goal(
        self, pose: jax.Array, local_goal: Optional[jax.Array]
    ) -> jax.Array:
        """What the robot block's goal-tracking terms aim at.

        `self.goal` unless local-goal tracking is on and a plan was
        offered. `local_goal is None` is a caller with no object plan to
        read (the direct-call tests, and any non-ADMM path), for which
        the global goal is the only defined answer.

        Inside the shaping-fade radius the target snaps back to
        `self.goal` even with the flag on: local-goal tracking exists so
        the robot is not penalized for following a plan that routes
        around something, and within `shaping_fade_dist` of the goal
        there is nothing left to route around -- x^{o*}_H is only H steps
        out and carries the object block's own residual error, so
        tracking it there asks the robot to stop short of g by exactly
        that residual.

        The gate is `shaping_fade` itself, not a second distance test, so
        the radius that means "the task is nearly over" cannot come to
        mean two different things. With `shaping_fade_dist <= 0` the fade
        is identically 1 and the gate is inert.

        One consequence worth knowing: an object block stuck under
        breakaway near the goal plans x^{o*}_H = x^o_0 (hold still), and
        inside the radius this overrides that with g -- the robot keeps
        pushing instead of settling for the stall.

        Resolved in one place because the running and terminal terms must
        aim at the same target -- they are the same tracking objective at
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
        weight_scale: jax.Array = 1.0,
    ) -> jax.Array:
        """Robot stage cost J_r (paper eq. 17).

        ``fade*w_robot_effort||u||^2 + ell_o + ell_r + obstacle
        + robot_contact``.

        The ADMM consensus penalty is *not* added here -- the ADMM layer adds
        it with the same `ConsensusSpace.penalty_cost` the object block uses.

        With `local_goal` tracking on, `ell_o` aims at the object block's
        horizon endpoint rather than the global goal. `ell_c` is left alone:
        it tracks the plan pointwise while `ell_o` now rewards reaching its
        end, which are different requests (pointwise tracking penalizes
        running ahead of schedule; endpoint tracking does not).
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        target = self.tracking_goal(pose, local_goal)
        # `weight_scale` = `time_ramp` at this horizon's start. Applied to
        # `ell_o` and the terminal term, NOT `ell_c`: letting the goal pull
        # away from the plan is the point.
        ell_o = weight_scale * se2_distance_sq(
            pose, target, self.q_pos, self.q_theta
        )
        ell_r = self._ell_r(state, pose, pusher_pos, obj_ref_t)
        # The OBJECT's proximity to obstacles, scored on the pose THIS
        # rollout produced. Same function and same weight the object block
        # uses (`PlanarPushingObject.running_cost`), deliberately: the two
        # blocks already share `ell_o` at shared gains, and a robot block
        # blind to obstacles will happily agree to a consensus wrench that
        # drives the block into one. Never faded, matching the object
        # block -- a goal beside an obstacle is exactly where routing the
        # block into it stays wrong.
        obj = self.object_model
        obstacle = obj.obstacles.exp_cost(
            obj.world_boundary(pose), obj.w_obstacle, obj.obstacle_decay
        )
        # Robot-vs-obstacle *contact*, a different quantity: the force the
        # robot's own body imparts, not the block's clearance.
        robot_contact = self._robot_contact_cost(state)
        # Squared command, faded on the same radius as `approach`/`align`
        # (`_ell_r` applies that fade to those two internally).
        effort = self.shaping_fade(pose) * self.w_robot_effort * jnp.sum(
            control**2
        )
        # No ell_c: the two blocks are coupled through the ADMM penalty
        # the layer adds, (rho/2)||A^r_t (-) z_t + y^r_t||^2, and nothing
        # else. Tracking the object block's plan pointwise scored the same
        # disagreement a second time under a different weight, against the
        # unilateral x^{o*}_t instead of the negotiated z_t.
        return ell_o + ell_r + obstacle + effort + robot_contact

    def robot_terminal_cost(
        self,
        state: mjx.Data,
        local_goal: Optional[jax.Array] = None,
        weight_scale: jax.Array = 1.0,
    ) -> jax.Array:
        """Heavier goal tracking, matching the object block's l_f.

        The term local-goal tracking changes most: `qf_*` are the heaviest
        weights in the robot block, and the terminal cost is not
        dt-weighted in the rollout while the stage costs are -- so this is
        where the mismatch between "what the plan asks for" and "the global
        goal" was priced highest.
        """
        pose = self._block_pose(state)
        target = self.tracking_goal(pose, local_goal)
        return weight_scale * se2_distance_sq(
            pose, target, self.qf_pos, self.qf_theta
        )
