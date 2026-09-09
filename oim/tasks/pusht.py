from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from oim import ROOT
from oim.objects import (
    PlanarPushingObject,
    library,
    rotate,
    se2_distance_sq,
    wrap_angle,
    wrench_weights,
)
from oim.objects.contact import (
    CONTACT_POINT_DIM,
    contact_point_to_wrench,
    project_contact_point,
    wrench_to_contact_point,
)
from oim.objects.sdf import Box
from oim.task_base import ConsensusTask, Task
from oim.utils.scenes import SCENES

# Worldbody geoms that are scenery rather than obstacles. Mirrors
# `tests/test_scenes.py`'s own `_SCENERY`.
_SCENERY_GEOMS = {"floor", "table"}

# Largest argument any cost term hands to `jnp.exp`, so an
# astronomical-means-veto penalty saturates instead of overflowing to `inf`.
# Two separate reasons this has to be capped, and the second one sets the
# value.
#
# 1. Overflow. float32 `exp` is `inf` past ~88, and an `inf` term times a
#    zero weight -- how every exponential guard here is switched off for an
#    ablation -- is NaN, not 0. One NaN sample makes the population max NaN,
#    so every softmax weight is NaN and the arm is commanded NaN from the
#    first control step. `_tip_height_cost`'s below-threshold branch is
#    selected by `jnp.where`, which evaluates BOTH branches, so the
#    discarded one overflows too -- in `oim.utils.costs`'s float64 replay
#    that surfaces as a RuntimeWarning on runs where the branch was never
#    taken at all. Any cap below ~88 fixes this.
#
# 2. float32 resolution, which is why the cap is 10 and not 80. These costs
#    are summed over the horizon and compared BETWEEN samples, and a barrier
#    only works if the rest of the cost survives being added to it. At
#    exp(36) = 4.3e15 it does not: float32 carries ~7 significant digits, so
#    `4.3e15 + 1.0 == 4.3e15` exactly. Every sample that trips the barrier
#    anywhere in its horizon then scores a bit-identical number, the softmax
#    cannot tell which of them does the task better, and that control step's
#    update is noise. Measured on a real run: the freeze's `sample_cost_min`
#    sat at 1e12-1e15 while command coherence was 0.05 on every joint.
#
#    exp(10) = 22026 keeps the barrier absolute -- 2e4 times the whole task
#    cost, no sample survives tripping it -- while `22026 + 0.01` still
#    resolves. That is the trade: large enough to veto, small enough that
#    vetoed samples remain rankable by everything else.
#
# `oim.utils.costs` reads this same constant, so the diagnostic replay and
# the planner cannot drift. MERGE NOTE (feat/real-mppi-obstacle -> main):
# this was 80.0 on main and 10.0 on the real branch; the real branch's
# measurement above is why the merged value is 10.0.
EXP_ARG_MAX = 10.0

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
    # The same idea in the contact parameterization's units. A separate key
    # rather than reusing `w_rate`: the channels are metres and newtons
    # there, not newtons and newton-metres, so one number cannot mean the
    # same thing in both. Read only when
    # `consensus="contact_point"`; see `PushT.object_rate_cost`.
    "w_contact_rate": [16.0, 16.0, 1.0],
    # Object-vs-obstacle clearance (see `PlanarPushingObject.obstacle_cost`).
    # Redefined 2026-09-07, per Shahid, to the same shape `support_cost`
    # (below) uses -- zero until a footprint point is within
    # `obstacle_margin` of the obstacle, then exponential in how far past
    # the margin it is (not quadratic, unlike support: crossing into an
    # obstacle's margin should be a much harder veto than nearing the
    # table edge, so MPPI never seriously considers a plan that does).
    # `obstacle_decay` is now that exponential's e-folding-ish scale
    # rather than an always-on decay length; `w_obstacle` unchanged in
    # role, still the weight at the point the penalty engages.
    "w_obstacle": 10.0,
    "obstacle_decay": 0.02,
    "obstacle_margin": 0.02,
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
    # Not cost weights. Both size the object block's *action*, and the
    # configs put them under `admm:` for that reason; they are kept here
    # so run files written before the move still replay, and so a task
    # built with neither argument behaves as it always did. `PushT`'s own
    # `wrench_fraction`/`contact_fraction` arguments take precedence.
    #
    # What one unit of object action is worth, as a fraction of the
    # friction-cone limit (`PlanarPushingObject.wrench_sample_fraction`).
    # `None` keeps the per-embodiment default. Read under
    # `consensus="wrench"` and `"object_pose"`, whose action IS the wrench.
    "wrench_fraction": None,
    # lambda's ceiling under `consensus="contact_point"`, as the same
    # fraction of `mu*m*g`. `None` falls back to `wrench_fraction`, which
    # is what every config did before this key existed. Its own key
    # because the two bound different things -- a coupled 3-channel wrench
    # against a single normal force -- and measure out differently: the
    # wrench ceiling is fraction*sqrt(3) on the coupled norm, while lambda
    # is one scalar that has to clear breakaway on its own.
    "contact_fraction": None,
    # Robot block only (paper eq. 20-22).
    "w_robot_effort": 0.05,  # squared control effort
    "w_approach": 40.0,  # approach: pull the tip toward the object
    "r0": 0.02,  # radius inside which approach goes slack
    # Which point `approach` pulls the tip toward. 0 (default: the
    # paper's eq. 20-22 form, and what every sim config runs) = block
    # origin. 1 = the footprint wall (SDF ring): the origin form's minimum
    # includes the column above the block, which was the measured
    # climb-onto-the-block failure (2026-08-28 15:58 run), while the ring
    # is exactly 0 over the footprint and matches the T's true shape on
    # every side; `r0` then means clearance beyond the wall. The real
    # config selects 1 explicitly (2026-09-07 per Shahid). The
    # wrench-informed-target mode (2) and the linear approach-power
    # option were both removed the same day -- approach is always the
    # quadratic form now, at whichever point this key selects.
    "approach_mode": 0.0,
    # Fold the tip's HEIGHT error into the approach distance (mode 1
    # only), so the term pulls at the actual contact pose {wall ring,
    # z = tip_target_z} instead of leaving z to the tip-height pull
    # alone. Gated to OUTSIDE the footprint: over the block a mid-height
    # z-target could only mean "press through the top face", which the
    # tip-height term alone already prices. Inert in mode 0.
    "approach_z": 0.0,
    "w_align": 15.0,  # stay behind the object relative to the reference
    "gamma0_deg": 15.0,  # alignment cone half-angle
    # Cross-solve smoothing of `align`'s reference (the ADMM path's own
    # object-plan endpoint; inert on the flat path, whose reference is
    # always the fixed global goal). Restored 2026-09-07 after briefly
    # being deleted along with the wrench-informed-target (mode 2)
    # machinery it used to also feed -- independent of that removal:
    # the object block's endpoint theta still moves solve to solve on
    # ADMM, and this is still what keeps `align`'s target from jittering
    # with it. 0.0 (default) = off, bit-identical to no smoothing.
    "wia_ref_alpha": 0.0,
    "w_tilt": 30.0,  # keep the stick pointing down (3D only)
    # Tip height, both above AND below the block's own mid-height
    # (`tip_target_z`): a single symmetric exponential, in CENTIMETRES
    # squared. Simplified 2026-09-07, per Shahid: previously quadratic at
    # or above mid-height and only exponential below it (see git history
    # for that piecewise form, and for the re-anchoring work it needed
    # once the mid-height anchor was found to cause near-goal stalls --
    # moot now, since a barrier this steep on both sides makes the
    # separate contact-z top-riding barrier redundant: the tip has no
    # reason to ever climb high enough to skim the top face in the first
    # place). Never faded -- staying at pushing height is a safety
    # property, not shaping that should relax near the goal.
    "w_z_tip_exp": 1.0,
    # Flat baseline only (`running_cost`/`terminal_cost`, not
    # `robot_running_cost`). Multiplier on q_theta/qf_theta, ramping from
    # 1x at pos_err >= theta_ramp_dist to this value at the goal -- 1.0 =
    # inert. A quadratic term's gradient near its own zero is small, so
    # once orientation is converged it does little to resist being
    # knocked back out by continued position-driven pushing; this keeps
    # its weight meaningful even at small error. See `_theta_ramp`.
    "q_theta_ramp": 1.0,
    # Radius [m] the above ramps over. 0 = inert (the ramp never
    # opens, whatever `q_theta_ramp` says).
    "theta_ramp_dist": 0.0,
    # Goal-tracking weight grows 1 + q_ramp_per_step * step, capped at
    # q_ramp_max. 0.0 = inert.
    #
    # Two mechanisms read these same two keys, on two disjoint call
    # paths: `time_ramp`/`weight_scale` inside `robot_running_cost`/
    # `robot_terminal_cost` (ADMM's robot block) and `_q_ramp_mult`
    # inside `running_cost` (the flat baseline). They now agree --
    # `1 + q_ramp_per_step * steps` capped at `q_ramp_max`, with `steps`
    # read once at the rollout's start -- so a task built for one path
    # and driven through the other gets the same ramp. The flat one used
    # to compound and to re-read the clock every step inside the horizon.
    "q_ramp_per_step": 0.0,
    "q_ramp_max": 5.0,
    # Fade approach, align, tilt, and the above-threshold branch of tip
    # height (the last internally, see `_tip_height_cost`) linearly as
    # ||p - p_g|| -> 0 (0 = disabled). Control effort fades too, in
    # `running_cost`/`robot_running_cost`, as does the ADMM consensus
    # penalty (`ADMM._admm_iteration` scales `rho` and the dual step by
    # this same factor). Never faded: tip_height's below-threshold
    # (exponential) branch, contact_z, `_robot_contact_cost`, the
    # xarm6-only pusher-obstacle hinge, and every goal/object term. See
    # `shaping_fade`.
    "shaping_fade_dist": 0.0,
    # How much heading error is forgiven outright, and how that forgiveness
    # shrinks as the object nears the goal -- flat baseline only, see
    # `_theta_slack`. A single stick cannot translate a T without also
    # rotating it, so a plain squared heading cost fines every push for a
    # rotation the robot has no way to avoid; far from the goal that fine
    # can exceed the position gain, and standing still wins. Forgiving a
    # bounded amount of heading error out there removes the fine without
    # giving up precision where it matters.
    # rad. 0.0 = inert (the default): the heading cost is then exactly the
    # plain squared one.
    "theta_slack_max": 0.0,
    # m. At or beyond this distance from the goal, the full `theta_slack_max`
    # is forgiven.
    "theta_slack_far_dist": 0.15,
    # m. At or below this distance, nothing is forgiven. MUST be <= the run's
    # goal_pos_tol: still forgiving heading error where position is already
    # inside tolerance would leave nothing in the cost pushing the two
    # success conditions to hold at the same time.
    "theta_slack_near_dist": 0.05,
    # Pusher-vs-obstacle hinge, scaled relative to w_obstacle and with its
    # own reach. xarm6 only -- see `_pusher_obstacle_cost`. 0.0 = inert,
    # which is the default: opt-in per config. `w_robot_contact` is the
    # other, reactive half (actual contact force); this one is the
    # preventive, geometric half.
    "pusher_obstacle_weight": 0.0,
    "pusher_obstacle_margin": 0.06,
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


def resolve_action_fractions(
    cost: Dict[str, Any],
    wrench_fraction: Optional[float],
    contact_fraction: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Settle the two object-action fractions against their legacy home.

    Neither is a cost weight -- both size the object block's *action* --
    so the configs put them under `admm:`. They are still accepted under
    `costs:`, where they used to live, so run files written before the
    move (which record the merged `task.costs`) keep replaying through
    `oim/worlds/real3d/scripts/plot_run_from_json.py`.

    Args:
        cost: The merged cost mapping, possibly carrying the legacy keys.
        wrench_fraction: Explicit value, or None to fall back.
        contact_fraction: Explicit value, or None to fall back.

    Returns:
        `(wrench_fraction, contact_fraction)`, each still None when
        neither location supplied one -- the caller's own default then
        applies.
    """
    return (
        cost["wrench_fraction"] if wrench_fraction is None else wrench_fraction,
        cost["contact_fraction"]
        if contact_fraction is None
        else contact_fraction,
    )


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

    # `q_pos`/`q_theta` carry a ramp in elapsed control steps, so the cost
    # functions must see the rollout's OWN start time rather than each
    # stepped state's clock -- see `_q_ramp_mult`.
    freeze_cost_time = True

    def __init__(
        self,
        impl: str = "jax",
        clutter: bool = False,
        planning_dt: Optional[float] = None,
        planning_iterations: Optional[int] = None,
        planning_ls_iterations: Optional[int] = None,
        robot: Literal["point", "xarm6"] = "point",
        consensus_source: Literal["twist", "twist_exact", "contact"] = "twist",
        twist_stick_speed: float = 0.005,
        consensus: Literal[
            "wrench", "contact_point", "object_pose"
        ] = "wrench",
        env: str = "clutter",
        push_object: str = library.SCENE_DEFAULT,
        goal: Optional[Sequence[float]] = None,
        costs: Optional[Dict[str, Any]] = None,
        wrench_fraction: Optional[float] = None,
        contact_fraction: Optional[float] = None,
        realized_wrench_clip: Optional[Sequence[float]] = None,
        push_speed: float = 0.05,
        local_goal: bool = False,
        local_goal_lookahead: float = 0.0,
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
                `"twist_exact"` inverts the same relation as `"twist"` but
                including the slip term the plant actually integrates; see
                `_consensus_from_twist_exact`.
            twist_stick_speed: Speed [m/s] below which the block counts as
                sticking, for `consensus_source="twist_exact"` only. Set it
                at the measured noise floor of the object twist -- on the
                lab rig, FoundationPose position noise is sigma ~ 1.2 mm and
                the twist is a finite difference through an alpha = 0.4 EMA,
                and the observed speed while the block is provably at rest
                has p95 = 2.8 mm/s, p99 = 5.2 mm/s, max 6.0 mm/s. Hence the
                5 mm/s default: it is the EMA settling tail, not the raw
                pose noise (which alone would give ~1.4 mm/s), that sets it.
            env: Which scene to load, by name from
                `oim.utils.scenes.SCENES` (only meaningful with
                `clutter=True`). Must support `robot`, or raises.
            push_object: WHAT gets pushed across it, independent of the
                scene. `SCENE_DEFAULT` leaves the scene's own MJCF alone;
                any other key of `oim.objects.library.PUSH_OBJECTS`
                rebuilds the block, its goal markers and its table
                friction, leaving the table, obstacles and goal pose
                exactly as the scene declared them.
            goal: Overrides the scene's own goal pose, world-frame SE(2)
                `[x, y, theta]`. Used by `examples/poses/<task>.yaml` to
                run one scene against several goals. The goal marker in
                the MJCF is a mocap body, moved separately by
                `oim.worlds.sim3d.build`; this sets what the *costs* aim at.
            costs: Overrides for any subset of `DEFAULT_COSTS`, normally
                the `costs:` block of `oim/configs/robots/{robot}.yaml`. One
                mapping feeds both ADMM blocks, so the shared goal-tracking
                weights cannot drift apart between them. Unknown keys raise.
            wrench_fraction: What one unit of object action is worth, as a
                fraction of the friction-cone limit -- so the largest
                wrench the block can propose is `fraction * w_limit`,
                whose normalized magnitude tops out at `fraction*sqrt(3)`
                against `PlanarPushingObject.step`'s threshold of 1. Read
                under `consensus="wrench"` and `"object_pose"`, whose
                action *is* the wrench. Normally the `admm:` block of the
                robot config -- not a cost weight, which is why it lives
                there. `None` falls back to the legacy `costs` key, then
                to the per-embodiment default.
            contact_fraction: lambda's ceiling under
                `consensus="contact_point"`, as a fraction of `mu*m*g`.
                Its own knob because the two bound different things: a
                coupled 3-channel wrench against a single normal force
                that has to clear breakaway on its own, so one number
                cannot be right for both. Same config location and same
                fallback chain as `wrench_fraction`, whose value it
                inherits when unset everywhere.
            consensus: What the two ADMM blocks agree on, and
                what the object block samples in -- one choice drives
                both, so the sampled decision always *is* the agreed
                quantity. `"wrench"` (default) is the paper's own choice,
                eq. 24: the block samples [f_x, f_y, tau] and A^r is the
                wrench the robot's rollout imparts. `"contact_point"`
                makes it [p_x, p_y, lambda] -- where on the boundary to
                push, in the object's *body* frame, and how hard along the
                inward normal. The wrench is then derived, w = J_c^T f, at
                the object's pose each step, so every proposal is
                realizable by construction: no pulling forces, no pure
                torques, nothing off the boundary.
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
            local_goal_lookahead: Distance [m] ahead along the object
                block's plan that the local goal sits, i.e. how tightly
                the robot is asked to follow the plan rather than only its
                endpoint. See `local_goal_from_plan`. 0 (default) keeps
                the endpoint, which is what shipped before this existed.
                Read only when `local_goal` is on.

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
        if consensus_source not in ("twist", "twist_exact", "contact"):
            raise ValueError(
                "consensus_source must be 'twist', 'twist_exact' or "
                f"'contact', got {consensus_source!r}"
            )
        if twist_stick_speed <= 0.0:
            raise ValueError(
                f"twist_stick_speed must be > 0, got {twist_stick_speed!r}"
            )
        if consensus_source == "contact" and robot != "point":
            raise ValueError(
                "consensus_source='contact' is only valid for robot='point'; "
                "an articulated arm's contact force appears as J^T f spread "
                "across its joints, not at a single pair of DOFs."
            )
        if consensus not in ("wrench", "contact_point", "object_pose"):
            raise ValueError(
                "consensus must be 'wrench', 'contact_point' or "
                f"'object_pose', got {consensus!r}"
            )

        cost = resolve_costs(costs)
        self.costs = cost
        wrench_fraction, contact_fraction = resolve_action_fractions(
            cost, wrench_fraction, contact_fraction
        )
        self.clutter = clutter
        self.robot = robot
        self.consensus_source = consensus_source
        self._twist_stick_speed = float(twist_stick_speed)
        self.consensus = consensus
        self.use_local_goal = local_goal
        self.local_goal_lookahead = float(local_goal_lookahead)
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
        # `push_object` rebuilds the scene's pushed object before the model
        # is compiled -- see `oim.objects.library`. `None` (the default) is
        # the scene's own MJCF untouched, which is what every recorded run
        # and every scene test loads.
        self.push_object = library.push_object(push_object)
        path = ROOT + "/models/" + scene_path
        if self.push_object is None:
            mj_model = mujoco.MjModel.from_xml_path(path)
        else:
            mj_spec = mujoco.MjSpec.from_file(path)
            library.apply_to_spec(mj_spec, self.push_object)
            mj_model = mj_spec.compile()
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
            # Obstacles are the worldbody geoms that are not scenery (the
            # pushed object and the goal marker each live in their own
            # body, so nothing else hangs off the world) -- PLUS any
            # "obs*"-named mocap body, which real obstacles calibrated
            # from ArUco detection now are (oim.worlds.real3d writes their
            # pose once at run start; mock keeps the MJCF default). Named
            # rather than "any mocap body", so goal/local_goal -- also
            # mocap -- are never swept in here.
            self.obstacle_geoms = jnp.array(
                sorted(
                    g
                    for g in range(mj_model.ngeom)
                    if mj_model.geom(g).name not in _SCENERY_GEOMS
                    and (
                        mj_model.geom_bodyid[g] == 0
                        or (
                            mj_model.body_mocapid[mj_model.geom_bodyid[g]] >= 0
                            and mj_model.body(mj_model.geom_bodyid[g]).name
                            .startswith("obs")
                        )
                    )
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
                # The pushed object's own shape and physics override the
                # scene's whenever one was selected: the scene still owns
                # the table, the obstacles and the goal POSE, but what is
                # being pushed across it -- and therefore its friction
                # budget -- comes from the object.
                footprint=(spec.footprint() if self.push_object is None
                           else self.push_object.footprint()),
                obstacles=spec.obstacles_for(robot),
                mu=spec.mu if self.push_object is None
                else self.push_object.mu,
                mass=spec.mass if self.push_object is None
                else self.push_object.mass,
                limit_surface_radius=(
                    spec.limit_surface_radius if self.push_object is None
                    else self.push_object.limit_surface_radius
                ),
                w_pos=cost["q_pos"],
                w_theta=cost["q_theta"],
                wf_pos=cost["qf_pos"],
                wf_theta=cost["qf_theta"],
                w_effort=cost["w_effort"],
                w_rate=cost["w_rate"],
                w_obstacle=cost["w_obstacle"],
                obstacle_decay=cost["obstacle_decay"],
                obstacle_margin=cost["obstacle_margin"],
                support=_support_region(mj_model),
                w_support=cost["w_support"],
                support_margin=cost["support_margin"],
                wrench_sample_fraction=(
                    (1.0 if robot == "xarm6" else 0.5)
                    if wrench_fraction is None
                    else wrench_fraction
                ),
                push_speed=push_speed,
                # Same three keys the flat path reads -- the two blocks
                # must forgive the same heading error or they aim at
                # different targets (see the q_*/qf_* rule above).
                theta_slack_max=float(cost["theta_slack_max"]),
                theta_slack_far_dist=float(cost["theta_slack_far_dist"]),
                theta_slack_near_dist=float(cost["theta_slack_near_dist"]),
                q_theta_ramp=float(cost["q_theta_ramp"]),
                theta_ramp_dist=float(cost["theta_ramp_dist"]),
            )
            self._realized_wrench_clip = (
                jnp.asarray(realized_wrench_clip, dtype=float)
                if realized_wrench_clip is not None
                else self.object_model.wrench_limit
            )
            # Cached here, as Python floats, because every reader is
            # called from inside a traced `optimize`: indexing a jnp
            # constant under trace yields a tracer, and `float()` on a
            # tracer raises. See `_contact_f_max`.
            #
            # `contact_fraction` scales the friction-cone limit directly
            # rather than `action_scale`, so raising lambda's ceiling for
            # a contact-point run no longer also widens the wrench box a
            # wrench run samples. Unset, it reads `action_scale[0]` --
            # exactly the old behaviour, so existing configs and run files
            # are unaffected.
            self._contact_fraction = contact_fraction
            self._contact_f_max = float(
                self.object_model.action_scale[0]
                if contact_fraction is None
                else contact_fraction * self.object_model.wrench_limit[0]
            )
            self._w_contact_rate = wrench_weights(cost["w_contact_rate"])
            self._contact_reach = float(
                self.object_model.footprint.bounding_radius
            )

            # Robot-level cost weights (paper eq. 20).
            self.w_robot_effort = cost["w_robot_effort"]
            self.w_approach, self.r0 = cost["w_approach"], cost["r0"]
            self.approach_z = bool(float(cost.get("approach_z", 0.0)))
            # 2026-09-07, per Shahid: modes 0 (origin) and 1 (SDF wall)
            # only now -- mode 2 (wrench-informed target) and the linear
            # approach-power option were both removed the same day, along
            # with the footprint-centroid/vertex setup mode 2 needed (see
            # git history if either is ever revisited). `.get` default 0
            # replays old configs/run files predating this key at the
            # origin-distance form.
            self.approach_mode = int(float(cost.get("approach_mode", 0.0)))
            self.w_align = cost["w_align"]
            self.gamma0 = jnp.cos(jnp.deg2rad(cost["gamma0_deg"]))
            self.wia_ref_alpha = float(cost["wia_ref_alpha"])
            # Not in the paper. Retuning w_tilt through 5/20/30/50 never
            # arrested the drift: over five 500-step runs the tilt angle
            # rises on 52-55% of steps (total variation ~8 rad for a net
            # ~1.3), and mean tilt rank-orders with final position error
            # across all five scenes. See `_tilt` -- the functional form,
            # not the weight, was the free parameter.
            #
            # Zero for the point pusher: its site cannot rotate, so `_tilt`
            # is a constant 2.0 -- cancels in every sampler, but was 60 of
            # the 60.6 total `_ell_r` in the cost figure. Forced here, not
            # in the config, so no point config can reintroduce it.
            self.w_tilt = 0.0 if robot == "point" else cost["w_tilt"]
            # Likewise zero for the point pusher: no z DOF, tip sits
            # exactly at `tip_target_z` in every point scene, so
            # `_tip_height_cost` is identically 0.
            point_tip = robot == "point"
            self.w_z_tip_exp = 0.0 if point_tip else cost["w_z_tip_exp"]
            self.w_robot_contact = float(cost["w_robot_contact"])
            self.pusher_obstacle_weight = float(
                cost["pusher_obstacle_weight"]
            )
            self.pusher_obstacle_margin = float(
                cost["pusher_obstacle_margin"]
            )
            self.q_theta_ramp = float(cost["q_theta_ramp"])
            self.theta_ramp_dist = float(cost["theta_ramp_dist"])
            self.shaping_fade_dist = float(cost["shaping_fade_dist"])
            self.theta_slack_max = float(cost["theta_slack_max"])
            self.theta_slack_far_dist = float(cost["theta_slack_far_dist"])
            self.theta_slack_near_dist = float(cost["theta_slack_near_dist"])
            # Target tip height: the block's own resting z, read from the
            # model rather than hardcoded. The sole anchor for
            # `_tip_height_cost` now that it is one symmetric exponential
            # -- see there.
            self.tip_target_z = float(mj_model.body("block").pos[2])
            self.q_pos, self.q_theta = cost["q_pos"], cost["q_theta"]
            self.qf_pos, self.qf_theta = cost["qf_pos"], cost["qf_theta"]
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

    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
    ) -> jax.Array:
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

        `q_pos`/`q_theta` are both scaled by `_q_ramp_mult` -- the flat
        baseline's own route to the same ramp `robot_running_cost` gets
        through `time_ramp`/`weight_scale`. `state.time` here is the
        ROLLOUT's start, not the stepped state's clock, because
        `freeze_cost_time` is set; see `_q_ramp_mult`.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        q_ramp = self._q_ramp_mult(state)
        q_theta = self.q_theta * self._theta_ramp(pose) * q_ramp
        ell_o = self._se2_cost(pose, self.q_pos * q_ramp, q_theta)
        obj = self.object_model
        obstacle = obj.obstacle_cost(pose) + obj.support_cost(pose)
        ell_r = self._ell_r(state, pose, pusher_pos, self.goal)
        # Faded (linearly, like align) -- recomputed here rather than
        # exposed from `_ell_r`, since that method is also
        # `terminal_cost`'s, which has no control to fade.
        effort = (
            self.shaping_fade(pose)
            * self.w_robot_effort
            * jnp.sum(control**2)
        )
        # Robot-vs-obstacle *contact*, the same term `robot_running_cost`
        # charges, so flat and ADMM price a collision identically. Never
        # faded: a collision near the goal is as wrong as one anywhere
        # else.
        robot_contact = self._robot_contact_cost(state)
        # Preventive half of the same concern: steer the tip around an
        # obstacle before it gets there. Never faded, same reasoning.
        pusher_obstacle = self._pusher_obstacle_cost(pusher_pos)
        return (
            ell_o + obstacle + ell_r + effort
            + robot_contact + pusher_obstacle
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost l_T(x_T) for plain (non-ADMM) MPC.

        Heavier SE(2) goal tracking (`qf_*`) plus the same l_r as the
        stage cost. Stage costs are dt-weighted in the rollout and the
        terminal is not, so this is where the pushing geometry is scored
        at full weight; a goal-only terminal let MPPI buy a better
        predicted pose by abandoning it.

        `_q_ramp_mult` is deliberately NOT applied here, unlike in
        `running_cost`. The real branch had added it during the merge
        window; reverted on the way in, because `xarm6.yaml` runs
        `q_ramp_per_step: 0.005` up to `q_ramp_max: 25.0` and letting that
        multiply the terminal weight too would have silently changed every
        sim run. It costs the real path nothing: no real config sets
        `q_ramp_per_step`, so the multiplier is 1.0 there either way.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        qf_theta = self.qf_theta * self._theta_ramp(pose)
        ell_f = self._se2_cost(pose, self.qf_pos, qf_theta)
        return (
            ell_f
            + self._ell_r(state, pose, pusher_pos, self.goal)
            + self._pusher_obstacle_cost(pusher_pos)
        )

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

        For `consensus="wrench"`, the friction-cone limit -- the
        largest wrench the support surface can transmit. For
        `"contact_point"`, the object's bounding radius in the two
        position channels and the largest normal force in lambda. Both
        normalize the ADMM penalty and residuals, keeping `rho`, `eps_r`
        and `eps_s` scale-free across the two choices.
        """
        if self.consensus == "contact_point":
            reach = self._contact_reach
            return jnp.array([reach, reach, self._contact_f_max])
        if self.consensus == "object_pose":
            # 1.0 rad, not pi: a yaw error of one radian sweeps the
            # footprint's edge through one body radius, so a normalized
            # residual of 1 is the same physical displacement in all three
            # channels. Normalizing yaw by pi instead would make half a
            # turn read the same as one radius of translation.
            reach = self._contact_reach
            return jnp.array([reach, reach, 1.0])
        return self.object_model.wrench_limit

    def object_consensus(
        self,
        obj_state: jax.Array,
        w: jax.Array,
        action: Optional[jax.Array] = None,
    ) -> jax.Array:
        """A^o: the block's own decision, whichever variable that is.

        A selection off U^o under `"wrench"` and `"contact_point"` --
        the paper's eq. 24 -- because there `consensus` drives the
        sampling space too, so what the block decides *is* what the blocks
        agree on. Under `"contact_point"` that is the action; `w` is the
        wrench derived from it and is not the consensus value.

        `"object_pose"` breaks that identity on purpose: the block still
        decides and rolls out a wrench, and A^o is the pose eq. 5 produced
        from it -- a *result* of U^o rather than a selection off it. That
        is what makes the two blocks agree on where the object ends up
        instead of on what pushes it there, and it is why the rate cost
        needs `object_rate_values` to find the decision again.
        """
        if self.consensus == "contact_point":
            return action
        if self.consensus == "object_pose":
            return obj_state
        return w

    def object_action_scale(self) -> jax.Array:
        """Map a unit sample from the object optimizer to a physical wrench.

        Identity under `"contact_point"`: that action already carries its
        own units (metres and newtons) and `project_object_action` bounds
        it, so there is nothing to rescale.
        """
        if self.consensus == "contact_point":
            return jnp.ones(CONTACT_POINT_DIM)
        return self.object_model.action_scale

    @property
    def object_action_dim(self) -> int:
        """3 either way: [f_x, f_y, tau] or [p_x, p_y, lambda]."""
        return (
            CONTACT_POINT_DIM
            if self.consensus == "contact_point"
            else self.consensus_dim
        )

    def object_action_bounds(self) -> Tuple[jax.Array, jax.Array]:
        """Box bounds on the block's decision.

        The wrench case defers to the base class's unit box, where
        `object_action_scale` carries the physics. The contact case needs
        its own box in real units: the point anywhere within the object's
        bounding radius (`project_object_action` puts it back on the
        boundary) and lambda unilateral.
        """
        if self.consensus != "contact_point":
            return super().object_action_bounds()
        reach, f_max = self._contact_reach, self._contact_f_max
        return (
            jnp.array([-reach, -reach, 0.0]),
            jnp.array([reach, reach, f_max]),
        )

    def initial_object_action(self) -> Optional[jax.Array]:
        """A real boundary point, not the origin.

        Taken from `sample_boundary` rather than by projecting an interior
        seed: the footprint's origin lies on its medial axis, where
        projection provably stalls (`Shape.project_to_boundary`) and the
        boundary normal every contact quantity depends on is undefined.
        Which face is picked barely matters -- MPPI's own Gaussian is
        sized against the object (`object_noise_level`), so its samples
        reach past the seed's face and the update migrates from here.
        """
        if self.consensus != "contact_point":
            return None
        samples = self.object_model.footprint.sample_boundary(4)
        start = samples[jnp.argmin(samples[:, 1])]
        # `concatenate`, not `jnp.array([...])`: under trace `start`'s
        # entries are tracers, which a Python list of scalars cannot hold.
        return jnp.concatenate(
            [start, jnp.array([0.25 * self._contact_f_max])]
        )

    def object_action_to_consensus(
        self, obj_state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Map the block's decision to the wrench that drives the rollout.

        Despite the name this returns the *wrench* in both modes, because
        that is what `object_dynamics` integrates. Under `"wrench"` the
        two coincide and this is also A^o; under `"contact_point"` A^o is
        the action itself and comes from `object_consensus`.

        The contact map is evaluated at `obj_state`, so one fixed contact
        point stays on the same material point of the object and the
        wrench it produces turns as the object rotates -- the behaviour a
        sampled world-frame wrench cannot express.
        """
        if self.consensus == "contact_point":
            return contact_point_to_wrench(
                self.object_model.footprint, obj_state, action
            )
        # Paper eq. 18 (Pi_F): project the sampled wrench into the limit
        # surface at the single gate everything else reads through --
        # dynamics, A^o, rate/effort costs. Consensus then carries a
        # feasible magnitude; sampling above the box (wrench_fraction > 1)
        # stays useful for reaching the surface in any direction.
        return self.object_model.project_wrench(
            action * self.object_action_scale()
        )

    def project_object_action(
        self, action: jax.Array, obj_state: Optional[jax.Array] = None
    ) -> jax.Array:
        """Contact point back onto the boundary, lambda into [0, f_max].

        Applied to every sample before it is rolled out, so the block can
        never score a wrench no point contact could produce. Unconditional,
        so `obj_state` is unused -- the constraint is on the action in the
        body frame, which does not depend on where the object is.
        """
        del obj_state
        if self.consensus != "contact_point":
            return action
        return project_contact_point(
            self.object_model.footprint, action, self._contact_f_max
        )

    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Quasi-static limit-surface dynamics (paper eq. 5).

        Always takes a wrench, in both modes: `object_action_to_consensus`
        has already derived it from the contact point.
        """
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

    def object_rate_values(
        self, wrenches: jax.Array, values: jax.Array
    ) -> jax.Array:
        """A^o under `"contact_point"`, the wrench otherwise.

        The block's decision is the wrench under both `"wrench"` and
        `"object_pose"` -- only the consensus variable differs -- and is
        the contact point under `"contact_point"`, where A^o is that same
        action. Charging A^o under `"object_pose"` would price the
        object's *pose* changing along the horizon, i.e. charge it for
        moving, which is the opposite of what a rate cost is for.
        """
        return values if self.consensus == "contact_point" else wrenches

    def object_rate_cost(
        self, values: jax.Array, prev: Optional[jax.Array] = None
    ) -> jax.Array:
        """Charge for the consensus decision changing along the sequence.

        Receives A^o, so under `"contact_point"` it charges for the
        contact *sliding* and for lambda chattering, weighted by
        `w_contact_rate` -- a separate key from `w_rate`, because those
        channels are metres and newtons rather than newtons and
        newton-metres and one number cannot mean the same thing in both.

        It is the *only* term in the whole formulation that knows
        relocating a contact is a real maneuver. The object block's
        rollout will happily teleport the contact across the object
        between consecutive steps -- it just recomputes w = J_c^T f at the
        new point -- while the arm has to retract, travel round and
        re-approach, which takes many control steps. Set this too low and
        the plan asks for a contact the robot cannot chase, so the robot
        chases a target that moves every step and makes no progress.

        Quadratic in the *normalized* step, so the price is strongly
        superlinear in distance: sliding along one face stays nearly free
        while hopping to another face does not.
        """
        if self.consensus != "contact_point":
            return self.object_model.rate_cost(values, prev)
        scale = self.consensus_scale()
        normalized = values / scale
        if prev is not None:
            normalized = jnp.concatenate(
                [(prev / scale)[None, :], normalized], axis=0
            )
        deltas = jnp.diff(normalized, axis=0)
        return jnp.sum(self._w_contact_rate * deltas**2)

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

    def _consensus_from_twist_exact(self, state: mjx.Data) -> jax.Array:
        """A^r by inverting the plant this repo actually integrates.

        Under the quasi-static plant (`PlanarPushingObject.step`) a moving
        block means the wrench is ON the limit surface: magnitude L,
        direction the twist's. So the inversion is

            w = L * xdot / ||xdot||        (any sliding speed)

        with no speed factor -- the twist's magnitude carries direction
        confidence only, not wrench magnitude. This matches |A^o|, which
        `project_wrench` also pins to L whenever the object block pushes,
        so the consensus compares like with like.

        STICKING. At rest the twist's direction is sensor noise; dividing
        by ||xdot|| there would attach a full cone-sized wrench to it.
        Below `twist_stick_speed` the estimate ramps linearly to zero --
        continuous at the origin, and an honest "no wrench delivered".
        """
        v = state.qvel[self.block_dofs]
        speed = jnp.linalg.norm(v)
        safe = jnp.maximum(speed, self._twist_stick_speed)
        return self.object_model.wrench_limit * (v / safe)

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
        """A^r: what the robot's rollout actually realized (paper eq. 23).

        Expressed in the world frame about the block's pose origin, in N and
        N.m -- the same frame, reference point and units the object block's
        A^o uses, so both ADMM blocks report the identical physical quantity.
        Under `consensus="object_pose"` it is the block's SE(2) pose
        instead, read straight off the state with no estimator at all;
        everything below concerns the two force-level variables only.

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
        if self.consensus == "object_pose":
            # No estimator and no clip: unlike a wrench, the pose is
            # *observed*. `consensus_source` and `_realized_wrench_clip`
            # exist only because the imparted wrench has to be inferred.
            return self.object_state_from_robot(state)
        if self.consensus_source == "contact":
            raw = self._consensus_from_contact(state)
        elif self.consensus_source == "twist_exact":
            raw = self._consensus_from_twist_exact(state)
        else:
            raw = self._consensus_from_twist(state)
        wrench = jnp.clip(
            raw, -self._realized_wrench_clip, self._realized_wrench_clip
        )
        if self.consensus == "contact_point":
            return self._contact_point_from_wrench(state, wrench)
        return wrench

    def _contact_point_from_wrench(
        self, state: mjx.Data, wrench: jax.Array
    ) -> jax.Array:
        """A^r as [p_x, p_y, lambda], from where the pusher is and what it did.

        The point is *known*, not estimated: the pusher's tip is a site on
        the robot, so its world position is read directly and taken into
        the object's body frame. Only lambda needs the force, and it comes
        from the same twist inversion the wrench mode uses -- the normal
        component of the realized force at that point. Deriving it from
        the wrench rather than from MJX's contact forces keeps this
        embodiment-agnostic: an arm's contact appears as J^T f spread
        across six joints, which is why `consensus_source="contact"` is
        restricted to the point pusher.

        Projected to the boundary because the tip has a radius and sits
        just outside the surface (and, between contacts, anywhere at all);
        the object block can only ever propose boundary points, so an
        unprojected A^r would report a standing disagreement that no
        amount of consensus could close.
        """
        pose = self._block_pose(state)
        p_body = self.object_model.footprint.project_to_boundary(
            rotate(-pose[2], self._pusher_pos(state) - pose[:2])
        )
        return wrench_to_contact_point(
            self.object_model.footprint, pose, p_body, wrench
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

    def _tip_height_cost(self, state: mjx.Data) -> jax.Array:
        """Symmetric exponential cost on tip height: not in the paper.

        Simplified 2026-09-07, per Shahid, from a piecewise form
        (quadratic at or above the block's mid-height, exponential
        below it -- see git history, including the re-anchoring work
        that piecewise form needed once anchoring it at mid-height was
        found to cause near-goal stalls) to one exponential, identical
        in shape on both sides of `tip_target_z`. A barrier this steep
        in both directions gives the tip no reason to climb high enough
        to skim the block's top face in the first place, which is what
        made the separate contact-z top-riding barrier redundant enough
        to remove outright (same date).

        Never faded: staying at pushing height is a safety property, not
        shaping that should relax near the goal. Centimetres, not
        metres, so a 1cm miss costs `w_z_tip_exp * exp(1)` -- matches the
        unit the old below-threshold branch always used. Caps its
        exponent at `EXP_ARG_MAX`; see there for why (a NaN softmax, not
        a numerical nicety).
        """
        z_tip = state.site_xpos[self.trace_site_ids[0], 2]
        gap_cm = 100.0 * jnp.abs(z_tip - self.tip_target_z)
        return self.w_z_tip_exp * jnp.exp(
            jnp.minimum(gap_cm**2, EXP_ARG_MAX)
        )

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
        iterations), real Newtons. Historical note: this used to feed a
        `contact_z` diagnostic bar mirroring `_contact_z_cost`, the
        top-riding barrier removed 2026-09-07 along with the rest of the
        contact-z mechanism (see `_tip_height_cost`) -- kept here as a
        raw, still-meaningful logged quantity (contact force between the
        stick and the block's top face) independent of whether anything
        currently costs it.
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

    def shaping_fade(self, pose: jax.Array) -> jax.Array:
        """Scale in [0, 1] on the near-goal-irrelevant terms.

        Linear from 1 at ``||p - p_g|| >= shaping_fade_dist`` (full
        shaping) down to 0 at the goal. ``shaping_fade_dist <= 0``
        disables the whole mechanism (the scale is identically 1).

        Faded (all shape the tip's route, which stops mattering once the
        object is one short correction from the goal): ``approach``,
        ``align``, ``tilt`` (all inside `_ell_r`), and ``effort`` (in
        `running_cost`/`robot_running_cost`, not here).

        Not faded (hard safety/task properties, not shaping):
        `_tip_height_cost` (simplified 2026-09-07 to one symmetric
        exponential, unfaded on both sides -- see there); the object's
        clearance terms, `obstacle_cost`/`support_cost` (a goal near an
        obstacle or the table edge is where driving the *block* into it
        stays wrong, and `robot_running_cost` charges the identical pair
        for the same reason); `robot_contact` and the xarm6-only
        pusher-obstacle hinge; ``ell_o``/``ell_c``.

        The ADMM consensus penalty IS faded, by the paragraph below --
        the one term that is scaled outside this class.

        Also read by the ADMM layer: `ADMM._admm_iteration` scales the
        consensus penalty (`rho` and the duals' step) by this same radius
        for both blocks at once, so inside it the two blocks stop
        negotiating a shared wrench and each optimizes its own objective.

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
        pos_err = jnp.linalg.norm(pose[:2] - self.goal[:2])
        return jnp.where(
            fade_dist > 0.0,
            jnp.clip(pos_err / fade_dist, 0.0, 1.0),
            jnp.asarray(1.0, dtype=pos_err.dtype),
        )

    def _pusher_obstacle_cost(self, pusher_pos: jax.Array) -> jax.Array:
        """Pusher-vs-obstacle clearance hinge -- xarm6 only.

        The same hinge `Obstacles.hinge_cost` gives the object side,
        applied to the pusher's own position instead of the block's
        footprint boundary. The object-side term keeps the *block* out of
        obstacles but does nothing about the pusher itself cutting through
        one on its way to "behind the object": `align` chases that
        position with no obstacle awareness of its own, and
        `_robot_contact_cost` only fires once contact has already
        happened, so nothing in the cost steers the tip around an
        obstacle in advance.

        Scaled relative to the object's own `w_obstacle` so the two stay
        commensurate when that is retuned, with its own (larger) reach --
        the pusher is a point, the block is a footprint, so the pusher
        needs to start turning away sooner in its own units.

        Inert at weight 0 (the default); early return so a run that opts
        out pays nothing.

        Args:
            pusher_pos: The pusher tip's world (x, y).

        Returns:
            The hinge cost, or a zero scalar when opted out.
        """
        if self.pusher_obstacle_weight == 0.0:
            return jnp.zeros(())
        obj = self.object_model
        return self.pusher_obstacle_weight * obj.obstacles.hinge_cost(
            pusher_pos, obj.w_obstacle, self.pusher_obstacle_margin
        )

    def _se2_slack_sq(
        self,
        pose: jax.Array,
        target: jax.Array,
        w_pos: float,
        w_theta: float,
    ) -> jax.Array:
        """`se2_distance_sq` against `target` with the heading slack.

        The slack radius is always measured to the GLOBAL goal (that is
        what `_theta_slack` reads), even under local-goal tracking --
        forgiveness is about how much of the task remains, not about the
        plan's endpoint. Bit-identical to the plain form at
        `theta_slack_max = 0`.
        """
        diff_pos = pose[..., :2] - target[:2]
        diff_theta = jnp.abs(wrap_angle(pose[..., 2] - target[2]))
        excess = jnp.maximum(diff_theta - self._theta_slack(pose), 0.0)
        # `_theta_ramp` (existing flat-path mechanism) applied here too,
        # so a low base q_theta far out still finishes the heading near
        # the goal. Inert at the default q_theta_ramp = 1.
        w_th = w_theta * self._theta_ramp(pose)
        return w_pos * jnp.sum(diff_pos**2, axis=-1) + w_th * excess**2

    def _theta_slack(self, pose: jax.Array) -> jax.Array:
        """How much heading error is forgiven at this distance, in radians.

        Linear in the distance to the goal: nothing is forgiven at
        `theta_slack_near_dist` and below, the full `theta_slack_max` at
        `theta_slack_far_dist` and beyond.

        Why the forgiveness shrinks instead of switching off (the
        contact-implicit baselines switch the heading weight itself, at a
        `cost_switching_threshold`): with a switch the heading is
        unconstrained outside the radius, so the object can arrive at the
        goal turned arbitrarily far and then has to be rotated in place --
        which for a T pushed by one stick necessarily spoils the position it
        just reached, and the run oscillates across the threshold. A
        shrinking allowance instead *bounds* the heading error: anything
        past the allowance is still charged at full weight, so the heading
        is squeezed down as position converges rather than ignored and then
        rescued. It is continuous too, so the sampler never sees a step in
        the cost.

        This is the opposite end of the problem from `_theta_ramp`, which
        raises the heading weight near the goal. Both can be on, but they
        pull against each other in the overlap, so treat that as a
        configuration to measure rather than assume.

        Inert (returns 0.0) at `theta_slack_max <= 0`.

        Args:
            pose: The object's SE(2) pose, shape (..., 3).

        Returns:
            The forgiven heading error in radians, shape (...).
        """
        if self.theta_slack_max <= 0.0:
            return jnp.zeros(())
        span = max(
            self.theta_slack_far_dist - self.theta_slack_near_dist, 1e-9
        )
        pos_err = jnp.linalg.norm(pose[..., :2] - self.goal[:2], axis=-1)
        opened = jnp.clip(
            (pos_err - self.theta_slack_near_dist) / span, 0.0, 1.0
        )
        return self.theta_slack_max * opened

    def _se2_cost(
        self, pose: jax.Array, w_pos: jax.Array, w_theta: jax.Array
    ) -> jax.Array:
        """`se2_distance_sq`, with the forgiven heading error subtracted.

        Same inputs and same output as `se2_distance_sq`, and identical to
        it whenever `_theta_slack` returns zero -- which is the default --
        so swapping the call in changes nothing until a config opts in.

        Only the excess beyond the allowance is charged, at the caller's own
        `w_theta`; the position half is untouched.

        Args:
            pose: The object's SE(2) pose, shape (..., 3).
            w_pos: Weight on the translational error.
            w_theta: Weight on the heading error past the allowance.

        Returns:
            The weighted squared distance, shape (...).
        """
        diff_pos = pose[..., :2] - self.goal[:2]
        diff_theta = jnp.abs(wrap_angle(pose[..., 2] - self.goal[2]))
        excess = jnp.maximum(diff_theta - self._theta_slack(pose), 0.0)
        return w_pos * jnp.sum(diff_pos**2, axis=-1) + w_theta * excess**2

    def _theta_ramp(self, pose: jax.Array) -> jax.Array:
        """Multiplier on q_theta/qf_theta, ramping up as position converges.

        1.0 at ``||p - p_g|| >= theta_ramp_dist``, ``q_theta_ramp`` at the
        goal. Inert (returns 1.0) if ``q_theta_ramp <= 1.0`` or
        ``theta_ramp_dist <= 0``. Deliberately its own radius, not
        `shaping_fade_dist` -- reusing that at a 3.0x multiplier was
        tried first and rejected (contact-shaping fading out over
        exactly the window this was ramping in caused more contact
        instability than it prevented); this is a milder 1.5x reusing
        the same shared radius, which survived multi-seed testing where
        the 3.0x version did not -- see Tasks.md for the full comparison
        if ever needed.

        Flat baseline only. A converged orientation has near-zero cost
        gradient at its own weight, so it does little to resist being
        knocked back out by a much larger position-error gradient in the
        same rollout cost; this keeps its effective weight from
        collapsing as its own error does.

        Args:
            pose: The object's SE(2) pose, (3,).

        Returns:
            A scalar in [1, q_theta_ramp].
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
        """Multiplier on q_pos/q_theta, growing with elapsed control steps.

        Flat baseline only (`running_cost`/`terminal_cost`). The ADMM
        track reads the same two config keys through `time_ramp`/
        `weight_scale` instead, and the two now agree on both halves of
        the formula:

        * ``min(1 + q_ramp_per_step * steps, q_ramp_max)`` -- LINEAR.
          This used to compound, ``(1 + q_ramp_per_step) ** steps``, which
          reached `q_ramp_max` in 646 steps where the ADMM path needs
          4800. Nothing justified the split; it was two mechanisms grown
          separately against the same two keys.
        * ``steps`` comes from the ROLLOUT's start time, not the stepped
          state's own clock: `TaskBase.freeze_cost_time` is True here, so
          `oim.alg_base.SamplingBasedController.eval_rollouts` hands the cost
          functions a state pinned to the time the rollout began. Reading
          it per step made the ramp keep growing inside the horizon, which
          weights step H above step 0 and tilts a plan toward its own
          tail.

        Inert (returns 1.0) if ``q_ramp_per_step <= 0`` or
        ``q_ramp_max <= 1.0``.
        """
        steps = state.time / self.dt
        grown = 1.0 + self.q_ramp_per_step * steps
        return jnp.where(
            (self.q_ramp_per_step > 0.0) & (self.q_ramp_max > 1.0),
            jnp.minimum(grown, self.q_ramp_max),
            jnp.asarray(1.0, dtype=grown.dtype),
        )

    def time_ramp(self, t: jax.Array) -> jax.Array:
        """Multiplier on goal tracking, growing with elapsed control steps.

        ``1 + q_ramp_per_step * (t / dt)``, capped at ``q_ramp_max``;
        inert at 0. Time-based, unlike `shaping_fade` which
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

        fade * (approach + align + tilt) + tip height. See
        `shaping_fade`.

        Approach, align, and tilt all fade, linearly, all reaching
        exactly 1 at shaping_fade_dist and exactly 0 at the goal:
        quadratic shaping costs relax near the goal, since the task is
        essentially done and holding posture that tightly stops
        mattering. `_tip_height_cost` stays unfaded: staying at pushing
        height should never go slack, even near the goal.
        Control effort is faded the same way too, but in
        `running_cost`/`robot_running_cost`, not here -- see those.

        Simplified 2026-09-07, per Shahid: approach is always the
        quadratic form now (the linear option and the wrench-informed
        target mode, 2, are both gone -- see git history for either),
        align always uses the plain "toward where the object must go"
        reference with no rotation-aware lever term, and align is never
        suppressed during top contact. All three removals are downstream
        of `_tip_height_cost` becoming a symmetric exponential the same
        date: none of the mechanisms they replace (top-riding gates,
        rotation-lever aiming to escape a stuck contact) have anything
        left to compensate for once the tip has no reason to climb onto
        the block in the first place.
        """
        if self.approach_mode == 1:
            # Distance to the WALL, not the origin: the xy SDF's outside
            # component, so the term is exactly 0 over the footprint and
            # its minimum is the pushing ring around the walls -- see
            # `approach_mode` in DEFAULT_COSTS for the failure the origin
            # form causes on a non-circular block.
            local = rotate(-pose[2], pusher_pos - pose[:2])
            sd_raw = self.object_model.footprint.sdf(local)
            sd = jnp.maximum(sd_raw, 0.0)
            gap = jnp.clip(sd - self.r0, 0.0, None)
            if self.approach_z:
                # Pull at the contact POSE, not just its xy ring: fold the
                # height error into the same distance -- but only OUTSIDE
                # the footprint. Over the block a mid-height target could
                # only press through the top face, which `_tip_height_cost`
                # already prices on its own.
                z_tip = state.site_xpos[self.trace_site_ids[0], 2]
                dz = jnp.where(sd_raw > 0.0, z_tip - self.tip_target_z, 0.0)
                gap = jnp.sqrt(gap**2 + dz**2 + 1e-18)
            approach = self.w_approach * gap**2
        else:
            d_ee = jnp.sum((pusher_pos - pose[:2]) ** 2)
            approach = self.w_approach * jnp.clip(
                d_ee - self.r0**2, 0.0, None
            )

        to_ref = obj_ref[:2] - pose[:2]
        to_object = pose[:2] - pusher_pos
        cos_angle = jnp.sum(to_object * to_ref) / (
            jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
        )
        align = self.w_align * jnp.clip(self.gamma0 - cos_angle, 0.0, None)

        tilt = self.w_tilt * self._tilt(state)
        tip_height = self._tip_height_cost(state)
        fade = self.shaping_fade(pose)
        return fade * (approach + align + tilt) + tip_height

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
            local_goal: The point of the object block's plan to aim at,
                already chosen by `local_goal_from_plan`, or None.

        Returns:
            The SE(2) pose to track, (3,).
        """
        if not self.use_local_goal or local_goal is None:
            return self.goal
        # 1 outside the fade radius, < 1 inside it.
        return jnp.where(self.shaping_fade(pose) < 1.0, self.goal, local_goal)

    def local_goal_from_plan(
        self, plan: jax.Array, pose: jax.Array
    ) -> jax.Array:
        """Pure pursuit along the object block's plan: the first planned
        pose at least `local_goal_lookahead` metres from where the object
        is now.

        The endpoint x^{o*}_H (the base class's answer, and what this
        returns at `local_goal_lookahead = 0`) says only where the plan
        finishes, so a plan that routes around an obstacle and one that
        drives straight through it are scored identically as long as they
        end together. A carrot a fixed distance ahead scores the ROUTE:
        the robot is pulled along the plan's own shape.

        No stored index and nothing to advance by hand -- the carrot is
        re-picked from `pose` at every rollout step, so it slides forward
        as the object closes on it, both within one rollout (the object
        moves along the horizon) and across control steps (the object
        moves in the world). Which is also what keeps it traceable.

        Degenerate case, and it is not rare: when NO planned pose is
        `local_goal_lookahead` away -- an object block planning to hold
        still under breakaway, whose whole 16-step plan once spanned
        0.071 m while the object sat 0.727 m from the goal -- `argmax`
        over an all-False mask returns 0, which would aim the robot at
        the object's own current pose and zero the tracking gradient
        exactly when it most needs to push. Falls back to the endpoint
        there, the same answer `local_goal_lookahead = 0` gives.

        Args:
            plan: The object block's nominal trajectory, (H, 3).
            pose: The object's SE(2) pose at this step, (3,).

        Returns:
            The SE(2) pose to aim at, (3,).
        """
        if not self.use_local_goal or self.local_goal_lookahead <= 0.0:
            return plan[-1]
        d = jnp.linalg.norm(plan[:, :2] - pose[:2], axis=1)
        # Forward of the closest planned pose, not forward of index 0:
        # distance alone is symmetric, so the first entry far enough away
        # can be the part of the plan the object has already covered --
        # from the middle of a 0.45 m plan, index 0 is 0.20 m BEHIND and
        # would win outright, dragging the robot back down the route.
        ahead = jnp.arange(plan.shape[0]) >= jnp.argmin(d)
        far = ahead & (d >= self.local_goal_lookahead)
        return jnp.where(jnp.any(far), plan[jnp.argmax(far)], plan[-1])

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
        + robot_contact + pusher_obstacle``.

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
        # Heading slack applied here too (inert at theta_slack_max = 0):
        # without it the robot block prices theta from zero, and a
        # polished theta (0.2 deg at pe 0.10, runs 17:39/17:54) makes
        # every predicted contact a pure loss -- the endgame deadlock.
        ell_o = weight_scale * self._se2_slack_sq(
            pose, target, self.q_pos, self.q_theta
        )
        # Shaping reference (approach target, align): the object plan's
        # ENDPOINT, the same `local_goal` the ADMM layer already hands in,
        # not the plan's k-th point. Early in the horizon plan[k] sits
        # millimetres from the current pose, so the demanded twist's
        # direction is noise -- measured on 141749: the k=0 landing target
        # jumped 20 mm per solve (p90 42 mm) while the endpoint-based one
        # moved 3 mm. The flat path already uses a fixed reference
        # (`self.goal` in `running_cost`); this makes the ADMM block do the
        # same with its own plan. Goal tracking (`ell_o`) is untouched.
        ref_r = obj_ref_t if local_goal is None else local_goal
        ell_r = self._ell_r(state, pose, pusher_pos, ref_r)
        # The OBJECT's proximity to obstacles and the table edge, scored
        # on the pose THIS rollout produced. Same function `running_cost`
        # calls (`PlanarPushingObject.obstacle_cost`/`support_cost`), not
        # a second reimplementation -- 2026-09-07, per Shahid, so flat and
        # ADMM cannot silently price the object block's own task
        # differently. Was previously two direct, lower-level calls here
        # (`Obstacles.exp_cost` for the obstacle term, bypassing
        # `obstacle_cost` entirely) that would have kept using the OLD
        # exponential-everywhere form even after `obstacle_cost` itself
        # was redefined, exactly the kind of drift this unification is
        # meant to close off. Never faded, matching the object block -- a
        # goal beside an obstacle is exactly where routing the block into
        # it stays wrong.
        obj = self.object_model
        obstacle = obj.obstacle_cost(pose) + obj.support_cost(pose)
        # Robot-vs-obstacle *contact*, a different quantity: the force the
        # robot's own body imparts, not the block's clearance.
        robot_contact = self._robot_contact_cost(state)
        # Preventive half of the same concern, the hinge `running_cost`
        # already charges the flat baseline: keep the TIP itself clear of
        # obstacles (and the base) before it gets there. Was missing from
        # this block only, so an ADMM run had nothing between the tip and
        # a cube until the contact force existed (09-06 201407/202157/
        # 205305 stopped with the tip on a cube). Inert at weight 0.
        pusher_obstacle = self._pusher_obstacle_cost(pusher_pos)
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
        return (
            ell_o + ell_r + obstacle + effort + robot_contact
            + pusher_obstacle
        )

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
        return weight_scale * self._se2_slack_sq(
            pose, target, self.qf_pos, self.qf_theta
        )
