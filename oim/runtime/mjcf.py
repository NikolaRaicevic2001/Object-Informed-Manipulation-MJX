"""MuJoCo scene inspection and setup, shared by every world with a model.

None of this is specific to a controller or to the 3D push-T world: it is
what any run needs to find a camera, address a mocap marker, place the
block, or build a separate execution model at a finer timestep than the
planner rolls out with. It lived in `oim.worlds.sim3d.build` while the 3D world
was the only caller; `oim.worlds.object_only.plant` was the second, and
reached across a package boundary to get it.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from oim.tasks.pusht import PushT

# Bodies that are not the object or its scenery. An object-level study --
# `oim.worlds.object_only`, or the object block of ADMM predicting with
# MJX -- has no robot in it, but the scene it borrows still contains one, so
# the arm is taken out of collision rather than left standing in the
# workspace as an obstacle the analytic model cannot see. Prefix match: the
# xArm6 contributes `xarm6_link*` and `xarm6_stick`.
ROBOT_BODY_PREFIXES = ("xarm6", "pusher")

# The support surface, taken out of collision for the same reason but with
# a sharper justification: the block's support friction is ALREADY modelled,
# as the `frictionloss` on its two slides and its hinge, set in the MJCF to
# exactly `mu*m*g` and `c*r*mu*m*g` precisely so the simulated block obeys
# the analytic limit surface (see the comment in `tee.xml`). Leaving the
# block resting on the table counts that friction a second time, through the
# contact.
#
# Measured on shelf_gap, force needed to break the block loose:
#
#     analytic limit surface           7.85 N   (= mu*m*g, by construction)
#     MuJoCo, support excluded         7.87 N   <- agrees to 0.3%
#     MuJoCo, block resting on table  11.16 N   <- 1.42x, double counted
#
# The root cause is the block's locked vertical DoF: the solver's
# normal force is then unbounded in a direction the block cannot move, so
# it is the contact constraint rather than Coulomb friction from the
# block's weight (the 11.16 N is the same with gravity off). The
# xarm6_pusht_tabletop scenes fix that at the source -- their block has the
# missing DoF and a condim="1" block<->table <pair>.
#
# RE-MEASURED 2026-08-19, and the exclusion is now a no-op EVERYWHERE, so
# `oim.runtime.object_mjx` no longer applies it (see that module's
# docstring). Breakaway, support kept (+gravity) vs excluded (g=0):
#
#     open_table (T_zs + condim=1 pair)   7.90 N  vs  7.90 N
#     clutter    (T_zs locked)            7.90 N  vs  7.90 N
#
# (Both on the 2.0 kg / mu 0.4 block those scenes carried then. open_table
# was re-scaled onto the lab block on 2026-08-25 and its budget is now
# mu*m*g = 0.2943 N; the CONCLUSION does not depend on the magnitude.)
#
# The locked-DoF scenes come out clean not because they were fixed but
# because their block never reaches its support: `pusht_clutter`'s hovers
# 10 mm above the floor plane and `xarm6_pusht_tabletop_real`'s 0.2 mm,
# both with zero block/support contacts at the start pose and no DoF that
# could create one. That is incidental geometry rather than a guarantee --
# a new locked-DoF scene whose block DID rest on its support would bring
# the 1.42x back, which is why this exclusion still exists as an option.
# AND SINCE 2026-08-19 THE ARGUMENT INVERTS FOR THE TABLETOP SCENES. Their
# block<->table pair is no longer `condim="1"`: it carries real Coulomb
# friction at the lab table's mu = 0.3 and their joints carry no
# `frictionloss` at all, so
# the support contact IS the object's friction there rather than a
# double-count of it. Excluding it does not remove a duplicate, it removes
# the friction outright and leaves the block sliding free. Both callers
# (`object_mjx_model`, `MujocoPlant`) therefore default to keeping it; this
# constant is now for the scenes that still lump friction into their joints
# (`clutter`, `xarm6_pusht_clutter`), where excluding the support is still
# exactly right. `xarm6_pusht_tabletop_real` used to be in that list and is
# NOT any more: it got a table and a block<->table `condim="3"` pair of its
# own, so its support contact IS its friction too.
SUPPORT_GEOM_NAMES = ("table", "floor", "ground")

# Joint config (degrees) putting the xArm6's stick tip near the clutter
# scene's block start; found via
# oim/models/xarm6_pusht_clutter/verify_reach.py. Fallback only, and used
# by `clutter` alone -- a scene with its own workspace (the tabletop
# family) defines a "start" keyframe instead, not another entry here.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]

# The point-mass pusher's own start, for the original `clutter` scene only
# (see `point_start_qpos`): block at the origin, pusher just below it. Two
# slides and a hinge for the block, then the pusher's x/y.
POINT_START_QPOS = [0.0, 0.0, 0.0, -0.05, -0.06]


def point_start_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    """Initial qpos for a point-robot scene.

    Reads the model's own "start" keyframe if it defines one -- the five
    tabletop scenes' point variants each do, matching their own workspace
    and obstacle layout -- falling back to `POINT_START_QPOS` otherwise
    (`clutter`, which has none). Mirrors `xarm6_start_qpos` exactly;
    before this, every point-robot scene silently reused
    `POINT_START_QPOS` regardless of what its own MJCF declared, so five
    different scenes all started from the same clutter-specific offset.

    Args:
        mj_model: The compiled scene.

    Returns:
        A full qpos vector.
    """
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "start")
    if key_id >= 0:
        return np.array(mj_model.key_qpos[key_id])
    return np.array(POINT_START_QPOS)


def xarm6_start_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    """Initial qpos for an xArm6 scene.

    Reads the model's own "start" keyframe if it defines one -- every scene
    besides the original clutter layout does, since each has a different
    workspace -- falling back to `XARM6_START_QPOS_DEG` otherwise. Generic
    over `mj_model.nq`, so it also covers the block's own qpos entries.

    Args:
        mj_model: The compiled scene.

    Returns:
        A full qpos vector.
    """
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "start")
    if key_id >= 0:
        return np.array(mj_model.key_qpos[key_id])
    qpos = np.zeros(mj_model.nq)
    qpos[:5] = np.radians(XARM6_START_QPOS_DEG)
    return qpos


def named_camera(
    mj_model: mujoco.MjModel, name: str = "front"
) -> Optional[str]:
    """`name` if the model defines a camera by that name, else `None`.

    `None` means the default free camera framing the whole scene. A scene
    the free camera frames badly (the tabletop family, viewed from the side
    by default) defines its own fixed camera instead of a special case in
    the caller.

    Args:
        mj_model: The compiled scene.
        name: Camera to look for.

    Returns:
        The camera name, or None.
    """
    if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
        return name
    return None


def mocap_id(mj_model: mujoco.MjModel, name: str) -> int:
    """The mocap index of body `name`, or -1 if the scene has no such body.

    Every marker write goes through this rather than a literal index.
    `goal` was mocap 0 only because it was the one mocap body in every
    scene; adding `local_goal` makes it a two-entry table ordered by
    declaration. The two happen to land at 0 and 1 in all seven scenes
    today, which is exactly the kind of coincidence the old literal relied
    on -- a scene that declares its marker before its goal would silently
    swap them.

    Args:
        mj_model: The compiled scene.
        name: Body name.

    Returns:
        The index into `mocap_pos`/`mocap_quat`, or -1 when the body is
        absent (scenes predating the marker) or is not a mocap body.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return -1
    return int(mj_model.body_mocapid[body_id])


def hide_body_geoms(mj_model: mujoco.MjModel, name: str) -> None:
    """Make every geom of body `name` fully transparent, in place.

    For a marker the run will not be driving. Editing alpha rather than
    moving the body out of frame keeps the fix local to rendering: the body
    stays where it is, so nothing that reads a position changes meaning.

    Args:
        mj_model: The model to edit -- pass the *execution* model, which is
            a deepcopy, not the task's own.
        name: Body name. Absent is a no-op.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return
    start = int(mj_model.body_geomadr[body_id])
    for geom in range(start, start + int(mj_model.body_geomnum[body_id])):
        mj_model.geom_rgba[geom][3] = 0.0


def set_mocap_se2(
    mj_data: mujoco.MjData, index: int, pose: Sequence[float]
) -> None:
    """Place mocap body `index` at an SE(2) pose, keeping its height.

    The markers are flat objects lying on the table, so only (x, y, yaw)
    ever move -- z stays at whatever the MJCF put the body at, which is the
    block's own resting height and differs per scene.

    Args:
        mj_data: The state to write into.
        index: A `mocap_id` result; negative is a no-op, so a caller need
            not branch on whether the scene has the marker.
        pose: World-frame `[x, y, theta]`.
    """
    if index < 0:
        return
    pose = np.asarray(pose, dtype=float)
    mj_data.mocap_pos[index][:2] = pose[:2]
    half = 0.5 * float(pose[2])
    mj_data.mocap_quat[index] = [np.cos(half), 0.0, 0.0, np.sin(half)]


def execution_model(
    task: PushT,
    robot: str,
    cfg: Dict[str, Any],
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """A separate, finer MuJoCo model for *executing* the plan.

    The planner rolls out at `planning_dt`; execution steps at
    `exec_timestep` with more solver iterations, so the closed loop is not
    graded against the same coarse integration it planned with.

    Args:
        task: The task whose `mj_model` to copy.
        robot: Embodiment, selecting the start pose.
        cfg: The config's `world3d` block.
        start: Object start pose, world-frame SE(2) `[x, y, theta]`,
            written straight into the block's slide/hinge `qpos`. `None`
            keeps the MJCF keyframe's own.
        goal: Goal pose, moving the `goal` mocap marker to match what the
            task's costs aim at. `None` keeps the MJCF's own. The costs
            themselves come from `PushT(goal=...)`; this is the marker and
            the goal-relative sensors that read off it.

    Returns:
        The execution model and its data, at the start pose.
    """
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = cfg["exec_timestep"]
    mj_model.opt.iterations = cfg["exec_iterations"]
    mj_model.opt.ls_iterations = cfg["exec_ls_iterations"]
    mj_data = mujoco.MjData(mj_model)
    if robot == "xarm6":
        mj_data.qpos[:] = xarm6_start_qpos(mj_model)
    else:
        mj_data.qpos[:] = point_start_qpos(mj_model)
    if start is not None:
        # The block's joints are two slides and a hinge anchored at the
        # origin, so qpos *is* the world pose -- the same invariant
        # `PushT._block_pose` relies on when it reads qpos back.
        mj_data.qpos[np.asarray(task.block_qpos_indices)] = np.asarray(
            start, dtype=float
        )
    if goal is not None:
        set_mocap_se2(mj_data, mocap_id(mj_model, "goal"), goal)
    # Park the local-goal marker on the block's start pose, so it is not
    # sitting at the world origin for the one frame before the first plan
    # exists. Absent in scenes without the marker, where this is a no-op.
    set_mocap_se2(
        mj_data,
        mocap_id(mj_model, "local_goal"),
        mj_data.qpos[np.asarray(task.block_qpos_indices)],
    )
    # Populate xpos/site_xpos/sensordata for whatever reads them before the
    # first step -- the initial log entry, and the goal-relative sensors.
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


def disable_collisions(
    mj_model: mujoco.MjModel,
    names: Sequence[str],
    geom: bool = False,
) -> None:
    """Take matching geoms out of collision, in place.

    Zeroes `contype`/`conaffinity` rather than deleting geoms, so everything
    still renders and every id in the model stays put -- ids that `PushT`
    has already cached against this scene.

    Explicit `<contact><pair>` elements bypass that filter, so any pair
    naming a disabled geom is deactivated too, by pushing its `gap` past
    any reachable penetration. Without this the tabletop scenes' frictionless
    block<->table pair survives into a model built to have no support, and
    with gravity zeroed it drives the block off the table surface.

    Shared by the two object-level models of the same scene, the CPU plant
    (`oim.worlds.object_only.plant`) and the MJX prediction backend
    (`oim.runtime.object_mjx`), so the two cannot end up simulating
    different worlds while claiming to differ only in integrator.

    Args:
        mj_model: The model to edit. Always a copy, never a task's own.
        names: Name prefixes to match.
        geom: Match the geom's own name rather than its body's. The support
            surface is a bare worldbody geom and so has no body name of its
            own; the robot's links are bodies.
    """
    disabled = set()
    for geom_id in range(mj_model.ngeom):
        name = (
            mj_model.geom(geom_id).name
            if geom
            else mj_model.body(mj_model.geom_bodyid[geom_id]).name
        )
        if name.startswith(tuple(names)):
            mj_model.geom_contype[geom_id] = 0
            mj_model.geom_conaffinity[geom_id] = 0
            disabled.add(geom_id)
    # `gap`, not `margin`: MJX Warp rejects a non-zero margin on box-box
    # pairs (MULTICCD). A contact enters the solver at dist < margin - gap,
    # so 1 m of gap is unreachable.
    for pair_id in range(mj_model.npair):
        if {
            int(mj_model.pair_geom1[pair_id]),
            int(mj_model.pair_geom2[pair_id]),
        } & disabled:
            mj_model.pair_gap[pair_id] = 1.0
