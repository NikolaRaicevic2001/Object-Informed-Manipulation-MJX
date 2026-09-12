"""Drawing a hardware run: the framing, the overlay and the recorder.

Shared by the viewer (`--live`) and the offscreen recorder (`--record`),
which draw the same scene from the same `mj_data_cpu`. Every entry point
here is a diagnostic -- the loops call them after the plan has gone out, so
a slow frame delays nothing the robot is waiting for.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, List, Optional

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from oim.runtime.overlay import BlockTrace, PlanOverlay, traces_for
from oim.runtime.video import OffscreenRecorder

# --live's refresh rate on `_run_overlapped`'s display thread, between
# solves. Not tied to control_rate: this is how often a human can usefully
# perceive an update, not a control-loop constraint like control_rate is.
_DISPLAY_HZ = 30.0

# --live's default framing, used when no --camera picks a model camera.
# `mjv_defaultFreeCamera` frames the whole MODEL, which on the real scenes
# means the 0.91 m of table leg and a wide margin of floor -- the table top,
# the only part anything happens on, ends up a small patch in the middle.
#
# Azimuth 180 stands the camera off the +x end of the table looking back
# along -x, which puts screen-right on +y: the table's long 1.523 m axis
# lies across the width, and the arm base (world origin) is at the far
# edge. That is the same standpoint the scenes' own "front" camera uses.
_VIEW_AZIMUTH = 180.0

_VIEW_ELEVATION = -27.0

# Raises the aim point off the table top so the frame holds the arm as well
# as the tabletop. Measured: at 0.10 the union of the table and every
# non-floor geom centres on the horizon (NDC 0.00) and spans y[-0.65,+0.65],
# so nothing is clipped and neither half of the frame is left empty. Aiming
# at the tabletop itself (0.0) rides the content high; past ~0.2 the table
# slides into the bottom third.
_VIEW_LOOKAT_Z = 0.10

# `_free_camera_distance` places the table's FAR corners on the frame edge;
# the NEAR corners, being closer, project 4.5% wider. Measured against
# mjv_updateScene and constant -- it does not move with aspect, elevation
# or lookat height.
_VIEW_NEAR_CORNER = 1.045

# Only used when the viewer has not sized its window yet (`Handle.viewport`
# reads back 0). Any real window replaces this on the first frame.
_VIEW_FALLBACK_ASPECT = 1.5

# Stand-in for `vis_lock` on the paths that have no second thread touching
# `mj_data_cpu` (the serial loop, and any run with neither --record nor
# --live). `contextlib.nullcontext` semantics, spelled out so
# `_visualize_step` needs no branch of its own.
_NULL_LOCK = contextlib.nullcontext()

def _free_camera_distance(
    model: mujoco.MjModel, aspect: float, elevation: float, lookat_z: float,
    half_span: float, half_depth: float,
) -> float:
    """How far back to stand so `half_span` just fills the frame width.

    At distance `d` the frustum is `d * tan(fovy/2) * aspect` wide, and the
    table's near edge sits `half_depth * cos(elevation)` closer than the aim
    point while raising the aim by `lookat_z` pulls it
    `lookat_z * sin(elevation)` nearer still -- so both shift the distance
    the width has to be solved at, not just the framing.
    """
    tan_h = math.tan(math.radians(model.vis.global_.fovy / 2.0)) * aspect
    tilt = math.radians(abs(elevation))
    return (
        half_span / tan_h
        - math.sin(tilt) * lookat_z
        + math.cos(tilt) * half_depth
    ) * _VIEW_NEAR_CORNER

def _frame_table(
    model: mujoco.MjModel, cam: Any, aspect: float,
    azimuth: float, elevation: float, distance: Optional[float],
) -> None:
    """Aim a free camera at the table, filling the width with its long axis.

    Falls back to `mjv_defaultFreeCamera` for any model without a `table`
    geom, so this stays safe for scenes it was not measured on.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if gid < 0:
        mujoco.mjv_defaultFreeCamera(model, cam)
        return
    pos = model.geom_pos[gid]      # the table is a worldbody geom, so this
    size = model.geom_size[gid]    # is already in world coordinates
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = [float(pos[0]), float(pos[1]), _VIEW_LOOKAT_Z]
    cam.distance = (
        distance if distance is not None else
        _free_camera_distance(
            model, aspect, elevation, _VIEW_LOOKAT_Z,
            half_span=float(size[1]), half_depth=float(size[0]),
        )
    )

def _visualize_step(
    vis_model: mujoco.MjModel,
    mjx_data: mjx.Data,
    mj_data_cpu: Optional[mujoco.MjData],
    recorder: Optional[OffscreenRecorder],
    overlay: Optional[PlanOverlay],
    viewer: Any,
    overlay_base: Optional[int],
    rollouts: Any,
    params: Any,
    admm: bool,
    show_samples: bool,
    show_optimal: bool,
    obj_plan: Optional[np.ndarray] = None,
    rob_plan: Optional[np.ndarray] = None,
    robot_trace: Optional[np.ndarray] = None,
    sync_viewer: bool = True,
    vis_lock: Any = None,
) -> List[BlockTrace]:
    """Push one frame to whichever of `recorder`/`viewer` are active.

    Real has no standing CPU `mujoco.MjData` the way the sim worlds do --
    the whole state is `mjx_data` -- so `mj_data_cpu` is written from it
    here, once, and shared by both destinations: the offscreen recorder
    (`OffscreenRecorder.capture`, exactly what every sim world already
    uses) and the live passive viewer (`viewer.sync`). A no-op if neither
    is set, so a run with neither `--record` nor `--live` pays nothing
    beyond the `is None` checks.

    Args:
        vis_model: The CPU model `mj_data_cpu` mirrors -- the shared
            deepcopy `run_real` builds, never `task.mj_model` itself.
        mjx_data: This control step's assembled state.
        mj_data_cpu: Reused every call; `None` iff both destinations are.
        recorder: The mp4 recorder, or `None`.
        overlay: The shared candidate/chosen-trajectory drawer, or `None`
            if neither `show_samples` nor `show_optimal` was asked for.
        viewer: A `mujoco.viewer` passive-viewer handle, or `None`.
        overlay_base: Fixed geom slot for the viewer's persistent scene
            (see `PlanOverlay.draw`); unused when drawing into the
            recorder's own scene, which is rebuilt every frame.
        rollouts: This step's sampled robot rollouts, for `show_samples`.
        params: What `optimize` just returned, for the object block's
            sampled population (ADMM only).
        admm: Whether `obj_plan`/`rob_plan` are meaningful -- a flat
            controller has no object block to draw one for.
        show_samples, show_optimal: As on `run_real`.
        obj_plan, rob_plan: The two blocks' predicted object trajectories
            (ADMM only), from `ADMM.nominal_plans`.
        robot_trace: The chosen end-effector path, from
            `ADMM.nominal_plans` or the flat controller's `nominal_trace`.
        sync_viewer: Draw and sync `viewer` here. `_run_overlapped` passes
            `False`: it still needs `viewer is not None` to trigger trace
            computation below, but its own display thread owns
            `viewer.sync()` exclusively (see there for why one caller).

    Returns:
        The traces just drawn (possibly `[]`) -- `_run_overlapped` reuses
        these for its own display thread, which redraws them at a higher
        rate than one per solve without recomputing `traces_for` itself.
    """
    if recorder is None and viewer is None:
        return []
    # Held across the write AND the mj_forward: the display thread's
    # viewer.sync() copies this same MjData, and a copy that lands
    # mid-mj_forward is the "stack is in use" abort.
    with vis_lock if vis_lock is not None else _NULL_LOCK:
        mj_data_cpu.qpos[:] = np.asarray(mjx_data.qpos)
        mj_data_cpu.qvel[:] = np.asarray(mjx_data.qvel)
        mj_data_cpu.mocap_pos[:] = np.asarray(mjx_data.mocap_pos)
        mj_data_cpu.mocap_quat[:] = np.asarray(mjx_data.mocap_quat)
        mj_data_cpu.time = float(mjx_data.time)
        mujoco.mj_forward(vis_model, mj_data_cpu)

    traces = []
    if overlay is not None:
        traces = traces_for(
            robot_chosen=robot_trace if show_optimal else None,
            object_chosen=obj_plan if (show_optimal and admm) else None,
            robot_object_chosen=rob_plan if (show_optimal and admm) else None,
            robot_samples=(
                np.asarray(rollouts.trace_sites)[:, :, 0, :]
                if show_samples
                else None
            ),
            object_samples=(
                np.asarray(params.object_samples)
                if show_samples and admm
                and getattr(params, "object_samples", None) is not None
                else None
            ),
        )
    with vis_lock if vis_lock is not None else _NULL_LOCK:
        if recorder is not None:
            recorder.set_plans(traces)
            recorder.capture(mj_data_cpu)
        if viewer is not None and sync_viewer:
            if overlay is not None:
                overlay.draw(viewer.user_scn, traces, base=overlay_base)
            viewer.sync()
    return traces
