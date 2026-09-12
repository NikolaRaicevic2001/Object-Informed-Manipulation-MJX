"""Moving the planner's obstacles to where the cameras say they are.

The MJCF places obstacles where they were measured once. A real table gets
bumped, so a run can start from a fresh ArUco/TF reading and shift both the
planner's obstacle field and the rendered scene to match, without editing
the model file. `live_real` goes further and generates its whole MJCF from
a calibration -- see `oim.worlds.real3d.live_scene`.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from scipy.spatial.transform import Rotation

from oim.objects import Box
from oim.runtime.mjcf import mocap_id
from oim.tasks.pusht import PushT

_OBSTACLE_NAMES = ("obs_1", "obs_2", "obs_3")

def _sample_obstacle_tf_live(
    interface: Any,
    names: Tuple[str, ...] = _OBSTACLE_NAMES,
    base_frame: str = "xarm_device",
    window_s: float = 1.5,
) -> Dict[str, Tuple[float, float, float]]:
    """The same xarm_device -> obs_N_center averaging
    Fork_FoundationPose/calibrate_obstacles.py does, but read directly off
    `interface`'s own already-running TF listener instead of a separate
    script + JSON file handoff.

    `Ros2Interface.__init__` already builds a `tf2_ros.Buffer` +
    `TransformListener` and spins them on a background thread before
    `run_real` ever reaches this call (see interface.py) -- the same TF
    tree the object pose is already read from. As long as
    `aruco_obstacle_node.py` + `aruco_tf_broadcaster.py` are running on the
    perception laptop and on the same ROS 2 domain (`setup_dds_env.sh` on
    both machines), `obs_N_center` is already arriving in that buffer for
    free; this just samples it. No new subscriptions, no rclpy.init(), no
    file ever touches disk.
    """
    import rclpy  # noqa: PLC0415
    from tf2_ros import (  # noqa: PLC0415
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )

    buffer = interface._tf_buffer  # noqa: SLF001 -- same listener object_pose uses
    result: Dict[str, Tuple[float, float, float]] = {}
    for name in names:
        target = f"{name}_center"
        samples = []
        t_end = time.monotonic() + window_s
        while time.monotonic() < t_end:
            try:
                tf = buffer.lookup_transform(base_frame, target, rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
                samples.append((t.x, t.y, yaw))
            except (LookupException, ConnectivityException, ExtrapolationException):
                pass
            time.sleep(0.02)
        if not samples:
            continue
        arr = np.asarray(samples)
        mean_xy = arr[:, :2].mean(axis=0)
        # Circular mean -- a plain average breaks across the +-pi wrap.
        mean_yaw = np.arctan2(np.sin(arr[:, 2]).mean(), np.cos(arr[:, 2]).mean())
        result[name] = (float(mean_xy[0]), float(mean_xy[1]), float(mean_yaw))
    return result

def _load_obstacle_calibration(
    source: str, interface: Any = None, verbose: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """`source` is either a path to a calibrate_obstacles.py JSON file, or
    the literal string `"live"` -- sample the obstacles' current pose
    directly off `interface`'s own TF connection instead (see
    `_sample_obstacle_tf_live`). `"live"` requires a real `Ros2Interface`
    (the mock has no TF tree to sample -- pass a JSON path there instead,
    which is how a calibrated layout is rehearsed before a hardware run).
    """
    if source != "live":
        with open(source) as f:
            return json.load(f)

    if interface is None or not hasattr(interface, "_tf_buffer"):
        raise ValueError(
            "--obstacle-calibration live requires a real Ros2Interface "
            "(the mock has no TF tree to sample) -- pass a JSON path from "
            "Fork_FoundationPose/calibrate_obstacles.py instead"
        )
    if verbose:
        print("[calibration] sampling obs_1/2/3 live over TF "
              "(xarm_device -> obs_N_center)...")
    calibration = _sample_obstacle_tf_live(interface)
    missing = set(_OBSTACLE_NAMES) - calibration.keys()
    if missing and verbose:
        print(f"[calibration] no live TF for {sorted(missing)} -- is "
              f"aruco_obstacle_node.py running on the perception laptop, "
              f"and are those tags in view? Continuing with the "
              f"{len(calibration)}/3 obstacles that did resolve.")
    return calibration

def apply_obstacle_calibration(
    task: PushT,
    base_data: mjx.Data,
    calibration: Dict[str, Tuple[float, float, float]],
    verbose: bool = True,
) -> mjx.Data:
    """Overwrite base_data's mocap_pos/mocap_quat from a loaded
    calibration (see `_load_obstacle_calibration` -- a live TF sample).

    Called once, before the control loop starts. `_assemble_state`
    builds every step's mjx_data via `base_data.replace(qpos=...,
    qvel=..., time=...)` -- it never touches mocap_pos/mocap_quat, so
    whatever is written here reaches every step of the run, both
    planning and rendering, for free. Mock runs the identical path (no
    camera involved): this just replaces the MJCF's own hardcoded
    default with a measured one.

    An obstacle with no matching mocap body in this scene (or vice
    versa -- calibration ran against a different scene, or a tag wasn't
    in view) is skipped, not fatal: it simply keeps the MJCF default,
    same as if this were never called.
    """
    # np.asarray on a JAX array can hand back a read-only view -- copy=True
    # forces a writable buffer.
    mocap_pos = np.array(base_data.mocap_pos, copy=True)
    mocap_quat = np.array(base_data.mocap_quat, copy=True)
    for name, (x, y, yaw) in calibration.items():
        idx = mocap_id(task.mj_model, name)
        if idx < 0:
            if verbose:
                print(f"[calibration] '{name}' has no mocap body in this "
                      f"scene -- skipped, keeping the MJCF default")
            continue
        mocap_pos[idx, 0] = x
        mocap_pos[idx, 1] = y
        # z untouched: calibration is SE(2) (see calibrate_obstacles.py),
        # same as every other planar pose in this task.
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        mocap_quat[idx] = [qw, qx, qy, qz]  # MuJoCo's wxyz, not scipy's xyzw
        if verbose:
            print(f"[calibration] {name}: mocap[{idx}] <- "
                  f"pos=({x:.4f}, {y:.4f})  yaw={np.degrees(yaw):.1f}deg")

    return base_data.replace(
        mocap_pos=jnp.asarray(mocap_pos), mocap_quat=jnp.asarray(mocap_quat)
    )

def apply_obstacle_calibration_to_planner(
    task: PushT,
    calibration: Dict[str, Tuple[float, float, float]],
    verbose: bool = True,
) -> None:
    """Mutate `task.object_model.obstacles.shapes` in place from the same
    loaded calibration `apply_obstacle_calibration` applies to `base_data`
    (see `_load_obstacle_calibration`).

    That function only overwrites `base_data.mocap_pos/mocap_quat`, which
    fixes collision physics but not this -- the planner's own analytic
    avoidance cost (`obstacle_cost`) reads `task.object_model.obstacles`
    directly, a *separate* obstacle list (`oim/utils/scenes.py`) that MJX
    collisions never touch. Must be called before `jit_optimize`'s first
    call: `task` is closed over (not passed as a traced argument), so any
    mutation after the first trace has no effect on the compiled planner.
    """
    shapes = task.object_model.obstacles.shapes
    for name, (x, y, yaw) in calibration.items():
        if not name.startswith("obs_"):
            continue
        idx = int(name.removeprefix("obs_")) - 1
        if not (0 <= idx < len(shapes)) or not isinstance(shapes[idx], Box):
            if verbose:
                print(f"[calibration] '{name}' has no matching planner Box "
                      f"-- skipped, keeping the scene default")
            continue
        old = shapes[idx]
        shapes[idx] = Box(center=[x, y], half_extents=old.half_extents, angle=yaw)
        if verbose:
            print(f"[calibration] {name}: planner obstacle[{idx}] <- "
                  f"center=({x:.4f}, {y:.4f})  angle={np.degrees(yaw):.1f}deg")
