"""The hardware I/O boundary for the real-robot push-T loop.

`run_real` never talks to ROS, the arm SDK or the camera directly. It only
calls a `RobotWorldInterface`, which hides *where* the state comes from and
*where* the control goes. Two implementations are provided:

- `MujocoMockInterface`: steps a MuJoCo simulation internally, so the whole
  `run_real` loop can be validated on a laptop with no hardware. This is the
  "feed it sim state, confirm it produces commands" milestone.
- `Ros2Interface`: the real one -- subscribes to `/joint_states`, reads the
  object's pose from a FoundationPose TF frame, and publishes joint velocity
  commands. Topics/frames/joint names default to the OI-MPPI lab setup; the
  few values still needing a check on the robot are marked `TODO(lab)`.

Both return the same `WorldState`, so `run_real` is identical for sim-mock
and hardware.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mujoco
import numpy as np

# MJX model joint names, as declared in models/xarm6/xarm6.xml and the block
# scene. The wrist-roll joint6 is welded/fixed, so there are only 5 actuated
# arm joints -- matching the 5 velocity actuators (motor1..5) and hence the
# planner's action dimension nu = 5. Used to resolve qpos/qvel addresses.
ARM_JOINT_NAMES: List[str] = [f"xarm6_joint{i}" for i in range(1, 6)]
BLOCK_JOINT_NAMES: List[str] = ["T_x", "T_y", "T_z"]  # slide x, slide y, hinge yaw

# The real xArm6 ROS driver names its joints joint1..joint6, NOT the MJX
# model's xarm6_joint1..5 -- hence a name mapping. joint6 is welded in the MJX
# scene so the planner emits 5 velocities; the real controller still wants 6,
# with joint6 = 0 (see send_velocity).
ROS_ARM_JOINT_NAMES: List[str] = [f"joint{i}" for i in range(1, 6)]
T_BLOCK_MESH_ORIGIN_OFFSET: Tuple[float, float] = (0.0, 0.030)

@dataclass
class WorldState:
    """One synchronized snapshot of everything the planner needs.

    All quantities are in the planner's world frame and SI units. The arm
    entries are ordered to match `ARM_JOINT_NAMES` (i.e. the actuator
    order); the object entries are planar SE(2), the only object DOFs the
    push-T task models.
    """

    arm_qpos: np.ndarray  # (5,) joint angles [rad]
    arm_qvel: np.ndarray  # (5,) joint velocities [rad/s]
    object_se2: np.ndarray  # (3,) block pose [x, y, yaw] in world frame
    object_twist: np.ndarray  # (3,) block twist [vx, vy, wz]
    time: float  # wall/sim clock [s]


@dataclass
class SceneAddresses:
    """qpos/qvel addresses for the arm and block, resolved once from a model.

    Both interfaces and `run_real` need to read/write the same slots of the
    combined MuJoCo state, so the lookup lives in one place. Looked up by
    joint name -- never assumed to be a fixed slice -- because the composed
    xarm6 scene compiles the arm's 5 joints first, so the block's SE(2) pose
    lands at qpos[5:8], not qpos[:3] (see oim/tasks/pusht.py).
    """

    arm_qpos_adr: np.ndarray  # (5,)
    arm_dof_adr: np.ndarray  # (5,)
    block_qpos_adr: np.ndarray  # (3,)
    block_dof_adr: np.ndarray  # (3,)

    @classmethod
    def from_model(cls, mj_model: mujoco.MjModel) -> "SceneAddresses":

        def qadr(name: str) -> int:
            return int(mj_model.joint(name).qposadr[0])

        def dadr(name: str) -> int:
            return int(mj_model.joint(name).dofadr[0])

        return cls(
            arm_qpos_adr=np.array([qadr(n) for n in ARM_JOINT_NAMES]),
            arm_dof_adr=np.array([dadr(n) for n in ARM_JOINT_NAMES]),
            block_qpos_adr=np.array([qadr(n) for n in BLOCK_JOINT_NAMES]),
            block_dof_adr=np.array([dadr(n) for n in BLOCK_JOINT_NAMES]),
        )


class RobotWorldInterface(ABC):
    """The one seam between the planner and the physical (or mock) world."""

    @abstractmethod
    def read_state(self) -> WorldState:
        """Return the latest synchronized state of arm + object."""

    @abstractmethod
    def send_velocity(self, u: np.ndarray) -> None:
        """Command the 5 arm joint velocities [rad/s], actuator order."""

    @abstractmethod
    def time(self) -> float:
        """Current clock in seconds."""

    def close(self) -> None:  # noqa: B027
        """Release hardware / stop threads. Default: nothing to do."""

    def stop(self) -> None:
        """Bring the arm to a halt. Default: one zero-velocity command."""
        self.send_velocity(np.zeros(len(ARM_JOINT_NAMES)))


# Mock: a MuJoCo simulation behind the same interface, for laptop testing.
class MujocoMockInterface(RobotWorldInterface):
    """Pretend-hardware backed by a MuJoCo sim, for developing `run_real`.

    `send_velocity` sets the sim's actuators and advances it by one control
    period; `read_state` reads the resulting state back. Swapping this for
    `Ros2Interface` should be the *only* change needed to move to hardware,
    which is the point of testing against it first.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        sim_steps_per_send: int,
        emulate_pose_only: bool = True,
    ) -> None:
        """
        Args:
            mj_model, mj_data: the execution sim (fine timestep).
            sim_steps_per_send: physics steps advanced per `send_velocity`,
                i.e. one replanning period's worth.
            emulate_pose_only: if True, derive the object twist by finite
                difference of the object pose (as real hardware must, from
                FoundationPose), rather than reading the sim's exact block
                qvel. Keeps the mock honest about the noisy-derivative issue.
        """
        self._model = mj_model
        self._data = mj_data
        self._n = max(1, sim_steps_per_send)
        self._adr = SceneAddresses.from_model(mj_model)
        self._emulate_pose_only = emulate_pose_only
        self._prev_se2: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        mujoco.mj_forward(mj_model, mj_data)

    def _object_se2(self) -> np.ndarray:
        return np.array(self._data.qpos[self._adr.block_qpos_adr])

    def read_state(self) -> WorldState:
        se2 = self._object_se2()
        t = float(self._data.time)
        if self._emulate_pose_only:
            twist = _finite_diff_se2(self._prev_se2, se2, self._prev_t, t)
        else:
            twist = np.array(self._data.qvel[self._adr.block_dof_adr])
        self._prev_se2, self._prev_t = se2, t
        return WorldState(
            arm_qpos=np.array(self._data.qpos[self._adr.arm_qpos_adr]),
            arm_qvel=np.array(self._data.qvel[self._adr.arm_dof_adr]),
            object_se2=se2,
            object_twist=twist,
            time=t,
        )

    def send_velocity(self, u: np.ndarray) -> None:
        # The mock's actuators are the same 5 velocity servos the planner
        # targets, so the command maps straight through.
        self._data.ctrl[:] = np.asarray(u)
        for _ in range(self._n):
            mujoco.mj_step(self._model, self._data)

    def time(self) -> float:
        return float(self._data.time)


# Real: ROS2 <-> xArm6 bridge (mirrors the OI-MPPI ros2_interface.py).
class Ros2Interface(RobotWorldInterface):
    """ROS2 <-> xArm6 bridge.

    Subscribes:
        /joint_states            (sensor_msgs/JointState)     -> arm state
        TF frame `object_frame`  (FoundationPose)             -> object pose
    Publishes:
        `velocity_command_topic` (std_msgs/Float64MultiArray) -> arm command

    `object_frame` defaults to the FoundationPose frame (pose_source=fp) and
    `watchdog_timeout` to 5*plan_dt. rclpy is imported lazily so this module
    still imports on a machine without ROS (a laptop running only the mock).
    """

    def __init__(
        self,
        world_frame: str = "world",
        object_frame: str = "fp_object_pose",
        object_origin_offset: Tuple[float,float] = (0.0, 0.0),
        object_z_band: Tuple[float, float] = (0.008, 0.026),
        object_tilt_max_deg: float = 30.0,
        yaw_jump_max_deg_s: float = 300.0,
        yaw_flip_recovery: bool = True,
        pose_reject_grace: float = 2.0,
        pose_reject_limit: float = 6.0,
        base_frame: str = "xarm_device",
        base_pos: Tuple[float, float] = (0.0, 0.0),
        base_yaw_deg: float = 0.0,
        base_z: float = 0.0,
        joint_states_topic: str = "/joint_states",
        velocity_command_topic: str = "velocity_controller/commands_nominal",
        enable_commands: bool = True,
        twist_filter_alpha: float = 0.4,
        watchdog_timeout: float = 0.3,
        object_staleness_limit: float = 1.5,
        startup_timeout: float = 30.0,
    ) -> None:
        import rclpy  # noqa: PLC0415
        import tf2_ros  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415
        from sensor_msgs.msg import JointState  # noqa: PLC0415
        from std_msgs.msg import Float64MultiArray  # noqa: PLC0415

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._Float64MultiArray = Float64MultiArray
        self._node = Node("oim_real3d_interface")

        self._world_frame = world_frame
        self._object_frame = object_frame
        self._object_origin_offset = tuple(float(v) for v in object_origin_offset)
        # Plausibility gates on the FoundationPose pose. The planner uses only
        # (x, y, yaw); the HEIGHT it throws away is therefore a free anomaly
        # detector, because a block resting on a table has exactly one
        # plausible mesh-origin height and a slipped fit does not respect it.
        self._object_z_band = (float(object_z_band[0]), float(object_z_band[1]))
        # Same idea one axis over. The block lies flat, so its body z-axis is
        # world z; anything past this is FoundationPose fitting the mesh on
        # edge or upside down, and the yaw such a fit reports is meaningless.
        # Near 180 deg it is not meaningless but INVERTED -- see gate 1b.
        self._object_tilt_max = float(np.radians(object_tilt_max_deg))
        # Ceiling on how fast this pusher can physically spin the block: the
        # tip moves at most ~0.1 m/s and contacts within ~45 mm of the block's
        # centre, so ~130 deg/s, and table friction bleeds that off in ~0.1 s.
        # The default leaves better than 2x margin over that.
        self._yaw_jump_max_rate = float(np.radians(yaw_jump_max_deg_s))
        self._yaw_flip_recovery = bool(yaw_flip_recovery)
        # 0 or pi. Sticky, because every frame from a flipped re-registration
        # carries the same flip -- see `_lookup_object_se2`.
        self._yaw_offset = 0.0
        # How long a run of gate-2 rejections is tolerated before the
        # REFERENCE is presumed to be the wrong one. Gate 2 is a relative
        # test, so a stream that disagrees with `_last_se2` for this long is
        # more likely right than the pose it is being compared against.
        self._pose_reject_grace = float(pose_reject_grace)
        # ... and how long a run of them is tolerated at all. Past this the
        # object is presumed GONE, not misread: knocked off the table, or
        # pushed somewhere FoundationPose can no longer fit it. Measured
        # failure this exists for (2026-08-27 15:57 run): the block was
        # launched off the table, every subsequent pose failed the height
        # gate, and `_hold` returned the same frozen SE(2) for 63 steps /
        # 41 s while the planner kept issuing saturated joint commands at
        # it. Nothing stopped that but a human on Ctrl-C.
        #
        # Distinct from `pose_reject_grace`, which asks "is the REFERENCE
        # wrong?" and re-baselines onto the incoming stream. That can only
        # fire from the yaw-rate gate, so a run of HEIGHT rejections -- the
        # exact signature of a block that has left the table -- never
        # reaches it. Must stay comfortably above the grace so re-baselining
        # gets its chance first.
        self._pose_reject_limit = float(pose_reject_limit)
        if self._pose_reject_limit <= self._pose_reject_grace:
            raise ValueError(
                f"pose_reject_limit ({self._pose_reject_limit}) must exceed "
                f"pose_reject_grace ({self._pose_reject_grace}); otherwise "
                "the run dies before a re-baseline can rescue it")
        self._reject_since: Optional[float] = None
        self._n_z_reject = 0
        self._n_tilt_reject = 0
        self._n_roll_flip = 0
        self._n_jump_reject = 0
        self._n_flip = 0
        self._n_rebaseline = 0
        self._alpha = twist_filter_alpha
        self._watchdog_timeout = watchdog_timeout
        # False = never publish a command (dry run), the same no-motion path as
        # OI-MPPI's enable_velocity_commands:=false. State/TF are still read.
        self._enable_commands = enable_commands

        # Latest arm state, filled by the subscription callback. Guarded by a
        # lock because rclpy spins on a background thread (see below).
        self._lock = threading.Lock()
        self._arm_qpos: Optional[np.ndarray] = None
        self._arm_qvel: Optional[np.ndarray] = None
        # For finite-difference object twist + low-pass filtering.
        self._prev_se2: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._twist_lp = np.zeros(3)
        self._last_cmd_time = time.monotonic()
        # Last good object pose, to hold through transient FoundationPose TF
        # gaps (up to object_staleness_limit) instead of crashing.
        self._staleness_limit = object_staleness_limit
        self._last_se2: Optional[np.ndarray] = None
        self._last_se2_time: Optional[float] = None

        # /joint_states publishes the arm joints (ROS_ARM_JOINT_NAMES =
        # joint1..joint5) in an arbitrary order -- reindex by name once.
        self._joint_index: Optional[List[int]] = None

        self._node.create_subscription(JointState, joint_states_topic,
                                       self._on_joint_states, 10)
        self._cmd_pub = self._node.create_publisher(Float64MultiArray,
                                                    velocity_command_topic, 10)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer,
                                                      self._node)

        # Publish the world -> base static transform ourselves, from the same
        # XARM6_BASE_POS/yaw the model uses, so it can't drift from a separate
        # script. FoundationPose's TF chain is rooted at the base
        # (base -> camera -> object); this link is what lets us look up the
        # object in the planner's world frame. (OI-MPPI published the
        # equivalent transform inside its ros2_interface node too.)
        self._publish_base_tf(tf2_ros, world_frame, base_frame,
                              base_pos, base_yaw_deg, base_z)

        # Safety watchdog: commands zero velocity if the control loop hasn't
        # sent anything within `watchdog_timeout`. It runs on the spin thread,
        # so it keeps firing even while the main thread is blocked in
        # `optimize` -- the planning call never disables the safety stop.
        self._node.create_timer(watchdog_timeout / 5.0, self._watchdog)

        # MJX runs in float32, where the spacing near a ROS epoch timestamp
        # (~1.79e9) is 128 s -- adding the 0.75 s plan horizon to it is a
        # no-op, so every knot in `tk` collapses onto the same value and the
        # plan loses its time axis entirely. Hand the planner a clock that
        # starts at zero instead. Only differences are ever used downstream.
        self._t0 = self._node.get_clock().now().nanoseconds * 1e-9

        # Spin rclpy on its own thread so read_state()/send_velocity() can be
        # called synchronously from run_real without blocking callbacks, and
        # so the subscription + watchdog run concurrently with planning.
        self._spin_thread = threading.Thread(target=rclpy.spin,
                                             args=(self._node,),
                                             daemon=True)
        self._spin_thread.start()

        # Block until /joint_states and the object TF are both available, so
        # the driver waits for the robot + FoundationPose to come up instead of
        # crashing on the first read.
        self._wait_until_ready(startup_timeout)

    def _wait_until_ready(self, timeout: float) -> None:
        """Wait for the first /joint_states and object TF (or time out)."""
        self._node.get_logger().info(
            "waiting for /joint_states and the object TF...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                have_arm = self._arm_qpos is not None
            try:
                self._lookup_object_se2()  # populates _last_se2 on success
                have_obj = True
            except Exception:  # noqa: BLE001
                have_obj = False
            if have_arm and have_obj:
                self._node.get_logger().info("robot + object TF ready")
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"timed out after {timeout}s waiting for /joint_states and TF "
            f"'{self._object_frame}' (is the robot bringup + FoundationPose up?)")

    def _publish_base_tf(self, tf2_ros, world_frame, base_frame,
                         base_pos, base_yaw_deg, base_z) -> None:
        """Broadcast the static world -> base transform.

        Skipped when the planner's world frame already IS the base frame
        (base-at-origin scenes like "box_clutter_real"): the transform is identity and
        FoundationPose's TF tree is already rooted at the base. base_pos/yaw/z
        are passed in as plain values by the caller -- this hardware I/O seam
        stays unaware of the SCENES registry (the placement lives there, is read
        into the task, and is handed here as values, like world_frame/base_z).
        """
        if world_frame == base_frame:
            return
        from geometry_msgs.msg import TransformStamped  # noqa: PLC0415
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        qx, qy, qz, qw = Rotation.from_euler(
            "z", base_yaw_deg, degrees=True
        ).as_quat()
        t = TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = world_frame
        t.child_frame_id = base_frame
        t.transform.translation.x = float(base_pos[0])
        t.transform.translation.y = float(base_pos[1])
        t.transform.translation.z = float(base_z)
        t.transform.rotation.x, t.transform.rotation.y = float(qx), float(qy)
        t.transform.rotation.z, t.transform.rotation.w = float(qz), float(qw)
        # Keep the broadcaster alive (a static TF is latched, but the object
        # must not be garbage-collected).
        self._static_bcaster = tf2_ros.StaticTransformBroadcaster(self._node)
        self._static_bcaster.sendTransform(t)

    # -- callbacks -----------------------------------------------------
    def _on_joint_states(self, msg) -> None:
        if self._joint_index is None:
            self._joint_index = [msg.name.index(n) for n in ROS_ARM_JOINT_NAMES]
        idx = self._joint_index
        with self._lock:
            self._arm_qpos = np.array([msg.position[i] for i in idx])
            self._arm_qvel = (np.array([msg.velocity[i] for i in idx])
                              if msg.velocity else np.zeros(len(idx)))

    def _lookup_object_se2(self) -> np.ndarray:
        """Read the object pose from TF and project 6D -> SE(2).

        FoundationPose gives a full 6D pose; the push-T task only models the
        planar (x, y, yaw) DOFs, so drop z/roll/pitch and keep the yaw about
        the table normal (see notes in oim/tasks/pusht.py).

        On a transient TF gap (a dropped FoundationPose frame) the last good
        pose is reused, up to `object_staleness_limit`; a longer outage raises
        rather than pushing on a stale pose.
        """
        from scipy.spatial.transform import Rotation  # noqa: PLC0415
        from tf2_ros import (  # noqa: PLC0415
            ConnectivityException,
            ExtrapolationException,
            LookupException,
        )

        try:
            tf = self._tf_buffer.lookup_transform(
                self._world_frame, self._object_frame, self._rclpy.time.Time())
        except (LookupException, ConnectivityException,
                ExtrapolationException) as exc:
            try:
                return self._hold("TF unavailable", fatal=True)
            except RuntimeError as err:
                raise err from exc

        p = tf.transform.translation
        q = tf.transform.rotation
        now = time.monotonic()

        # GATE 1 -- height. See `object_z_band` in __init__.
        lo, hi = self._object_z_band
        if not (lo <= p.z <= hi):
            self._n_z_reject += 1
            if self._n_z_reject % 10 == 1:
                self._node.get_logger().warn(
                    f"pose rejected: z={p.z:+.4f} outside [{lo:.3f}, {hi:.3f}]"
                    f" ({self._n_z_reject} so far)")
            if self._reject_since is None:
                self._reject_since = now
            return self._hold("implausible height")

        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        rpy = rot.as_euler("xyz")

        # GATE 1b -- out-of-plane tilt, and the mesh flip hiding inside it.
        #
        # `tilt` is the angle between the block's own z-axis and world z, read
        # straight off the rotation matrix rather than out of the Euler
        # triple, which goes singular exactly here.
        #
        # The T's footprint is MIRROR-symmetric about its stem axis, and the
        # mesh origin offset is measured along that same axis (hence the
        # offset's x component is 0). So the mesh rotated by pi about that
        # axis projects to an identical footprint: FoundationPose cannot tell
        # the two apart from depth alone, and neither can gate 1 -- the mesh
        # origin is at mid-thickness, so a flip leaves z untouched.
        #
        # What the flip does change is everything downstream. Writing the
        # stem axis as body y, R_flip = Rz(t) Ry(pi) = Rz(t + pi) Rx(pi), so
        # such a fit arrives as roll ~= 180 with the yaw component reading
        # t + pi -- pi away from the block's true heading. That wrong yaw then
        # rotates `object_origin_offset` by pi as well, which turns a +30 mm
        # correction into -30 mm: 60 mm of position error on top of a heading
        # that is backwards. This is the cold-start failure the runs showed
        # (rpy=(-180.0, -0.7, +176.3) for the first ~30 steps).
        #
        # Corrected, not rejected, and NOT sticky: unlike the crossbar
        # re-registration in gate 2, this is measured every frame rather than
        # guessed, so it needs no memory and self-clears the moment the fit
        # rights itself. Applied BEFORE gate 2 so the de-flipped yaw is what
        # the rate test sees -- otherwise the flip also trips gate 2 and the
        # run spends its grace period fighting a pose it could have fixed.
        tilt = float(np.arccos(np.clip(rot.as_matrix()[2, 2], -1.0, 1.0)))
        flipped = tilt > np.pi - self._object_tilt_max
        if flipped:
            self._n_roll_flip += 1
            if self._n_roll_flip % 20 == 1:
                self._node.get_logger().warn(
                    f"pose is upside down: tilt={np.degrees(tilt):.0f} deg; "
                    f"correcting yaw by pi ({self._n_roll_flip} so far)")
        elif tilt > self._object_tilt_max:
            # Between the two thresholds there is no ambiguity to resolve --
            # a block on a table is not on edge, so the fit has come off the
            # object entirely and nothing about it can be salvaged.
            self._n_tilt_reject += 1
            if self._n_tilt_reject % 10 == 1:
                self._node.get_logger().warn(
                    f"pose rejected: tilt={np.degrees(tilt):.0f} deg over "
                    f"{np.degrees(self._object_tilt_max):.0f}"
                    f" ({self._n_tilt_reject} so far)")
            if self._reject_since is None:
                self._reject_since = now
            return self._hold("implausible tilt")

        yaw = _wrap(rpy[2] + self._yaw_offset + (np.pi if flipped else 0.0))

        # GATE 2 -- angular rate, and the 180-degree flip hiding inside it.
        #
        # A jump of very nearly pi is the specific failure a re-registration
        # produces: the T's CROSSBAR is symmetric, so a fresh fit that loses
        # the stem lands exactly one pi out, with a perfectly plausible height
        # and position -- gate 1 cannot see it. That case is CORRECTED rather
        # than rejected, and the correction is sticky because every later
        # frame from the same fit carries it. It is also self-undoing: guess
        # wrong and the next frame is another impossible jump the other way.
        #
        # NOT handled by wrapping yaw modulo pi. The block is not pi-symmetric
        # -- crossbar at +y, stem at -y -- so the goal pose, the footprint and
        # every push direction distinguish the two. Folding them together
        # would make the planner accept a goal it has not reached.
        if self._last_se2 is not None and self._last_se2_time is not None:
            # Capped, not just floored. `_last_se2_time` only advances on an
            # ACCEPTED pose, so during a rejection run dt grows and the
            # allowed jump grows with it -- the gate would go slack exactly
            # when tracking is worst. 0.2 s is ~2 replans.
            dt = min(max(now - self._last_se2_time, 1e-3), 0.2)
            d = _wrap(yaw - float(self._last_se2[2]))
            if abs(d) > self._yaw_jump_max_rate * dt:
                if (self._yaw_flip_recovery
                        and abs(d) > np.pi - np.radians(25.0)):
                    self._yaw_offset = _wrap(self._yaw_offset + np.pi)
                    yaw = _wrap(yaw + np.pi)
                    self._n_flip += 1
                    self._node.get_logger().warn(
                        f"180-degree pose flip: yaw jumped "
                        f"{np.degrees(d):+.0f} deg in {dt * 1e3:.0f} ms; "
                        f"correcting by pi ({self._n_flip} so far)")
                elif (self._reject_since is not None
                      and now - self._reject_since > self._pose_reject_grace):
                    # Everything has disagreed with the reference for longer
                    # than a bad frame lasts, so the reference is the outlier.
                    # Take the incoming pose as truth and carry on; refusing
                    # forever would leave the planner on a frozen pose, which
                    # is the same failure with none of the visibility.
                    self._n_rebaseline += 1
                    self._node.get_logger().warn(
                        f"pose re-baselined after "
                        f"{now - self._reject_since:.1f}s of rejections; "
                        f"accepting yaw {np.degrees(yaw):+.0f} deg "
                        f"({self._n_rebaseline} so far)")
                    self._reject_since = None
                else:
                    if self._reject_since is None:
                        self._reject_since = now
                    self._n_jump_reject += 1
                    if self._n_jump_reject % 10 == 1:
                        self._node.get_logger().warn(
                            f"pose rejected: yaw jumped {np.degrees(d):+.0f} "
                            f"deg in {dt * 1e3:.0f} ms, over "
                            f"{np.degrees(self._yaw_jump_max_rate):.0f} deg/s"
                            f" ({self._n_jump_reject} so far)")
                    return self._hold("implausible angular rate")

        dx, dy = self._object_origin_offset
        c, s = np.cos(yaw), np.sin(yaw)
        se2 = np.array([p.x + c * dx - s * dy, p.y + s * dx + c * dy, yaw])
        self._pose_log_n = getattr(self, "_pose_log_n", 0) + 1
        if self._pose_log_n % 20 == 1:  # ~1 Hz at 20 Hz TF
            self._node.get_logger().info(
                f"TF raw ({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f}) "
                f"rpy=({np.degrees(rpy[0]):+.1f},{np.degrees(rpy[1]):+.1f},"
                f"{np.degrees(rpy[2]):+.1f})d yaw "
                f"{np.degrees(yaw):+.1f}d -> planner SE(2) "
                f"({se2[0]:+.4f}, {se2[1]:+.4f})  tilt="
                f"{np.degrees(tilt):.0f}d  [rejected z={self._n_z_reject} "
                f"tilt={self._n_tilt_reject} jump={self._n_jump_reject} "
                f"roll_flips={self._n_roll_flip} flips={self._n_flip} "
                f"rebase={self._n_rebaseline}]")
        self._reject_since = None
        self._last_se2, self._last_se2_time = se2, now
        return se2

    def _hold(self, reason: str, fatal: bool = False) -> np.ndarray:
        """Reuse the last good pose.

        `fatal=False` -- a GATE rejection. Holds, but only for
        `pose_reject_limit`. FoundationPose disagreeing with itself for one
        frame is a bad frame, not a dead system, and ending a hardware run
        over it costs more than planning against a pose a moment old --
        which is why this holds at all. Disagreeing for SECONDS is not that:
        it is a block that has left the table or the camera, and holding
        then leaves the planner driving at full command toward a pose that
        no longer exists anywhere. The counters and throttled warnings keep
        the short case visible; the re-baseline in `_lookup_object_se2`
        breaks a run of them when the reference is the wrong one; this limit
        ends the run when neither applies.

        `fatal=True` -- no TF arriving at all. Still raises past
        `object_staleness_limit`: nothing is coming, so there is no reason to
        expect a recovery, and the arm must not keep planning against a frozen
        pose forever.
        """
        if self._last_se2 is not None:
            if not fatal:
                held = (0.0 if self._reject_since is None
                        else time.monotonic() - self._reject_since)
                if held <= self._pose_reject_limit:
                    return self._last_se2
                raise RuntimeError(
                    f"object pose from '{self._object_frame}' rejected "
                    f"continuously for {held:.1f}s ({reason}); the block has "
                    f"most likely left the table or the camera's view. "
                    f"[rejected z={self._n_z_reject} tilt={self._n_tilt_reject}"
                    f" jump={self._n_jump_reject}]")
            if (self._last_se2_time is not None
                    and time.monotonic() - self._last_se2_time
                    <= self._staleness_limit):
                return self._last_se2
        raise RuntimeError(
            f"object pose from '{self._object_frame}' unusable ({reason}); "
            f"no usable pose within {self._staleness_limit}s")

    # -- interface -----------------------------------------------------
    def read_state(self) -> WorldState:
        with self._lock:
            arm_qpos = self._arm_qpos
            arm_qvel = self._arm_qvel
        if arm_qpos is None:
            raise RuntimeError("no /joint_states received yet")

        se2 = self._lookup_object_se2()
        t = self._node.get_clock().now().nanoseconds * 1e-9 - self._t0
        raw_twist = _finite_diff_se2(self._prev_se2, se2, self._prev_t, t)
        # Low-pass the finite-difference twist: dividing a jittery pose
        # estimate by a small dt amplifies noise, and this twist feeds the
        # `twist` consensus estimator directly.
        self._twist_lp = (self._alpha * raw_twist +
                          (1.0 - self._alpha) * self._twist_lp)
        self._prev_se2, self._prev_t = se2, t
        return WorldState(
            arm_qpos=arm_qpos,
            arm_qvel=arm_qvel,
            object_se2=se2,
            object_twist=self._twist_lp.copy(),
            time=t,
        )

    def send_velocity(self, u: np.ndarray) -> None:
        if not self._enable_commands:  # dry run: read state/TF, publish nothing
            return
        # The planner emits 5 velocities; the real controller expects 6, with
        # the welded wrist-roll joint6 commanded to 0.
        cmd = [float(x) for x in np.asarray(u)] + [0.0]
        msg = self._Float64MultiArray()
        msg.data = cmd
        self._cmd_pub.publish(msg)
        self._last_cmd_time = time.monotonic()

    def _watchdog(self) -> None:
        """Zero the arm if no fresh command arrived within the timeout."""
        if not self._enable_commands:
            return
        if time.monotonic() - self._last_cmd_time > self._watchdog_timeout:
            msg = self._Float64MultiArray()
            msg.data = [0.0] * (len(ROS_ARM_JOINT_NAMES) + 1)
            self._cmd_pub.publish(msg)

    def time(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9 - self._t0

    def stop(self, hold: float = 0.5, rate: float = 50.0) -> None:
        """Publish zero velocity repeatedly until the arm is certainly stopped.

        One publish is not enough. rclpy's publish is asynchronous and
        `destroy_node` tears the writer down without waiting for delivery, so a
        single stop command can be dropped -- and the arm's velocity controller
        HOLDS the last velocity it received, so a dropped zero means the arm
        keeps moving after this process exits. Publishing for `hold` seconds
        also overrides anything the publisher thread queued on its way out.
        """
        if not self._enable_commands:
            return
        zeros = np.zeros(len(ARM_JOINT_NAMES))
        deadline = time.monotonic() + hold
        while time.monotonic() < deadline:
            self.send_velocity(zeros)
            time.sleep(1.0 / rate)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._node.destroy_node()


def _wrap(angle: float) -> float:
    """Wrap an angle to (-pi, pi].

    Local rather than `oim.objects.wrap_angle` on purpose: this module stays
    importable without JAX so the mock runs on a laptop with no GPU stack.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _finite_diff_se2(
    prev: Optional[np.ndarray],
    curr: np.ndarray,
    prev_t: Optional[float],
    t: float,
) -> np.ndarray:
    """Twist [vx, vy, wz] from two SE(2) poses; zero on the first call.

    The yaw component is unwrapped so a +/-pi crossing doesn't produce a
    spurious spike. Short sampling periods make this a good local-derivative
    approximation; the caller is responsible for any further filtering.
    """
    if prev is None or prev_t is None or t <= prev_t:
        return np.zeros(3)
    dt = t - prev_t
    d = curr - prev
    d[2] = (d[2] + np.pi) % (2.0 * np.pi) - np.pi  # wrap yaw difference
    return d / dt


def clamp_velocity(u: np.ndarray, limit: float = 0.2) -> np.ndarray:
    u = np.asarray(u)
    peak = np.max(np.abs(u))
    if peak > limit:
        u = u * (limit / peak)  # scale all joints together, ratio preserved
    return u
