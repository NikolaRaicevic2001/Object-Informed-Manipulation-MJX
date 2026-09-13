"""Ros2Interface pose gates: a sticky 180-degree yaw correction must not
outlive the FoundationPose flip it was applied for.

Sequences replay FoundationPose poses logged on the real rig. The interface is
built without ROS: TF, the logger and the clock are faked."""

import sys
import types
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from oim.worlds.real3d import interface
from oim.worlds.real3d.interface import Ros2Interface

OFFSET = (0.0, 0.025)


class _Tf:
    frame = None

    def lookup_transform(self, *_args):
        return self.frame


@pytest.fixture
def gate(monkeypatch):
    tf2_ros = types.ModuleType("tf2_ros")
    for name in ("LookupException", "ConnectivityException",
                 "ExtrapolationException"):
        setattr(tf2_ros, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "tf2_ros", tf2_ros)
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(interface, "time",
                        SimpleNamespace(monotonic=lambda: clock.t))
    warns = []
    logger = SimpleNamespace(warn=warns.append, info=lambda _msg: None)

    g = object.__new__(Ros2Interface)
    g._node = SimpleNamespace(get_logger=lambda: logger)
    g._tf_buffer = _Tf()
    g._rclpy = SimpleNamespace(time=SimpleNamespace(Time=lambda: None))
    g._world_frame, g._object_frame = "xarm_device", "fp_object_pose"
    # Constructor defaults.
    g._object_origin_offset = OFFSET
    g._object_z_band = (-0.015, 0.045)
    g._object_tilt_max = np.radians(30.0)
    g._yaw_jump_max_rate = np.radians(300.0)
    g._pos_jump_max_rate = 0.25
    g._pose_reject_confirm = 3
    g._pose_reject_grace = 4.0
    g._pose_reject_limit = 10.0
    g._staleness_limit = 1.5
    g._pose_median3 = True
    g._yaw_flip_recovery = True
    g._se2_hist = deque(maxlen=3)
    g._last_raw = None
    g._last_se2 = g._last_se2_time = g._reject_since = None
    g._consistent_n = 0
    g._yaw_offset = 0.0
    for counter in ("_n_stream_rebase", "_n_z_reject", "_n_tilt_reject",
                    "_n_roll_flip", "_n_jump_reject", "_n_flip",
                    "_n_rebaseline"):
        setattr(g, counter, 0)
    return g, clock, warns


def _mesh_origin(planner_xy, yaw_deg):
    """FP mesh origin that maps to `planner_xy` at heading `yaw_deg`."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    dx, dy = OFFSET
    return (planner_xy[0] - (c * dx - s * dy),
            planner_xy[1] - (s * dx + c * dy))


def _feed(g, clock, mesh_xy, raw_yaw_deg, n=1):
    """Publish one upright FP pose and read it `n` times, 0.2 s apart."""
    h = np.radians(raw_yaw_deg) / 2.0
    g._tf_buffer.frame = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=mesh_xy[0], y=mesh_xy[1], z=0.01),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=np.sin(h), w=np.cos(h))))
    for _ in range(n):
        clock.t += 0.2
        g._lookup_object_se2()


def _yaw_err_deg(se2, yaw_deg):
    return abs(np.degrees(interface._wrap(se2[2] - np.radians(yaw_deg))))


def test_flip_correction_survives_fp_staying_off_and_clears_on_recovery(gate):
    g, clock, warns = gate
    _feed(g, clock, _mesh_origin((0.399, 0.3108), 57.0), 57.0, n=5)

    # FP re-registers pi out, position still plausible: corrected.
    _feed(g, clock, (0.3856, 0.3061), -121.5)
    assert _yaw_err_deg(g._last_se2, 58.5) < 1.0

    # The flipped fit slides 78 mm and is re-baselined while still flipped.
    _feed(g, clock, _mesh_origin((0.432, 0.280), 66.0), 66.0 - 180.0, n=6)
    assert sum("re-baselined" in w for w in warns) == 1
    assert _yaw_err_deg(g._last_se2, 66.0) < 1.0

    # FP recovers: pi back and 120 mm away, re-baselined again.
    _feed(g, clock, (0.5190, 0.2418), 67.0, n=8)
    assert sum("re-baselined" in w for w in warns) == 2
    assert abs(interface._wrap(g._yaw_offset)) < 1e-9
    assert _yaw_err_deg(g._last_se2, 67.0) < 1.0
    assert _yaw_err_deg(g.peek_object_se2(), 67.0) < 1.0


def test_flip_correction_clears_when_the_rebaselined_stream_is_not_flipped(gate):
    g, clock, warns = gate
    _feed(g, clock, _mesh_origin((0.55, 0.19), -36.0), -36.0, n=5)
    _feed(g, clock, (0.5477, 0.1921), 163.3)
    assert _yaw_err_deg(g._last_se2, -16.7) < 1.0

    # FP back near the pre-flip heading, 82 mm away.
    _feed(g, clock, (0.5007, 0.1247), -45.0, n=6)
    assert any("re-baselined" in w for w in warns)
    assert abs(interface._wrap(g._yaw_offset)) < 1e-9
    assert _yaw_err_deg(g._last_se2, -45.0) < 1.0


def test_rebaseline_does_not_introduce_a_flip_correction(gate):
    g, clock, warns = gate
    _feed(g, clock, _mesh_origin((0.40, 0.20), 0.0), 0.0, n=5)
    _feed(g, clock, _mesh_origin((0.50, 0.20), 180.0), 180.0, n=6)
    assert any("re-baselined" in w for w in warns)
    assert abs(interface._wrap(g._yaw_offset)) < 1e-9
    assert _yaw_err_deg(g._last_se2, 180.0) < 1.0
