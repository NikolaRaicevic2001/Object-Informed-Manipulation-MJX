"""Loop-timing stamps: read-side and publish-side lists stay aligned per
step, and the summary reads the intended differences."""

import numpy as np

from oim.worlds.real3d import timing
from oim.worlds.real3d.interface import WorldState


def _world(stamps):
    return WorldState(
        arm_qpos=np.zeros(5), arm_qvel=np.zeros(5), object_se2=np.zeros(3),
        object_twist=np.zeros(3), time=0.0, stamps=stamps,
    )


def test_stamps_default_to_nan_without_an_interface_that_has_them() -> None:
    log = {}
    timing._log_read_stamps(log, _world(None))
    for key in timing._STAMP_READ_KEYS:
        assert len(log[key]) == 1 and np.isnan(log[key][0])
    timing._log_publish_stamps(log, {})
    assert np.isnan(log["ros_cmd_pub"][0]) and log["cmd_pub_index"] == [-1]
    assert timing._stamp_summary(log, 0) == ""


def test_stamps_align_and_summarise() -> None:
    log = {}
    # Three steps: the joint state is 8 ms old at read, our callback got it
    # 3 ms after the driver stamped it, and the first command went out
    # 300 ms after read_state on both clocks.
    for k in range(3):
        base = 10.0 + k
        timing._log_read_stamps(log, _world({
            "ros_js_stamp": base, "ros_js_recv": base + 0.003,
            "ros_read": base + 0.008,
            "perf_js_recv": 500.0 + k + 0.003, "perf_read": 500.0 + k + 0.008,
        }))
    pub = {0: (10.308, 500.308, 0), 1: (11.308, 501.308, 1)}  # step 2 unsent
    timing._log_publish_stamps(log, pub)
    assert len(log["ros_cmd_pub"]) == 3 and log["cmd_pub_index"] == [0, 1, -1]
    line = timing._stamp_summary(log, 1)
    assert "js_age=8ms" in line and "transport=3ms" in line
    assert "e2e_ros=308ms" in line and "e2e_perf=300ms" in line
    assert "first_idx=1" in line
    assert timing._stamp_summary(log, 2) == ""
    summary = timing._stamp_summary(log)
    assert summary.startswith("timing (median/p95)")
    assert "e2e_ros=308/308ms" in summary
