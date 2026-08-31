"""FoundationPose pre-flight check -- gate a real run on a sane pose stream.

Watches the RAW object TF (before the interface's gates, which would hold or
reject exactly the frames this check needs to see) for a few seconds while
the block sits STILL, and grades it PASS / WARN / FAIL. Wired into
`run_real` so a LIVE run refuses to send its first command over a bad fit;
also runnable standalone before a session.

Catches, before a run is wasted, the three FP failure modes the 2026-08-29
runs hit:

  1. Upside-down / mirror fit -- tilt ~180 deg. A STEADY 100% is only a
     WARN: gate 1b un-flips the yaw exactly, and on this rig the published
     TF carries a fixed 180-deg offset from FP's own estimate (FP debug
     window and rviz disagree), so the run is fine. Flicker or a partial
     flip fraction is a FAIL: re-seat the block and/or re-register FP.
  2. Fit oscillation between competing minima -- yaw hopping tens of degrees
     between frames on a static block (the 83/119/146 deg flicker of run
     194342). The interface gates then reject the true fit and rebase onto
     phantoms mid-run.
  3. Floated bbox -- the SAM2 bbox drifts slightly off the object: the
     planar condition breaks, so z and tilt wobble while xy still tracks.
     No auto-reset fires for this, so it silently poisons a whole run.

Standalone usage (ROS env sourced, FP + bringup running, block placed and
untouched):

    python -m oim.worlds.real3d.fp_preflight
    python -m oim.worlds.real3d.fp_preflight --seconds 8 --z-band 0.002 0.020

Exit code: 0 = PASS, 1 = WARN (usable but look at it), 2 = FAIL (do not
launch; re-seat the block / re-register FP and re-check).
"""
from __future__ import annotations

import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np


def _wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _read_raw(tf_buffer, rclpy_mod, world_frame: str, object_frame: str,
              tilt_max: float):
    """One raw TF lookup -> (stamp, x, y, z, tilt, yaw, flipped) or None.

    Mirrors the interface's gate-1b definitions (tilt off the rotation
    matrix; yaw corrected by pi on an upside-down fit) but applies no gate:
    the point is to SEE the raw stream, bad frames included.
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    from tf2_ros import (  # noqa: PLC0415
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )
    try:
        tf = tf_buffer.lookup_transform(world_frame, object_frame,
                                        rclpy_mod.time.Time())
    except (LookupException, ConnectivityException, ExtrapolationException):
        return None
    p, q = tf.transform.translation, tf.transform.rotation
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
    tilt = float(np.arccos(np.clip(rot.as_matrix()[2, 2], -1.0, 1.0)))
    flipped = tilt > np.pi - tilt_max
    yaw = _wrap(rot.as_euler("xyz")[2] + (np.pi if flipped else 0.0))
    stamp = (tf.header.stamp.sec, tf.header.stamp.nanosec)
    return stamp, float(p.x), float(p.y), float(p.z), tilt, yaw, flipped


def collect_and_check(
    tf_buffer,
    rclpy_mod,
    *,
    world_frame: str,
    object_frame: str,
    z_band: Tuple[float, float],
    tilt_max: float,
    seconds: float = 5.0,
    min_fps: float = 5.0,
    startup_timeout: float = 15.0,
    pump: Optional[Callable[[], None]] = None,
) -> Tuple[int, List[str]]:
    """Observe the raw stream for `seconds` and grade it.

    `pump` is called between lookups to let TF callbacks run: the gate path
    passes nothing (the interface already spins rclpy on a background
    thread, so a plain sleep suffices); the standalone CLI passes a
    spin_once wrapper.

    Returns (verdict, lines): verdict 0 = PASS, 1 = WARN, 2 = FAIL, lines
    already formatted for printing.
    """
    pump = pump or (lambda: time.sleep(0.02))
    lines: List[str] = [
        f"[preflight] watching raw TF '{object_frame}' for {seconds:.0f}s "
        f"-- the block must be STILL"]

    deadline = time.monotonic() + startup_timeout
    first = None
    while first is None and time.monotonic() < deadline:
        pump()
        first = _read_raw(tf_buffer, rclpy_mod, world_frame, object_frame,
                          tilt_max)
    if first is None:
        lines.append(f"[preflight] FAIL: no object TF within "
                     f"{startup_timeout:.0f}s -- is FoundationPose (and the "
                     f"TF broadcaster) up?")
        return 2, lines

    rows, last_stamp = [], None
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        pump()
        r = _read_raw(tf_buffer, rclpy_mod, world_frame, object_frame,
                      tilt_max)
        if r is not None and r[0] != last_stamp:
            last_stamp = r[0]
            rows.append((time.monotonic(),) + r[1:])

    n = len(rows)
    if n < 2:
        lines.append(f"[preflight] FAIL: only {n} fresh frame(s) in "
                     f"{seconds:.0f}s -- FP is up but not streaming.")
        return 2, lines

    t = np.array([r[0] for r in rows])
    xy = np.array([[r[1], r[2]] for r in rows])
    z = np.array([r[3] for r in rows])
    tilt = np.array([r[4] for r in rows])
    yaw = np.array([r[5] for r in rows])
    flipped = np.array([r[6] for r in rows])

    fps = (n - 1) / max(t[-1] - t[0], 1e-6)
    # Off-plane angle regardless of flip direction: a clean mirror fit has
    # tilt ~pi but is still planar once corrected.
    tilt_eff = np.degrees(np.minimum(tilt, np.pi - tilt))
    dyaw = np.array([_wrap(b - a) for a, b in zip(yaw[:-1], yaw[1:])])
    dt = np.maximum(np.diff(t), 1e-3)
    yaw_rate_max = float(np.degrees(np.abs(dyaw / dt)).max())
    yaw_med = float(np.median(yaw))
    yaw_spread = float(np.degrees(
        np.abs([_wrap(v - yaw_med) for v in yaw])).max())
    xy_ptp = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    z_med, z_ptp = float(np.median(z)), float(z.max() - z.min())

    lines.append(
        f"[preflight] {n} frames over {t[-1]-t[0]:.1f}s ({fps:.1f} fps)  "
        f"xy=({np.median(xy[:,0]):+.4f},{np.median(xy[:,1]):+.4f}) "
        f"ptp={xy_ptp*1000:.1f}mm")
    lines.append(
        f"[preflight] z med={z_med:+.4f} ptp={z_ptp*1000:.1f}mm | tilt "
        f"med={np.median(tilt_eff):.1f}d max={tilt_eff.max():.1f}d | "
        f"flipped {int(flipped.sum())}/{n}")
    lines.append(
        f"[preflight] yaw med={np.degrees(yaw_med):+.1f}d "
        f"spread={yaw_spread:.1f}d max_rate={yaw_rate_max:.0f}d/s "
        f"(flip-corrected; static block, so all motion here is fit noise)")

    fails: List[str] = []
    warns: List[str] = []

    if fps < min_fps:
        fails.append(f"frame rate {fps:.1f} fps < {min_fps:.0f} -- FP "
                     f"stalling")
    if flipped.any():
        frac = float(flipped.mean())
        if frac >= 0.95:
            # A STEADY upside-down stream is not a broken fit: the interface's
            # gate 1b measures this per frame and un-flips the yaw exactly
            # (Ry(pi) about the stem axis is what it was written for), so the
            # planner sees the right heading. It is a fixed 180-deg offset
            # somewhere between FP's estimate and the published TF (the FP
            # debug window and rviz disagree by exactly this). Launchable,
            # but fix the publisher: while every frame is "flipped", a real
            # mirror flip arrives looking NORMAL and the flip counters lie.
            warns.append(f"stream is upside-down on {100*frac:.0f}% of frames "
                         f"(steady) -- gate 1b corrects the yaw, run is OK; "
                         f"fix the FP TF publisher (fixed 180-deg offset)")
        elif frac > 0.5:
            fails.append(f"UPSIDE-DOWN fit on {100*frac:.0f}% of frames with "
                         f"gaps -- block physically flipped, or FP in the "
                         f"mirror minimum: re-seat the block and re-register")
        else:
            fails.append(f"flip FLICKER on {100*frac:.0f}% of frames -- fit "
                         f"oscillating across the mirror: re-register")
    tilt_med = float(np.median(tilt_eff))
    if tilt_med > np.degrees(tilt_max):
        fails.append("fit is off the object entirely (tilt over limit)")
    elif tilt_med > 10.0:
        warns.append(f"median tilt {tilt_med:.0f}d on a flat block -- fit "
                     f"is skewed")

    lo, hi = z_band
    if not (lo <= z_med <= hi):
        fails.append(f"median z {z_med:+.4f} outside [{lo:.3f},{hi:.3f}] -- "
                     f"the run's z gate would reject every frame")
    if z_ptp > 0.012:
        fails.append(f"z wobble {z_ptp*1000:.0f}mm on a static block -- "
                     f"floated-bbox signature: re-register")
    elif z_ptp > 0.006:
        warns.append(f"z wobble {z_ptp*1000:.0f}mm -- bbox may be starting "
                     f"to float")

    if yaw_rate_max > 30.0 or yaw_spread > 8.0:
        fails.append(f"yaw unstable (spread {yaw_spread:.1f}d, rate "
                     f"{yaw_rate_max:.0f}d/s) on a static block -- fit "
                     f"hopping between minima: re-register")
    elif yaw_spread > 3.0:
        warns.append(f"yaw spread {yaw_spread:.1f}d -- marginal")

    if xy_ptp > 0.015:
        fails.append(f"xy jitter {xy_ptp*1000:.0f}mm on a static block")
    elif xy_ptp > 0.006:
        warns.append(f"xy jitter {xy_ptp*1000:.0f}mm -- marginal")

    for m in fails:
        lines.append(f"[preflight] FAIL: {m}")
    for m in warns:
        lines.append(f"[preflight] warn: {m}")
    if fails:
        lines.append("[preflight] VERDICT: FAIL -- not launching. Re-seat "
                     "the block / re-register FP, then retry.")
        return 2, lines
    if warns:
        lines.append("[preflight] VERDICT: WARN -- proceeding, but keep an "
                     "eye on it.")
        return 1, lines
    lines.append("[preflight] VERDICT: PASS -- good to launch.")
    return 0, lines


def preflight_gate(interface, seconds: float = 5.0,
                   verbose: bool = True) -> None:
    """Gate used by `run_real`: check the stream, raise on FAIL.

    Reads the frames/thresholds off the live `Ros2Interface` so the check
    grades exactly what the run would see; its rclpy spin thread is already
    pumping TF, so plain sleeps suffice. A non-hardware interface (mock) has
    no TF buffer and is skipped silently.
    """
    tf_buffer = getattr(interface, "_tf_buffer", None)
    if tf_buffer is None or seconds <= 0.0:
        return
    verdict, lines = collect_and_check(
        tf_buffer, interface._rclpy,
        world_frame=interface._world_frame,
        object_frame=interface._object_frame,
        z_band=interface._object_z_band,
        tilt_max=interface._object_tilt_max,
        seconds=seconds)
    if verbose or verdict:
        for ln in lines:
            print(ln)
    if verdict >= 2:
        raise RuntimeError(
            "FP pre-flight FAILED -- refusing to start the arm on a bad "
            "pose stream (rerun with --preflight 0 to override).")


def main() -> int:
    """Standalone CLI: own node + TF listener, self-pumped."""
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="observation window after the first frame "
                         "(default 5)")
    ap.add_argument("--world-frame", default="world")
    ap.add_argument("--object-frame", default="fp_object_pose")
    ap.add_argument("--z-band", type=float, nargs=2, default=(0.002, 0.020),
                    metavar=("LO", "HI"),
                    help="plausible mesh-origin height band, m (keep in "
                         "sync with the run's object_z_band)")
    ap.add_argument("--tilt-max-deg", type=float, default=30.0,
                    help="off-plane tilt limit, same meaning as the "
                         "interface's object_tilt_max_deg (default 30)")
    ap.add_argument("--min-fps", type=float, default=5.0,
                    help="minimum fresh-frame rate to accept (default 5)")
    args = ap.parse_args()

    import rclpy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415

    if not rclpy.ok():
        rclpy.init()
    node = Node("oim_fp_preflight")
    buf = tf2_ros.Buffer()
    _listener = tf2_ros.TransformListener(buf, node)  # noqa: F841

    verdict, lines = collect_and_check(
        buf, rclpy,
        world_frame=args.world_frame,
        object_frame=args.object_frame,
        z_band=tuple(args.z_band),
        tilt_max=float(np.radians(args.tilt_max_deg)),
        seconds=args.seconds,
        min_fps=args.min_fps,
        pump=lambda: rclpy.spin_once(node, timeout_sec=0.02))
    node.destroy_node()
    for ln in lines:
        print(ln)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
