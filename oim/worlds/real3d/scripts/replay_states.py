"""Replay a saved states JSON in a MuJoCo viewer (or render it to mp4).

Both the sim (`examples/clutter.py`) and the mock/real driver
(`examples/pusht/pusht_real.py`) write the same states file: `dynamic.qpos` holds
the full MuJoCo configuration at every frame. This script loads the matching
scene model and plays those frames back, so you can *watch* a run that was
logged headless (e.g. over SSH on the lab).

Two output modes:

    # Interactive window -- run locally on the Mac after scp-ing the JSON.
    python oim/worlds/real3d/scripts/replay_states.py \
        RUN_states.json --scene box_clutter

    # Offscreen render to mp4 -- run on the lab (no display needed, uses the
    # GPU's EGL context), then scp the mp4 to watch it.
    MUJOCO_GL=egl python oim/worlds/real3d/scripts/replay_states.py \
        RUN_states.json --scene box_clutter --mp4 run.mp4
Only mujoco + numpy are needed (no JAX/GPU for playback), so the interactive
path runs fine on a laptop.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np

# The oim package dir. Resolved from __file__ so playback needs no oim/JAX
# install -- just mujoco + numpy, which run fine on a laptop with no pixi
# env. Four levels up: scripts/ -> real3d/ -> worlds/ -> oim/. Counting
# levels breaks silently when the file moves, so it is asserted below.
_OIM = str(Path(__file__).resolve().parents[3])
assert os.path.isdir(os.path.join(_OIM, "models")), (
    f"expected the oim package dir, got {_OIM}"
)

# Scene name -> (scene MJCF, base_pos, base_yaw_deg). The base placement is NOT
# in the XML: PushT mutates body_pos/body_quat of xarm6_link_base in code
# (oim/tasks/pusht.py), so we must reproduce it here or the whole arm is drawn
# offset from where it really was -- e.g. clutter's base at (0.2, 0.75).
# Real-table scenes all share the arm base at the world origin (xarm_device
# frame) and live at models/xarm6_pusht_tabletop_real/{name}.xml -- to add a
# new real scene, just put its name here.
_REAL_SCENES = ["box_clutter", "open_table_real", "single_obstacle_real"]

SCENES = {
    "clutter": (os.path.join(_OIM, "models/xarm6_pusht_clutter/scene.xml"),
                (0.2, 0.75), -90.0),
    **{n: (os.path.join(_OIM, f"models/xarm6_pusht_tabletop_real/{n}.xml"),
           (0.0, 0.0), 0.0) for n in _REAL_SCENES},
}


def place_base(model, base_pos, base_yaw_deg):
    """Reproduce PushT's in-code base placement (pusht.py lines ~195-203)."""
    bid = model.body("xarm6_link_base").id
    model.body_pos[bid] = [base_pos[0], base_pos[1], 0.0]
    yaw = math.radians(base_yaw_deg)
    model.body_quat[bid] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def load_frames(states_path):
    """Return (qpos_frames [N, nq], control_dt) from a states JSON."""
    with open(states_path) as f:
        payload = json.load(f)
    qpos = np.asarray(payload["dynamic"]["qpos"], dtype=float)
    control_dt = float(payload["static"].get("control_dt", 0.05))
    return qpos, control_dt


def replay_interactive(model, frames, control_dt, speed, loop=True):
    """Play frames in a passive viewer window (needs a display)."""
    import mujoco.viewer  # noqa: PLC0415  (only needed for the window path)

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        first = True
        while viewer.is_running() and (loop or first):
            first = False
            for frame in frames:
                if not viewer.is_running():
                    break
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(control_dt / max(speed, 1e-6))
            time.sleep(0.5)  # brief pause, then loop the trajectory again


def replay_mp4(model, frames, control_dt, speed, out_path):
    """Render frames offscreen to an mp4 (needs MUJOCO_GL=egl/osmesa)."""
    import imageio.v2 as imageio  # noqa: PLC0415

    data = mujoco.MjData(model)
    fps = max(1, round(speed / control_dt))
    # The offscreen framebuffer defaults to 640x480; enlarge it to match the
    # render size, or mujoco.Renderer raises "Image width > framebuffer width".
    width, height = 1280, 720
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    with imageio.get_writer(out_path, fps=fps, macro_block_size=8) as writer:
        for frame in frames:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            writer.append_data(renderer.render())
    renderer.close()
    print(f"wrote {out_path} ({len(frames)} frames @ {fps} fps)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("states", help="path to a *_states_*.json file")
    p.add_argument("--scene", choices=sorted(SCENES),
                   help="scene the run used (picks the model + base placement)")
    p.add_argument("--model", help="explicit scene .xml (overrides --scene; "
                                   "no base placement applied)")
    p.add_argument("--mp4", help="render to this mp4 instead of a window")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (2.0 = twice real time)")
    p.add_argument("--start", type=int, default=0, help="first frame to play")
    p.add_argument("--end", type=int, default=None,
                   help="stop before this frame (max frame); default = all")
    p.add_argument("--once", action="store_true",
                   help="play once instead of looping")
    args = p.parse_args()

    if args.model:
        model = mujoco.MjModel.from_xml_path(args.model)
    elif args.scene:
        xml, base_pos, base_yaw = SCENES[args.scene]
        model = mujoco.MjModel.from_xml_path(xml)
        place_base(model, base_pos, base_yaw)  # match PushT, or the arm is offset
    else:
        p.error(f"give --scene {{{','.join(sorted(SCENES))}}} or --model path.xml")

    frames, control_dt = load_frames(args.states)
    if frames.shape[1] != model.nq:
        raise SystemExit(
            f"qpos width {frames.shape[1]} != model nq {model.nq}; "
            f"wrong --scene for this states file?"
        )
    total = len(frames)
    frames = frames[args.start:args.end]
    print(f"{total} frames total, playing [{args.start}:{args.end}] "
          f"= {len(frames)} frames, nq={model.nq}, control_dt={control_dt}s")

    if args.mp4:
        replay_mp4(model, frames, control_dt, args.speed, args.mp4)
    else:
        replay_interactive(model, frames, control_dt, args.speed,
                           loop=not args.once)


if __name__ == "__main__":
    main()
