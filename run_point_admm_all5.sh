#!/bin/bash
cd ~/shahid_ws/Object-Informed-Manipulation-MJX
UV=~/.local/bin/uv
export MUJOCO_GL=egl
export PATH="$HOME/.local/bin:$PATH"

run() {
  echo "=== $(date) : $* ==="
  $UV run python -u "$@"
  echo "=== exit code $? ==="
}

for scene in open_table single_obstacle shelf_gap ycb_clutter icra_sign; do
  run examples/${scene}.py --robot point --warp --record --show-samples --show-optimal --horizon 35 --samples 96 admm --headless --seed 5 --steps 1500
done

echo "=== ALL DONE $(date) ==="
