#!/usr/bin/env bash
# C3+ baseline matrix: four tabletop scenes x five seeds, one folder of results.
#
#   ./c3_sweep.sh
#
# Everything the sweep produces -- run JSONs, per-run plots, mp4s -- ends up
# under oim/results/c3_matrix/, because `oim/run_launch` writes each cell
# through the scene's own example script and those scatter their output
# across results/runs and recordings/. Anything from an earlier C3 sweep is
# moved to c3_archive_<stamp>/ first, so the folder holds one matrix and
# `run_eval` cannot average this run together with an older one.
set -u
export PYTHONUNBUFFERED=1
# Headless: no DISPLAY, and mujoco's default GL backend needs one. Every
# cell that records a video dies in seconds without this.
export MUJOCO_GL=${MUJOCO_GL:-egl}
# C3 factorises an LCS every step through cuSolver, which allocates its own
# GPU workspace. JAX grabs 75% of the card up front by default and leaves
# too little, which fails as `gpusolverDnCreate ... cuSolver internal error`.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
cd "$(dirname "$0")"

STEPS=${STEPS:-1200}
OUT=oim/results/c3_matrix
STAMP=$(date +%Y%m%d_%H%M)
LOG="oim/results/c3_sweep_${STAMP}.log"
mkdir -p oim/results
exec > >(tee -a "$LOG") 2>&1

echo "=== C3 matrix $(date) ==="
echo "branch : $(git rev-parse --abbrev-ref HEAD)  $(git rev-parse --short HEAD)"
echo "steps  : $STEPS   (seeds come from oim/configs/sweeps/c3.yaml)"
echo "out    : $OUT"
echo "log    : $LOG"

# --- move any earlier C3 output out of the way -------------------------
ARCHIVE="oim/results/c3_archive_${STAMP}"
moved=0
for d in oim/results/runs oim/recordings "$OUT"; do
    [ -d "$d" ] || continue
    for f in "$d"/*_c3_*; do
        [ -e "$f" ] || continue
        mkdir -p "$ARCHIVE"
        mv "$f" "$ARCHIVE"/ && moved=$((moved + 1))
    done
done
echo "archived $moved earlier C3 files$([ $moved -gt 0 ] && echo " -> $ARCHIVE")"

# --- the sweep ---------------------------------------------------------

python -m oim.run_launch --config c3 --set steps="$STEPS" || true

# --- collect ------------------------------------------------------------
echo
echo "=== collecting into $OUT ==="
mkdir -p "$OUT"
found=0
for d in oim/results/runs oim/recordings oim/results; do
    [ -d "$d" ] || continue
    for f in "$d"/*_c3_*; do
        [ -e "$f" ] || continue
        case "$f" in "$OUT"/*) continue ;; esac
        mv "$f" "$OUT"/ && found=$((found + 1))
    done
done
echo "collected $found files"
ls "$OUT" | sed 's/^/  /' | head -40
echo "  ..."
echo "json: $(ls "$OUT"/*.json 2>/dev/null | wc -l)   " \
     "png: $(ls "$OUT"/*.png 2>/dev/null | wc -l)   " \
     "mp4: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l)"

# --- score --------------------------------------------------------------
echo
echo "=== scoring ==="
python -m oim.run_eval --runs-dir "$OUT" --group-by task || true

echo
echo "=== done $(date) ==="
echo "log: $LOG"
