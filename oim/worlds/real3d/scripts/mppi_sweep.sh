#!/usr/bin/env bash
# flat-MPPI sweep over the REAL-table scenes, through the real3d driver in mock
# mode (examples/pusht/pusht_real.py). One scene per invocation, sequentially:
# MJX/XLA compiles per scene model, so parallel runs fight over the GPU and the
# compilation cache.
#
#   # the run: real-T-scaled costs (needs real_mppi_parity.patch applied)
#   VARIANT=real_costs EXTRA="--config xarm6_real" bash scripts/mppi_sweep.sh
#
#   # decomposition: same sampler, sim cost weights
#   VARIANT=sim_costs bash scripts/mppi_sweep.sh
#
#   # the as-is baseline is the pre-patch commit, not a flag:
#   #   git stash && VARIANT=asis bash scripts/mppi_sweep.sh && git stash pop
#
#   # time one run before committing to a sweep
#   SCENES=open_table_real STEPS=100 VARIANT=smoke \
#       EXTRA="--config xarm6_real" bash scripts/mppi_sweep.sh
#
# Every run is rendered to an mp4 beside its log (RECORD=0 to skip): reading a
# stall off the numbers alone has been misleading more than once, and 30 s of
# video answers "what is the arm actually doing" that a table cannot.
#
# Env: VARIANT SCENES STEPS VEL SEED EXACT(1/0) EXTRA OUTROOT RECORD

set -u
VARIANT=${VARIANT:-asis}
SCENES=${SCENES:-"open_table_real single_obstacle_real box_clutter"}
STEPS=${STEPS:-1000}
VEL=${VEL:-0.5}
SEED=${SEED:-5}
EXACT=${EXACT:-1}
EXTRA=${EXTRA:-}
OUTROOT=${OUTROOT:-oim/results/sweeps}
RECORD=${RECORD:-1}
REPLAY=oim/worlds/real3d/scripts/replay_states.py

ENTRY=examples/pusht/pusht_real.py
[ -f "$ENTRY" ] || { echo "error: run from the repo root (no $ENTRY)" >&2; exit 1; }

OUT="$OUTROOT/$VARIANT"; mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'variant\tscene\tstatus\telapsed_s\tresult_json\tlog\n' > "$MANIFEST"

EXACT_FLAG=""; [ "$EXACT" = "1" ] && EXACT_FLAG="--exact-twist"

echo "=== flat-MPPI sweep '$VARIANT' ==="
echo "scenes: $SCENES"
echo "steps $STEPS  vel $VEL  seed $SEED  exact-twist $EXACT  extra: ${EXTRA:-(none)}"
echo "out   : $OUT"; echo

T_ALL=$(date +%s)
for scene in $SCENES; do
    LOG="$OUT/${scene}.log"
    echo "--- $scene   -> $LOG"
    T0=$(date +%s)
    # shellcheck disable=SC2086
    python "$ENTRY" --mock $EXACT_FLAG --scene "$scene" --algorithm mppi \
        --vel-limit "$VEL" --steps "$STEPS" --seed "$SEED" $EXTRA \
        > "$LOG" 2>&1
    RC=$?; T1=$(date +%s); EL=$((T1 - T0))
    JSON=$(grep -m1 '^saved run to ' "$LOG" | sed 's/^saved run to //')
    if [ $RC -ne 0 ]; then
        STATUS="failed(rc=$RC)"; echo "    FAILED after ${EL}s:"; tail -n 15 "$LOG" | sed 's/^/      /'
    elif [ -z "$JSON" ]; then
        STATUS="no_json"; echo "    finished in ${EL}s but wrote no run JSON"
    else
        STATUS="ok"; echo "    ok in ${EL}s -> $JSON"
        grep -E '^(step|goal reached|.*stuck -- kicked)' "$LOG" | tail -n 1 | sed 's/^/    /'
        echo "    kicks: $(grep -c 'stuck -- kicked' "$LOG")"
    fi
    if [ "$RECORD" = "1" ] && [ -n "$JSON" ] && [ -f "$REPLAY" ]; then
        MP4="$OUT/${scene}.mp4"
        if MUJOCO_GL=${MUJOCO_GL:-egl} python "$REPLAY" "$JSON" \
                --scene "$scene" --mp4 "$MP4" --once >> "$LOG" 2>&1; then
            echo "    video: $MP4"
        else
            echo "    video failed -- see the tail of $LOG"
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$VARIANT" "$scene" "$STATUS" "$EL" "$JSON" "$LOG" >> "$MANIFEST"
    echo
done
echo "=== done in $(( $(date +%s) - T_ALL ))s ==="
echo "manifest: $MANIFEST"
echo "next: python oim/worlds/real3d/scripts/analyze_mppi_runs.py $MANIFEST -o $OUT"
