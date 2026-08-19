#!/usr/bin/env bash
# Run several single-variable variants of one scene, unattended.
#
# "Change one variable at a time" is a rule about *interpretation*, not about
# doing them one evening at a time: each line below is its own run with exactly
# one thing changed from the baseline, so they can all be launched at once and
# read against each other afterwards. The mock is deterministic under jax with a
# fixed seed, so the comparison is exact.
#
# Edit the VARIANTS list, then:
#   SCENE=single_obstacle_real bash oim/worlds/real3d/scripts/mppi_variations.sh
#
# Env: SCENE STEPS VEL SEED EXACT(1/0) BASE OUTROOT

set -u
SCENE=${SCENE:-single_obstacle_real}
STEPS=${STEPS:-1000}
VEL=${VEL:-0.5}
SEED=${SEED:-5}
EXACT=${EXACT:-1}
BASE=${BASE:---config xarm6_real}          # applied to every variant
OUTROOT=${OUTROOT:-oim/results/variations}

# label|extra flags.  The first row must be the untouched baseline.
VARIANTS=(
  "base|"
  "H1a_decay020|--cost obstacle_decay=0.02"
  "H1b_decay035|--cost obstacle_decay=0.035"
  "H2_horizon32|--horizon 32"
  "H3_wobs40|--cost w_obstacle=40"
  "H4_gamma45|--cost gamma0_deg=45"
  "H5_pusherobs|--cost pusher_obstacle_weight=0.5"
  "H6_qramp|--cost q_ramp_per_step=0.002 --cost q_ramp_max=12.0"
  "H7_samples256|--num-samples 256"
  "H8_fade025|--cost shaping_fade_dist=0.25"
)

ENTRY=examples/pusht/pusht_real.py
[ -f "$ENTRY" ] || { echo "error: run from the repo root (no $ENTRY)" >&2; exit 1; }

OUT="$OUTROOT/$SCENE"; mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'variant\tscene\tstatus\telapsed_s\tresult_json\tlog\n' > "$MANIFEST"
EXACT_FLAG=""; [ "$EXACT" = "1" ] && EXACT_FLAG="--exact-twist"

echo "=== variations on '$SCENE' (${#VARIANTS[@]} runs) ==="
echo "base flags: $BASE   steps $STEPS  vel $VEL  seed $SEED  exact $EXACT"
echo "out: $OUT"; echo

T_ALL=$(date +%s)
for row in "${VARIANTS[@]}"; do
    label="${row%%|*}"; extra="${row#*|}"
    LOG="$OUT/${label}.log"
    echo "--- $label   ${extra:-(baseline)}"
    T0=$(date +%s)
    # shellcheck disable=SC2086
    python "$ENTRY" --mock $EXACT_FLAG --scene "$SCENE" --algorithm mppi \
        --vel-limit "$VEL" --steps "$STEPS" --seed "$SEED" $BASE $extra \
        > "$LOG" 2>&1
    RC=$?; EL=$(( $(date +%s) - T0 ))
    JSON=$(grep -m1 '^saved run to ' "$LOG" | sed 's/^saved run to //')
    if [ $RC -ne 0 ]; then
        STATUS="failed(rc=$RC)"; echo "    FAILED after ${EL}s:"; tail -n 12 "$LOG" | sed 's/^/      /'
    elif [ -z "$JSON" ]; then
        STATUS="no_json"; echo "    ${EL}s, no run JSON"
    else
        STATUS="ok"
        echo "    ok in ${EL}s"
        grep -E '^(step|goal reached)' "$LOG" | tail -n 1 | sed 's/^/    /'
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$SCENE" "$STATUS" "$EL" "$JSON" "$LOG" >> "$MANIFEST"
    echo
done
echo "=== done in $(( $(date +%s) - T_ALL ))s ==="
echo "manifest: $MANIFEST"
echo "next: python oim/worlds/real3d/scripts/analyze_mppi_runs.py $MANIFEST -o $OUT"
