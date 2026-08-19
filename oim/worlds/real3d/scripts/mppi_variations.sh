#!/usr/bin/env bash
# Run several single-variable variants of one scene, unattended.
#
# "Change one variable at a time" is a rule about *interpretation*, not about
# doing them one at a time by hand: each row below is its own run with exactly
# one thing changed from the baseline, so they can all be launched together and
# read against each other afterwards. The mock is deterministic under jax with a
# fixed seed, so the comparison is exact.
#
#   PROFILE=near_goal SEED=1 SCENE=open_table_real \
#       bash oim/worlds/real3d/scripts/mppi_variations.sh
#   PROFILE=obstacle SCENE=single_obstacle_real \
#       bash oim/worlds/real3d/scripts/mppi_variations.sh
#
# Env: PROFILE SCENE STEPS VEL SEED EXACT(1/0) BASE OUTROOT

set -u
PROFILE=${PROFILE:-near_goal}
SCENE=${SCENE:-open_table_real}
STEPS=${STEPS:-800}
VEL=${VEL:-0.5}
SEED=${SEED:-1}
EXACT=${EXACT:-1}
BASE=${BASE:---config xarm6_real}          # applied to every variant
OUTROOT=${OUTROOT:-oim/results/variations}

# The endgame profile. Diagnosis: every seed freezes bit-exact at
# pos_err 0.07-0.10 with theta already inside tolerance, so the question is
# what stops the last few centimetres of translation -- not how the run gets
# there. H9* test whether the near-goal heading weight is what forbids it.
NEAR_GOAL=(
  "base|"
  "H9a_rampoff|--cost q_theta_ramp=1.0"
  "H9b_qth25|--cost q_theta=25 --cost qf_theta=25"
  "H9c_qth16|--cost q_theta=16 --cost qf_theta=16"
  "H10_vel10|--vel-limit 1.0"
  "H6_qramp|--cost q_ramp_per_step=0.002 --cost q_ramp_max=12.0"
  "H8_fade025|--cost shaping_fade_dist=0.25"
  "H2_horizon32|--horizon 32"
  "H7_samples256|--num-samples 256"
)

# The detour profile. Diagnosis: the block parks beside an obstacle and the
# obstacle term stays pinned high for hundreds of steps, so the question is
# what would make a detour -- which costs more now and pays later -- win.
OBSTACLE=(
  "base|"
  "H1a_decay020|--cost obstacle_decay=0.02"
  "H1b_decay035|--cost obstacle_decay=0.035"
  "H5_pusherobs|--cost pusher_obstacle_weight=0.5"
  "H6_qramp|--cost q_ramp_per_step=0.002 --cost q_ramp_max=12.0"
  "H2_horizon32|--horizon 32"
  "H3_wobs40|--cost w_obstacle=40"
  "H4_gamma45|--cost gamma0_deg=45"
  "H7_samples256|--num-samples 256"
)

case "$PROFILE" in
  near_goal) VARIANTS=("${NEAR_GOAL[@]}") ;;
  obstacle)  VARIANTS=("${OBSTACLE[@]}") ;;
  *) echo "error: PROFILE must be near_goal or obstacle" >&2; exit 1 ;;
esac

ENTRY=examples/pusht/pusht_real.py
[ -f "$ENTRY" ] || { echo "error: run from the repo root (no $ENTRY)" >&2; exit 1; }

OUT="$OUTROOT/${PROFILE}_${SCENE}_seed${SEED}"; mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'variant\tscene\tstatus\telapsed_s\tresult_json\tlog\n' > "$MANIFEST"
EXACT_FLAG=""; [ "$EXACT" = "1" ] && EXACT_FLAG="--exact-twist"

echo "=== '$PROFILE' variations on '$SCENE', seed $SEED (${#VARIANTS[@]} runs) ==="
echo "base flags: $BASE   steps $STEPS  vel $VEL  exact $EXACT"
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
        STATUS="ok"; echo "    ok in ${EL}s"
        grep -E '^(step|goal reached)' "$LOG" | tail -n 1 | sed 's/^/    /'
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$SCENE" "$STATUS" "$EL" "$JSON" "$LOG" >> "$MANIFEST"
    echo
done
echo "=== done in $(( $(date +%s) - T_ALL ))s ==="
echo "manifest: $MANIFEST"
echo "next: python oim/worlds/real3d/scripts/analyze_mppi_runs.py $MANIFEST -o $OUT"
