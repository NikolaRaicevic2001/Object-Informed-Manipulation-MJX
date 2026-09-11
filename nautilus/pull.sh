#!/usr/bin/env bash
# Pull result directories off the Nautilus PVC, resumably.
#
#   ./nautilus/pull.sh oim-mppi-open-table-rtx4090        # sync (default)
#   ./nautilus/pull.sh --runs-only 'oim-mppi-*'           # JSONs only, no mp4s
#   ./nautilus/pull.sh --check 'oim-*'                    # report, copy nothing
#   ./nautilus/pull.sh --list                             # what is on the PVC
#
# Why not `kubectl cp`: it streams one tar through the API server and has no
# resume, so a mid-transfer "connection reset by peer" loses the whole tree
# and the retry starts over -- which is why a big directory never lands. It
# also aborts on "file changed as we read it" while a Job is still writing.
#
# This compares a remote manifest (path + size) against the local one, fetches
# ONLY what is missing or truncated, and does it in small batches so a reset
# costs one batch instead of everything. Re-run it until it says complete;
# each run picks up where the last left off.
set -uo pipefail

NS="${OIM_NS:-erl-ucsd}"
REMOTE_ROOT="/nikola-volume/oim"
CONTAINER="gpu-container"
DEST="${HOME}/Downloads"
BATCH=40           # files per tar; small enough that a reset is cheap
RETRIES=4
RUNS_ONLY=0; POD=""; LIST=0; CHECK=0

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)      DEST="$2"; shift 2 ;;
    --pod)       POD="$2"; shift 2 ;;
    --ns)        NS="$2"; shift 2 ;;
    --runs-only) RUNS_ONLY=1; shift ;;
    --batch)     BATCH="$2"; shift 2 ;;
    --retries)   RETRIES="$2"; shift 2 ;;
    --check)     CHECK=1; shift ;;
    --list)      LIST=1; shift ;;
    -h|--help)   usage 0 ;;
    -*)          echo "unknown flag: $1" >&2; usage 1 ;;
    *)           break ;;
  esac
done

# Any pod mounting the PVC will do -- it is ReadWriteMany and every oim pod
# mounts it at the same path, so this need not be the pod that produced the
# data (that Job may have finished and gone).
if [[ -z "$POD" ]]; then
  POD=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null \
        | awk '$3=="Running" && $1 ~ /^oim-/ {print $1; exit}')
fi
[[ -z "$POD" ]] && { echo "no Running oim-* pod in '$NS'; pass --pod NAME" >&2; exit 1; }
echo "pod: $POD   dest: $DEST"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# kubectl's OWN stderr used to be discarded here along with the remote
# command's, so an unreachable API server arrived as an empty result and was
# reported as "nothing on the PVC matches" -- the one diagnosis it is not.
# Seen 2026-09-11: intermittent `dial tcp 67.58.53.148:443: i/o timeout`
# against directories that were plainly there.
#
# Each remote command below silences its own EXPECTED noise inside the remote
# shell (a glob matching nothing, a directory that is not there), so whatever
# reaches this file is kubectl's and worth printing. A file, not a variable:
# `remote` is called inside `$( )` and in pipelines, both subshells, whose
# assignments the caller never sees.
ERR="$tmp/kubectl.err"
remote() {
  local rc
  kubectl -n "$NS" exec "$POD" -c "$CONTAINER" -- sh -c "$1" 2>"$ERR"
  rc=$?
  [[ -s "$ERR" ]] && sed 's/^/      /' "$ERR" >&2
  return $rc
}

# True when the last `remote` never reached the pod, as opposed to running
# there and exiting non-zero (which is how "no match" legitimately arrives).
unreachable() { [[ -s "$ERR" ]]; }

if [[ $LIST -eq 1 ]]; then remote "ls -1 $REMOTE_ROOT"; exit 0; fi
[[ $# -eq 0 ]] && { echo "no directories given" >&2; usage 1; }

DIRS=$(remote "cd $REMOTE_ROOT 2>/dev/null && ls -d $* 2>/dev/null")
if [[ -z "$DIRS" ]]; then
  if unreachable; then
    echo "could not reach $POD (see above) -- the PVC was never listed" >&2
  else
    echo "nothing on the PVC matches: $*" >&2
  fi
  exit 1
fi

SUB=""; [[ $RUNS_ONLY -eq 1 ]] && SUB="/runs"
mkdir -p "$DEST"
status=0

for d in $DIRS; do
  rel="$d$SUB"
  # Manifest: "<size> <path relative to REMOTE_ROOT>", one file per line.
  remote "cd $REMOTE_ROOT && test -d '$rel' && find '$rel' -type f -printf '%s %p\n'" \
    | sed 's/[[:space:]]*$//' > "$tmp/remote.txt"
  # Same distinction as above: an empty manifest means an empty directory
  # only if we actually got to look at one.
  if unreachable; then
    echo "  $d: could not reach $POD -- skipped" >&2; status=1; continue
  fi
  want=$(wc -l < "$tmp/remote.txt")
  if [[ "$want" -eq 0 ]]; then echo "  $d: nothing under ${SUB:-/}"; continue; fi

  # A local file counts as present only if its SIZE matches -- a transfer cut
  # mid-file leaves a short one with the right name, which a count-only check
  # would happily accept.
  : > "$tmp/missing.txt"
  while read -r sz path; do
    [[ -z "$path" ]] && continue
    lsz=$(stat -c %s "$DEST/$path" 2>/dev/null || echo -1)
    [[ "$lsz" != "$sz" ]] && printf '%s\n' "$path" >> "$tmp/missing.txt"
  done < "$tmp/remote.txt"
  miss=$(wc -l < "$tmp/missing.txt")

  if [[ "$miss" -eq 0 ]]; then echo "  $d: complete ($want files)"; continue; fi
  if [[ $CHECK -eq 1 ]]; then
    echo "  $d: $((want-miss))/$want present, $miss missing or truncated"
    sed 's/^/      /' "$tmp/missing.txt" | head -5
    [[ "$miss" -gt 5 ]] && echo "      ... and $((miss-5)) more"
    status=1; continue
  fi

  echo "  $d: $((want-miss))/$want present, fetching $miss in batches of $BATCH"
  # `split` overwrites batch.aa, batch.ab ... but leaves any batch file the
  # PREVIOUS directory made and this one does not reach, which the glob below
  # would then re-fetch and count as progress.
  rm -f "$tmp"/batch.*
  split -l "$BATCH" "$tmp/missing.txt" "$tmp/batch."
  fetched=0
  for b in "$tmp"/batch.*; do
    for attempt in $(seq 1 "$RETRIES"); do
      # shellcheck disable=SC2046  # word splitting is the point: one arg per file
      if kubectl -n "$NS" exec "$POD" -c "$CONTAINER" -- \
           tar cf - --warning=no-file-changed -C "$REMOTE_ROOT" $(tr '\n' ' ' < "$b") \
           2>"$ERR" | tar xf - -C "$DEST" 2>/dev/null; then
        fetched=$((fetched + $(wc -l < "$b"))); break
      fi
      sleep $((attempt * 2))          # back off; resets cluster in bursts
    done
  done
  echo "      fetched $fetched/$miss  ->  $DEST/$rel"
  if [[ "$fetched" -lt "$miss" ]]; then
    # Why it stopped, not just that it did -- the last attempt's kubectl
    # stderr, which this loop used to discard.
    [[ -s "$ERR" ]] && sed 's/^/      /' "$ERR" >&2
    echo "      INCOMPLETE -- re-run to resume" >&2; status=1
  fi
done
exit $status
