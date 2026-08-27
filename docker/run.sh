#!/usr/bin/env bash
# Run the sweep image locally.
#
#   ./docker/run.sh                                    # interactive shell
#   ./docker/run.sh uv run python -m oim.run_launch    # one command, then exit
#   ./docker/run.sh --baked                            # image's code, not yours
#   ./docker/run.sh --no-gpu pytest tests/test_scenes.py
#
# Env: OIM_IMAGE, OIM_TAG, OIM_JAX_CACHE_DIR.
set -euo pipefail

IMAGE="${OIM_IMAGE:-nikolaraicevic2001/contact-mpc}"
TAG="${OIM_TAG:-latest}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Same path oim/__init__.py picks natively, so host and container share it.
CACHE="${OIM_JAX_CACHE_DIR:-$HOME/.cache/oim/jax}"

mount_repo=1
cpu_only=0
gpu_args=(--gpus all)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baked)  mount_repo=0; shift ;;
        --no-gpu) gpu_args=(); cpu_only=1; shift ;;
        --tag)    TAG="$2"; shift 2 ;;
        --)       shift; break ;;
        *)        break ;;
    esac
done

args=(--rm)
# -t fails without a terminal (pipe, CI).
[[ -t 0 ]] && args+=(-it)
args+=("${gpu_args[@]}")
# Explicit CPU. Does not silence jaxlib's cuInit traceback -- it probes the
# CUDA plugin at import regardless; the device list after it is the answer.
[[ "$cpu_only" == 1 ]] && args+=(-e JAX_PLATFORMS=cpu)

mkdir -p "$CACHE"
args+=(-v "$CACHE:/jax-cache" -e OIM_JAX_CACHE_DIR=/jax-cache)

if [[ "$mount_repo" == 1 ]]; then
    # Anonymous volume on .venv: the bind mount would otherwise hide the
    # one the image built, and uv would rebuild it on every start.
    args+=(-v "$REPO:/workspace:z" -v /workspace/.venv)
fi

exec docker run "${args[@]}" "${IMAGE}:${TAG}" "${@:-bash}"
