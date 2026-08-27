#!/usr/bin/env bash
# Run the image locally with this repo bind-mounted at /workspace -- the
# image has no code, so the mount is what supplies it.
#
#   ./docker/run.sh                              # shell
#   ./docker/run.sh python -m oim.run_launch     # one command, then exit
#   ./docker/run.sh --no-gpu pytest tests/test_scenes.py
#
# `python` is already the image's venv, so no `uv run`.
set -euo pipefail

IMAGE="${OIM_IMAGE:-nikolaraicevic2001/contact-mpc}"
TAG="${OIM_TAG:-latest}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Same path oim/__init__.py picks natively, so host and container share it.
CACHE="${OIM_JAX_CACHE_DIR:-$HOME/.cache/oim/jax}"

gpu_args=(--gpus all)
cpu_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-gpu) gpu_args=(); cpu_only=1; shift ;;
        --tag)    TAG="$2"; shift 2 ;;
        *)        break ;;
    esac
done

args=(--rm)
# -t fails without a terminal (pipe, CI).
[[ -t 0 ]] && args+=(-it)
args+=("${gpu_args[@]}")
[[ "$cpu_only" == 1 ]] && args+=(-e JAX_PLATFORMS=cpu)

mkdir -p "$CACHE"
args+=(-v "$CACHE:/jax-cache" -e OIM_JAX_CACHE_DIR=/jax-cache)
# The anonymous volume shadows the host's .venv, so a stray `uv` in the
# container (which runs as root) cannot rewrite it and leave it root-owned.
args+=(-v "$REPO:/workspace:z" -v /workspace/.venv)

exec docker run "${args[@]}" "${IMAGE}:${TAG}" "${@:-bash}"
