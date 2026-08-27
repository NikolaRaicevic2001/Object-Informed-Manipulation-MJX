#!/usr/bin/env bash
# Build and push the sweep image.
#
#   ./docker/build.sh                 # :latest, build and push
#   ./docker/build.sh v2 --no-push    # build only
set -euo pipefail

IMAGE="${OIM_IMAGE:-nikolaraicevic2001/contact-mpc}"
TAG="${1:-latest}"
PUSH=1
[[ "${2:-}" == "--no-push" ]] && PUSH=0

# Context is the repo root, where .dockerignore lives.
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t "${IMAGE}:${TAG}" .
echo "built ${IMAGE}:${TAG}"

if [[ "${PUSH}" == 1 ]]; then
    docker push "${IMAGE}:${TAG}"
    echo "pushed ${IMAGE}:${TAG}"
fi
