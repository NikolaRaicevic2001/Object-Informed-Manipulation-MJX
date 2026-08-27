# Image for running `oim` sweeps. Build from the repo root: ./docker/build.sh
#
# CUDA comes from `jax[cuda13]`'s own wheels, so the base supplies only the
# driver interface, EGL for MuJoCo's offscreen renderer, and ffmpeg for
# `--record`.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        ca-certificates \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

WORKDIR /workspace

# Deps before source, so a code edit does not re-resolve them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    # `graphics`: without it the runtime injects no EGL vendor ICD and the
    # offscreen renderer finds no device.
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    # nautilus/job.yaml points this at the PVC so the cache outlives the pod.
    OIM_JAX_CACHE_DIR=/workspace/.jax-cache \
    PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "-m", "oim.run_launch", "--config", "ablation"]
