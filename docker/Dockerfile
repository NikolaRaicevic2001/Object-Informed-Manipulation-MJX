# Dependencies only. Build: ./docker/build.sh
#
# The repo is not copied in -- `nautilus/launch.py` clones it into
# /workspace at pod start:
#
#     change oim/ or examples/  ->  relaunch
#     change pyproject/uv.lock  ->  rebuild and push this image
#
# CUDA comes from `jax[cuda13]`'s wheels; the base adds only EGL for
# MuJoCo's offscreen renderer, ffmpeg for `--record`, and git for the clone.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        ca-certificates \
        git \
        ffmpeg \
        libegl1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

# Apart from the code, so the clone cannot shadow it. README.md is here
# only because pyproject.toml's `readme =` points at it; the hash lets
# `launch.py` warn when the clone's uv.lock has moved on.
WORKDIR /opt/oim
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project \
    && sha256sum uv.lock | cut -d' ' -f1 > /opt/oim/uv.lock.sha256

# `oim` is imported off PYTHONPATH, never installed: pure Python, no entry
# points. WORKDIR creates /workspace empty, which the clone needs.
WORKDIR /workspace

ENV VIRTUAL_ENV=/opt/oim/.venv \
    PATH=/opt/oim/.venv/bin:$PATH \
    PYTHONPATH=/workspace \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    # Without `graphics` the runtime injects no EGL vendor ICD.
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONUNBUFFERED=1

CMD ["bash"]
