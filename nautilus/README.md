# Nautilus

Two things run here: an interactive pod to work in, and a Job that runs a
sweep from [`oim/configs/sweeps/`](../oim/configs/sweeps/) to completion.
`launch.py` renders one of the two templates and submits it; what actually
gets run is decided by the sweep config, not duplicated here.

The image holds **dependencies only**; the pod clones this repo into
`/workspace` at start, so a code change needs a relaunch and not a rebuild.
Change `pyproject.toml` or `uv.lock` and you do have to rebuild the image.

```bash
python nautilus/launch.py pod                        # a GPU and a shell
python nautilus/launch.py job                        # the whole ablation
python nautilus/launch.py job --shard task           # one Job per task
python nautilus/launch.py job --only algorithm=mppi  # a slice of it
python nautilus/launch.py job --dry-run              # print, submit nothing

# pin the GPU model, or the code (either command)
python nautilus/launch.py pod --gpu-type NVIDIA-GeForce-RTX-4090
python nautilus/launch.py job --ref my-branch
```

| File | |
| --- | --- |
| `launch.py` | renders a template and `kubectl apply`s it |
| `pod.yaml` | interactive pod: image, one GPU, the PVC, `sleep` |
| `job.yaml` | batch Job: same, but runs the sweep and exits |
| `persistant_storage.yaml` | the PVC itself; apply once |
| `legacy/` | the PLS-VLA scripts this replaced, kept for reference |

| Flag | |
| --- | --- |
| `--config` | sweep config under `oim/configs/sweeps/` (default `ablation`) |
| `--shard AXIS` | one Job per value of that axis, each with the matching `--only` |
| `--only K=V` | passed to `run_launch --only`; repeatable |
| `--set K=V` | passed to `run_launch --set`; repeatable |
| `--image` | default `nikolaraicevic2001/contact-mpc:latest` |
| `--gpu`, `--cpu`, `--memory` | override the template; unset keeps its values |
| `--gpu-type MODEL` | pin the GPU model (`nvidia.com/gpu.product`); repeatable, replaces the template's list |
| `--repo`, `--ref` | what to clone, and the branch/tag/SHA (default `main`) |
| `--name`, `--dry-run` | |

The image is built from [`docker/`](../docker/):

```bash
./docker/build.sh               # contact-mpc:latest, build and push
./docker/build.sh v2 --no-push  # build :v2 only
./docker/run.sh                 # shell in that image, repo bind-mounted
```

Results go to the PVC, not into the image: the container symlinks
`oim/results` and `oim/recordings` to `/nikola-volume/oim/<name>/`
before starting, so a finished Job — or a pod you exec into and run by
hand — leaves its run files behind. The JAX compilation cache is shared
at `/nikola-volume/oim/jax-cache` — every sweep cell is its own process,
so without it each one recompiles from scratch.

The clone is shallow and authenticates with `GIT_ACCESS_TOKEN`, read from
the `github-token-nikola` secret (`optional`, so a namespace without it
still starts the pod). `git pull` inside the pod works too.

Both templates carry a `nodeAffinity` listing the GPU models a run may land
on, floored at 24 GB — `--warp` disables JAX preallocation so MuJoCo Warp
can build its CUDA graphs, and 16 GB is not enough for the xArm6 scene.
`--gpu-type` replaces that list, so it reaches a model the template does
not name; `gpu_summary.txt` is where the names come from.

`--shard task` filters on `script=`, not `task=`, because a `task:` entry
is the mapping `{script: open_table}` and `--only` matches flat keys.
`launch.py` handles that; the shards are exact and exhaustive (3 × 470 =
1410 cells for the current `ablation.yaml`).

The image puts this repo at `/workspace`, which `launch.py` hardcodes as
`IMAGE_WORKDIR`; `docker/Dockerfile` must keep that `WORKDIR`.
