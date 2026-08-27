# Nautilus

Two things run here: an interactive pod to work in, and a Job that runs a
sweep from [`oim/configs/sweeps/`](../oim/configs/sweeps/) to completion.
`launch.py` renders one of the two templates and submits it; what actually
gets run is decided by the sweep config, not duplicated here.

```bash
python nautilus/launch.py pod                        # a GPU and a shell
python nautilus/launch.py job                        # the whole ablation
python nautilus/launch.py job --shard task           # one Job per task
python nautilus/launch.py job --only algorithm=mppi  # a slice of it
python nautilus/launch.py job --dry-run              # print, submit nothing
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
| `--name`, `--dry-run` | |

The image is built from [`docker/`](../docker/):

```bash
./docker/build.sh               # contact-mpc:latest, build and push
./docker/build.sh v2 --no-push  # build :v2 only
./docker/run.sh                 # shell in that image, repo bind-mounted
```

Results go to the PVC, not into the image: the container symlinks
`oim/results` and `oim/recordings` to `/nikola-volume/oim/<job-name>/`
before starting, so a finished Job leaves its run files behind. The JAX
compilation cache is shared at `/nikola-volume/oim/jax-cache` — every
sweep cell is its own process, so without it each one recompiles from
scratch.

`--shard task` filters on `script=`, not `task=`, because a `task:` entry
is the mapping `{script: open_table}` and `--only` matches flat keys.
`launch.py` handles that; the shards are exact and exhaustive (3 × 470 =
1410 cells for the current `ablation.yaml`).

The image puts this repo at `/workspace`, which `launch.py` hardcodes as
`IMAGE_WORKDIR`; `docker/Dockerfile` must keep that `WORKDIR`.
