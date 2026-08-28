# Nautilus

An interactive pod to work in, and a Job that runs a sweep from
[`oim/configs/sweeps/`](../oim/configs/sweeps/) to completion. `launch.py`
renders one of the two templates and `kubectl apply`s it.

The image holds dependencies only; the pod clones this repo into
`/workspace` at start, so a code change is a relaunch, not a rebuild.

```bash
python nautilus/launch.py pod                        # a GPU and a shell
python nautilus/launch.py pod --name my-pod          # custom name
python nautilus/launch.py job                        # the whole ablation
python nautilus/launch.py job --shard task           # one Job per task
python nautilus/launch.py job --only algorithm=mppi  # a slice of it
python nautilus/launch.py job --ref my-branch        # code other than main
python nautilus/launch.py job --dry-run              # print, submit nothing

kubectl exec -it <name> -- /bin/bash
```

| File | |
| --- | --- |
| `launch.py` | renders a template and submits it |
| `templates/pod.yaml` | interactive pod: image, one GPU, the PVC, `sleep` |
| `templates/job.yaml` | batch Job: same, but runs the sweep and exits |
| `templates/persistent_storage.yaml` | the PVC itself; apply once |
| `gpu_summary.py` | writes `gpu_summary.txt`; needs a working `kubectl` |
| `gpu_summary.txt` | cluster GPU snapshot, where `--gpu-type` names come from |

| Flag | |
| --- | --- |
| `--config` | sweep config under `oim/configs/sweeps/` (default `ablation`) |
| `--shard AXIS` | one Job per value of that axis, each with the matching `--only` |
| `--only K=V` | passed to `run_launch --only`; repeatable |
| `--set K=V` | passed to `run_launch --set`; repeatable |
| `--repo`, `--ref` | what to clone; `--ref` takes a branch/tag/SHA, default is the repo's default branch |
| `--gpu-type MODEL` | pin the GPU model; repeatable, replaces the template's list |
| `--gpu`, `--cpu`, `--memory` | override the template; unset keeps its values |
| `--image` | default `nikolaraicevic2001/contact-mpc:latest` |
| `--name` | resource name and results directory; default is generated |
| `--dry-run` | print the manifests, submit nothing |

## Results

Everything lands on the PVC, one directory per pod or Job.

| Path | |
| --- | --- |
| `/nikola-volume/oim/<name>/runs` | run files, symlinked as `oim/results/runs` |
| `/nikola-volume/oim/<name>/recordings` | plots and mp4s, symlinked as `oim/recordings` |
| `/nikola-volume/oim/jax-cache` | shared compilation cache; every sweep cell is its own process |

The symlinks are made before anything runs, so `oim` writes beside its own
package exactly as it does locally, and a run typed by hand after
`kubectl exec` is kept too.

Pulling a run down. Prefer `runs/` alone: it is what `run_eval` reads, and
it skips the recordings a still-running Job is writing.

```bash
kubectl exec <pod> -- ls /nikola-volume/oim        # what is there
kubectl cp <pod>:/nikola-volume/oim/<name>/runs ./oim/results/<name>
uv run python -m oim.run_eval --runs-dir ./oim/results/<name>
```

Everything, including recordings:

```bash
kubectl exec <pod> -- sh -c \
  'tar cf - -C /nikola-volume/oim <name> 2>/dev/null; exit 0' \
  | tar xf - -C ./oim/results
```

`kubectl cp` is tar over an exec stream, so a file changing mid-read (a Job
still writing an mp4) kills it with `unexpected EOF`. The `exit 0` above is
what survives that; the file in flight arrives truncated.
