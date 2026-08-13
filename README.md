# Object-Informed Manipulation MJX

GPU-accelerated planning for **non-prehensile manipulation**, on
[JAX](https://jax.readthedocs.io/) and
[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

**Object-informed MPPI** splits long-horizon pushing into an *object-level*
planner (what contact wrench does the object need?) and a *robot-level*
planner (how do I produce it?), coordinated by ADMM until the two agree on
the wrench. Both blocks accept any sampler from the library below as their
inner solver.

<p align="center">
  <img src="img/humanoid.gif" width="30%" />
  &nbsp;&nbsp;
  <img src="img/cube.gif" width="30%" />
</p>

- [Setup](#setup) · [Algorithms](#algorithms) · [Running](#running) · [Method](#method) ·
  [Code layout](#code-layout) · [Extending](#extending) ·
  [Citation](#citation)

## Setup

Python ≥ 3.12, CUDA 13, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NikolaRaicevic2001/Object-Informed-Manipulation-MJX.git
cd Object-Informed-Manipulation-MJX && uv sync
```

| | |
| --- | --- |
| `uv run <cmd>` | Run in the environment (or `source .venv/bin/activate`) |
| `uv run pytest` | Tests |
| `uv run ruff check .` | Lint |

## Algorithms

| Algorithm | Description | Import |
| --- | --- | --- |
| **[ADMM object-informed MPPI](#method)** | **Hierarchical object/robot decomposition, coordinated to consensus on the contact wrench.** | [`oim.algs.ADMM`](oim/algs/admm.py) |
| [Predictive sampling](https://arxiv.org/abs/2212.00541) | Take the lowest-cost rollout. | [`oim.algs.PredictiveSampling`](oim/algs/predictive_sampling.py) |
| [MPPI](https://arxiv.org/abs/1707.02342) | Exponentially weighted average of rollouts. | [`oim.algs.MPPI`](oim/algs/mppi.py) |
| [CEM](https://en.wikipedia.org/wiki/Cross-entropy_method) | Fit a Gaussian to the `n` elite rollouts. | [`oim.algs.CEM`](oim/algs/cem.py) |
| [DIAL-MPC](https://arxiv.org/abs/2409.15610) | MPPI with dual-loop annealed covariance. | [`oim.algs.DIAL`](oim/algs/dial.py) |
| [MPPI-CMA](https://arxiv.org/pdf/2506.22087) | MPPI with an adaptive sampling distribution. | [`oim.algs.MppiCma`](oim/algs/mppi_cma.py) |
| [MTP](https://arxiv.org/abs/2505.01059) | Structured tensor sampling + local CEM update. | [`oim.algs.MTP`](oim/algs/mtp.py) |
| [CBO](https://en.wikipedia.org/wiki/Consensus_based_optimization) | SDE pulling samples toward a consensus point. | [`oim.algs.CBO`](oim/algs/cbo.py) |
| [Evosax](https://github.com/RobertTLange/evosax/) | 30+ evolution strategies (CMA-ES, DE, …). | [`oim.algs.Evosax`](oim/algs/evosax.py) |


## Running

Planar pushing: drive an object to an SE(2) goal past static obstacles.
**One script per task** under `examples/`, and three programs, one per job,
so an expensive step never repeats for a cheap one:

| Program | Runs | Writes | Reads |
| --- | --- | --- | --- |
| `examples/<task>.py` | one experiment | `results/runs/*.json` | — |
| [`examples/object_only.py`](examples/object_only.py) | the object block alone, any scene | `results/object/*.json` | — |
| [`oim/run_launch.py`](oim/run_launch.py) | a sweep, one subprocess per cell | `results/sweeps/*.json` | the sweep config |
| [`oim/run_eval.py`](oim/run_eval.py) | nothing | `results/eval/*.{json,md,tex}` | run files |

### Tasks

| `examples/` | Object → goal | In the way |
| --- | --- | --- |
| `clutter.py` | T, 45° turn | disc, box, triangle — and the only 3D scene with a point-mass embodiment |
| `open_table.py` | T, 180° flip | nothing — the unobstructed baseline |
| `single_obstacle.py` | T, 180° flip | one 0.1 m cube on the direct path |
| `shelf_gap.py` | T, 180° flip | two shelves; the gap is exactly as wide as the T is long |
| `ycb_clutter.py` | T, 180° flip | that cube plus spam can, sugar box, mustard bottle |
| `icra_sign.py` | C, 90° turn | seven glyphs spelling *ICRA 2026*; the goal is the empty C slot |
| `clutter2.py` | T, 90° turn | three pudding boxes — the lab's measured table, the only scene that runs on hardware |
| `pusht2d_clutter.py` | T, 45° turn | 2D, 41 mm clearance |
| `pusht2d_corridor.py` | T | 2D, a 15 mm horizontal channel |
| `pusht2d_gate.py` | T | 2D, a 5 mm vertical slot, then a turn |


### Single runs

```bash
# 3D: 300 headless steps on the shelves, Warp rollouts, mp4 + trajectory overlay
uv run python examples/shelf_gap.py --warp --record --show-samples --show-optimal admm --headless --steps 100

# 3D: the point mass instead of the arm, flat MPPI baseline
uv run python examples/clutter.py --robot point mppi --headless --steps 200

# 2D: animated, eager for a debugger
uv run python examples/pusht2d_gate.py --animate --no-jit admm --n-admm 12 --rho 20
```

| Flag | Default | |
| --- | --- | --- |
| ***before the algorithm*** | | |
| `--robot` | scene's own | *3D:* embodiment, limited to those the scene has an MJCF for. Also picks the config: `oim/configs/{robot}.yaml` |
| `--samples`, `--horizon` | from config | Rollouts per block, consensus horizon $H$ (shared by both blocks) |
| `--object-samples` | `sampler.object.num_samples` | Rollouts for the object block alone |
| `--warp` | off | *3D:* [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) rollouts. Also disables JAX's GPU preallocation, which Warp needs |
| `--record` | off | *3D:* mp4 (needs `ffmpeg`); with `--headless`, renders offscreen |
| `--show-samples` | from config | *3D:* overlay the sampled candidate rollouts, as thin lines |
| `--show-optimal` | from config | *3D:* overlay the chosen trajectory, as a thick line. Independent of `--show-samples`: either, both, or neither |
| `--start`, `--goal` | random | *3D:* pose key from [`examples/poses/<task>.yaml`](examples/poses/) — five of each per task. Unset draws one (seeded by `--seed`); the run file records which |
| `--no-plot` | off | Skip the summary figure |
| `--contact-action`, `--no-relocate`, `--no-obstacles`, `--no-jit`, `--animate` | off | *2D:* object-block parameterization, contact search, obstacles, eager mode, gif |
| ***after the algorithm*** (`admm`, or *3D:* `mppi`/`ps`) | | |
| `--steps`, `--seed` | from config | Control steps, RNG seed |
| `--n-admm`, `--rho`, `--gamma` | from config | *`admm`:* max iterations, penalty $\rho$, proximal weight $\gamma$ |
| `--robot-opt`, `--object-opt` | `mppi` | *3D `admm`:* inner solver per block — `mppi`/`cem`/`ps`/`cbo` |
| `--rho-torque` | 10.0 | *3D `admm`:* penalty on the torque component alone, split from `--rho` |
| `--consensus` | `wrench` | *3D `admm`:* what the blocks agree on — `wrench` or `pose` |
| `--consensus-alpha` | from config | *3D `admm`:* EMA on $A^o, A^r$ across rounds (1.0 = raw) |
| `--local-goal` | off | *`admm`:* robot block tracks $x^{o*}_H$ instead of $g$ — see [Local goal](#local-goal) |
| `--headless` | off | *3D:* no viewer; run `--steps` and save a run file |
| *config only* | | No flag: $\epsilon_r$, $\epsilon_s$, noise annealing, per-method sampler parameters, execution timestep, goal tolerances, 2D physics |

The overlay works for every 3D algorithm — each samples a population and
reduces it to one trajectory. Blocks are told apart by color, samples from
the chosen path by width, and both are composited into the mp4 as well as
the viewer:

| Block | Samples | Chosen | Drawn by |
| --- | --- | --- | --- |
| object | pale cyan | strong blue | `admm` |
| robot | pale amber | strong orange | `admm`, `mppi`, `ps` |

The plan's *endpoint* $x^{o*}_H$ is drawn instead as a faint blue ghost of
the object (the `local_goal` mocap body in each scene's MJCF) — an SE(2)
pose has an orientation and a line cannot show it. Always on for `admm`,
independent of the two flags above and of `--local-goal`; hidden for flat
baselines, which have no object block to read it from.

| Output | When |
| --- | --- |
| `results/runs/*.json` | *3D:* `--headless`, *2D:* always — settings, scene, per-step states/controls/wrenches/residuals/timings |
| `recordings/*.png` | unless `--no-plot` — trajectory + diagnostics + per-step cost terms |
| `recordings/*.mp4` | *3D:* `--record` (needs `ffmpeg`) |
| `recordings/*.gif` | *2D:* `--animate`; *object-only:* `--record` |

### The object block alone

Whether the two blocks agree is only worth asking once the object block can
solve its own half. `examples/object_only.py` runs it with no robot and no
ADMM — the same `PushT` and the same
[`ObjectSubproblem`](oim/algs/admm.py), with $\rho$ and $\gamma$ set to
zero, so it is the object block ADMM uses rather than a re-implementation.

```bash
uv run python examples/object_only.py --scene shelf_gap --robot xarm6 --record
```

**The plant is the model**: `object_dynamics` both predicts and executes,
so there is no model error at all. That upper-bounds what the object block
can achieve, and the gap to an ADMM run is what the robot and the consensus
cost. The deadzone therefore applies to execution too, which is why the
figure plots $\lVert w\rVert/D^{-1}$ against its threshold.

| Flag | |
| --- | --- |
| `--scene`, `--robot` | any `SCENES` key; no robot is simulated, but it picks the config and the scene variant |
| `--plant` | `analytic` (the model executes itself) or `mujoco` (same wrench, simulator dynamics) — see below |
| `--iterations` | optimizer passes per step — set to `n_admm` for like-for-like |
| `--wrench-fraction` | overrides `costs.wrench_fraction`; decides whether the block can move the object at all, and `analytic` and `mujoco` need different values (see implementation notes) |
| `--record`, `--show-samples`, `--show-optimal`, `--fps` | gif, one frame per control step, and what it overlays. Required for any recording — `record: true` in a sweep config passes exactly this flag |
| `--video-width`, `--video-height` | with `--plant mujoco`, `--record` also writes an mp4 of the simulated scene from its own camera |
| `--stride`, `--no-jit`, `--no-plot` | figure density, eager mode, skip the figure |

The trajectories are in the **gif**, not the PNG: one static frame carrying
every step's horizon hides the one thing the panel is for. Colours match
the 3D overlay — pale cyan candidates, strong blue chosen, endpoint marked.

`check_action_budget` prints the reachable $\max\lVert w\rVert/D^{-1}$ at
build time and warns when it is under 1, because that failure and "the
planner cannot find a route" look identical in the log. Run files go to
`results/object/`, not `results/runs/` — no robot and no replanning rate,
so they must not average into `run_eval`'s tables. Evaluate them on their
own with `--runs-dir`:

```bash
uv run python -m oim.run_eval --runs-dir oim/results/object --plot
```

#### Analytic vs MuJoCo dynamics

`--plant mujoco` keeps the planner, sampler, costs, projection and warm
start identical and swaps only what executes the chosen wrench: it is
written to `qfrc_applied` on the block's `T_x`/`T_y`/`T_z` DoFs, which for
two world-axis slides and a hinge about $z$ *is* the planar wrench of
eq. 5. The log then carries `pred_pos_err`/`pred_theta_err`, the per-step
gap between what eq. 5 predicted and what the simulator did — zero by
construction under `--plant analytic`. The plant takes the arm and the
support surface out of collision: the block's support friction is already
modelled as its joints' `frictionloss`, so leaving it resting on the table
counts that friction twice (measured breakaway 11.16 N against 7.87 N).

Measured on `shelf_gap`, displacement under a wrench held for 1 s:

| $w/D^{-1}$ | $\lVert w/D^{-1}\rVert$ | analytic | MuJoCo |
| --- | --- | --- | --- |
| $[1, 0, 0]$ | 1.00 | 0.000 m | 0.006 m |
| $[1, 1, 0]$ | 1.41 | 0.414 m | 0.007 m |
| $[1.5, 0, 0]$ | 1.50 | 0.500 m | 0.984 m |
| $[2, 2, 0]$ | 2.83 | 1.828 m | 1.290 m |

| Difference | |
| --- | --- |
| **Threshold shape** | eq. 5 measures friction with the coupled norm $\lVert w/D^{-1}\rVert$ (an ellipsoid); `frictionloss` measures it per DoF (a box). An ellipsoid is the right shape for a block on a table, so closing this means giving the *simulator* a coupled cone. Row 2 above is 1.41× over the ellipsoid and exactly *at* the box. Not a corner case: the action box is the unit cube, so the optimizer's preferred actions are its corners. **Consequence for `--wrench-fraction`:** at 1.0 the most the block can put on any one channel is exactly that channel's friction, so under `--plant mujoco` the net generalized force is ~0 and the object does not move — `pos_err` 0.732 after 200 steps on `open_table`. At 2.0 the same run reaches the goal at step 91. `check_action_budget` warns on this whenever the plant is `mujoco`. |
| **Inertia** | eq. 5 is quasi-static; MuJoCo integrates 2 kg. Since `step` began subtracting friction the two agree on *shape* — the ratio is a constant **9.7** across every over-threshold wrench, where the old gating form ran 569 → 13.7 — so all that is left is one scale factor, $D$'s quasi-static velocity scale against real acceleration. One 0.05 s step from rest at 1.5× is 0.025 m predicted vs 0.003 m realized; held 1 s, 0.50 m vs 0.98 m. |
| **Coupling** | $D$ is diagonal, so a pure force cannot rotate the object. The block's CoM is offset from its joint origin, so in MuJoCo a pure $+x$ force for 1 s yields $-0.021$ rad. |
| **Contact** | analytic obstacles are a soft cost hinge and can be planned through; MuJoCo stops the block dead. |
| **Soft deadzone** | `frictionloss` has solver compliance, so a sub-threshold wrench creeps ~5 mm/s instead of sticking. |

`--plant mujoco --record` writes an **mp4 of the simulated scene** beside
the usual gif, rendered from the plant's own `MjData` at the physics rate
(so playback is real time) through the same `OffscreenRecorder` and
`PlanOverlay` the 3D runners use — candidates and chosen plan composited in
the same colours, one block instead of three. The arm appears parked and
out of collision, and the block floats a hair above the table — both
deliberate, see above.

Where the trajectories live: **gif** (2D, matplotlib) and **mp4** (3D,
MuJoCo). The summary **PNG** deliberately has none — one static frame
carrying every step's horizon is unreadable.

So `--plant mujoco` needs its own `--wrench-fraction` (2.0), not the
analytic default of 1.0. With it the closed loop converges. Wrench jitter
is *not* a factor here — at fraction 1.0 the executed wrench is already
smooth (median turn 12.8°, 0% reversals) because the object barely moves,
and raising `w_rate` 50× changes nothing. Reconciling the two threshold
*shapes* — a coupled friction cone in MuJoCo rather than three independent
per-DoF elements — remains open.

### Sweeps

```yaml
# oim/configs/run_launch_config.yaml
sweep:
  task: [{ script: shelf_gap }, { script: clutter, robot: point }]
  algorithm: [admm, mppi]
  horizon: [5, 15, 25]
  seed: [0, 1, 2, 3, 4]
fixed: { steps: 200, headless: true }
```

Every combination runs as its own subprocess. `task` names the script and
any flags for it; an empty list drops the axis, and an axis that is not
sweepable is rejected up front rather than ignored.

`start`/`goal` are axes too, drawn from [`examples/poses/`](examples/poses/):
five starts and five goals per task, each checked clear of that scene's
obstacles. Sweeping them varies the problem; sweeping `seed` alone only
redraws the sampler's noise against a fixed one.

`object_only` sweeps too, with two differences it declares for itself
(`SWEEP_WORLD` and `sweep_parser()` in place of an `Experiment`): the scene
is an axis rather than the script, and there is no algorithm subcommand, so
the `algorithm`/`n_admm`/`rho`/… axes are dropped for its cells.

```yaml
sweep:
  task: [{ script: object_only }]
  scene: [open_table, shelf_gap, icra_sign]
  wrench_fraction: [0.5, 1.0]
fixed: { robot: xarm6, steps: 400, iterations: 4 }
```

Any axis a script has no flag for is dropped before cells are deduplicated,
so a mixed sweep does not run the same command once per value of something
that script never sees.

```bash
uv run python -m oim.run_launch                        # the whole product
uv run python -m oim.run_launch --dry-run              # print, run nothing
uv run python -m oim.run_launch --only algorithm=admm  # narrow it
uv run python -m oim.run_launch --warp --set steps=50  # override `fixed:`
```

| Flag | |
| --- | --- |
| `--dry-run` | print each cell's exact command, run none |
| `--only KEY=A,B` | keep only matching cells; repeatable |
| `--set KEY=VALUE` | override `fixed:` for this sweep; unknown keys rejected up front |
| `--warp` | shorthand for `--set warp=true` |
| `--stop-on-error` | abort on the first failure instead of skipping it |

### Evaluation

One block per task, one row per method, then a `Mean` block averaging each
method over the tasks — the paper's table. **Everything not grouped on and
not ablated is averaged into the cell**, so a sweep over seeds still gives
one number per (task, method). Whatever still varied is printed above the
table; `--filter` pins a value, `--ablate` folds a field into the method
label (e.g. `admm(mppi/mppi) rho=1.0`), and `--group-by` splits it into
row blocks instead.

```bash
uv run python -m oim.run_eval                          # every run
uv run python -m oim.run_eval --format latex           # paper-ready tabular
uv run python -m oim.run_eval --filter algorithm=admm,mppi
uv run python -m oim.run_eval --group-by task horizon  # split a setting into blocks
uv run python -m oim.run_eval --pos-tol 0.02           # re-score, no re-running

# Ablation: one method row per rho; pin the other ADMM knobs, optional curves
uv run python -m oim.run_eval --ablate rho \
    --filter n_admm=4 --filter gamma=0.1 --filter consensus_alpha=1.0 \
    --format latex --plot
```

| Flag | |
| --- | --- |
| `--filter KEY=A,B` | keep matching runs; repeatable. One field's values OR-ed, different fields AND-ed |
| `--ablate FIELD …` | fold these fields into the method label so each value is its own row (pin the rest with `--filter`) |
| `--group-by` | fields forming each block (default `task`). Methods are always the rows inside |
| `--plot` | write a step-curve figure under `results/eval/` ($\epsilon_d$, $\epsilon_\theta$, ADMM primal/dual residuals) |
| `--pos-tol`, `--theta-tol` | re-score success against a new tolerance |
| `--format` | `text` (default), `markdown`, `latex`. A human-readable `.txt` is always written; this flag adds a second file when not `text` |

| Column | Paper | |
| --- | --- | --- |
| `SR` | SR | fraction reaching both tolerances, re-derived from the final pose |
| `eps_d` | $\epsilon_d$ | position error averaged **over the trajectory**, not the final value, so it stays large even at SR 1.0 |
| `eps_d^s` | $\epsilon_d^s$ | same, successful trials only; blank if none succeeded |
| `theta` | — | orientation error, same trajectory mean; not in the paper |
| `steps` | $N$ | control steps to first meet both tolerances; a trial that never does is censored at the run's configured `steps` |
| `f (Hz)` | $\bar{f}$ | wall-clock planning rate, from the recorded `compute_time` |
| `T (s)` | $T$ | *simulated* time (`steps_run × dt`), machine-independent; a failed trial is credited the slowest time across every loaded run, so methods stay comparable |


## Method

A robot manipulates an unactuated rigid object through contact. Rather than
one monolithic contact-implicit problem, this is **two subsystems coupled
only through the contact wrench**, reconciled by ADMM.

### Subsystem models

| | Dynamics | |
| --- | --- | --- |
| Robot | $x^r_{t+1} = f_r(x^r_t) + g_r(x^r_t) u^r_t + h_r(x^r_t) w^r_t$ | input-affine |
| Object | $x^o_{t+1} = f_o(x^o_t) + h_o(x^o_t) w^o_t$ | unactuated; moves only via the wrench |

The object planner treats $w^o_t \triangleq [f_x, f_y, \tau]^\top$ as a
*decision variable* rather than the outcome of complementarity constraints.
For quasi-static planar pushing the **ellipsoidal limit surface** closes the
model with no simulator in the loop:

```math
\dot{x}^o = D\,w^o, \qquad
x^o_{t+1} = x^o_t + \Delta t\, D\, w^o_t, \qquad
D^{-1} = \mathrm{diag}\big(\mu m g,\ \mu m g,\ c\,r\,\mu m g\big)
```

| $\mu$ | $m$ | $g$ | $c$ | $r$ | $D^{-1}$ |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 2.0 kg | 9.81 m/s² | 1.0 | 0.06 m | $(7.848,\ 7.848,\ 0.47088)$ |

$D^{-1}$ is the **friction-cone limit** — the largest wrench the support can
transmit. Every scene with an object model sets the block joints'
`frictionloss` to it, so the simulated block and the analytic model share one
friction budget. Reused throughout as the natural normalizer.

### Consensus problem

Over horizon $H$, with $\mathbf{Z} = \{z_t\}$ shared:

```math
\begin{aligned}
\min_{\mathbf{W}^o,\, \mathbf{U}^r,\, \mathbf{Z}} \quad
& \sum_{t=0}^{H-1}\Big(\ell_o(x^o_t) + \ell_r(x^r_t, u^r_t)\Big) + \ell_f(x^o_H) \\
\text{s.t.}\quad
& x^o_{t+1} \text{ via } w^o_t, \quad x^r_{t+1} \text{ via } u^r_t, \quad (x^r_t, u^r_t) \in \mathcal{C}, \\
& w^o_t = z_t, \qquad \hat{w}^o_t = z_t .
\end{aligned}
```

| Extraction map | Value | Source |
| --- | --- | --- |
| $A^o(\mathbf{U}^o)_t$ | $w^o_t$ | read off the object block's decision variable |
| $A^r(\mathbf{U}^r)_t$ | $\hat{w}^o_t$ | read off the robot block's MJX rollout |

$\hat{w}^o_t$ is **not** a decision variable — it is whatever the rollout
produces once $\mathbf{U}^r$ is applied. Splitting $\mathbf{W}^o$ from
$\mathbf{U}^r$ through $z_t$ is what lets the two planners run independently.

### ADMM iteration

The $N = 2$ case of global-variable-consensus ADMM. Each iteration $l$:

| # | Step | |
| --- | --- | --- |
| 1 | $\mathbf{U}^{i,(l+1)} = \mathrm{arg\,min} \big\{ J_i + \tfrac{\gamma}{2}\lVert \mathbf{U}^i - \mathbf{U}^{i,(l)}\rVert^2 + \tfrac{\rho}{2}\sum_t \lVert A^i_t - z^{(l)}_t + y^{i,(l)}_t\rVert^2 \big\}$ | both blocks, $i \in \{o,r\}$; penalties evaluated inside the sampler's rollout cost |
| 2 | $z^{(l+1)}_t = \tfrac{1}{2}\big( A^o_t + y^{o,(l)}_t + A^r_t + y^{r,(l)}_t \big)$ | $\Pi_\mathcal{Z} = \mathrm{id}$, so a plain average |
| 3 | $y^{i,(l+1)}_t = y^{i,(l)}_t + A^i_t - z^{(l+1)}_t$ | duals integrate disagreement |
| 4 | $\rho \leftarrow 2\rho$ if $\lVert r\rVert > 10\lVert d\rVert$; $\rho/2$ if $\lVert d\rVert > 10\lVert r\rVert$ | from $r = [A^o - z; A^r - z]$, $d = \rho(z^{(l+1)} - z^{(l)})$ |

The proximal term $\gamma > 0$ is inertia between iterations: it prevents
the radical trajectory shifts non-convex contact dynamics induce, and
supplies the strong convexity non-convex ADMM convergence results require.

> **Given** $x_0$, previous $(\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}^o, \mathbf{Y}^r)$, parameters $\rho, \gamma$
> 1. Warm-start all five by shifting one step
> 2. **for** $l = 0 \dots N_{\mathrm{ADMM}}-1$: steps 1–4 above
> 3. &nbsp;&nbsp;&nbsp;&nbsp; **break** if $\lVert r\rVert \le \epsilon_r$ and $\lVert d\rVert \le \epsilon_s$
> 4. Apply $u^r_0$, shift, observe $x_1$

Shipped in [`oim/configs/`](oim/configs/); $y_{\max} = 2\mu m g = 15.696$ and
$\epsilon_r = \epsilon_s = 0.5$ in both.

| | $N_{\mathrm{ADMM}}$ | $\rho_0$ | $\rho_\tau$ | $\gamma$ | $H$ | samples |
| --- | --- | --- | --- | --- | --- | --- |
| `point.yaml` | 8 | 10.0 | 10.0 | 0.1 | 15 | 64 |
| `xarm6.yaml` | 4 | 10.0 | 10.0 | 0.1 | 32 | 128 |

$\rho$ is a **per-dimension vector** $\mathrm{diag}(\rho_0,\rho_0,\rho_\tau)$
(`--rho-torque`, the paper's anisotropic $P$), so the penalty can pull
harder on orientation agreement than on position independently of the cost.

### Consensus variable

`--consensus` selects what the blocks agree on — [README_ADMM](README_ADMM.md)
§4 is the argument for the second.

| | $A^o$ | $A^r$ |
| --- | --- | --- |
| `wrench` (default) | the block's own decision $w^o_t$ | the wrench the rollout imparts, inferred and clipped |
| `pose` | eq. 5 integrated, hence **affine** in $\mathbf{U}^o$ | the object's SE(2) pose, read from the state |

`pose` makes $\mathcal{Z} = SE(2)$, so $\Pi_\mathcal{Z}$ is the angle wrap,
the duals are twists, and eq. 27's average is taken about a base point
([`PoseConsensus`](oim/algs/admm.py)). It also drops $\ell_c$, which the
penalty then subsumes.

### Costs

All SE(2) tracking shares one weighted squared distance
([`se2_distance_sq`](oim/objects/planar_pushing.py)), angle wrapped to
$(-\pi,\pi]$:

```math
d^2_w(x,g) = w_p\lVert p - p^g\rVert^2 + w_\theta \,\mathrm{wrap}(\theta-\theta^g)^2
```

**Object block** — $b_j(x^o)$ are the footprint's boundary samples in world
frame, $\delta$ the clearance margin. The clearance term is geometric, not a
simulator contact force: this block has no simulator.

```math
\ell_o(x^o_t, w_t) = d^2_{q}(x^o_t, g)
+ w_{\text{obs}} \sum_{j} \max\big(\delta - \mathrm{sdf}(b_j(x^o_t)),\ 0\big)^2
+ r_o \lVert w_t\rVert^2 ,
\qquad \ell_f = d^2_{q_f}(x^o_H, g)
```

**Robot block** — $x^{o*}_t$ is the object planner's nominal trajectory from
this iteration (paper eq. 17). The last term is the *same* clearance hinge,
applied to the pusher's own position: the object block keeps the block clear
of obstacles, but nothing kept the pusher from being commanded straight
through a shelf on its way to "behind the block".

```math
J_r(x^r_t, u^r_t) = r_r \lVert u^r_t\rVert^2
+ \underbrace{d^2_{q}(x^o_t, g)}_{\text{goal}}
+ \underbrace{d^2_{q}(x^o_t, x^{o*}_t)}_{\ell_c\ \text{coupling}}
+ \ell_r
+ w_{\text{obs}} \max\big(\delta - \mathrm{sdf}(p^{ee}_t),\ 0\big)^2 ,
\qquad J_{r,f} = d^2_{q_f}(x^o_H, g)
```

#### Local goal

`--local-goal` / `admm.local_goal:`, off by default. The object block routes
toward $g$ around obstacles over $H$ steps while the robot block is scored
against $g$ *directly* — so anything the plan does that is not straight at
the goal, the robot block is penalized for following. The flag retargets
the robot block's two goal-tracking terms onto the plan's own endpoint:

```math
d^2_{q}(x^o_t, g) \to d^2_{q}(x^o_t, x^{o*}_H),
\qquad
J_{r,f} = d^2_{q_f}(x^o_H, g) \to d^2_{q_f}(x^o_H, x^{o*}_H)
```

| | |
| --- | --- |
| $\ell_c$ kept | it tracks the plan *pointwise* (penalizing running ahead), the retargeted term rewards reaching its end. Dropping it here is the next ablation |
| $\phi$ **not** retargeted | it means "the task is nearly over"; against a target $H$ steps away it would read $\approx 0$ every step and switch off align, tilt and tip height for the whole run |
| Needs a live object plan | while the block is stuck under breakaway, $x^{o*}_H = x^o_0$ and the flag asks the robot to hold the object still |

**Contact shaping** $\ell_r$ — what makes the tip a *pusher* rather than
merely something nearby. $\phi$ fades the three posture terms as the object
nears its goal; approach never fades, since the tip has to stay on the block
to push it at all.

```math
\ell_r = \underbrace{w_{ee}\max\big(\lVert p^{ee}_t - p^o_t\rVert^2 - r_0^2,\ 0\big)}_{\text{approach}}
+ \phi(x^o_t)\Big[\, w_{\text{align}} \psi_{\text{align}}
+ \underbrace{w_{\text{tilt}} \big(1 - \cos\psi_{\text{tilt}}\big)
+ w_{z}(z^{ee}_t - z^\ast)^2}_{\text{3D only}} \,\Big]
```

```math
\psi_{\text{align}} = \max\big(\gamma_0 - \cos\angle(p^o_t - p^{ee}_t,\ p^{o*}_t - p^o_t),\ 0\big),
\quad \cos\psi_{\text{tilt}} = -R^{ee}_{33},
\quad \phi = \mathrm{clip}\!\big(\lVert p^o - p^g\rVert / d_{\text{fade}},\, 0,\, 1\big)
```

Approach pulls the pusher in but goes slack inside $r_0$;
$\psi_{\text{align}}$ keeps it *behind* the object relative to the reference;
$\psi_{\text{tilt}}$ is the stick's $z$-axis against world $-z$, zero pointing
straight down. Tilt is penalized as $1-\cos\psi_{\text{tilt}}$, not as
$\psi_{\text{tilt}}$: a linear penalty has a constant restoring gradient and
never arrested the measured drift at any weight.

**Flat baseline** — the same terms, with the goal $g$ standing in for
$x^{o*}$ (there is no object plan) and both clearance hinges kept, so a
baseline is not handicapped by lacking obstacle awareness the ADMM object
block has. Its terminal cost is $d^2_{q_f} + \ell_r$, **not** goal tracking
alone: stage costs are $\Delta t$-weighted and the terminal is not, so the
terminal is where the pushing geometry is scored at full weight. Replacing it
with a goal-only $\ell_f$ let MPPI buy a better predicted pose by abandoning
that geometry — measured on `open_table` (xArm6), 0.07 m → 0.98 m final
error.

Weights are [`oim/configs/{robot}.yaml`](oim/configs/)'s `costs:` block over
[`DEFAULT_COSTS`](oim/tasks/pusht.py); one mapping feeds both blocks, so the
shared goal-tracking weights cannot drift apart between them. The object
block is identical in both worlds; only $\ell_r$ differs, and only where the
embodiment does — the 2D disc has no orientation or height to shape and no
fade. $w_{\text{tilt}}$, $w_z$, $\phi$ and the pusher hinge are not in the
paper.

### Object action parameterization

By default the object block's decision variable *is* the consensus variable:
it samples $w^o_t$ directly, box-bounded (see implementation notes). That
bounds the wrench's *magnitude* but not its *direction* — the sampler may
still propose a pure torque or a pulling force.

A task may instead decide a **contact action**
$a_t = [p_x, p_y, f_n, f_t]$ — where to push in the body frame, and with what
normal/tangential force — and derive the wrench through the contact Jacobian,
making every reachable $A^o$ realizable by construction:

```math
A^o_t = J_c(p_t)^\top f = \begin{bmatrix} f \\ (p^{c}_t - p^o_t) \times f \end{bmatrix},
\quad f = f_n\hat{n}(p_t) + f_t\hat{t}(p_t),
\quad p_t \in \partial\mathcal{O},\ 0 \le f_n \le f_{\max},\ |f_t| \le \mu_c f_n
```

$z$ is still the 3-vector wrench, so nothing in the ADMM layer changes; only
$\dim(a) = 4 \ne \dim(z) = 3$. Sampling must respect the geometry: points are
perturbed, re-projected onto the boundary, then rejection-filtered on normal
alignment (an unfiltered step can hop to the opposite face, reversing the
wrench). That makes the proposal local, so a separate CEM search over the
whole boundary re-chooses the contact point each step — without it the block
can slide along one face but never decide to push from elsewhere, which is
what routing around an obstacle requires.

**2D only, and off by default** (`--contact-action` opts in): where the robot
touches is the robot block's concern, and making the object planner choose it
duplicates that job in the wrong subproblem. A task opts in by overriding
`object_action_dim`, `object_action_bounds`, `object_action_to_consensus`,
`project_object_action`, `sample_object_actions`, `initial_object_action`; the
3D `PushT` overrides none of them and has no flag.

### Implementation notes

Where the implementation departs from the formulation above.

| | What, and why |
| --- | --- |
| **Penalty normalization** | Penalty and residuals are divided by $D^{-1}$ before squaring. Unnormalized, ~10 N forces give ~10² against task costs of ~1, and the robot optimizes wrench matching instead of reaching the object. Identical diagonal preconditioning on both blocks, so the fixed point is unchanged and $\rho, \epsilon_r, \epsilon_s$ become scale-free. |
| **$A^r$ is inferred, not read** | Default `consensus_source="twist"` inverts the limit surface, $\hat{w}^o = D^{-1}\dot{x}^o$, rather than reading MJX's contact force. Backend- and embodiment-agnostic, and continuous through contact breaks, where the literal force is exactly zero and chatters. `"contact"` reads `qfrc_constraint` literally, matching the paper, but is only valid for the point pusher — an arm's contact appears as $J^\top f$ spread across its joints. |
| **$A^r$ clipped to $D^{-1}$** | A rigid-body solver reports up to ~16× the friction-cone limit at contact onset. Unclipped, that outlier drags $z$ outside the object block's own feasible set, which it can then never match, and the disagreement outlives the spike by several steps. |
| **Consensus smoothed across rounds** | $z$ and both residuals are computed from an EMA of $A^o, A^r$ with weight `consensus_alpha`, re-zeroed every control step. Each round's $A$ is one noisy resampling estimate, not a converged proposal, so raw disagreement is dominated by resampling variance. Ships at 1.0 (raw, as the paper); 0.2 measured better. |
| **Object action cannot reach breakaway** | `object_action_bounds` is the unit box and `object_action_to_consensus` scales by `wrench_sample_fraction`$\cdot D^{-1}$, so the largest expressible wrench is $\lVert w/D^{-1}\rVert \le \texttt{fraction}\sqrt3$ — against the deadzone's threshold of 1. Below 1 the block cannot move the object at all and MPPI converges *to* $w=0$: with every rollout frozen, effort is the only term still varying across samples. Since `step` began subtracting friction, **1.0 is also not enough** — a saturated single channel then nets exactly zero force — so both configs ship `costs.wrench_fraction: 2.0`. Measured across all five scenes: 15/15 reach the goal at 1.5/2.0/3.0, none at 1.0. This is the un-implemented $\Pi_\mathcal{F}$ of [README_ADMM](README_ADMM.md) §1; `examples/object_only.py --wrench-fraction` isolates it. |
| **Object MPPI temperature must be read against the cost spread** | $\lambda$ is meaningless in isolation: far below the *spread* of rollout costs the softmax collapses onto one sample, so the "weighted average" is an argmax over white noise and the mean re-randomizes every control step. The spread is set by the sampler's `noise_level`, not by $\lambda$ — at the old object `noise_level: 0.5`, `shelf_gap`+`xarm6` gave cost std 31.9 and **ESS 1.0/128** at $\lambda = 0.5$; at `0.25` the same $\lambda$ gives cost std 4.5 and **ESS 15.0/16**. So lowering the noise fixed the collapse without touching $\lambda$, and raising $\lambda$ alone does not (it drives the averaged wrench below the breakaway deadzone instead — measured: frozen at `pos_err` 0.540). `simobj.report_softmax_ess` prints this at the start of every object-only run. |
| **Nothing couples $w_t$ to $w_{t+1}$** | The object block samples one independent knot per timestep under a zero-order hold, and the effort term sees only $\lVert w_t\rVert$, so a sequence that reverses every step is free. `w_rate` charges $\sum_i w_{\mathrm{rate},i}(\Delta w_i / D^{-1}_i)^2$ — the cheapest stand-in for the fact that reversing a push means relocating the contact, which this block cannot represent (it does not model *where* it pushes). Being quadratic is the point: spreading a change over $k$ steps costs $1/k$ of jumping it. Weighted per channel `[f_x, f_y, τ]`, because $\tau_{\max} = 0.471$ vs $F_{\max} = 7.848$ means a shared weight taxes rotation hardest exactly where the goal needs it. Ships at 0 in `DEFAULT_COSTS`, i.e. the paper's cost. |
| **Penalty is not $\Delta t$-weighted** | Both blocks compute $\Delta t\,\ell + \tfrac{\rho}{2}\lVert\cdot\rVert^2$. They agree, so the fixed point is well defined, but the penalty's effective weight scales as $1/\Delta t$ — changing the planning timestep silently re-tunes $\rho$. |
| **Residuals unnormalized by horizon** | $\lVert r\rVert$ is a Frobenius norm over $(2H,3)$, not an RMS, so it grows like $\sqrt{2H}$. Residuals are $O(1)$ at both horizons in use, so $\epsilon_r = \epsilon_s = 0.5$; the paper's $0.05$ is unreachable here and the early exit would never fire. |
| **Variance annealing additive, and off** | Most samplers expose no mutable covariance, so the wrappers *add* $\mathrm{clip}(\kappa\lVert r\rVert, \sigma_{\min}, \sigma_{\max})$ rather than replacing $\Sigma_u$. The upper clip is required ($\kappa\lVert r\rVert$ is otherwise a positive feedback loop), and since $\lVert r\rVert$ does not converge here the clip binds permanently — so $\kappa = 0$. Measured over 600 steps at identical seed: final position error 4.65 with annealing on, 2.01 off. |
| **$\rho$ and $\lVert r\rVert$ persist** | `rho_init` is only the $t=0$ value; both are carried in the policy parameters and never reset, so they drift over a run. $\lVert d\rVert$ is re-initialized to $\infty$ each step, so the first iteration never exits early. |
| **Dual anti-windup** | Duals clipped to $\pm y_{\max}$. This is why the $z$-update keeps the dual terms: $\sum_i y^i = 0$ is an ADMM invariant that would make $z = \tfrac12(A^o + A^r)$ equivalent, but clipping breaks it. |
| **Warm start is not a pure shift** | $z, y^o, y^r$ and a direct-wrench object block shift by one and zero-fill the tail; a structured action space repeats the last value instead, since zero need not be feasible there (a zero contact point is the object's origin, not on its boundary). The robot mean is re-interpolated onto shifted spline knot times, not shifted. |
| **Horizons shared, sample counts not** | The formulation permits $H^c \le \min(H^o, H^r)$; the implementation enforces $H^o = H^r = H^c$, because $z$, both duals and both $A$ sequences are all $(H, \dim)$ — one `--horizon` sets all of them. Sample counts *are* independent (each block reweights its own population; only the $(H,\dim)$ consensus values cross between them), so `--object-samples` / `sampler.object.num_samples` splits them. Worth splitting: an object rollout integrates a 3-vector in closed form, a robot rollout steps MJX over the whole scene. |
| **Tilt is degenerate for the 3D point pusher** | $\psi_{\text{tilt}}$ reads the trace site's rotation, which for the point pusher never changes, so its raw contribution is a constant $2w_{\text{tilt}}$. That cancels in every sampler's cost differences only while $\phi \equiv 1$; with `shaping_fade_dist > 0` (both configs ship 0.15) the constant is scaled by $\phi(x^o_t)$ and becomes a pose-dependent term — an extra pull toward the goal, worth $2w_{\text{tilt}}$ at the fade radius, that no longer cancels. |
| **Limit surface has a real deadzone, and friction is subtracted** | $x^o_{t+1} = x^o_t + \Delta t\, D\, w^o_t$ extends proportionally through $w^o = 0$, but $D^{-1}$ is the friction-cone limit, not a soft compliance: a wrench below it should produce zero motion. The first fix zeroed sub-threshold wrenches and passed the **full** wrench above threshold, which made the map discontinuous — one-step displacement jumped 0 → $\Delta t \cdot 1 = 0.05$ m, *exactly* the goal tolerance, so no correction smaller than the target ball was representable and the near-goal choice was freeze or overshoot. `step` now subtracts the friction instead, $s = \lVert w^o/D^{-1}\rVert$, $\dot{x}^o = D w^o \max(0, 1 - 1/s)$ — the standard Coulomb form, and what `frictionloss` already does. Motion goes continuously to zero at the cone: $s = 1.05$ gives 2.5 mm where the gated form gave 52.5 mm. |
| **`project_gate_pos` now ships at 0 (off)** | `project_object_action` snapped any nonzero sample up to $\lVert w^o/D^{-1}\rVert = 1$ within $\lVert p - p_g\rVert$ of the goal. It existed only because the gated deadzone had no motion smaller than 0.05 m, so near the goal the choice was freeze or overshoot and this picked overshoot — the opposite of what fine control needs. With friction subtracted a smaller sampled force gives a smaller step, and the snap is actively harmful: measured across all five scenes at `wrench_fraction: 2.0`, gate 0.1 reached **4/5** in 461–558 steps, gate 0.0 reached **5/5** in 51–161. Kept but defaulted off, since it documents the failure mode. |

## Code layout

```
oim/
├── alg_base.py           SamplingBasedController: warm-start, spline knots,
│                           parallel rollouts, domain randomization, risk
├── task_base.py          Task; ConsensusTask (the ADMM contract)
├── risk.py               AverageCost, WorstCase, CVaR, …
├── open_loop.py          offline trajectory optimization + playback
│
├── algs/                 every sampler shares sample_knots / update_params
│   ├── admm.py           ADMM loop; ConsensusSpace, Wrench/PoseConsensus;
│   │                       ObjectSubproblem, RobotSubproblem;
│   │                       RobotRollout / MJXRollout  ← the 2D/3D seam
│   ├── mppi.py  cem.py  predictive_sampling.py  cbo.py
│   └── dial.py  mppi_cma.py  mtp.py  evosax.py
│
├── objects/              analytic, simulator-free — shared by 2D and 3D
│   ├── sdf.py            Shape/Circle/Box/Capsule/Polygon, sdf_and_grad
│   ├── planar_pushing.py limit-surface dynamics + object-level costs
│   └── contact.py        w = J_cᵀf, friction-cone projection, sampling
│
├── sim2d/                analytic 2D world — no MuJoCo anywhere
│   ├── engine.py         Sim2DState/Sim2DModel, resolve_contact
│   ├── task.py           PushT2D          scenarios.py  clutter/corridor/gate
│   └── run.py            build_admm_2d, run_2d
│
├── simobj/               the object block with no robot and no ADMM
│   └── run.py            build_object_only, run_object
│
├── sim3d/                MuJoCo drivers
│   ├── build.py          task + controller + execution model, so a flat
│   │                       baseline is built exactly like ADMM's
│   ├── deterministic.py  run_interactive: viewer, replanning, recording
│   ├── run.py            run_3d_admm / run_3d_plain: headless + logging
│   ├── plan_overlay.py   samples/chosen path per block, viewer and video
│   └── asynchronous.py   controller and simulator in separate processes
│
├── experiment.py         Experiment + main(): the CLI, closed loop,
│                           recording, run file and plot every
│                           examples/ script shares
├── real3d/               hardware: RobotWorldInterface, run_real (see below)
├── run_launch.py         sweep driver;  run_eval.py  post-hoc metrics
├── configs/              point.yaml, xarm6.yaml (defaults per robot);
│                         run_launch_config.yaml (the sweep definition)
├── tasks/  models/       MuJoCo tasks; MJCF scenes and meshes
└── utils/                scenes.py (the 3D scene registry), plotting.py,
                          costs.py (per-term cost decomposition), poses.py,
                          spline, video, results.py, metrics.py
```

One `ADMM.optimize(state, params)` call, top to bottom:

| Stage | Code | What crosses the boundary |
| --- | --- | --- |
| Warm-start | `ADMM.optimize` | previous $\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}$, shifted |
| Object block | `ObjectSubproblem` → `PlanarPushingObject.step` | samples $\mathbf{W}^o$, rolls out analytically, returns $A^o$ |
| Robot block | `RobotSubproblem` → `RobotRollout.step` | samples $\mathbf{U}^r$, rolls out in MJX **or** 2D, returns $A^r$ |
| Consensus + duals | `ConsensusSpace` | $z, y^o, y^r$, normalized by `consensus_scale()` |
| Convergence | `jax.lax.while_loop` | $\lVert r\rVert, \lVert d\rVert$; adapt $\rho$; exit test |


## Running on the real xArm6

`oim/real3d/` runs the ADMM push-T controller on a physical UFACTORY xArm6.
The planner (`ADMM.optimize`), the `PushT` task and the MJX rollouts are the
simulation path's; only the outer loop's I/O changes:

```
sim3d:  mjx_data <- mj_data ;      mj_data.ctrl = u ; mujoco.mj_step(...)
real3d: mjx_data <- ROS sensors ;  publish u to the arm's velocity controller
```

| File | Role |
| --- | --- |
| [`oim/real3d/interface.py`](oim/real3d/interface.py) | `RobotWorldInterface` (the I/O seam): `MujocoMockInterface` for laptop testing, `Ros2Interface` for hardware |
| [`oim/real3d/run_real.py`](oim/real3d/run_real.py) | the closed loop -- the hardware counterpart of `sim3d/run.py::_run` |
| [`examples/pusht_real.py`](examples/pusht_real.py) | entry point |
| [`oim/real3d/scripts/`](oim/real3d/scripts/) | RViz scene markers, state replay, contact analysis |

> **`pusht_real.py` does not read `oim/configs/`.** It builds its own task
> and controller rather than calling `build_admm_3d`, so it runs
> `DEFAULT_COSTS` and its own hardcoded `HORIZON`/knots — *not* the retuned
> `costs:` block, `rho_torque` or `realized_wrench_clip` a simulated xArm6
> run uses. A sim/real comparison is therefore not yet like-for-like.

MJX is still used on hardware -- it is the planner's internal predictive model,
run on the GPU every control step. The planner is a plain jitted JAX function,
so it runs in-process rather than behind an RPC server the way the Isaac-Gym
OI-MPPI stack needed (that split existed for Isaac's per-process sim context,
which MJX does not have).

### Environment

`pusht_real.py` needs **one environment holding both ROS 2 (`rclpy`, `tf2_ros`)
and the CUDA JAX stack (`jax[cuda13]`, `mujoco-mjx`, `oim`)**. ROS 2 Humble's
own `rclpy` is built for Python 3.10 while `oim`/JAX need ≥ 3.12, so sourcing
`/opt/ros/humble/setup.bash` into the uv venv does not work. RoboStack ships
`ros-humble-*` as conda packages for whichever Python you pick, which is what
[`oim/real3d/pixi.toml`](oim/real3d/pixi.toml) uses:

```bash
cd oim/real3d && pixi install && pixi shell
pip install -e /path/to/Object-Informed-Manipulation-MJX --no-deps
```

`pixi.lock` is committed, so `pixi install` reproduces the exact environment
these runs were made in.

### Laptop dry-run (no hardware)

Drives a MuJoCo sim through the hardware interface, so the whole loop -- state
assembly, command mapping, logging -- runs with no robot and no ROS:

```bash
python examples/pusht_real.py --mock --scene clutter2 --steps 200 \
    --command-mode stream
```

This is where to test **behaviour** changes -- cost weights, horizon, sampler
budget -- before spending robot time on them. Anything that lives in the cost
landscape reproduces here; only calibration (frame offsets, stick geometry,
safety limits) needs hardware.

One caveat: the mock starts the arm at the scene's `arm_start_deg`, which for
`clutter2` is right behind the block. To reproduce a run that started somewhere
else, change that value to the pose you actually started from -- otherwise the
mock is answering a different question.

Runs write `oim/results/pusht3d_xarm6_mock_clutter2_admm_*_states_*.json`, the
same schema a simulation run produces, so the two compare entry-for-entry.

### Running on the robot

FoundationPose runs on the **perception laptop** as its own stack (its own
repo, Docker, conda `my` env -- camera + mask + pose node + TF broadcaster; see
its readme). It publishes the `fp_object_pose` TF, which `Ros2Interface` reads
by default (`object_frame`; use `"sam6d_object"` for SAM-6D).

Laptop and desktop are two machines, so connect their ROS 2 over the LAN
first: on **both**, in every terminal, `source oim/real3d/scripts/setup_dds_env.sh`
(matches `RMW_IMPLEMENTATION` + `ROS_DOMAIN_ID`). Then check `ros2 topic list`
shows the other host. On a subnet where multicast is blocked you will need a
CycloneDDS XML as well.

On the **desktop**, four terminals:

**Terminal 1 -- robot bring-up:**
```bash
# Inside keti_ws
./scripts/run_docker

# Inside the container
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch main real_xarm6.launch.py
```

**Terminal 2 -- planner / driver (pixi env):**
```bash
cd oim/real3d && pixi shell && cd ../..

# No motion: reads state and the object TF, publishes nothing
python examples/pusht_real.py --scene clutter2 --dry-run --steps 1 --warp

# Live
python examples/pusht_real.py --scene clutter2 --steps 200 \
    --warp --command-mode stream --num-samples 64 --vel-limit 0.4
```

`--warp` is the MuJoCo Warp rollout backend: ~9x faster than the JAX backend
and effectively required for a usable replan rate (see *Real-time and speed*).
`--vel-limit` caps the joint velocity and is applied to **both** the published
command and the planner's own sample bounds -- the two must match, or the
planner predicts motion the arm will not produce. `--replan-rate` is a
mock-only knob; the hardware loop replans as fast as it solves.

**Terminal 3 -- scene markers (optional):**
```bash
cd oim/real3d && pixi shell
cd ../.. && python oim/real3d/scripts/publish_scene_markers.py \
    --scene-xml oim/models/xarm6_pusht_clutter_2/scene.xml \
    --frame xarm_device \
    --start 0.381,0.343,0   # read the live pose with
                            # `ros2 run tf2_ros tf2_echo xarm_device fp_object_pose`
```

**Terminal 4 -- RViz (optional):**
```bash
rviz2 -d oim/real3d/scripts/real3d.rviz
```

### Bring-up & calibration checklist

The mock validates everything except the physical setup. Redo this whenever the
stick, the camera mount or the table moves.

**1. Stick geometry.** Measure flange-to-tip length and rod diameter, and put
them in [`oim/models/xarm6/xarm6.xml`](oim/models/xarm6/xarm6.xml). MuJoCo's
capsule `size` is `(radius, half-length of the *cylinder*)`, and the two
rounded ends add `radius` each, so the capsule spans
`pos ± (half-length + radius)`. Set that span equal to the measured length:

```xml
<!-- 179.4 mm flange-to-tip, 11.1 mm diameter -->
<body name="xarm6_stick" pos="0 0 0">
  <geom name="xarm6_stick" type="capsule" size="0.00555 0.08415"
        pos="0 0 0.0897" material="xarm6_stick" mass="0.05"/>
  <site name="xarm6_tip" pos="0 0 0.1794" size="0.003" rgba="1 0 0 1"/>
</body>
```

**2. Table height (`base_z`).** The MJCF puts the scene floor at model z = 0,
but the arm base is not at table level. Calibrate from **joint angles only**, so
the result does not depend on the controller's TCP offset or on the camera:

- rest the stick tip on the table, record the joint angles
- run them through the model's own FK -- that z is where the model thinks the
  table is
- set `base_z = −(that z)` in the scene, which drops the arm by the same amount
  and puts the model floor on the real table

For `clutter2` this gives `base_z = -0.0111`. Getting it wrong is not subtle:
before it was calibrated the model floor sat 32 mm low, so `tip_target_z` -- the
block's mid-height in the model -- landed on the table surface in reality and
the arm drove itself into the table.

**3. Verify.** At any pose, the model's FK tip and the controller's reported TCP
should agree to **~1 mm on x, y and z**. If x and y agree but z does not, the
stick geometry or `base_z` is still wrong.

**4. Safety boundary.** The controller enforces it on **its own TCP**, in its
own frame -- and that frame's z = 0 is the robot base plane, *not* the table. On
this setup the table reads about −18 mm there. Since the block is ~60 mm tall,
its mid-height -- where the pusher has to be -- reads about **+12 mm**. Set the
boundary **below** that: any higher and it stops the arm before it can reach the
block, which looks exactly like a planner failure.

**5. Joint mapping.** Jog each joint and confirm `joint{i}` ↔ `xarm6_joint{i}`
and the sign convention (CW = +) match the model. The real wrist-roll `joint6`
is welded in the MJX scene and always commanded 0.

**6. Scene placement.** Run the marker script (Terminal 3 above) and nudge the
physical obstacles and block onto the drawn markers. This is the calibration
that makes the plan mean anything.

**7. If the block is swapped.** `mu`, `mass` and `limit_surface_radius` in the
scene and `frictionloss` in the MJCF are tied: the friction-cone limit `mu·m·g`
must equal the block joints' `frictionloss`, or the analytic object model and
the simulated one describe different physics.

### Real-time and speed

The hardware loop is *overlapped*: a publisher thread streams the current plan
at the control rate while the main thread solves the next one. What makes that
safe is a single relationship -- **the plan has to be longer than the solve**:

| | plan horizon | solve | margin |
| --- | --- | --- | --- |
| JAX backend | 0.75 s | ~1.3 s | **plan runs out** |
| `--warp` | 0.75 s | ~0.15 s | 5x |
| `--warp`, ADMM exits early | 0.75 s | 30–60 ms | 12–25x |

With `--warp` the arm replans at **6–30 Hz**, against 20 Hz in sim. ADMM exits
early whenever the primal residual is under `eps_r`/`eps_s`, which is why the
solve time varies -- a rising solve time means the two blocks have stopped
agreeing.

Three things this loop depends on, none of them obvious:

- **The publisher thread must never call into JAX.** It indexes a numpy table
  the solver prepared. Calling `interp` from the publisher while the solver runs
  on the same device segfaults the Warp backend, which captures CUDA graphs.
- **The planner's clock has to start near zero.** MJX runs in float32, where the
  spacing near a ROS epoch timestamp (~1.79e9) is 128 s -- adding a 0.75 s
  horizon to one is a no-op, and every knot in the plan collapses onto a single
  value. `Ros2Interface` offsets the clock by its own start time for this reason.
- **When a plan expires the arm stops.** Past the horizon the publisher sends
  zeros rather than holding the last sample. The interface watchdog cannot catch
  a stalled solver on its own, because the publisher is still sending.

Levers, in the order worth trying: `--warp` first, then `--num-samples`, then
`--n-admm`. `--vel-limit` is not a free knob -- the horizon is measured in time,
so lowering the speed shrinks how far the arm can plan to reach within it, and
below the block's friction threshold (`mu·m·g`) a push moves nothing at all.
Raise `HORIZON` alongside it if you need to go slower.

## Citation

```bibtex
@article{raicevic2026objectinformed,
  title   = {Object-Informed Model Predictive Path Integral Control
             for Non-Prehensile Robot Manipulation},
  author  = {Raicevic, Nikola and Kim, Hyomuk and Mulla, Shahid and
             Radhakrishnan, Bharath Raam and Yu, Chenbin and
             Lee, Ki Myung Brian and Atanasov, Nikolay},
  year    = {2026}
}
```

A fork of [**Hydrax**](https://github.com/vincekurtz/hydrax) by Vince Kurtz,
which provides the sampling-based MPC framework — controller/task
abstractions, spline parameterization, parallel MJX rollouts, domain
randomization, and every non-ADMM algorithm above. The ADMM object-informed
layer is our addition. Hydrax is itself inspired by
[MJPC](https://github.com/google-deepmind/mujoco_mpc).

```bibtex
@misc{kurtz2024hydrax,
  title  = {Hydrax: Sampling-based model predictive control on GPU
            with JAX and MuJoCo MJX},
  author = {Kurtz, Vince},
  year   = {2024},
  note   = {https://github.com/vincekurtz/hydrax}
}
```

The xArm6 model derives from [UFACTORY](https://www.ufactory.cc/)'s
published URDF; the Unitree G1 model is from
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) (see
[`oim/models/g1/LICENSE`](oim/models/g1/LICENSE)). Motion-capture references
come from [LocoMuJoCo](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets).

## License

MIT — see [LICENSE](LICENSE).
