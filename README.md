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
| `--samples`, `--horizon` | from config | Rollouts per block, consensus horizon $H$ |
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

| Output | When |
| --- | --- |
| `results/runs/*.json` | *3D:* `--headless`, *2D:* always — settings, scene, per-step states/controls/wrenches/residuals/timings |
| `recordings/*.png` | unless `--no-plot` — trajectory + residuals + accumulated cost terms |
| `recordings/*.gif` / `*.mp4` | `--animate` / `--record` |

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
| `eps_d` | $\epsilon_d$ | position error, all trials |
| `eps_d^s` | $\epsilon_d^s$ | position error, successful trials only; blank if none succeeded |
| `theta` | — | orientation error; not in the paper |
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

| $N_{\mathrm{ADMM}}$ | $\rho_0$ | $\gamma$ | $\epsilon_r = \epsilon_s$ | $y_{\max}$ | $H$ | samples |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 10.0 | 0.1 | 0.5 | $2\mu m g = 15.696$ | 25 (xArm6), 15 (point) | 64 |

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
| **Object bound is not $D^{-1}$** | `object_action_bounds` returns $\pm D^{-1}$ in physical units, but `object_action_to_consensus` then multiplies by `action_scale` $= \tfrac12 D^{-1}$, which assumes a unit sample. The two conventions disagree, so the realized box is $\pm\tfrac12 (D^{-1})^2$ — 3.92× the friction-cone limit in force, 0.235× in torque. |
| **Penalty is not $\Delta t$-weighted** | Both blocks compute $\Delta t\,\ell + \tfrac{\rho}{2}\lVert\cdot\rVert^2$. They agree, so the fixed point is well defined, but the penalty's effective weight scales as $1/\Delta t$ — changing the planning timestep silently re-tunes $\rho$. |
| **Residuals unnormalized by horizon** | $\lVert r\rVert$ is a Frobenius norm over $(2H,3)$, not an RMS, so it grows like $\sqrt{2H}$. Residuals are $O(1)$ at both horizons in use, so $\epsilon_r = \epsilon_s = 0.5$; the paper's $0.05$ is unreachable here and the early exit would never fire. |
| **Variance annealing additive, and off** | Most samplers expose no mutable covariance, so the wrappers *add* $\mathrm{clip}(\kappa\lVert r\rVert, \sigma_{\min}, \sigma_{\max})$ rather than replacing $\Sigma_u$. The upper clip is required ($\kappa\lVert r\rVert$ is otherwise a positive feedback loop), and since $\lVert r\rVert$ does not converge here the clip binds permanently — so $\kappa = 0$. Measured over 600 steps at identical seed: final position error 4.65 with annealing on, 2.01 off. |
| **$\rho$ and $\lVert r\rVert$ persist** | `rho_init` is only the $t=0$ value; both are carried in the policy parameters and never reset, so they drift over a run. $\lVert d\rVert$ is re-initialized to $\infty$ each step, so the first iteration never exits early. |
| **Dual anti-windup** | Duals clipped to $\pm y_{\max}$. This is why the $z$-update keeps the dual terms: $\sum_i y^i = 0$ is an ADMM invariant that would make $z = \tfrac12(A^o + A^r)$ equivalent, but clipping breaks it. |
| **Warm start is not a pure shift** | $z, y^o, y^r$ and a direct-wrench object block shift by one and zero-fill the tail; a structured action space repeats the last value instead, since zero need not be feasible there (a zero contact point is the object's origin, not on its boundary). The robot mean is re-interpolated onto shifted spline knot times, not shifted. |
| **Horizons** | The formulation permits $H^c \le \min(H^o, H^r)$; the implementation enforces $H^o = H^r = H^c$. |
| **Tilt is degenerate for the 3D point pusher** | $\psi_{\text{tilt}}$ reads the trace site's rotation, which for the point pusher never changes, so its raw contribution is a constant $2w_{\text{tilt}}$. That cancels in every sampler's cost differences only while $\phi \equiv 1$; with `shaping_fade_dist > 0` (both configs ship 0.15) the constant is scaled by $\phi(x^o_t)$ and becomes a pose-dependent term — an extra pull toward the goal, worth $2w_{\text{tilt}}$ at the fade radius, that no longer cancels. |
| **Limit surface has a real deadzone** | $x^o_{t+1} = x^o_t + \Delta t\, D\, w^o_t$ extends proportionally through $w^o = 0$, but $D^{-1}$ is the friction-cone limit, not a soft compliance: a wrench below it should produce zero motion, not a small one. Root cause of a persistent near-goal stall, since the object's optimal wrench under quadratic effort cost shrinks continuously with position error, and once it falls under $D^{-1}$ the real block does not move even though the model still predicts progress. `PlanarPushingObject.step()` now zeroes any wrench with $\lVert w^o / D^{-1} \rVert < 1$ before integrating. |
| **Object action snapped to breakaway near the goal** | The deadzone fix alone was not enough: the optimizer's own exploration noise, tuned for smooth tracking rather than escaping a deadzone, rarely sampled far enough past a small mean to land above threshold. Widening it did, but made every sample noisier, not only the near-goal ones. `project_object_action` instead rescales any nonzero sample up to $\lVert w^o / D^{-1} \rVert = 1$ in the same direction, gated on $\lVert p - p_g \rVert < 0.3$ m so it does not affect small pulls unrelated to goal proximity, e.g. the ADMM proximal term. |

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
│   ├── admm.py           ADMM loop; ConsensusSpace, WrenchConsensus;
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
├── run_launch.py         sweep driver;  run_eval.py  post-hoc metrics
├── configs/              point.yaml, xarm6.yaml (defaults per robot);
│                         run_launch_config.yaml (the sweep definition)
├── tasks/  models/       MuJoCo tasks; MJCF scenes and meshes
└── utils/                scenes.py (the 3D scene registry), plotting.py,
                          poses.py (examples/poses/*.yaml), spline, video,
                          results.py (run files), metrics.py
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

`oim/real3d/` runs the same ADMM push-T controller on a physical UFACTORY
xArm6. The planner (`ADMM.optimize`), the task cost and the MJX rollouts are
reused verbatim from the simulation path; only the outer loop's I/O changes:

```
sim3d:  mjx_data <- mj_data ;      mj_data.ctrl = u ; mujoco.mj_step(...)
real3d: mjx_data <- ROS sensors ;  publish u to the arm's velocity controller
```

| File | Role |
| --- | --- |
| [`oim/real3d/interface.py`](oim/real3d/interface.py) | `RobotWorldInterface` (the I/O seam): `MujocoMockInterface` for laptop testing, `Ros2Interface` for hardware |
| [`oim/real3d/run_real.py`](oim/real3d/run_real.py) | the closed loop -- the hardware counterpart of `sim3d/run.py::_run` |
| [`examples/pusht_real.py`](examples/pusht_real.py) | entry point; same controller build as `examples/clutter.py` |
| [`oim/real3d/scripts/`](oim/real3d/scripts/) | RViz scene markers, state replay, contact analysis |

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
