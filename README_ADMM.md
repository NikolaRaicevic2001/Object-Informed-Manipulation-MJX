# ADMM: audit and proposed revisions

Notation follows the paper draft. Equation and section numbers in
parentheses — (6), (24), §IV-D — refer to it. Symbols carried over verbatim:

| | |
| --- | --- |
| $\mathbf{U}^o = \{w^o_t\}_{t=0}^{H^c-1}$, $\mathbf{U}^r = \{u^r_t\}_{t=0}^{H^c-1}$ | the two ADMM blocks (§IV-A) |
| $\mathbf{Z} = \{z_t\}$, $\mathbf{Y}^o$, $\mathbf{Y}^r$ | consensus variable and scaled duals |
| $A^o$, $A^r$ | extraction maps (24) |
| $H^o, H^r, H^c$; $K^o, K^r$; $N_{\mathrm{ADMM}}$ | horizons, samples, inner iterations |
| $\mathbf{P} = \operatorname{diag}(\rho_f,\rho_f,\rho_\tau)$, $\gamma$ | anisotropic penalty (§IV-D), proximal weight |
| $J_o, J_r$; $\ell_o,\ell_r,\ell_c,\ell_f$ | block costs (25)-(26); stage costs (16)-(17) |
| $r^{(l)}_m$, $d^{(l)}_m$, $m \in \{f,\tau\}$ | primal/dual residuals, per block |
| $D = \operatorname{diag}(\mu mg,\mu mg,cr\mu mg)^{-1}$ | limit-surface compliance (4) |
| $\Pi_\mathcal{Z}$, $\Pi_\mathcal{F}$ | consensus-set and limit-surface projections (14), (18) |
| $\omega^{(k)}$, $S^{(k)}$, $\lambda$, $\Sigma_u$ | MPPI weights, cost, temperature, covariance (7)-(10) |
| $\beta$, $\kappa$, $\Sigma_{\min}$ | dual momentum (28), variance schedule (§IV-D) |

New symbols introduced below: $S \in \mathbb{R}^{H^c\times H^c}$ strictly
lower-triangular ones (discrete integration); $\ominus,\oplus$ for $SE(2)$
difference and increment with $\theta$ wrapped to $(-\pi,\pi]$.

---

## 1. Where the implementation departs from the draft

Read this first — several of the draft's stated mechanisms are not in the
code, and two of the gaps are load-bearing for §3's measurements.

| Draft | Implementation | Consequence |
| --- | --- | --- |
| **(18)** every sampled wrench projected onto the limit-surface ellipsoid, $\tilde w^{(k)} = \Pi_\mathcal{F}(\bar w + \epsilon^{(k)})$ | **Not implemented.** Samples are box-clipped instead | See below — the realized box is not $\mathcal{F}$, and is not even contained in it |
| **(28)** dual update carries Heavy-Ball momentum $\beta(y^{i,(l)}-y^{i,(l-1)})$ to filter stick-slip noise | **Not implemented.** Duals are hard-clipped to $\pm y_{\max}=2\mu mg$ instead | The clip breaks the ADMM invariant $\sum_i y^i = 0$, which is why (27) must keep its dual terms; $\beta$ would not have broken it |
| **§IV-D** $\rho_f$ and $\rho_\tau$ adapted *independently* from their own $r^{(l)}_m, d^{(l)}_m$, $m\in\{f,\tau\}$ | One scalar residual over all $p^o$ components drives one scalar rule, applied to all of $\mathbf{P}$ | The anisotropic penalty is anisotropic only in its *initial value*; it cannot rebalance force against torque online |
| **§IV-D** $\Sigma^{(l+1)}_{u,m} = \max(\Sigma_{\min},\kappa\lVert r^{(l+1)}_m\rVert)$ replaces the covariance | Extra noise is **added** to the sampler's own $\Sigma_u$, and ships at $\kappa = 0$ | The variance schedule is inert. Measured: with it on, final $\lVert p^o-p^g\rVert$ 4.65 vs 2.01 off |
| **(19)** $\ell_o$ penalizes simulator contact force, $w^o_f(\max(\lambda_t-f_0,0))^2$ | Geometric hinge on the footprint's boundary samples, $w_{\text{obs}}\sum_j\max(\delta-\mathrm{sdf}(b_j),0)^2$ | Different physical quantity. Necessary — the object block has no simulator to report $\lambda_t$ — but it is a departure |
| **(23)** $\psi_{\text{tilt}} = \sqrt{\varrho^2+\varphi^2}$ | $1-\cos\psi_{\text{tilt}} = 1+R^{ee}_{33}$ | Deliberate: a linear penalty has constant restoring gradient and never arrested the measured drift at any $w_{\text{tilt}}$ |
| **§IV-A** $H^c \le \min(H^o,H^r)$ | $H^o = H^r = H^c$ enforced | — |

**The missing $\Pi_\mathcal{F}$ is a live bug, not just an omission.** The
object block's box bound returns $\pm D^{-1}$ in physical units, but the
action-to-wrench map then multiplies by $\tfrac12 D^{-1}$, so the realized
sample set is

```math
\Big\{\,w : |w_j| \le \tfrac12 (D^{-1}_{jj})^2 \,\Big\}
\;=\;
\big\{|f_x|,|f_y| \le 3.92\,f_{\max}\big\} \times \big\{|\tau| \le 0.235\,\tau_{\max}\big\},
```

against $\mathcal{F}$'s $f_{\max}=\mu mg = 7.848$, $\tau_{\max}=cr\mu mg =
0.471$. The block can propose forces **3.92×** past the friction cone and is
simultaneously starved to **0.235×** its torque authority. Compounding it, a
breakaway deadzone was added to (5) that zeroes any $\lVert w^o/D^{-1}\rVert
< 1$ — so the torque channel *alone can never reach threshold*, and the
object block cannot propose a pure rotation, on a scene set that is entirely
90° and 180° turns. Implementing (18) as written fixes both, and is
independent of everything below.

---

## 2. Assumption audit

Hong, Luo & Razaviyayn [6] — the result the draft cites for the proximal term
$\gamma$ — requires all of the following of (11):

| # | Assumption | Status |
| --- | --- | --- |
| A1 | $A_i x_i = z$ **linear** | **Violated.** $A^r(\mathbf{U}^r)_t = \hat w^o_t$ is a simulator rollout composed with twist inversion and a clip |
| A2 | $F_i$ has **Lipschitz gradient** | **Violated** for $J_r$; rigid contact is discontinuous |
| A3 | Each (13) update is an **exact minimization** | **Violated.** One MPPI pass; the (9) softmax average can increase $J_i$ |
| A4 | $\rho$ **above a Lipschitz-set threshold** | **Unknown and unenforced** |
| A5 | $\rho$ **fixed or non-decreasing** | **Violated.** The §IV-D rule can halve $\rho_m$ — and does, see §3 |
| A6 | Duals updated **unclipped** | **Violated** by the clip that replaced (28)'s $\beta$ |
| A7 | $\mathcal{L}_\rho$ **bounded below** | Plausible; costs are nonnegative |

The draft's own §IV-B acknowledges A1's failure — "$A^r(\mathbf{U}^r)_t =
\hat w^o_t$ implicitly depends on the object state $x^o_t$, which technically
violates the strict block-separability assumption" — and argues $\gamma$
restores it within a trust region. That argument bounds $\lVert \mathbf{U}^r
- \mathbf{U}^{r,(l)}\rVert$, but $A^r$ is discontinuous in $\mathbf{U}^r$: an
arbitrarily small control change can make or break contact and move $\hat
w^o_t$ by the full $\pm D^{-1}$. A trust region does not tame a
discontinuity, only a large derivative.

---

## 3. What the recorded runs show

**Point robot, `shelf_gap`, $H^c{=}32$, $K^o{=}K^r{=}128$,
$N_{\mathrm{ADMM}}{=}4$, seed 1**, matched against flat MPPI at *identical*
seed, start, goal, costs and sampler budget — only the decomposition varies:

| | ADMM | flat MPPI |
| --- | --- | --- |
| control steps to goal | 875 | **342** |
| planning rate | 6.27 Hz | **37.44 Hz** |
| planning time to goal | $\approx 140$ s | $\approx 9$ s |

Residuals over that run. $\lVert r\rVert$ is Frobenius over $(2H^c,p^o) =
(64,3)$ entries normalized by $D^{-1}$, so per-entry RMS is $\lVert
r\rVert/8$ — a fraction of the maximum transmissible wrench:

| Quantity | Value | Reading |
| --- | --- | --- |
| mean $\lVert r\rVert$ | 4.79 | per-entry RMS **0.60** |
| min $\lVert r\rVert$ | 1.73 | per-entry RMS 0.22, the best it ever gets |
| mean $\lVert d\rVert$ | 9.68 | |
| steps with $\lVert r\rVert \le \epsilon_r$ | **0 / 875** | the (Alg. 4, line 9) break never fires |
| $\rho$ adaptations | **1 / 875** (step 12, $10\to5$) | static for the remaining 862 |
| $\lVert r\rVert$ over $l = 0..3$ | 4.92, 4.98, 4.87, 4.73 | $-3.8\%$ across the whole $N_{\mathrm{ADMM}}$ budget |

**xArm6**, last committed run per scene (ADMM $H^c{=}32,K{=}128$; MPPI
$H{=}16,K{=}64$ — the baseline had *half* the budget):

| Scene | ADMM $\lVert p^o-p^g\rVert$, 500 steps | MPPI, 1000 steps |
| --- | --- | --- |
| open_table | 0.631 | **0.171** |
| single_obstacle | 0.436 | **0.130** |
| shelf_gap | 0.604 | **0.509** |
| ycb_clutter | 0.176 | **0.138** |
| icra_sign | **0.013** (reached) | 1.012 |

Four conclusions:

1. **The blocks never agree.** $A^o$ and $A^r$ differ by 22–60% of $D^{-1}$
   at every $t$, for the whole run. Not slow convergence — no convergence.
2. **$N_{\mathrm{ADMM}}$ buys nothing.** $\lVert r\rVert$ moves 3.8% over
   four inner iterations. Iterations 2–4 are ~75% of the cost for ~3% of the
   progress.
3. **The §IV-D penalty rule is inert in the useful direction.** It fired once,
   *downward*, in the first 12 steps — violating A5 at the only moment it
   acted — then never again. The rule needs a 10× residual ratio; the
   observed ratio is 2–4×.
4. **ADMM loses to its own baseline on 4 of 5 scenes**, at 6–12× the per-step
   cost. `icra_sign` is the exception, and it is the scene whose difficulty is
   object-level routing among seven fixed glyphs — precisely what the
   decomposition exists to provide.

Conclusion 4 is the case for repairing the layer rather than dropping it.

---

## 4. Proposal 1 — take the object trajectory as the consensus variable

Replace §IV-A's $z_t \triangleq w^o_t \in \mathcal{Z} \triangleq
\mathbb{R}^{p^o}$ with

```math
z_t \;\triangleq\; x^o_t \;\in\; \mathcal{Z} \;\triangleq\; SE(2).
```

The blocks are unchanged — $\mathbf{U}^o = \{w^o_t\}$, $\mathbf{U}^r =
\{u^r_t\}$ — but the extraction maps (24) become

```math
A^o(\mathbf{U}^o)_t = x^o_t
= x^o_0 + \Delta t\, D \sum_{k=0}^{t-1} w^o_k ,
\qquad
A^r(\mathbf{U}^r)_t = x^o_t \ \text{read from the simulator state.}
```

Written over the horizon, $A^o$ is (5) integrated, which is a matrix product:

```math
A^o(\mathbf{U}^o) \;=\; \mathbf{1}_{H^c}\,(x^o_0)^\top \;+\; \Delta t\; S\, \mathbf{U}^o D .
```

**$A^o$ is affine in $\mathbf{U}^o$** — A1 holds exactly on the object side,
not by the triviality of a selection matrix but by the linearity of the limit
surface (4). And $A^r$ is now the object's pose taken straight from the
rollout: no twist inversion $\hat w = D^{-1}\dot x^o$, no clip to $\pm
D^{-1}$, no per-embodiment estimator.

Because $\mathcal{Z} = SE(2)$ is a manifold, the draft's remark that
"$\Pi_\mathcal{Z}$ becomes the identity mapping, reducing (27) to a simple
average" **no longer holds**. $\Pi_\mathcal{Z}$ is the angle wrap, the duals
live in the tangent space (they are twists, $\mathbb{R}^3$, not poses), and
(27)-(28) are taken about a base point $\bar z_t = z^{(l)}_t$:

```math
z^{(l+1)}_t = \bar z_t \,\oplus\, \tfrac{1}{2}\Big[\big(A^o(\mathbf{U}^{o,(l+1)})_t \ominus \bar z_t\big) + y^{o,(l)}_t + \big(A^r(\mathbf{U}^{r,(l+1)})_t \ominus \bar z_t\big) + y^{r,(l)}_t\Big],
```

```math
y^{i,(l+1)}_t = y^{i,(l)}_t + \big(A^i(\mathbf{U}^{i,(l+1)})_t \ominus z^{(l+1)}_t\big) + \beta\big(y^{i,(l)}_t - y^{i,(l-1)}_t\big),
\qquad i \in \{o,r\} .
```

$\mathbf{P}$ keeps its §IV-D role and its anisotropy, with the block index
$m$ ranging over position and orientation rather than force and torque:

```math
\mathbf{P} = \operatorname{diag}(\rho_p,\rho_p,\rho_\theta),
\qquad m \in \{p,\theta\},
\qquad
r^{(l)} = \begin{bmatrix} A^o \ominus \mathbf{Z} \\ A^r \ominus \mathbf{Z}\end{bmatrix},
\quad d^{(l)} = \mathbf{P}\big(\mathbf{Z}^{(l+1)} \ominus \mathbf{Z}^{(l)}\big).
```

The §IV-D argument for anisotropy carries over verbatim — metres and radians
are as incommensurable as newtons and newton-metres — with normalization
$\operatorname{diag}(\varsigma,\varsigma,1)$ for a characteristic length
$\varsigma$ (the object's bounding radius), so $\rho_m$ and the tolerances
stay scale-free and $\lVert r\rVert$ reads in body lengths and radians.

**Two consequences worth stating separately.**

*(17)'s $\ell_c$ and (26)'s penalty become the same term.* The robot block
currently carries both

```math
\ell_c(x^o_t, x^{o*}_t) = d^2(x^o_t, x^{o*}_t)
\qquad\text{and}\qquad
\tfrac{1}{2}\big\lVert A^r(\mathbf{U}^r)_t - z_t + y^{r}_t\big\rVert^2_{\mathbf{P}} ,
```

and under $A^r(\mathbf{U}^r)_t = x^o_t$ the second *is* the first, with $z_t$
in place of $x^{o*}_t$ and a dual offset. So $\ell_c$ should be deleted and
its job taken over by the ADMM penalty. This removes a double-counted
coupling and disentangles $w^o_d, w^o_\theta$ — which presently act as both
goal-tracking *and* coupling weights — from $\mathbf{P}$, which is what is
supposed to set coupling strength. The reference the robot block tracks stops
being $x^{o*}$ (one block's unilateral proposal) and becomes $\mathbf{Z}$
(the negotiated one), which is what a consensus method should have been doing
from the start.

*The object subproblem (25) becomes least-squares.* With $A^o$ affine,
$J_o + \tfrac\gamma2\lVert\cdot\rVert^2 + \tfrac12\lVert\cdot\rVert^2_\mathbf{P}$
is quadratic in $\mathbf{U}^o$ apart from $\ell_o$'s obstacle hinge and the
deadzone. Its minimizer has a closed form — a banded solve, $O(H^c)$ —
instead of $K^o$ sampled rollouts. Not required here; it is the door
Proposal 3 walks through.

| | |
| --- | --- |
| **Main idea** | Set $z_t \triangleq x^o_t \in SE(2)$ instead of $z_t \triangleq w^o_t \in \mathbb{R}^{p^o}$. $A^o$ becomes (5) integrated, $\mathbf{1}(x^o_0)^\top + \Delta t\,S\mathbf{U}^oD$ — affine; $A^r$ becomes the object pose read from the simulator state. Duals move to the tangent space, $\Pi_\mathcal{Z}$ becomes the angle wrap, and (17)'s $\ell_c$ is absorbed into (26)'s penalty. |
| **Main benefit** | Both blocks report the *same directly-observed state* rather than one decision and one force estimate. (i) A1 holds exactly on the object side, and A2 improves on the robot side: $x^o_t$ is an integral of the dynamics, continuous through the contact breaks where $\hat w^o_t$ is discontinuous — which is also the stick-slip noise (28)'s $\beta$ exists to filter, so $\beta$ and the dual clip that replaced it may both become unnecessary, restoring A6. (ii) It deletes the twist inversion and the $\pm D^{-1}$ clip, the entire reason xArm6's half of the consensus is structurally noisier than the point robot's — an arm cannot read $\hat w^o_t$ from a single pair of DOFs at all, so it is forced onto model inversion. (iii) A zero-residual point *exists*: both blocks propose trajectories of the same object in $SE(2)^{H^c}$, whereas the object's reachable wrench set and the robot's realizable wrench set need not intersect. (iv) $\lVert r\rVert$ becomes interpretable in metres and radians, and equals the divergence between the two blocks' predicted object paths that the overlay already draws. |
| **Main drawback** | $A^r$ remains a simulator rollout, so A1 is repaired on one side only — better conditioning, not a proof. The $SE(2)$ structure is a real source of bugs: every difference must use $\ominus$, and **the tabletop goal is $\theta^g = \pi$, sitting exactly on the branch cut**, so an unwrapped subtraction yields a $2\pi$ residual precisely at the goal. $\mathbf{Z}$ and $\mathbf{Y}^i$ stop being the same kind of object (pose vs. twist). Compute is unchanged. Every recorded wrench-consensus run becomes incomparable and must be regenerated. |

---

## 5. Proposal 2 — read $A^r$ off the population already rolled out

After the robot update (26), the implementation re-simulates the updated
nominal $\bar{\mathbf{U}}^r$ **alone** to evaluate $A^r(\mathbf{U}^{r,(l+1)})$
for (27) — a second, batch-1 rollout of $H^c$ steps, sequential with the
batched one over $K^r$ samples. On a GPU a batch-1 rollout is latency-bound,
so it does not cost $1/K^r$ of the batched rollout; it costs a large fraction
of it.

Every sample's $A^r(\mathbf{U}^{r,(k)})$ is already computed inside the
rollout that scores $S^{(k)}$. Reuse it with the same weights (10) that
already define the update (9):

```math
\widehat{A^r}_t \;=\; \sum_{k=1}^{K^r} \omega^{(k)} A^r\big(\mathbf{U}^{r,(k)}\big)_t ,
\qquad
\omega^{(k)} = \frac{\exp\!\big(-\tfrac1\lambda(S^{(k)}-S_{\min})\big)}{\sum_j \exp\!\big(-\tfrac1\lambda(S^{(j)}-S_{\min})\big)} .
```

**The cost is measurable from §3.** With $C_r$ the batched robot rollout,
$C_o$ the object block and $C_1$ the batch-1 rollout,

```math
\frac{C_{\mathrm{ADMM}}}{C_{\mathrm{flat}}}
= \frac{N_{\mathrm{ADMM}}\,(C_o + C_r + C_1)}{C_r} = \frac{37.44}{6.27} = 5.97
\;\Longrightarrow\;
\frac{C_o + C_1}{C_r} \approx 0.49 \quad (N_{\mathrm{ADMM}}=4).
```

The object block and the extra rollout together are **~33% of ADMM's per-step
cost** — and the object block is analytic, so nearly all of it is one
redundant simulator call.

| | |
| --- | --- |
| **Main idea** | Estimate $A^r$ for (27) as the MPPI-weighted average $\sum_k \omega^{(k)} A^r(\mathbf{U}^{r,(k)})$ over the samples already rolled out, instead of re-simulating $\bar{\mathbf{U}}^{r,(l+1)}$ in a separate batch-1 rollout. |
| **Main benefit** | Free — the values are already computed while scoring $S^{(k)}$. Removes one sequential simulator call per inner iteration $l$, projected **$\approx 1.5\times$** end to end (6.27 → ~9.3 Hz on the §3 run). Also removes an inconsistency: (9) executes the weighted mean, so scoring consensus against the rollout of that mean measures a trajectory the robot never commits to. |
| **Main drawback** | $\sum_k \omega^{(k)} A^r(\mathbf{U}^{(k)}) \neq A^r(\sum_k \omega^{(k)}\mathbf{U}^{(k)})$ — the average of rollouts is not the rollout of the average, and the gap grows with contact nonlinearity and sample spread. It is the same approximation (9) already makes for the control, so no new class of error, but $A^r$ becomes a biased estimate of what will execute. The 0.49 figure is fitted from one measurement and absorbs $C_o$, so the realized gain will be somewhat under $1.5\times$. |

---

## 6. Proposal 3 — spend inner iterations on the block that is free

(25) and (26) each run **once** per inner iteration $l$, with $K^o = K^r$.
But the object block has no simulator: $K^o$ evaluations of (5) against $K^r$
full contact solves is a ratio of $10^{-2}$–$10^{-3}$. The budget is
allocated as though the two cost the same.

Let $M$ be object updates per robot update. Per control step,

```math
C \;=\; N_{\mathrm{ADMM}}\big(M\,C_o + C_r\big), \qquad C_o \ll C_r ,
```

so $M \approx 5$–$10$ is nearly free. Each round then hands (27) an object
proposal refined against fixed $(z^{(l)}, y^{o,(l)}, \mathbf{P})$ until it
stops moving, rather than one MPPI pass over one fresh batch.

That targets a known defect directly. The consensus EMA that the
implementation added (not in the draft) exists because each round's $A^o$ is
a single noisy resampling estimate rather than a converged proposal — so the
disagreement entering (27) is dominated by sampling variance rather than by
the blocks genuinely disagreeing. Converging the *cheap* block removes that
variance at its source on one side instead of filtering it afterwards.

Under Proposal 1 this goes further: $A^o$ affine makes (25) least-squares, so
$M$ sampled sweeps can be replaced by one banded solve for everything but the
obstacle hinge.

| | |
| --- | --- |
| **Main idea** | Decouple the blocks' update counts: run (25) $M \approx 5$–$10$ times per single (26), against fixed $(z^{(l)}, y^{o,(l)}, \mathbf{P})$. Optionally, under Proposal 1, replace those sweeps with the closed-form least-squares solution of (25)'s quadratic part. |
| **Main benefit** | The expensive block's budget is untouched; cost rises by $M C_o \ll C_r$, a few percent. In exchange $A^o$ becomes converged, which attacks the resampling variance the consensus EMA currently papers over and makes $\lVert r\rVert$ measure genuine block disagreement rather than sampler noise. (13) is sequential by construction, so unequal sweep counts are legitimate ADMM. |
| **Main drawback** | Converging one block against a stale $z^{(l)}$ can overshoot: the object block commits harder to a proposal the robot has not been consulted on, which can *increase* the disagreement it is meant to reduce, and it interacts with $\gamma$ (now anchoring $M$ sweeps rather than one). Adds a knob. The least-squares variant is a substantial rewrite and must still handle $\ell_o$'s obstacle hinge and the deadzone, both nonconvex, by sampling or linearization. |

---

## 7. Proposal 4 — exit on stagnation, and measure $\lVert r\rVert$ per entry

Two defects in Alg. 4's line 9, both visible in §3.

*The tolerance is horizon-dependent.* $\lVert r\rVert$ is Frobenius over
$2H^c$ entries, so it grows like $\sqrt{2H^c}$ and the tolerance silently
re-tunes with the horizon. This is why the draft's $\epsilon_r = 0.05$ was
raised to $0.5$ in the code and is *still* unreachable. Report the RMS:

```math
\lVert r\rVert_{\mathrm{RMS}} = \frac{\lVert r\rVert_F}{\sqrt{2H^c}},
\qquad \text{§3: } \tfrac{4.79}{8} = 0.60 .
```

*The test asks the wrong question.* Line 9 breaks when the blocks **agree**.
They never do, so all $N_{\mathrm{ADMM}}$ iterations always run. But an
iteration is worth paying for while it *changes* something, and §3 shows
$\lVert r\rVert$ moving 3.8% across the whole budget. Break on stagnation of
$\mathbf{Z}$ instead:

```math
\text{break if}\quad
\big\lVert \mathbf{Z}^{(l+1)} \ominus \mathbf{Z}^{(l)}\big\rVert_{\mathrm{RMS}} \le \epsilon_z
\quad\text{or}\quad
\lVert r^{(l+1)}\rVert_{\mathrm{RMS}} \le \epsilon_r .
```

On the §3 traces this fires at $l = 1$–$2$, an effective $\bar N \approx 1.5$
against a budget of 4.

| | |
| --- | --- |
| **Main idea** | Break Alg. 4's loop on **stagnation of $\mathbf{Z}$**, not only on agreement, and report $r^{(l)}_m, d^{(l)}_m$ as per-entry RMS so $\epsilon_r,\epsilon_s,\epsilon_z$ are independent of $H^c$. |
| **Main benefit** | Turns $N_{\mathrm{ADMM}}$ from a fixed cost into a budget spent only while it buys progress: projected **$\approx 2.7\times$** on the §3 run ($4 \to 1.5$ effective iterations), at zero implementation risk — both quantities already exist, since $d^{(l)} = \mathbf{P}(\mathbf{Z}^{(l+1)}-\mathbf{Z}^{(l)})$ *is* the stagnation measure up to $\mathbf{P}$. Makes $\epsilon_r$ meaningful for the first time. |
| **Main drawback** | Stagnation is not convergence: exiting on a flat residual bakes the disagreement in rather than surfacing it, and if Proposals 1/3 make the iteration genuinely productive, an aggressive $\epsilon_z$ would cut it off before it pays. The threshold is scene- and $\mathbf{P}$-dependent, so it needs a floor $N_{\min}\ge 2$ to avoid degenerating into flat MPPI with one extra penalty term. |

---

## 8. Cost projection

```math
\frac{C_{\mathrm{ADMM}}}{C_{\mathrm{flat}}}
= \bar N\left(\frac{M\,C_o + C_r + [\,C_1\,]}{C_r}\right)
```

| Configuration | $\bar N$ | ratio | projected rate (§3 run) |
| --- | --- | --- | --- |
| today | 4 | 5.97 | 6.27 Hz |
| + P2 (drop $C_1$) | 4 | $\approx 4.0$ | $\approx 9.3$ Hz |
| + P4 (stagnation break) | $\approx 1.5$ | $\approx 1.5$ | $\approx 25$ Hz |
| + P3 ($M=8$) | $\approx 1.5$ | $\approx 1.7$ | $\approx 22$ Hz |
| flat MPPI, same $K,H$ | — | 1.00 | 37.44 Hz |

Projections from a model fitted to one measurement, not results. The point is
the direction: **the decomposition can be brought within ~1.7× of flat
MPPI's per-step cost without touching $K^o$, $K^r$ or $H^c$**, because two
thirds of the current overhead is one redundant simulator call plus inner
iterations that change nothing. At 22 Hz against a 20 Hz control rate the
method is realizable on this hardware; at 6 Hz it is not.

---

## 9. Comparison and order

| | P1: $z_t = x^o_t$ | P2: free $A^r$ | P3: $M$ object sweeps | P4: stagnation break |
| --- | --- | --- | --- | --- |
| Assumption repaired | A1 (object side), A2, possibly A6 | — | partially A3 | — |
| Effect on $\lVert r\rVert$ | large; makes a zero exist | none | moderate | none (changes when we stop) |
| Effect on cost | none | $\div 1.5$ | $\times 1.1$ | $\div 2.7$ |
| Effort | moderate | low | low → high | low |
| Invalidates recorded runs | yes | marginally | marginally | no |

1. **(18) as written**, plus **P2** and **P4**. All three are cheap, none
   commits to a structural change, and together they decide whether the layer
   is fast enough to be worth improving.
2. **P1**. The structural fix — do it once sweeping $\mathbf{P}$ and
   $\varsigma$ is affordable.
3. **P3**, the $M>1$ sweeps first; the least-squares object block only if P1
   lands and (25) is then the bottleneck.

**Deferred: a descent safeguard on (13).** Accepting a block update only on
sufficient decrease of $\mathcal{L}_\rho$ would repair A3 and A5 and make
$\mathcal{L}_\rho$ monotone. It costs one extra rollout per block (~1.5%) and
the acceptance rate is the only cheap way to learn whether MPPI is a descent
method here at all. Deferred rather than dropped because its value is mostly
*informational*: under P4 a rejected step and a stagnated step lead to the
same action.

**Withdrawn: a contact-point parameterization of $\mathbf{U}^o$.** An earlier
version of this document proposed the object block decide $a_t = [p_x,p_y,
f_n,f_t]$ with $A^o = J_c(p)^\top f$, so every proposal is realizable by
construction. That is contrary to the decomposition's premise — the object
planner should not model how the robot produces the motion, and §III-A's whole
argument is that the coupling enters *only* through $w^o_t$. P1 dissolves the
issue anyway. Note this is **not** the same as (18): the ellipsoid
$\mathcal{F}$ is a property of the object and its support surface, not of the
robot's mechanism, so implementing (18) respects the premise while a contact
parameterization does not.

---

## 10. What would still not be proven

Even with all four, a convergence proof is **out of reach while the simulator
defines $J_r$**: A2 needs a Lipschitz gradient and rigid contact is
discontinuous. P1 improves conditioning — an integrated pose is far smoother
than a differentiated-then-clipped wrench — but does not make it Lipschitz.

Two defensible positions, both worth stating rather than glossing:

1. **Claim what is true.** With P1 and the descent safeguard, the iteration is
   a monotone descent method on $\mathcal{L}_\rho$ whose fixed points are
   stationary points of a well-defined lifted problem. That is strictly more
   than the draft currently supports.
2. **Prove it where it is provable.** `oim/sim2d` has an *analytic* contact
   model — the wrench is computed in closed form and (4) is linear, so under
   P1 both $A^o$ and $A^r$ are tractable and A2 is plausible. State and verify
   the theorem there, and present the 3D results as empirical transfer.

Position 2 is the stronger paper, and the codebase is already set up for it:
one `ADMM` class drives both worlds, so a theorem in 2D and experiments in 3D
are the same algorithm, not an analogy.

---

## References

Numbering follows the paper.

[6] Hong, Luo & Razaviyayn, *Convergence analysis of ADMM for a family of
nonconvex problems*, SIAM J. Optim. 26(1), 2016.

[7] Ouyang, He, Tran & Gray, *Stochastic alternating direction method of
multipliers*, ICML 2013.
