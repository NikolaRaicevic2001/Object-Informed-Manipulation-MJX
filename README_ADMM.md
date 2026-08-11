# ADMM: theoretical audit and three proposed fixes

What this document is: an honest account of what the implemented ADMM layer
actually optimizes, which of the cited convergence results apply to it
(none, as written), what the recorded diagnostics say about why, and three
concrete modifications ranked by how much they buy.

Read `README.md` §Method first for notation. This document only argues about
the ADMM layer; the cost functions, limit-surface model and scene set are
taken as given.

---

## 1. The algorithm as implemented

The paper poses (eq. 6) a consensus problem over the wrench:

```math
\min_{\mathbf{W}^o,\mathbf{U}^r,\mathbf{Z}} \sum_t \big(\ell_o + \ell_r\big) + \ell_f
\quad\text{s.t.}\quad w^o_t = z_t,\qquad \hat{w}^o_t = z_t
```

and solves it with the $N=2$ global-variable-consensus form (eq. 11):

```math
\min_{x_1,\dots,x_N,z} \sum_i F_i(x_i) \quad\text{s.t.}\quad A_i x_i = z
```

The two constraints are **not the same kind of object**, and this is the
root of everything below:

| Block | Constraint | Form | Linear in the block variable? |
| --- | --- | --- | --- |
| object | $A^o(\mathbf{U}^o)_t = w^o_t = z_t$ | selection matrix | yes |
| robot | $A^r(\mathbf{U}^r)_t = \hat w^o_t = \Phi_t(\mathbf{U}^r) = z_t$ | MJX rollout, then twist inversion, then a clip | no, and not continuous |

$\Phi$ is a physics simulator composed with $\hat w = D^{-1}\dot x^o$ and a
clip to $\pm D^{-1}$. It is discontinuous wherever contact makes or breaks —
which, in a pushing task, is everywhere that matters.

So problem (6) is not an instance of (11). It is

```math
\min_{x_o,x_r,z} F_o(x_o) + F_r(x_r) \quad\text{s.t.}\quad x_o = z,\ \ \Phi(x_r) = z
```

a **nonlinearly-coupled** consensus problem. Every result cited in the paper
is for linearly-coupled problems.

---

## 2. Assumption audit

Hong, Luo & Razaviyayn (2016) [5] — the convergence result the paper leans on
for the proximal term — needs all of the following. Status in this codebase:

| # | Assumption | Needed for | Status here |
| --- | --- | --- | --- |
| A1 | Constraints $A_i x_i = z$ are **linear** | The whole framework; the AL descent lemma | **Violated.** $A^r$ is a simulator rollout |
| A2 | Each $F_i$ has **Lipschitz gradient** | Bounding the dual ascent by the primal descent | **Violated** for the robot block; contact is discontinuous |
| A3 | Each block update is an **exact minimization** (or gives sufficient decrease) | The descent lemma | **Violated.** One MPPI pass over one fresh batch, a softmax-weighted average that can increase the objective |
| A4 | $\rho$ **exceeds a threshold** set by the Lipschitz constants | Making the AL decrease rather than increase | **Unknown and unenforced.** $\rho_0=10$, and the adaptive rule may halve it |
| A5 | $\rho$ is **fixed or non-decreasing** | Monotonicity of the AL | **Violated in principle.** The rule can halve $\rho$ |
| A6 | Duals updated as $y \leftarrow y + Ax - z$, **unclipped** | $\sum_i y^i = 0$; the KKT interpretation of a fixed point | **Violated.** Duals are clipped to $\pm y_{\max}=2\mu mg$ |
| A7 | AL is **bounded below** | Existence of a limit | Plausible; costs are nonnegative |

Ouyang et al. (2013) [6] — cited for the variance schedule — is a stochastic
ADMM result and requires **decaying** step sizes. The implementation ships
$\kappa = 0$ (annealing off entirely) and lets $\rho$ grow, which is the
opposite schedule.

**Conclusion of the audit.** Six of seven assumptions fail. The algorithm is
a reasonable *heuristic* coordination layer, but no convergence claim in the
literature currently covers it, and — more practically — its fixed points are
not KKT points of eq. 6, because of A1 and A6 alone.

One structural consequence worth stating separately. With $N=2$ and the
standard $z$-update, ADMM maintains the invariant

```math
\textstyle\sum_i y^{i,(l+1)} = \sum_i y^{i,(l)} + \sum_i A^i x_i - 2z^{(l+1)} = 0
```

so $z^{(l+1)} = \tfrac12(A^o + A^r)$ exactly, and the dual terms in the
$z$-update are redundant. Clipping the duals breaks this invariant, which is
precisely why the implementation must keep those terms. That is a symptom,
not a design choice.

---

## 3. What the recorded runs actually show

From `oim/results/runs/pusht3d_xarm6_shelf_gap_admm_mppi_mppi_*.json`
($H=35$, $N_{\mathrm{ADMM}}=8$, $\rho_0=10$, $\gamma=0.1$, 100 control steps).
$\lVert r\rVert$ is Frobenius over $(2H,3)=(70,3)$ normalized entries, so the
per-entry RMS below is **as a fraction of the friction-cone limit**:

| Quantity | Value | Reading |
| --- | --- | --- |
| mean $\lVert r\rVert$ | 5.065 | per-entry RMS **0.350** |
| min $\lVert r\rVert$ | 3.188 | per-entry RMS 0.220, the best it ever gets |
| mean $\lVert d\rVert$ | 20.231 | |
| mean $\lVert z^{(l+1)}-z^{(l)}\rVert$ | 2.023 | per-entry RMS **0.197 per inner iteration** |
| $\rho$ over the run | 10.0 → 10.0, never changed | |
| $\rho$ rule fired up | 0 / 100 steps | |
| $\rho$ rule fired down | 0 / 100 steps | |
| early exit fired | 0 / 100 steps | $\lVert r\rVert \ge 3.19 \gg \epsilon_r = 0.5$ |

Three facts follow directly, and they are not subtle:

1. **The two blocks never agree.** They differ by ~35% of the maximum
   transmissible wrench, at every timestep of the horizon, for the entire
   run. This is not "converging slowly", it is not converging.
2. **The consensus variable never settles.** $z$ moves by ~20% of the
   friction-cone limit *per inner iteration*, at iteration 8 as much as at
   iteration 1.
3. **The adaptive penalty rule is inert.** It requires a 10× ratio between
   residuals; the actual ratio hovers around 4. $\rho$ has never once
   adapted in a recorded run. The "adaptive" in "adaptive penalty" is
   currently decorative.

So the ADMM layer is paying 10–25× the per-step cost of flat MPPI (2.3–5.9 Hz
vs 57–60 Hz) to run eight inner iterations that do not converge to anything.
That is the performance gap to attack.

---

## 4. Idea 1 — Lift the realized wrench into a decision variable

Give the robot block its own *declared* wrench $\mathbf{V}=\{v_t\}$ alongside
$\mathbf{U}^r$, constrain $v_t = z_t$ (linear), and move the simulator into
the objective as a realizability penalty:

```math
F_r(\mathbf{U}^r,\mathbf{V}) = J_r(\mathbf{U}^r)
+ \frac{\mu}{2}\sum_t \big\lVert \Phi_t(\mathbf{U}^r) - v_t \big\rVert^2 ,
\qquad A^r(\mathbf{U}^r,\mathbf{V}) = \mathbf{V} .
```

The $\mathbf{V}$-update is a strictly convex quadratic with a **closed-form
solution** — no sampling, no simulator:

```math
v^\ast_t = \frac{\mu\,\Phi_t(\mathbf{U}^r) + \rho\,(z_t - y^r_t)}{\mu + \rho}
```

a convex combination of what the robot *can* do and what consensus *wants*.
As $\mu\to\infty$, $v\to\Phi$ and the current algorithm is recovered exactly,
so this is a strict generalization with one new knob.

| Question | Answer |
| --- | --- |
| What changes | Robot block decides $(\mathbf{U}^r, \mathbf{V})$; $A^r$ becomes the linear projection onto $\mathbf{V}$; the simulator moves from the constraint into $F_r$ |
| Why it is good | It is the only change that makes the problem an actual instance of the consensus form the whole method is built on |
| How it helps the numbers we see now | $z$ is now averaging two genuine decision variables instead of one decision and one measurement, so the $z$-thrash (0.197/iteration) has a mechanism to damp; the robot block gains an explicit way to say "I cannot deliver that" instead of silently failing to |
| What it fixes in the theory | A1 outright. Also repairs the meaning of a fixed point: at convergence $w^o = v = z$ with $\lVert\Phi - v\rVert$ bounded by $O(\rho/\mu)$, which is a stationarity statement about eq. 6 |
| Assumption restored, and the result that needs it | A1, required by Hong et al. [5] and by every consensus-ADMM result. Makes the existing citation honest rather than aspirational |
| Extra compute per ADMM iteration | Essentially zero. The $\mathbf{V}$-update is one closed-form vector operation on an $(H,3)$ array; $\Phi$ is already computed |
| Implementation effort | Moderate. `ADMMParams` gains a $\mathbf{V}$ field, `RobotSubproblem` gains the $\mu$ term in its rollout cost and a closed-form update, `WrenchConsensus.z_update` is unchanged |
| Main risk | The robot block can now satisfy consensus *on paper* by moving $v$ while $\mathbf{U}^r$ does nothing. Guarded by $\mu$: too small and the plan is fiction |
| New knob, and what it trades | $\mu$: exactness against stiffness. Large $\mu$ recovers today's behavior including its stiffness; small $\mu$ is well-conditioned but tolerates an unrealizable plan. Start at $\mu = 10\rho$ |
| How we validate it | Sweep $\mu \in \{1,10,100\}\times\rho$ on `shelf_gap`; plot $\lVert r\rVert$ and $\lVert\Phi - v\rVert$ separately. The second is the new honesty metric |
| Falsifiable prediction | $\lVert r\rVert$ drops below 1.0 (per-entry RMS < 0.07) within 8 iterations for $\mu \le 10\rho$, while $\lVert\Phi-v\rVert$ stays bounded. If $\lVert r\rVert$ falls only because $v$ decoupled from $\Phi$, the idea has failed and the second plot will show it |

---

## 5. Idea 2 — Make each block update an actual descent step

The proximal term $\tfrac{\gamma}{2}\lVert \mathbf{U}-\mathbf{U}^{(l)}\rVert^2$
exists to supply sufficient decrease. If the subproblem were solved exactly,
optimality of $\mathbf{U}^{(l+1)}$ against the feasible point
$\mathbf{U}^{(l)}$ gives

```math
F(\mathbf{U}^{(l+1)}) + \tfrac{\gamma}{2}\lVert \mathbf{U}^{(l+1)}-\mathbf{U}^{(l)}\rVert^2 \le F(\mathbf{U}^{(l)})
```

which is exactly the descent lemma every nonconvex ADMM proof runs on. **MPPI
does not solve it exactly.** Its update is a softmax-weighted average of $K$
perturbations, which can and does increase the objective. So $\gamma = 0.1$
currently guarantees nothing at all.

The fix is to *check* rather than assume: evaluate $\mathcal{L}_\rho$ at the
proposed nominal and at the previous one, and accept only on sufficient
decrease. Pair it with a monotone non-decreasing $\rho$ and a floor.

| Question | Answer |
| --- | --- |
| What changes | After each MPPI update, accept $\mathbf{U}^{(l+1)}$ only if $\mathcal{L}_\rho$ decreases by at least $\tfrac{\gamma}{2}\lVert\Delta\mathbf{U}\rVert^2$; otherwise keep $\mathbf{U}^{(l)}$. Separately, make $\rho$ non-decreasing with a floor $\rho_{\min}$ |
| Why it is good | It converts $\gamma$ from a hopeful regularizer into an enforced guarantee, at almost no cost, using quantities already computed |
| How it helps the numbers we see now | Directly attacks the $z$-thrash and the non-monotone residual. A rejected step cannot make consensus worse, so $\lVert r\rVert$ becomes non-increasing within a control step instead of wandering |
| What it fixes in the theory | A3 and A5. Gives a monotonically decreasing $\mathcal{L}_\rho$, hence bounded iterates and $\lVert r\rVert \to 0$ along a subsequence under the remaining assumptions |
| Assumption restored, and the result that needs it | A3 (sufficient decrease) and A5 (monotone $\rho$), both required by Hong et al. [5]. A4 becomes enforceable once $\rho_{\min}$ exists |
| Extra compute per ADMM iteration | One extra rollout per block, against 64 samples already rolled out. About 1.5% |
| Implementation effort | Low. The costs are already computed inside `_eval_rollouts_one`; this adds a comparison and a `jnp.where` on the params pytree |
| Main risk | Over-rejection stalls the iteration: if MPPI rarely produces a descent step, the algorithm freezes at the warm start and behaves like open-loop. Mitigated by logging the acceptance rate |
| New knob, and what it trades | $\rho_{\min}$, and the acceptance tolerance. Strict acceptance buys theory and risks stalling; loose acceptance is closer to today |
| How we validate it | Log the per-iteration acceptance rate and $\mathcal{L}_\rho$. A healthy run shows monotone $\mathcal{L}_\rho$ and acceptance in the 40–90% band |
| Falsifiable prediction | $\mathcal{L}_\rho$ becomes monotone within a control step, and acceptance stays above 30%. If acceptance collapses below 10%, MPPI is not a descent method on this problem and the whole proximal-ADMM framing needs rethinking — which would itself be the most valuable thing we could learn |

---

## 6. Idea 3 — Make the object block propose only realizable wrenches

The object block samples $w^o$ from a box. The robot can only realize
wrenches of the form $J_c(p)^\top f$ for a contact point $p$ on the object
boundary that the arm can actually reach, with $f$ inside the friction cone —
a strictly smaller, state-dependent, non-convex set. The object block's
unconstrained optimum is generally *outside* it (the cheapest wrench is
usually a straight pull toward the goal, which no pusher can produce).

With finite $\rho$ the compromise $z$ therefore sits outside the robot's
reachable set, the robot cannot match it by construction, and $\lVert r\rVert$
has a floor. **A floor of 0.220 per-entry RMS is exactly what is measured.**

Two sub-fixes, one of which is a bug:

1. **The box is wrong.** `object_action_bounds` returns $\pm D^{-1}$ in
   physical units, but `object_action_to_consensus` then multiplies by
   `action_scale` $=\tfrac12 D^{-1}$, which assumes a unit sample. The
   realized box is $\pm\tfrac12(D^{-1})^2$ — **3.92× the friction-cone limit
   in force and 0.235× in torque**. The block can propose forces no support
   surface could transmit, and is simultaneously starved of torque authority.
2. **The parameterization already exists and is switched off.** The contact
   action $a=[p_x,p_y,f_n,f_t]$ with $A^o = J_c(p)^\top f$ makes every
   proposal realizable by construction. It is implemented, tested, and
   `--contact-action` opts in — but only in the 2D world, and off by default.

| Question | Answer |
| --- | --- |
| What changes | Fix the bounds/scale mismatch, then make the contact-action parameterization the default and port it to the 3D task |
| Why it is good | It removes a structural reason the residual cannot reach zero, rather than tuning around it. And most of the work is already done |
| How it helps the numbers we see now | Attacks the residual floor directly. The measured floor (0.220) is the signature of a proposal set larger than the realizable set |
| What it fixes in the theory | Not an assumption of [5], but a well-posedness question upstream of it: it makes the consensus target lie in the intersection of both blocks' reachable sets, so a zero-residual solution exists to converge *to* |
| Assumption restored, and the result that needs it | None directly. It repairs the premise instead: without it, $\lVert r\rVert \to 0$ is not merely unproven but unachievable, so no amount of theory in Ideas 1 and 2 would help |
| Extra compute per ADMM iteration | Real. The contact-action path adds boundary projection, rejection sampling on normal alignment, and a CEM search over the boundary each step. Measured cost not yet established in 3D |
| Implementation effort | High for the 3D port (needs a reachability-aware contact model for the arm); trivial for the bounds bug |
| Main risk | Behavioral. Fixing the bounds changes every run's results, so all recorded evaluations must be regenerated. The 3D contact model may not be worth its cost |
| New knob, and what it trades | $\mu_c$, $f_{\max}$, and the CEM budget. Richer parameterization against per-step cost |
| How we validate it | Fix the bounds alone first and re-run the five scenes; that is cheap and isolates one variable. Then compare `--contact-action` against direct-wrench in 2D, where both already run |
| Falsifiable prediction | The bounds fix alone moves min $\lVert r\rVert$ measurably below 3.19. If it does not, the residual floor is caused by the model mismatch between the analytic limit surface and MJX rather than by the proposal set, and the right fix is a different one |

---

## 7. Comparison

| | Idea 1: lift $\hat w$ | Idea 2: descent safeguard | Idea 3: realizable proposals |
| --- | --- | --- | --- |
| Fixes assumption | A1 | A3, A5, enables A4 | none (repairs the premise) |
| Theory gained | Problem becomes a real consensus instance | Monotone $\mathcal{L}_\rho$, subsequence convergence | A zero-residual point exists |
| Expected effect on $\lVert r\rVert$ | Large | Moderate, and makes it monotone | Large if the floor is proposal-set driven |
| Compute cost | Negligible | ~1.5% | Substantial in 3D |
| Implementation effort | Moderate | Low | Low (bug) to high (3D port) |
| Invalidates recorded runs | No | No | Yes |
| Standalone value | High | High | High |

None of the three conflicts with the others; they touch different parts of
the iteration. Idea 2 is also the cheapest way to find out whether the other
two are working, because it makes $\mathcal{L}_\rho$ a meaningful progress
signal for the first time.

## 8. Recommended order

1. **Idea 3's bounds bug.** One line, isolates one variable, and the
   falsifiable prediction tells us whether the residual floor is even a
   proposal-set problem before anyone builds a 3D contact model.
2. **Idea 2.** Cheap, low-risk, and instruments everything that follows. The
   acceptance rate alone will tell us whether MPPI is a descent method here.
3. **Idea 1.** The real theoretical fix, worth doing once we can see whether
   it helps.
4. **Idea 3's 3D contact-action port**, only if step 1 says the proposal set
   is the binding constraint.

Also worth doing regardless, because both are nearly free:

| Change | Reason |
| --- | --- |
| Remove the dual clip once Idea 1 lands | Restores $\sum_i y^i = 0$ (A6) and makes the $z$-update the plain average the paper states |
| Replace the 10× ratio in the $\rho$ rule | It has fired 0 times in 100 steps; the observed ratio hovers near 4 |
| Ship `consensus_alpha = 0.2` | Measured better than the shipped 1.0; smooths the resampling variance that currently dominates the residual |
| Report $\lVert r\rVert$ as RMS, not Frobenius | Makes $\epsilon_r$ horizon-independent; today $\epsilon_r = 0.5$ is unreachable at $H=35$ by construction |

## 9. What would still not be proven

Even with all three, a full convergence proof is **out of reach while MJX
defines $F_r$**: assumption A2 needs a Lipschitz gradient, and rigid-body
contact is discontinuous. No amount of restructuring the outer loop fixes
that.

Two honest positions are available, and they are worth stating in the paper
rather than glossing:

1. **Claim what is true.** With Ideas 1 and 2, the iteration is a monotone
   descent method on $\mathcal{L}_\rho$ whose fixed points are stationary
   points of a well-defined lifted problem. That is a real and defensible
   claim, and it is strictly more than the paper currently supports.
2. **Prove it where it is provable.** The 2D world (`oim/sim2d`) has an
   *analytic* contact model — `resolve_contact` computes the wrench in closed
   form, and the limit surface is linear. A2 is plausible there. That is
   where a theorem could actually be stated and verified numerically, with
   the 3D results presented as empirical transfer.

Position 2 is the stronger paper, and the codebase is already set up for it:
the same `ADMM` class drives both worlds, so a theorem in 2D and experiments
in 3D are the same algorithm, not an analogy.

---

## References

Numbering follows the paper.

[5] Hong, Luo & Razaviyayn, *Convergence analysis of alternating direction
method of multipliers for a family of nonconvex problems*, SIAM J. Optim.
26(1), 2016.

[6] Ouyang, He, Tran & Gray, *Stochastic alternating direction method of
multipliers*, ICML 2013.
