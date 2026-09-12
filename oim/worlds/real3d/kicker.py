"""The no-progress kick: what to do when the arm has stopped solving.

A sampling planner that has lost contact can sit at a local minimum
indefinitely -- the samples that would re-approach the object all cost more
than standing still. This detects that (no progress over a window) and
perturbs the sampling mean, which the NEXT solve starts from; nothing the
arm is currently executing changes.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import jax
import numpy as np


class _StuckKicker:
    """Detect-and-kick, ported from `oim.worlds.sim3d.run._run_plain`.

    Same logic, same 1e-4 threshold, same perturbation. Wrapped in a class
    only because this module has two loops (`_run_serial` and
    `_run_overlapped`) where the sim has one, so inlining it would mean two
    copies. If sim and real should ever share one implementation, this is the
    piece to hoist into `oim/runtime/`.

    Why it exists: MPPI's softmax-weighted mean update can settle into a blend
    that commits to neither "found the contact angle that breaks stiction" nor
    "back off and re-approach" -- the object stops moving entirely while the
    arm keeps jiggling around the same pose. The sim's flat loop perturbs the
    sampling mean after `stuck_kick_steps` consecutive no-progress control
    steps; this driver had no equivalent, so the same controller stalls here
    where it recovers there.

    Like the sim's version, this only ever touches the traced `params` pytree,
    never a `self.` attribute on the controller -- `jit_optimize` is a jitted
    bound method, so a mutated `self.x` would be silently ignored.

    Reads its two numbers off the controller. `MPPI` carries them; `ADMM`
    does not -- it holds an MPPI as its ROBOT sub-optimizer, and that is where
    the two live, so a plain `getattr(ctrl, ...)` read 0 and this class was
    silently inert on the whole ADMM path. Measured cost of that on the two
    2026-08-27 mock runs: 131 and 124 consecutive frozen control steps, 31% of
    each run, in exactly the state it exists to break -- while the setup dump
    printed `stuck_kick=100x2.0` as though it were armed.
    """

    # Matches the exact-zero signature real stiction produces in MJX/Warp
    # (object_velocity goes bit-exact 0.0, not a gradual decay) -- not a
    # tolerance chosen to catch merely "slow" progress. On HARDWARE that
    # signature never occurs: FoundationPose jitter moves pos_err/theta_err
    # by more than EPS on most steps, so the consecutive-step streak reset
    # every time and the kicker was silently inert -- the 2026-08-29 15:57
    # success run dwelled for 185 s of its 263 s (70%), including 5-8 s
    # frozen episodes, and fired exactly ONE kick. The WINDOW test below
    # replaces the streak for that reason: it asks whether NET progress
    # over the last `stuck_kick_steps` solves is under the noise scale,
    # which jitter cannot fool in either direction. A genuinely slow push
    # (2 mm/s over the ~3.5 s window = 7 mm) still clears it.
    EPS = 2e-3
    WINDOW_POS = 5e-3      # net |d pos_err| under 5 mm over the window ...
    WINDOW_THETA = 3.5e-2  # ... AND net |d theta_err| under ~2 deg = stuck
    # ... AND the tip itself went nowhere. The block not moving is NOT
    # stuck while the ARM is travelling: without this gate the first
    # hardware run fired every ~15 solves DURING THE INITIAL APPROACH
    # (block untouched, tip covering centimetres per window) and knocked
    # the arm off its own approach each time (2026-08-29, kicks at steps
    # 14/29/46/63/78). 30 mm net over the ~3.5 s window is above hover
    # wiggle's net drift but far below any real approach or repositioning
    # leg, so only a genuinely parked arm still counts as stuck.
    WINDOW_TIP = 3e-2

    def __init__(self, ctrl: Any) -> None:
        source = ctrl
        if not hasattr(source, "stuck_kick_steps"):
            # ADMM: the knobs belong to the robot block's own optimizer.
            source = getattr(
                getattr(ctrl, "robot_subproblem", None), "optimizer", ctrl
            )
        self.steps = int(getattr(source, "stuck_kick_steps", 0) or 0)
        self.scale = float(getattr(source, "stuck_kick_scale", 0.0) or 0.0)
        self.count = 0
        self.kicks = 0
        self._prev = None
        # Rolling window of (pos_err, theta_err), one entry per solve.
        self._hist: deque = deque(maxlen=max(self.steps, 1))

    def maybe_kick(self, params: Any, pos_err: float, theta_err: float,
                   step: int, verbose: bool,
                   tip_xy: Any = None) -> Any:
        """Return `params`, perturbed if the run has been frozen long enough."""
        if self.steps <= 0 or not hasattr(params, "mean"):
            return params
        if tip_xy is not None:
            tx, ty = float(tip_xy[0]), float(tip_xy[1])
        else:  # caller without tip logging: gate passes, old behaviour
            tx = ty = float("nan")
        self._hist.append((pos_err, theta_err, tx, ty))
        if len(self._hist) < self._hist.maxlen:
            return params
        p0, t0, x0, y0 = self._hist[0]
        block_stuck = (
            abs(pos_err - p0) < self.WINDOW_POS
            and abs(theta_err - t0) < self.WINDOW_THETA
        )
        tip_moved = (
            not np.isnan(x0)
            and float(np.hypot(tx - x0, ty - y0)) >= self.WINDOW_TIP
        )
        if (not block_stuck) or tip_moved:
            return params
        # Fire, then start a fresh window so the kick gets `steps` solves
        # to show progress before it can fire again.
        self._hist.clear()
        kick_rng, rng = jax.random.split(params.rng)
        inner = getattr(params, "robot_params", None)
        self.count = 0
        self.kicks += 1
        if verbose:
            print(f"step {step:4d}  stuck -- kicked ({self.kicks})")
        if inner is not None and hasattr(inner, "mean"):
            # ADMM: `ADMMParams.mean` is a read-only PROPERTY forwarding to
            # `robot_params.mean`, so `params.replace(mean=...)` would raise
            # -- it is not a field. Kick the field it delegates to. The
            # consensus variable and both duals are deliberately left alone:
            # the robot block is the one that has stopped exploring, and
            # resetting z would also discard whatever the object block has
            # agreed to.
            kick = self.scale * jax.random.normal(kick_rng, inner.mean.shape)
            return params.replace(
                robot_params=inner.replace(mean=inner.mean + kick), rng=rng
            )
        kick = self.scale * jax.random.normal(kick_rng, params.mean.shape)
        return params.replace(mean=params.mean + kick, rng=rng)
