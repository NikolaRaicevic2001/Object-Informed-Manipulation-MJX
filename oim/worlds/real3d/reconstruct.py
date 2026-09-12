"""Rebuilding the expensive log series after the loop, not during it.

The plans and the per-term costs are functions of state the log already
holds, so paying for them per control step -- an extra rollout of each ADMM
block, plus a forward pass -- buys nothing the run itself uses. The loop
records the block means instead (tiny), and this replays them once the arm
has stopped. A viewer or recorder needs them live, and then this is skipped.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from oim.runtime.logs import finalize_log
from oim.worlds.real3d.diagnostics import _COST_TERM_KEYS, _cost_terms_jnp


def _logged_states(log: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    """(qpos, qvel, time) of every logged control step, initial entry dropped."""
    n = len(log["object_pose"]) - 1
    qpos = np.asarray(log["qpos"][1:n + 1], dtype=np.float32)
    qvel = np.asarray(log["qvel"][1:n + 1], dtype=np.float32)
    t = np.asarray(log["time"][1:n + 1], dtype=np.float32)
    return qpos, qvel, t

def _recon_batch(task: Any) -> Optional[int]:
    """How many logged states one reconstruction kernel evaluates at once.

    64 under the JAX backend. Under MuJoCo Warp the contact arenas are
    shared across the batch and sized for the loop's own rollouts, so the
    post-run map goes one state at a time -- the same call shape the loop
    itself used, which is known to fit. None, not 1: any `batch_size` makes
    `lax.map` vmap the body, which buys nothing at width 1 and is unsafe
    over the control projection (see `_reconstruct_plans`).
    """
    return None if getattr(task.model, "impl", "jax") == "warp" else 64

def _reconstruct_cost_terms(task: Any, base_data: Any,
                            log: Dict[str, Any]) -> None:
    """Fill the `c_*` series from the logged states, after the loop.

    The same `_cost_terms_jnp` the console print uses, mapped over every
    logged (qpos, qvel, time) with forward kinematics in between --
    identical numbers to evaluating it live, minus the per-step cost.
    """
    if "c_goal" not in log:
        return
    qpos, qvel, t = _logged_states(log)
    if qpos.shape[0] == 0:
        return

    def one(args: Any) -> jax.Array:
        q, v, tt = args
        d = base_data.replace(qpos=q, qvel=v, time=tt)
        return _cost_terms_jnp(task, mjx.forward(task.model, d))

    vals = np.asarray(
        jax.lax.map(one, (qpos, qvel, t), batch_size=_recon_batch(task))
    )
    for i, key in enumerate(_COST_TERM_KEYS):
        log[key] = [float(v) for v in vals[:, i]]

def _reconstruct_plans(task: Any, base_data: Any, log: Dict[str, Any],
                       params: Any, jit_plans: Any,
                       knots: List[Tuple[Any, Any, Any]]) -> bool:
    """Fill `object_plan` / `robot_plan` from the logged means, after the loop.

    `jit_plans` (`ADMM.nominal_plans`) is one extra rollout of each block
    per step -- the most expensive diagnostic in the loop. Each step's
    plan is a function of the logged state and that step's block means
    (+ knot times), so it is recomputed here from those; with a viewer or
    recorder the loop computed it live instead and this is skipped.

    Returns False (and leaves the series untouched) if nothing was
    collected, so the caller can drop the plans from the run file.
    """
    if not knots:
        return False
    qpos, qvel, t = _logged_states(log)
    n = min(qpos.shape[0], len(knots))
    if n == 0:
        return False
    om = jnp.asarray(np.stack([k[0] for k in knots[:n]]))
    rm = jnp.asarray(np.stack([k[1] for k in knots[:n]]))
    tk = jnp.asarray(np.stack([k[2] for k in knots[:n]]))

    def one(args: Any) -> Tuple[jax.Array, jax.Array]:
        q, v, tt, om_i, rm_i, tk_i = args
        d = mjx.forward(task.model, base_data.replace(qpos=q, qvel=v, time=tt))
        p = params.replace(
            object_params=params.object_params.replace(mean=om_i),
            robot_params=params.robot_params.replace(mean=rm_i, tk=tk_i),
        )
        obj_plan, rob_plan, _trace = jit_plans(d, p)
        return obj_plan, rob_plan

    # `nominal_plan` projects its controls, and that QP runs in a local
    # `jax.enable_x64` context. vmap ignores it -- the batching rules
    # re-canonicalize to float32 while the explicit casts stay float64, and
    # the trace dies on a float64/float32 `lax.mul`. A `batch_size` is a
    # vmap, so a projected run maps one state at a time (scan, no vmap).
    batch = (None if getattr(task, "control_projector", None) is not None
             else _recon_batch(task))
    obj, rob = jax.lax.map(
        one, (qpos[:n], qvel[:n], t[:n], om, rm, tk), batch_size=batch,
    )
    log["object_plan"] = [np.asarray(x) for x in np.asarray(obj)]
    log["robot_plan"] = [np.asarray(x) for x in np.asarray(rob)]
    return True

def _plan_knots(params: Any) -> Tuple[Any, Any, Any]:
    """This step's block means and knot times, copied to the host (tiny)."""
    return jax.device_get((
        params.object_params.mean, params.robot_params.mean,
        params.robot_params.tk,
    ))

def _finish(log: Dict[str, Any], task: Any, base_data: Any, reached: bool,
            admm: bool, params: Any, jit_plans: Any,
            knots: Optional[List[Any]], verbose: bool) -> Dict[str, Any]:
    """Post-loop reconstruction of the deferred series, then `finalize_log`."""
    t0 = time.perf_counter()
    show_plans = admm
    try:
        _reconstruct_cost_terms(task, base_data, log)
    except Exception as exc:  # noqa: BLE001 -- never lose the run file
        print(f"[log] cost-term reconstruction failed: {exc!r}")
        for key in _COST_TERM_KEYS:
            log[key] = []
    if admm and knots is not None:
        try:
            show_plans = _reconstruct_plans(
                task, base_data, log, params, jit_plans, knots
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[log] plan reconstruction failed: {exc!r}")
            show_plans = False
        if not show_plans:
            log.pop("object_plan", None)
            log.pop("robot_plan", None)
    if verbose:
        print(f"[log] post-run reconstruction: {time.perf_counter() - t0:.1f}s")
    return finalize_log(log, task, reached, show_plans=show_plans, admm=admm)
