"""Close the loop on the object-level subproblem, once per control step.

BY DEFAULT THE PLANT IS THE MODEL. `object_dynamics` both predicts and
executes, so the loop is a receding-horizon MPC with no model error
whatsoever. That is the point: it upper-bounds what the object block can
achieve, and the gap between this and an ADMM run is what the robot and the
consensus cost. It also means the breakaway threshold in
`PlanarPushingObject.step` applies to execution as well as to prediction,
which is why the diagnostics plot draws the wrench against it.

Pass a `MujocoPlant` (`oim.worlds.object_only.plant`) to break that identity
and have the simulator execute the same wrench instead. Only the plant
changes -- sampler, costs, projection and warm start are the same objects,
built by `oim.worlds.object_only.build` either way -- so the `pred_pos_err`
column the run then carries is model error and nothing else.
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from oim.algs import ObjectSubproblem
from oim.algs.admm import shift_object_actions
from oim.objects import wrap_angle
from oim.tasks.pusht import PushT
from oim.utils.series import finite_difference
from oim.worlds.object_only.build import report_softmax_ess
from oim.worlds.object_only.plant import AnalyticPlant, ObjectPlant


def run_object(
    task: PushT,
    block: ObjectSubproblem,
    params: Any,
    obj_state0: jnp.ndarray,
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    verbose: bool = True,
    jit: bool = True,
    log_samples: bool = True,
    plant: Optional[ObjectPlant] = None,
    on_plan: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
) -> Dict[str, Any]:
    """Receding-horizon loop over the object block alone.

    Each step solves for a wrench sequence, applies its first entry to the
    plant, and warm-starts by shifting -- the object half of an ADMM control
    step, with the consensus and the robot removed.

    Args:
        task: The `PushT` from `build_object_only`.
        block: Its `ObjectSubproblem`.
        params: The object optimizer's initial policy parameters.
        obj_state0: Object start pose, world-frame SE(2).
        max_steps: Maximum control steps.
        goal_pos_tol: Positional tolerance for declaring success.
        goal_theta_tol: Angular tolerance for declaring success.
        verbose: Print per-step progress.
        jit: Whether to jit the solve. False steps through the object
            dynamics and the sampler in a debugger, which is most of the
            reason to run this world at all.
        log_samples: Keep each step's sampled candidate trajectories for
            the plot. `(num_samples, H, 3)` per step is by far the largest
            thing in the log, so it can be switched off for a long run.
        plant: What executes the chosen wrench. `None` builds an
            `AnalyticPlant`, i.e. the model executes itself and there is no
            model error -- the loop this module shipped with. Pass a
            `MujocoPlant` to plan with the limit surface and be graded by
            the simulator; see `oim.worlds.object_only.plant`.
        on_plan: Called each control step with `(plan, samples)` -- the
            chosen object trajectory and the candidates behind it -- just
            before the wrench is executed. Exists so a MuJoCo recording can
            composite the plans into the frames captured *during* that
            step, which is exactly their period of validity; the log alone
            could not do that, since it is only complete at the end.

    Returns:
        A log dict in the same shape the other runners produce: `time`,
        `object_pose`, `object_velocity`, `wrench`, `object_plan`,
        `pos_err`, `theta_err`, `compute_time`, `reached` -- plus
        `object_samples` when asked for, and always `predicted_pose`,
        `pred_pos_err`, `pred_theta_err`: what the planner's own model said
        the executed wrench would do, against what the plant did with it.
        Identically zero under `AnalyticPlant`, which is the point of
        recording it for both.
    """
    horizon = params.mean.shape[0]
    dim = task.consensus_dim
    # The ADMM inputs, switched off. Shapes must still be right: they are
    # traced into the penalty even though it evaluates to zero.
    z = jnp.zeros((horizon, dim))
    dual = jnp.zeros((horizon, dim))
    rho = jnp.zeros(dim)
    noise_scale = jnp.zeros(())

    def _solve(
        obj_state: jnp.ndarray, prm: Any, rng: jax.Array
    ) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
        """One object-block update; returns params, its plan, its samples."""
        new_params, _a_obj, ref_states, samples = block.optimize(
            obj_state, prm, z, dual, rho, prm.mean, noise_scale, rng
        )
        return new_params, ref_states, samples

    solve = jax.jit(_solve) if jit else _solve
    # The planner's own one-step model, kept separate from the plant so the
    # gap between them can be measured. Under `AnalyticPlant` they are the
    # same function and the gap is exactly zero.
    predict = jax.jit(task.object_dynamics) if jit else task.object_dynamics
    if plant is None:
        plant = AnalyticPlant(task, jit=jit)

    obj_state = jnp.asarray(plant.reset(obj_state0), dtype=float)
    goal = np.asarray(task.goal)
    rng = jax.random.key(0)
    if verbose:
        report_softmax_ess(task, block, params, obj_state)

    log: Dict[str, Any] = {
        "time": [0.0],
        "object_pose": [np.asarray(obj_state)],
        "wrench": [],
        "object_plan": [],
        "pos_err": [],
        "theta_err": [],
        "compute_time": [],
        "predicted_pose": [],
        "pred_pos_err": [],
        "pred_theta_err": [],
    }
    if log_samples:
        log["object_samples"] = []
    reached = False

    for step in range(max_steps):
        rng, step_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        params, plan, samples = solve(obj_state, params, step_rng)
        jax.block_until_ready(params)
        log["compute_time"].append(time.perf_counter() - t0)
        if step == 0 and verbose:
            _report_plan_span(np.asarray(plan), np.asarray(samples))

        if on_plan is not None:
            on_plan(np.asarray(plan), np.asarray(samples))

        # The decision actually executed: the first entry of the projected
        # nominal, mapped through the same action -> wrench map the rollout
        # used, so the applied wrench is the one the plan was scored on.
        action = task.project_object_action(params.mean, obj_state)[0]
        wrench = task.object_action_to_consensus(obj_state, action)
        # Predict before executing: both start from the same pose, so their
        # difference is one step of model error and nothing else.
        predicted = np.asarray(predict(obj_state, wrench), dtype=float)
        obj_state = jnp.asarray(plant.step(np.asarray(wrench)), dtype=float)

        pos_err, theta_err = _log_step(
            log,
            task,
            step,
            obj_state,
            wrench,
            plan,
            samples if log_samples else None,
            goal,
            verbose,
            predicted,
        )
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

        # Warm start, identically to ADMM's own object warm start.
        params = params.replace(
            mean=shift_object_actions(task, params.mean)
        )

    log["reached"] = reached
    log["plant"] = plant.name
    for key in (
        "time", "object_pose", "wrench", "object_plan", "predicted_pose"
    ):
        log[key] = np.array(log[key])
    if log_samples:
        log["object_samples"] = np.array(log["object_samples"])
    # Matches the other worlds' logs, where the object twist is recorded:
    # forward Euler on the pose, so the difference quotient is exactly the
    # velocity the model applied.
    log["object_velocity"] = finite_difference(
        log["object_pose"], float(task.dt), angle_col=2
    )
    return log


# A plan shorter than this spans well under a pixel at any sensible axis
# scale, so it is drawn but cannot be seen -- and an overlay that renders
# nothing looks exactly like one that was never switched on. Same constant
# and the same reasoning as `oim.worlds.sim3d.run._VISIBLE_SPAN_M`.
_VISIBLE_SPAN_M = 5e-3


def _report_plan_span(plan: np.ndarray, samples: np.ndarray) -> None:
    """Print how far the first plan and its candidates actually reach.

    Both spans are worth having, because they fail apart: a spread of
    candidates around a motionless nominal means the optimizer is still
    exploring and merely preferring stillness, while candidates that are
    *all* motionless mean no sampled action can move the object at all --
    the deadzone-reachability failure `check_action_budget` predicts.

    Args:
        plan: The object block's nominal, (H, 3).
        samples: Its candidate trajectories, (num_samples, H, 3).
    """
    span = float(np.linalg.norm(plan[-1, :2] - plan[0, :2]))
    spans = np.linalg.norm(samples[:, -1, :2] - samples[:, 0, :2], axis=-1)
    print(
        f"first plan: nominal spans {span:.4f} m; candidates span "
        f"{float(spans.min()):.4f}-{float(spans.max()):.4f} m"
    )
    if float(spans.max()) < _VISIBLE_SPAN_M:
        print(
            "  every candidate is motionless -- the object block cannot "
            "move the object, not merely choosing not to. See the action "
            "budget warning above."
        )
    elif span < _VISIBLE_SPAN_M:
        print(
            "  the nominal is motionless while candidates are not: the "
            "optimizer is choosing stillness, so this is the cost, not "
            "reachability."
        )


def _log_step(
    log: Dict[str, Any],
    task: PushT,
    step: int,
    obj_state: jnp.ndarray,
    wrench: jnp.ndarray,
    plan: jnp.ndarray,
    samples: Optional[jnp.ndarray],
    goal: np.ndarray,
    verbose: bool,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    """Append one control step to the log; return its two goal errors.

    The progress line reports the wrench as a fraction of the friction-cone
    limit rather than in newtons, because the number that matters is
    whether it cleared 1.0: below that `PlanarPushingObject.step` holds the
    object still, so a run can look busy in the cost panel and not be
    moving at all.
    """
    log["time"].append((step + 1) * float(task.dt))
    log["object_pose"].append(np.asarray(obj_state))
    log["wrench"].append(np.asarray(wrench))
    log["object_plan"].append(np.asarray(plan))
    log["predicted_pose"].append(predicted)
    if samples is not None:
        log["object_samples"].append(np.asarray(samples))

    pos_err = float(np.linalg.norm(np.asarray(obj_state)[:2] - goal[:2]))
    theta_err = float(
        abs(float(wrap_angle(float(obj_state[2]) - float(goal[2]))))
    )
    log["pos_err"].append(pos_err)
    log["theta_err"].append(theta_err)

    # One step of model error: predicted-vs-realized, both from the pose
    # the step started at.
    realized = np.asarray(obj_state, dtype=float)
    pred_pos = float(np.linalg.norm(predicted[:2] - realized[:2]))
    pred_theta = float(
        abs(float(wrap_angle(float(predicted[2]) - float(realized[2]))))
    )
    log["pred_pos_err"].append(pred_pos)
    log["pred_theta_err"].append(pred_theta)

    if verbose and step % 10 == 0:
        limit = np.asarray(task.object_model.wrench_limit)
        normalized = float(np.linalg.norm(np.asarray(wrench) / limit))
        held = "  (below breakaway: held)" if normalized < 1.0 else ""
        model_gap = (
            "" if pred_pos + pred_theta < 1e-9
            else f"  model_err={pred_pos:.4f}m/{pred_theta:.4f}rad"
        )
        print(
            f"step {step:4d}  pos_err={pos_err:.4f}  "
            f"theta_err={theta_err:.4f}  "
            f"|w|/limit={normalized:.3f}{held}{model_gap}"
        )
    return pos_err, theta_err
