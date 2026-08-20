"""The two object-only plants must be swappable, and honestly labelled.

`oim.worlds.object_only.plant` exists so that a difference between two
object-only runs
is a difference in *dynamics* and cannot be a difference in planner. These
tests pin the parts of that claim which would otherwise fail silently: the
analytic plant reporting a nonzero model error, or the MuJoCo plant quietly
simulating something other than the block (support friction of a different
shape than the analytic cone assumes, or a robot the run does not have).
"""

from typing import Any, Dict

import numpy as np
import pytest

from oim.experiment import load_config
from oim.tasks.pusht import PushT
from oim.worlds.object_only.plant import AnalyticPlant, MujocoPlant, build_plant

SCENE, ROBOT = "shelf_gap", "xarm6"


@pytest.fixture(scope="module")
def cfg() -> Dict[str, Any]:
    """The xArm6 config, whose tabletop scenes carry real table friction."""
    return load_config(ROBOT)


@pytest.fixture(scope="module")
def task(cfg: Dict[str, Any]) -> PushT:
    """One task for every test: compiling the scene dominates the runtime."""
    return PushT(
        impl="jax",
        clutter=True,
        planning_dt=cfg["world3d"]["planning_dt"],
        robot=ROBOT,
        consensus_source="twist",
        env=SCENE,
        costs=cfg.get("costs"),
    )


def _plant(kind: str, task: PushT, cfg: Dict[str, Any]) -> Any:
    return build_plant(
        kind, task, ROBOT, cfg["world3d"], control_dt=float(task.dt)
    )


def test_build_plant_rejects_an_unknown_name(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """A typo must not fall back to a plant the run then mislabels."""
    with pytest.raises(ValueError, match="unknown plant"):
        _plant("mjx", task, cfg)


def test_analytic_plant_is_the_task_model_exactly(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """It must agree with `object_dynamics` bit for bit.

    This is what makes the `pred_pos_err` column meaningful: it reads zero
    under the analytic plant *by construction*, so any nonzero value under
    another plant is model error and not bookkeeping drift.
    """
    plant = _plant("analytic", task, cfg)
    pose = np.asarray(task.start, dtype=float)
    plant.reset(pose)
    wrench = 1.5 * np.asarray(task.object_model.wrench_limit)
    expected = np.asarray(task.object_dynamics(pose, wrench))
    assert plant.step(wrench) == pytest.approx(expected)


def test_mujoco_plant_matches_the_analytic_breakaway_force(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """Per channel, the two thresholds are the same number.

    The MJCF sets each joint's `frictionloss` to the friction-cone limit
    for exactly this reason. If the support surface were left in collision
    it would be counted twice and this would land near 1.4x instead.

    Below the threshold MuJoCo creeps rather than sticking -- `frictionloss`
    is a solver constraint with compliance, not the hard `jnp.where` the
    analytic model uses, and it drifts ~5 mm over the second sampled here.
    The bound below is set from that measurement, so the test pins the
    creep as a known quantity instead of asserting a zero that never held.
    """
    plant = _plant("mujoco", task, cfg)
    limit = float(np.asarray(task.object_model.wrench_limit)[0])
    start = np.asarray(task.start, dtype=float)

    def travels(multiple: float) -> float:
        plant.reset(start)
        for _ in range(20):
            pose = plant.step(np.array([multiple * limit, 0.0, 0.0]))
        return abs(float(pose[0] - start[0]))

    assert travels(0.9) < 0.01  # held, up to solver creep
    assert travels(1.5) > 0.1  # clearly moving: 20x further


def test_mujoco_and_analytic_disagree_on_a_multi_axis_wrench(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """The shape mismatch, pinned as a fact rather than a bug.

    `PlanarPushingObject.step` measures friction with the coupled norm
    `||w/D^-1||` -- an ellipse, isotropic in the push direction. The
    simulator no longer measures it per DoF: the tabletop scenes' support
    friction is the table contact's, and MuJoCo's default `cone` is
    *pyramidal*, a square inscribed in that ellipse with its corners on the
    world axes. So the two now disagree in the opposite direction from
    before, and only off-axis. Travel under a wrench held 1 s, this scene:

        |w| / mu*m*g    0 deg    15 deg   30 deg   45 deg
        0.75           0.000 m  0.000 m  0.035 m  0.086 m
        0.90           0.000    0.146    0.325    0.384
        1.00           0.002    0.320    0.503    0.503
        1.10           0.208    0.494    0.556    0.527

    On axis the simulator sticks to exactly `mu*m*g`, which is what keeps
    `wrench_limit` meaningful; at 45 degrees it breaks away at 0.71 of it,
    the pyramid's inscribed radius. Under `frictionloss` the bias ran the
    other way -- a box *containing* the ellipse, so the simulated block
    stayed put where the analytic one slid (0.414 m against 0.007 m at
    [1, 1, 0]).

    Asserted so that a future change which reconciles them is noticed here
    rather than silently altering what the object-only comparison means.
    Setting `opt.cone = mjCONE_ELLIPTIC` does reconcile them exactly on CPU
    -- stuck below 1.0 and sliding above it at every angle -- and is not
    used, because the same option returns NaN under MJX at 3.7x to 14x the
    time. See `oim.runtime.object_mjx`.
    """
    limit = np.asarray(task.object_model.wrench_limit)
    start = np.asarray(task.start, dtype=float)

    def travels(wrench: np.ndarray) -> Dict[str, float]:
        out = {}
        for kind in ("analytic", "mujoco"):
            plant = _plant(kind, task, cfg)
            plant.reset(start)
            for _ in range(20):
                pose = plant.step(np.asarray(wrench, dtype=float))
            out[kind] = float(
                np.linalg.norm(np.asarray(pose)[:2] - start[:2])
            )
        return out

    # On axis, at exactly the limit: both hold. This is the agreement the
    # analytic cone is calibrated on, so it has to survive.
    on_axis = travels(np.array([limit[0], 0.0, 0.0]))
    assert on_axis["analytic"] < 0.01
    assert on_axis["mujoco"] < 0.01

    # At 45 degrees and *inside* the ellipse (norm 0.99): the analytic
    # block holds and the simulated one runs. The pyramid is inscribed.
    diagonal = np.array([0.7 * limit[0], 0.7 * limit[1], 0.0])
    assert np.linalg.norm(diagonal / limit) < 1.0
    inside = travels(diagonal)
    assert inside["analytic"] < 0.01
    assert inside["mujoco"] > 0.1


def test_mujoco_plant_leaves_the_task_model_untouched(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """Editing collisions and gravity must not reach the shared task.

    `PushT.mj_model` is what every other world builds from, and the plant
    disables the robot, disables the support surface and zeroes gravity.
    Doing that in place would silently change ADMM runs built afterwards.
    """
    before = task.mj_model.geom_contype.copy()
    gravity = np.array(task.mj_model.opt.gravity)
    _plant("mujoco", task, cfg)
    assert np.array_equal(task.mj_model.geom_contype, before)
    assert np.array_equal(np.array(task.mj_model.opt.gravity), gravity)


def test_mujoco_plant_takes_the_robot_out_of_collision(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """An object-only run has no robot, so its arm cannot be an obstacle."""
    plant: MujocoPlant = _plant("mujoco", task, cfg)
    model = plant.mj_model
    arm = [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith("xarm6")
    ]
    assert arm, "fixture scene should have an arm to disable"
    assert not any(model.geom_contype[g] for g in arm)
    # The block itself must still collide, or the obstacles do nothing.
    block = [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name == "block"
    ]
    assert all(model.geom_contype[g] for g in block)


def test_mujoco_plant_rejects_a_mismatched_control_step(
    task: PushT, cfg: Dict[str, Any]
) -> None:
    """A control step that is not a whole number of substeps must raise.

    Rounding it would put the two plants on different control rates while
    the run file reported one.
    """
    with pytest.raises(ValueError, match="whole multiple"):
        MujocoPlant(
            task,
            ROBOT,
            cfg["world3d"],
            control_dt=float(cfg["world3d"]["exec_timestep"]) * 2.5,
        )


def test_analytic_plant_reset_is_exact(task: PushT) -> None:
    """Reset places the object where asked; there is no other state."""
    plant = AnalyticPlant(task, jit=False)
    pose = np.array([0.3, -0.2, 1.1])
    assert plant.reset(pose) == pytest.approx(pose)
