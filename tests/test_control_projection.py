"""Frozen CBF projections, tilt relaxation and sampling integration."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import FixedProjector
from mujoco import mjx

from oim import experiment
from oim.alg_base import Trajectory
from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.algs.admm import RobotSubproblem
from oim.control_projection import (
    ControlProjector,
    ProjectionConfig,
    ProjectionConstraints,
    configure_projection,
    height_ceiling,
    make_constraints,
    project_controls,
    reject_invalid_tapes,
)
from oim.task_base import Task
from oim.tasks.pusht import PushT
from oim.worlds.sim3d.build import build_admm_3d, build_flat_3d


def _constraints() -> ProjectionConstraints:
    # z velocity >= 0; outward x velocity >= z velocity.
    return ProjectionConstraints(
        cbf_a=jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0]]),
        cbf_b=jnp.zeros(2),
        tilt_a=jnp.array([1.0, 0.0, 0.0]),
        tilt_b=jnp.array(0.1),
        u_min=-jnp.ones(3),
        u_max=jnp.ones(3),
    )


def test_batched_floor_feasibility_matches_float64_residual() -> None:
    """GPU matrix-product precision must not reject boundary solutions."""
    row = jnp.array([0., -.5811344, -.3749607, -.0002925, -.0755278])
    c = ProjectionConstraints(
        cbf_a=jnp.stack((row, jnp.zeros(5))),
        cbf_b=jnp.array([.0084166, 1.]),
        tilt_a=jnp.zeros(5), tilt_b=jnp.array(0.),
        u_min=jnp.full(5, -.15), u_max=jnp.full(5, .15),
    )
    u = jax.random.normal(jax.random.key(0), (4096, 5)) * .15
    out, diag = jax.jit(
        ControlProjector(ProjectionConfig(mode="qpax")).project
    )(u, c)
    residual = np.asarray(out, dtype=np.float64) @ np.asarray(row) + .0084166
    assert np.min(residual) >= -1e-6
    assert np.all(diag.valid)


def test_infeasible_qpax_correction_is_bounded_and_rejected() -> None:
    """An infeasible pair of CBFs must not be reported feasible."""
    c = _constraints().replace(cbf_b=jnp.array([0.0, -3.0]))
    out, diag = ControlProjector(ProjectionConfig(mode="qpax")).project(
        jnp.zeros((2, 3, 3)), c
    )
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= 1.0)
    assert not np.any(diag.valid)
    assert np.all(
        np.isinf(reject_invalid_tapes(jnp.zeros((2, 4)), diag)[:, -1])
    )


def test_ceiling_and_clf_rows_include_distance_rate() -> None:
    """Approach lowers the ceiling; moving sliders change the CBF bias."""
    config = ProjectionConfig(z_min=0.1, z_near=0.15, z_far=0.25)
    distance = jnp.array(0.085)
    ceiling, slope = height_ceiling(distance, config)
    assert float(slope) > 0
    np.testing.assert_allclose(ceiling, 0.2, atol=1e-6)
    angle = 0.2
    axis = jnp.array([np.sin(angle), 0.0, -np.cos(angle)])
    c = make_constraints(
        jnp.array([0.0, 0.0, 0.18]),
        axis,
        jnp.eye(3),
        jnp.eye(3),
        distance,
        jnp.array([1.0, 0.0]),
        jnp.array(-0.1),
        -jnp.ones(3),
        jnp.ones(3),
        config,
    )
    np.testing.assert_allclose(c.cbf_a[1], [slope, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(
        c.cbf_b[1], config.cbf_alpha * (0.2 - 0.18) - 0.1 * slope, atol=1e-6
    )
    # Positive rotation about y restores a down-pointing tilted axis.
    assert float(c.tilt_a[1]) < 0
    assert float(c.tilt_b) > 0
    for distance, expected in [(-1.0, config.z_near), (1.0, config.z_far)]:
        height, slope = height_ceiling(jnp.array(distance), config)
        np.testing.assert_allclose(height, expected)
        assert float(slope) == 0.0


def test_qpax_keeps_cbfs_hard_and_relaxes_tilt() -> None:
    """Incompatible tilt descent is paid for with CLF slack."""
    pytest.importorskip("qpax")
    c = _constraints()
    u = jnp.array([[[-1.0, 0.0, -0.5], [1.0, 0.0, 1.0]]])
    out, diag = jax.jit(
        ControlProjector(ProjectionConfig(mode="qpax")).project
    )(u, c)
    assert np.all(diag.valid)
    assert np.all(diag.tilt_slack > 0)
    assert np.max(np.asarray(diag.hard_violation)) <= 1e-5
    # This problem reduces to x=z and slack=x+.1 in the second lane.
    np.testing.assert_allclose(out[0, 1], [1 / 12, 0.0, 1 / 12], atol=2e-4)
    np.testing.assert_allclose(out[0, 0], 0.0, atol=2e-4)


@pytest.mark.parametrize("weight", [0.0, 100.0])
def test_qpax_preserves_nominal_xy_when_vertical_repair_is_possible(
    weight: float,
) -> None:
    """A coupled floor constraint can be repaired horizontally or vertically."""
    pytest.importorskip("qpax")
    c = _constraints().replace(
        cbf_a=jnp.array([[1., 0., 1.], [0., 0., 0.]]),
        cbf_b=jnp.array([-1., 1.]),
        tilt_a=jnp.zeros(3), tilt_b=jnp.array(0.),
        xy_jacobian=jnp.array([[1., 0., 0.], [0., 1., 0.]]),
        u_min=jnp.full(3, -2.), u_max=jnp.full(3, 2.),
    )
    out, diag = jax.jit(ControlProjector(
        ProjectionConfig(mode="qpax", xy_weight=weight)
    ).project)(jnp.array([.4, 0., 0.]), c)
    # The QP optimum for x+z >= 1, with W=diag(1+weight,1,1).
    dx = .6 / (weight + 2.)
    np.testing.assert_allclose(out, [.4 + dx, 0., .6 - dx], atol=1e-3)
    assert bool(diag.valid)


def test_qpax_weighted_free_clf_solution() -> None:
    """The exact inactive-CBF path must use the same Cartesian metric."""
    pytest.importorskip("qpax")
    c = _constraints().replace(
        cbf_b=jnp.full(2, 10.),
        xy_jacobian=jnp.array([[1., 0., 0.], [0., 1., 0.]]),
    )
    out, diag = jax.jit(ControlProjector(
        ProjectionConfig(mode="qpax", xy_weight=100.)
    ).project)(jnp.array([.4, 0., 0.]), c)
    np.testing.assert_allclose(out, [.4 - 5. / 111., 0., 0.], atol=1e-6)
    assert bool(diag.valid)


def test_weighted_qp_batch_keeps_precision_local() -> None:
    """A broad feasible batch must survive KKT and physical checks on GPU."""
    pytest.importorskip("qpax")
    config = ProjectionConfig(
        mode="qpax", xy_weight=100., z_min=.02, z_near=.03, z_far=.13
    )
    c = make_constraints(
        jnp.array([0., 0., .025]), jnp.array([0., 0., -1.]),
        jnp.array([[.3, .2, 0., .1, .1], [0., .1, .5, .1, .1],
                   [0., -.5, -.3, 0., -.1]]),
        jnp.zeros((3, 5)), jnp.array(.085), jnp.array([0., 1.]),
        jnp.array(0.), jnp.full(5, -.15), jnp.full(5, .15), config,
    )
    u = jnp.clip(jax.random.normal(jax.random.key(0), (4096, 5)) * .2,
                 -.15, .15).astype(jnp.float32)
    original_x64 = jax.config.x64_enabled
    out, diag = jax.jit(ControlProjector(config).project)(u, c)
    assert jax.config.x64_enabled == original_x64
    assert out.dtype == jnp.float32
    assert diag.iterations.dtype == jnp.int32
    assert np.all(diag.converged)
    assert np.all(diag.valid)
    residual = (np.asarray(out, dtype=np.float64)
                @ np.asarray(c.cbf_a, dtype=np.float64).T
                + np.asarray(c.cbf_b, dtype=np.float64))
    assert np.min(residual) >= -config.feasibility_tol


@pytest.mark.parametrize("tilt", [0.0, 0.002, 0.2])
def test_qpax_inactive_cbfs_do_not_add_drift(tilt: float) -> None:
    """With loose hard bounds the soft-CLF optimum is known exactly."""
    pytest.importorskip("qpax")
    c = _constraints().replace(
        cbf_b=jnp.full(2, 10.0),
        tilt_a=jnp.array([tilt, 0.0, 0.0]),
        tilt_b=jnp.array(tilt**2),
    )
    out, diag = jax.jit(
        ControlProjector(ProjectionConfig(mode="qpax")).project
    )(jnp.zeros(3), c)
    expected = -10.0 * tilt**3 / (1 + 10.0 * tilt**2)
    np.testing.assert_allclose(out, [expected, 0.0, 0.0], atol=1e-8)
    assert bool(diag.valid)


def test_qpax_single_halfspace_matches_analytic_solution() -> None:
    """Check QP optimality against the closed-form scalar projection."""
    pytest.importorskip("qpax")
    c = _constraints().replace(
        cbf_a=jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        cbf_b=jnp.array([0.0, 1.0]),
        tilt_a=jnp.zeros(3),
        tilt_b=jnp.array(-1.0),
    )
    u = jnp.array([[0.2, -0.3, -0.6], [0.4, 0.1, 0.5]])
    out, diag = jax.jit(
        ControlProjector(ProjectionConfig(mode="qpax")).project
    )(u, c)
    assert np.all(diag.valid)
    np.testing.assert_allclose(
        out, [[0.2, -0.3, 0.0], [0.4, 0.1, 0.5]], atol=3e-4
    )


def test_qpax_iteration_limit_is_not_silently_accepted() -> None:
    """A deliberately unfinished QP reports failure with finite fallback."""
    pytest.importorskip("qpax")
    p = ControlProjector(ProjectionConfig(mode="qpax", max_iter=1))
    out, diag = jax.jit(p.project)(
        jnp.array([[-1.0, 0.0, -0.5]]), _constraints()
    )
    assert not np.any(diag.valid)
    assert not np.any(diag.converged)
    np.testing.assert_array_equal(out, [[-1.0, 0.0, -0.5]])


def test_disabled_projection_is_structural_identity() -> None:
    """Disabled mode returns the original tape and no diagnostics."""
    u = jnp.ones((2, 4, 3))
    out, diag = project_controls(SimpleNamespace(), None, u)
    assert out is u
    assert diag is None


def test_nonfinite_nominal_is_rejected_without_poisoning_physics() -> None:
    """Invalid input cannot introduce NaNs into the state integration."""
    p = ControlProjector(ProjectionConfig(mode="qpax"))
    out, diag = p.project(jnp.array([[jnp.nan, 0.0, -1.0]]), _constraints())
    assert np.all(np.isfinite(out))
    assert not np.any(diag.valid)


def test_projection_flag_composes_with_other_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved CLI choice reaches the shared builders and run metadata."""
    captured = []
    monkeypatch.setattr(
        experiment, "_run_3d", lambda exp, args: captured.append(args)
    )
    experiment.main(
        experiment.Experiment(world="3d", scene="clutter"),
        [
            "--robot",
            "xarm6",
            "--control-projection",
            "qpax",
            "--cbf-floor-alpha", "0.5",
            "--cbf-slider-alpha", "1.5",
            "--cbf-z-near", "0.04",
            "--cbf-distance-far", "0.2",
            "--cbf-xy-weight", "250",
            "--gamma0-deg",
            "42",
            "mppi",
        ],
    )
    assert captured[0].cfg["control_projection"]["mode"] == "qpax"
    assert captured[0].cfg["control_projection"]["floor_alpha"] == 0.5
    assert captured[0].cfg["control_projection"]["slider_alpha"] == 1.5
    assert captured[0].cfg["control_projection"]["z_near"] == 0.04
    assert captured[0].cfg["control_projection"]["distance_far"] == 0.2
    assert captured[0].cfg["control_projection"]["xy_weight"] == 250.0
    assert captured[0].cfg["costs"]["gamma0_deg"] == 42.0
    assert (
        experiment.load_config("xarm6")["control_projection"]["mode"] == "qpax"
    )


@pytest.mark.parametrize("mode", ["off", "qpax"])
def test_both_builders_honor_projection_configuration(mode: str) -> None:
    """Shared configuration works for the actual xArm flat and ADMM builders."""
    if mode == "qpax":
        pytest.importorskip("qpax")
    cfg = experiment.load_config("xarm6")
    cfg["control_projection"] = {"mode": mode}
    common = dict(warp=False, horizon=4, samples=2, seed=0)
    flat, *_ = build_flat_3d(
        "ps", "clutter", "xarm6", cfg, control_dt=0.05, **common
    )
    admm, *_ = build_admm_3d(
        "clutter",
        "xarm6",
        cfg,
        robot_opt="ps",
        object_opt="ps",
        n_admm=2,
        rho=1.0,
        gamma=0.0,
        **common,
    )
    for task in (flat, admm):
        projector = getattr(task, "control_projector", None)
        if mode == "off":
            assert projector is None
        else:
            assert projector.config.mode == mode
            assert projector.config.z_near == task.tip_target_z


class _Task(Task):
    """Small translational plant for testing actual rollout dispatch."""

    def __init__(self, impl: str = "jax") -> None:
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.05" gravity="0 0 0"/>
        <worldbody><body><joint name="x" type="slide" axis="1 0 0"/>
        <joint name="y" type="slide" axis="0 1 0"/>
        <joint name="z" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".01" mass="1"/><site name="tip"/>
        </body></worldbody><actuator>
        <velocity joint="x" kv="1" ctrlrange="-1 1"/>
        <velocity joint="y" kv="1" ctrlrange="-1 1"/>
        <velocity joint="z" kv="1" ctrlrange="-1 1"/>
        </actuator></mujoco>""")
        super().__init__(model, trace_sites=["tip"], impl=impl)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        return jnp.sum(control**2) + jnp.sum(state.qpos**2)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return jnp.sum(state.qpos**2)

    def robot_running_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
        weight_scale: jax.Array,
        ref_pose: jax.Array | None = None,
    ) -> jax.Array:
        return self.running_cost(state, control)

    def robot_terminal_cost(
        self, state: mjx.Data, weight_scale: jax.Array
    ) -> jax.Array:
        return self.terminal_cost(state)

    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        return state.qpos

    def object_state_from_robot(self, state: mjx.Data) -> jax.Array:
        return state.qpos


@pytest.mark.parametrize("mode", ["qpax"])
@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_flat_and_admm_use_projected_tapes_and_nominal_knots(
    mode: str, impl: str
) -> None:
    """Samples, nominal consensus and drawings use the same projection."""
    if mode == "qpax":
        pytest.importorskip("qpax")
    if impl == "warp" and jax.default_backend() != "gpu":
        pytest.skip("Warp rollout integration requires a GPU")
    task = _Task(impl)
    task.control_projector = FixedProjector(
        ProjectionConfig(mode=mode), _constraints()
    )
    opt = PredictiveSampling(
        task,
        num_samples=2,
        noise_level=0.1,
        num_knots=2,
        plan_horizon=0.15,
        num_randomizations=2,
    )
    params = opt.init_params()
    state = task.make_data()
    knots = jnp.broadcast_to(jnp.array([-0.5, 0.1, 0.2]), (2, 2, 3))
    rng = jax.random.key(1)
    flat = jax.jit(opt.rollout_with_randomizations)(
        state, params.tk, knots, rng
    )
    np.testing.assert_array_equal(flat.knots, knots)
    assert np.all(flat.projection.valid)
    assert flat.projection.valid.shape == (2, 3)
    assert not np.allclose(flat.controls[:, 0], knots[:, 0])
    # No ADMM penalty: compare states and control-effort costs directly.
    robot = RobotSubproblem(
        task, opt, WrenchConsensus(max_dual=2.0), proximal_weight=0.0
    )
    zero = jnp.zeros((3, 3))
    admm = jax.jit(robot.rollout_with_randomizations)(
        state, params.tk, knots, rng, zero, zero, jnp.zeros(3), knots[0]
    )
    np.testing.assert_allclose(flat.controls, admm.controls)
    np.testing.assert_allclose(flat.costs, admm.costs, atol=1e-6)
    nominal = params.replace(mean=knots[0])
    consensus = jax.jit(robot.nominal_realized_consensus)(state, nominal)
    np.testing.assert_allclose(consensus, admm.consensus_values[0], atol=1e-6)
    obj, trace = jax.jit(robot.nominal_plan)(state, nominal)
    np.testing.assert_allclose(obj, consensus, atol=1e-6)
    np.testing.assert_allclose(trace, flat.trace_sites[0, :-1, 0], atol=1e-6)
    # Exercise the compiled optimizer scan and diagnostic pytree too.
    _, optimized = jax.jit(opt.optimize)(state, params)
    assert optimized.projection.valid.shape == (2, 3)


class _PlanarProjector(ControlProjector):
    """Two-dimensional constraints for the cheap full ADMM integration test."""

    def prepare(self, state: mjx.Data) -> ProjectionConstraints:
        return ProjectionConstraints(
            cbf_a=jnp.array([[0.0, 1.0], [1.0, -1.0]]),
            cbf_b=jnp.array([0.3, 0.3]),
            tilt_a=jnp.array([1.0, 0.0]),
            tilt_b=jnp.array(0.01),
            u_min=-jnp.ones(2),
            u_max=jnp.ones(2),
        )


@pytest.mark.parametrize(
    "mode,lagged", [("qpax", "off"), ("qpax", "robot")]
)
def test_full_admm_loop_carries_projection_diagnostics(
    mode: str, lagged: str
) -> None:
    """Shared prepared rows and diagnostics survive the outer ADMM loop."""
    if mode == "qpax":
        pytest.importorskip("qpax")
    task = PushT(clutter=True, planning_dt=0.05)
    task.control_projector = _PlanarProjector(ProjectionConfig(mode=mode))
    common = dict(num_samples=4, noise_level=0.1, plan_horizon=0.75)
    robot = PredictiveSampling(task, num_knots=4, **common)
    obj = PredictiveSampling(
        make_object_shim(task, dt=0.05), num_knots=15, **common
    )
    ctrl = ADMM(
        task,
        robot,
        obj,
        WrenchConsensus(max_dual=2.0),
        n_admm=2,
        eps_r=0.0,
        eps_s=0.0,
        lagged_consensus=lagged,
    )
    params, rollouts = jax.jit(ctrl.optimize)(
        task.make_data(), ctrl.init_params()
    )
    assert rollouts.projection.valid.shape == (4, 15)
    assert np.all(np.isfinite(params.mean))
    assert np.all(np.isfinite(params.z))
    assert np.any(np.all(rollouts.projection.valid, axis=-1))


@pytest.mark.parametrize("kind", ["mppi", "cem", "ps", "cbo"])
def test_optimizers_exclude_failed_projections(kind: str) -> None:
    """Failed samples never influence the mean, even with too few elites."""
    task = _Task()
    constructors = {
        "mppi": lambda: MPPI(
            task, num_samples=2, noise_level=0.1, temperature=1.0
        ),
        "cem": lambda: CEM(
            task, num_samples=2, num_elites=2, sigma_start=0.1, sigma_min=0.01
        ),
        "ps": lambda: PredictiveSampling(task, num_samples=2, noise_level=0.1),
        "cbo": lambda: CBO(
            task,
            num_samples=2,
            initial_noise_level=0.1,
            temperature=1.0,
            consensus_weight=1.0,
            noise_weight=0.1,
        ),
    }
    opt = constructors[kind]()
    params = opt.init_params()
    knots = jnp.stack(
        (jnp.ones_like(params.mean) * 0.2, jnp.ones_like(params.mean) * 0.8)
    )
    _, diag = ControlProjector(ProjectionConfig(mode="qpax")).project(
        jnp.zeros((2, 1, 3)), _constraints()
    )
    rollouts = Trajectory(
        controls=knots[:, :1],
        knots=knots,
        costs=jnp.array([[0.0, 0.0], [0.0, jnp.inf]]),
        trace_sites=jnp.zeros((2, 2, 1, 3)),
        projection=diag.replace(valid=jnp.array([[True], [False]])),
    )
    updated = jax.jit(opt.update_params)(params, rollouts)
    np.testing.assert_allclose(updated.mean, 0.2, atol=1e-6)
    corrupt = rollouts.replace(knots=knots.at[1].set(jnp.nan))
    updated = jax.jit(opt.update_params)(params, corrupt)
    np.testing.assert_allclose(updated.mean, 0.2, atol=1e-6)
    if kind == "cbo":
        assert np.all(np.isfinite(updated.samples))
    rejected = rollouts.replace(
        costs=jnp.full((2, 2), jnp.inf),
        projection=diag.replace(valid=jnp.zeros((2, 1), dtype=bool)),
    )
    updated = jax.jit(opt.update_params)(params, rejected)
    np.testing.assert_array_equal(updated.mean, params.mean)
    if kind == "cem":
        np.testing.assert_array_equal(updated.cov, params.cov)
    if kind == "cbo":
        np.testing.assert_array_equal(updated.samples, params.samples)


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_xarm_preparation_matches_mujoco_and_refreshes_kinematics(
    impl: str,
) -> None:
    """The frozen rows use current qpos, not stale observed site arrays."""
    if impl == "warp" and jax.default_backend() != "gpu":
        pytest.skip("Warp data preparation requires a GPU")
    task = PushT(robot="xarm6", clutter=True, impl=impl)
    configure_projection(task, {"mode": "qpax"})
    projector = task.control_projector
    data = mujoco.MjData(task.mj_model)
    data.qpos[1] += 0.1
    data.qpos[5:8] += [0.02, -0.01, 0.2]
    data.qvel[:] = 0.03
    mujoco.mj_forward(task.mj_model, data)
    state = task.make_data().replace(
        qpos=jnp.array(data.qpos), qvel=jnp.array(data.qvel)
    )
    c = jax.jit(projector.prepare)(state)
    jacp, jacr = (
        np.zeros((3, task.mj_model.nv)),
        np.zeros((3, task.mj_model.nv)),
    )
    mujoco.mj_jacSite(task.mj_model, data, jacp, jacr, task.tip_site_id)
    np.testing.assert_allclose(
        c.cbf_a[0], jacp[2, task.robot_dof_adr], atol=1e-6
    )
    axis = data.site_xmat[task.tip_site_id].reshape(3, 3)[:, 2]
    expected = np.cross([0.0, 0.0, -1.0], axis) @ jacr[:, task.robot_dof_adr]
    np.testing.assert_allclose(c.tilt_a, expected, atol=1e-6)
    np.testing.assert_allclose(
        c.cbf_b[0],
        projector.config.cbf_alpha
        * (data.site_xpos[task.tip_site_id, 2] - projector.config.z_min),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "settings",
    [
        dict(mode="bad"),
        dict(z_near=-1.0),
        dict(distance_near=0.2),
        dict(max_iter=0),
        dict(tilt_weight=-1.0),
    ],
)
def test_invalid_configuration_fails_early(settings: dict) -> None:
    """Reject invalid geometry, solver and mode settings before JIT."""
    with pytest.raises(ValueError):
        replace(ProjectionConfig(), **settings)
