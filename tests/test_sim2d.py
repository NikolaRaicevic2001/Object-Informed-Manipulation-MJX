import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oim.algs import ADMM, MPPI, MJXRollout, WrenchConsensus, make_object_shim
from oim.objects import (
    Box,
    Capsule,
    Circle,
    ObstacleField,
    PlanarPushingObject,
    Polygon,
    Shape,
    contact_action_to_wrench,
    contact_force_to_com_wrench,
    estimate_contact_point,
    pack_contact_action,
    project_contact_action,
    rotate,
    sample_contact_actions,
    t_shape_footprint,
    unpack_contact_action,
)
from oim.sim2d import (
    PushT2D,
    build_admm_2d,
    build_scenario,
    list_scenarios,
    resolve_contact,
    run_2d,
)
from oim.tasks.pusht import PushT
from oim.utils.metrics import goal_errors, trial_metrics
from oim.utils.results import RunName, load_run, save_run

GOAL = [0.50, 0.48, float(jnp.pi / 4)]
OBSTACLES = [
    Circle(center=[0.08, 0.32], radius=0.04),
    Box(center=[0.38, 0.10], half_extents=[0.04, 0.035], angle=0.25),
    Polygon(jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])),
]


def _task(**kwargs) -> PushT2D:
    return PushT2D(footprint=t_shape_footprint(), goal=GOAL, **kwargs)


# ----------------------------------------------------------------------
# SDF extensions
# ----------------------------------------------------------------------


def test_sdf_and_grad_returns_unit_normals() -> None:
    """Gradients must be unit-length, including on the boundary itself."""
    shape = t_shape_footprint()
    pts = jnp.array([[0.5, 0.5], [0.0, -0.2], [0.015, -0.05]])
    d, grad = shape.sdf_and_grad(pts)
    assert d.shape == (3,)
    assert grad.shape == (3, 2)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.allclose(jnp.linalg.norm(grad, axis=-1), 1.0, atol=1e-4)


def test_sdf_and_grad_single_point_shape() -> None:
    """A single (2,) query returns a scalar distance and a (2,) gradient."""
    d, grad = t_shape_footprint().sdf_and_grad(jnp.array([0.5, 0.5]))
    assert d.shape == ()
    assert grad.shape == (2,)


def test_normals_are_exact_on_flat_faces() -> None:
    """The inward normal on a flat face must be that face's normal.

    Regression test: computing the gradient by autodiff gave a *wrong*
    normal (45 degrees off) for points lying exactly on an edge, because
    `Polygon.sdf` is `sqrt(dist2)` and `dist2 == 0` there. Contact points
    are projected onto the boundary before use, so that is the common case,
    and a wrong normal silently corrupts every derived contact wrench.
    """
    shape = t_shape_footprint()
    cases = [
        ((0.0, -0.105), (0.0, 1.0)),  # stem tip, pushes +y
        ((-0.015, -0.05), (1.0, 0.0)),  # stem left face, pushes +x
        ((0.015, -0.05), (-1.0, 0.0)),  # stem right face, pushes -x
        ((-0.049, 0.045), (0.0, -1.0)),  # crossbar top, pushes -y
        ((0.09, 0.030), (-1.0, 0.0)),  # crossbar right end, pushes -x
    ]
    for point, expected in cases:
        _, grad = shape.sdf_and_grad(jnp.array(point))
        inward = -grad
        assert jnp.allclose(inward, jnp.array(expected), atol=1e-3), (
            f"normal at {point}: got {inward}, expected {expected}"
        )


def test_contact_search_picks_a_face_that_pushes_toward_the_goal() -> None:
    """The boundary search must choose the correct *side* of the object.

    This is the capability the local rejection sampler cannot provide: it
    keeps every sample on the incumbent's face, so only a global search can
    decide to push from the opposite side when the goal is behind the
    object. Checked by the sign of the induced force against the direction
    the object actually needs to travel.
    """
    task = _task()
    shape = task.footprint
    goal = jnp.asarray(task.goal)[:2]

    for xy in [(0.0, 0.0), (0.9, 0.9), (0.9, 0.48), (0.5, -0.4)]:
        pose = jnp.array([xy[0], xy[1], 0.0])
        p = estimate_contact_point(
            shape=shape,
            pose=pose,
            goal_cost_fn=task._object_goal_cost,
            rng=jax.random.key(0),
            candidates=task._boundary_candidates,
            f_n=2.0,
        )
        assert jnp.abs(shape.sdf(p[None, :])[0]) < 1e-3  # on the boundary

        w = contact_action_to_wrench(
            shape, pose, pack_contact_action(p, jnp.array(2.0), jnp.array(0.0))
        )
        needed = goal - pose[:2]
        alignment = jnp.dot(w[:2], needed) / (
            jnp.linalg.norm(w[:2]) * jnp.linalg.norm(needed) + 1e-9
        )
        assert float(alignment) > 0.0, (
            f"object at {xy} needs {needed} but the chosen contact pushes "
            f"{w[:2]} (alignment {float(alignment):+.3f})"
        )


@pytest.mark.parametrize(
    "shape",
    [
        t_shape_footprint(),
        Circle(center=[0.0, 0.0], radius=0.05),
        Box(center=[0.0, 0.0], half_extents=[0.04, 0.02], angle=0.3),
        Capsule(a=[-0.05, 0.0], b=[0.05, 0.0], radius=0.01),
    ],
)
def test_sdf_gradient_matches_finite_differences(shape: Shape) -> None:
    """Every shape's reported gradient must match a coarse FD of its own sdf.

    A cheap, general consistency check between `sdf` and `sdf_and_grad` --
    exactly the class of test that catches a gradient silently disagreeing
    with the distance field it is supposed to differentiate. Sampled away
    from faces and the medial axis, where the true gradient is genuinely
    discontinuous or undefined.
    """
    pts = jnp.array(
        [[0.13, 0.09], [-0.11, 0.07], [0.08, -0.12], [-0.09, -0.14]]
    )
    _, grad = shape.sdf_and_grad(pts)

    h = 1e-3
    fd = jnp.stack(
        [
            (
                shape.sdf(pts + jnp.array([h, 0.0]))
                - shape.sdf(pts - jnp.array([h, 0.0]))
            )
            / (2 * h),
            (
                shape.sdf(pts + jnp.array([0.0, h]))
                - shape.sdf(pts - jnp.array([0.0, h]))
            )
            / (2 * h),
        ],
        axis=-1,
    )
    fd = fd / jnp.linalg.norm(fd, axis=-1, keepdims=True)
    assert jnp.allclose(grad, fd, atol=1e-2), f"{grad} vs {fd}"


def test_capsule_sdf_matches_segment_distance() -> None:
    """The capsule is the set within `radius` of its segment."""
    cap = Capsule(a=[-0.05, 0.0], b=[0.05, 0.0], radius=0.01)
    # Beside the middle of the segment: distance is vertical offset - radius.
    assert jnp.allclose(cap.sdf(jnp.array([[0.0, 0.03]]))[0], 0.02, atol=1e-6)
    # Past an end cap: distance is measured from the endpoint.
    assert jnp.allclose(cap.sdf(jnp.array([[0.09, 0.0]]))[0], 0.03, atol=1e-6)
    # On the axis: inside by the full radius.
    assert jnp.allclose(cap.sdf(jnp.array([[0.0, 0.0]]))[0], -0.01, atol=1e-6)
    assert cap.bounding_radius == pytest.approx(0.06)


def test_project_to_boundary_lands_on_boundary() -> None:
    """Points outside the shape project onto its surface."""
    shape = t_shape_footprint()
    pts = jnp.array([[0.3, 0.3], [-0.2, 0.0], [0.0, -0.4]])
    proj = shape.project_to_boundary(pts)
    assert jnp.max(jnp.abs(shape.sdf(proj))) < 1e-4


@pytest.mark.parametrize(
    "shape",
    [
        t_shape_footprint(),
        Circle(center=[0.1, 0.2], radius=0.05),
        Box(center=[0.0, 0.0], half_extents=[0.04, 0.02], angle=0.3),
        Capsule(a=[-0.05, 0.0], b=[0.05, 0.0], radius=0.01),
    ],
)
def test_projection_reaches_boundary_from_degenerate_interior_points(
    shape: Shape,
) -> None:
    """Projection must land on the surface from *anywhere*, not just outside.

    Regression test for two distinct failures, both of which returned
    interior points while looking successful:

    * The generic gradient iteration oscillates in a 2-cycle on a polygon's
      medial axis -- the T footprint's origin mapped to (0, 0.015) and
      straight back, forever, every iterate 15 mm inside the shape.
    * A closed-form projection can pick a degenerate fallback direction: a
      capsule query lying exactly on its spine has no offset to normalize,
      and stepping along a fixed axis slides *down* the spine, staying
      inside.

    `project_contact_action` relies on this to guarantee contact points sit
    on the surface, so an interior result silently corrupts every derived
    contact wrench.
    """
    # Interior seeds, including each shape's own most degenerate point.
    queries = jnp.array(
        [
            [0.0, 0.0],  # T's medial axis / circle centre / capsule spine
            [0.02, 0.0],
            [0.1, 0.2],  # circle's exact centre
            [0.0, 0.015],  # the T's 2-cycle partner
        ]
    )
    proj = shape.project_to_boundary(queries)
    residual = jnp.max(jnp.abs(shape.sdf(proj)))
    assert float(residual) < 1e-4, (
        f"{type(shape).__name__}: projected points are {residual:.2e} off "
        f"the boundary -- {proj}"
    )


def test_bounding_radius_contains_shape() -> None:
    """Every boundary sample must sit inside the bounding disc."""
    for shape in [t_shape_footprint(), *OBSTACLES]:
        if not isinstance(shape, Polygon):
            continue
        r = jnp.linalg.norm(shape.vertices - shape.center, axis=-1)
        assert float(jnp.max(r)) <= shape.bounding_radius + 1e-6


# ----------------------------------------------------------------------
# Contact parameterization
# ----------------------------------------------------------------------


def test_wrench_map_matches_hand_computation() -> None:
    """W = J_c^T f: force passes through, torque is the moment arm cross."""
    pose = jnp.array([1.0, 2.0, 0.0])
    p_world = jnp.array([1.5, 2.0])  # 0.5 m along +x from the pose origin
    f_world = jnp.array([0.0, 3.0])  # 3 N along +y
    w = contact_force_to_com_wrench(pose, p_world, f_world)
    assert jnp.allclose(w[:2], f_world)
    assert jnp.allclose(w[2], 0.5 * 3.0)  # r_x f_y - r_y f_x


def test_projection_enforces_friction_cone_and_unilaterality() -> None:
    """The projection must make every contact action physically realizable."""
    shape = t_shape_footprint()
    raw = jnp.array([[0.5, 0.5, -3.0, 10.0], [0.0, 0.0, 99.0, -99.0]])
    proj = project_contact_action(shape, raw, mu_c=0.5, f_max=4.0)
    p, f_n, f_t = unpack_contact_action(proj)

    assert jnp.max(jnp.abs(shape.sdf(p))) < 1e-4  # on the boundary
    assert jnp.all(f_n >= 0.0)  # contacts push, never pull
    assert jnp.all(f_n <= 4.0)
    assert jnp.all(jnp.abs(f_t) <= 0.5 * f_n + 1e-5)  # inside the cone


def test_sampled_actions_are_always_feasible() -> None:
    """Every sample -- not just the mean -- must satisfy the constraints."""
    shape = t_shape_footprint()
    nominal = jnp.broadcast_to(jnp.array([0.015, -0.105, 1.0, 0.0]), (15, 4))
    s = jax.jit(
        lambda k: sample_contact_actions(
            shape, nominal, k, 64, 0.012, 0.7, 0.3, 0.5, 4.0, 0.7
        )
    )(jax.random.key(0))

    assert s.shape == (64, 15, 4)
    p, f_n, f_t = unpack_contact_action(s)
    assert jnp.all(jnp.isfinite(s))
    assert jnp.max(jnp.abs(shape.sdf(p.reshape(-1, 2)))) < 1e-4
    assert jnp.all(f_n >= 0.0) and jnp.all(f_n <= 4.0)
    assert jnp.all(jnp.abs(f_t) <= 0.5 * f_n + 1e-5)


def test_zero_force_gives_zero_wrench() -> None:
    """No normal or tangential force means no wrench, whatever the point."""
    shape = t_shape_footprint()
    action = pack_contact_action(
        jnp.array([0.015, -0.105]), jnp.array(0.0), jnp.array(0.0)
    )
    w = contact_action_to_wrench(shape, jnp.array([0.1, 0.2, 0.3]), action)
    assert jnp.allclose(w, 0.0, atol=1e-6)


# ----------------------------------------------------------------------
# 2D physics engine
# ----------------------------------------------------------------------


def test_no_contact_means_no_wrench_and_no_object_motion() -> None:
    """A robot far from the object must not disturb it."""
    task = _task()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(1.0, 1.0))
    out = task.rollout.step(task.model, state, jnp.array([0.1, 0.0]))
    assert jnp.allclose(out.wrench, 0.0)
    assert jnp.allclose(out.object_pose, state.object_pose)


def test_pushing_moves_the_object_along_the_push_direction() -> None:
    """Pressing on the stem tip from below must drive the object +y."""
    task = _task()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(0.0, -0.115))
    out = task.rollout.step(task.model, state, jnp.array([0.0, 0.6]))
    assert float(out.wrench[1]) > 0.0
    assert float(out.object_pose[1]) > float(state.object_pose[1])


def test_resolve_contact_respects_the_friction_cone() -> None:
    """The tangential force the solver reports stays inside the cone."""
    task = _task()
    pose = jnp.zeros(3)
    # Deep penetration with strong tangential sliding.
    w = resolve_contact(
        task.footprint,
        task.model,
        pose,
        jnp.array([0.0, -0.10]),
        jnp.array([0.6, 0.6]),
        dt=0.05,
    )
    assert jnp.all(jnp.isfinite(w))
    assert (
        float(jnp.linalg.norm(w[:2]))
        <= task.f_max * jnp.sqrt(1.0 + task.mu_c**2) + 1e-4
    )


def test_engine_step_is_jittable_and_vmappable() -> None:
    """The rollout backend has to survive the same transforms MJX does."""
    task = _task(obstacles=OBSTACLES)
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(0.0, -0.13))
    controls = jnp.tile(jnp.array([0.0, 0.5]), (8, 1))
    batched = jax.jit(
        jax.vmap(lambda u: task.rollout.step(task.model, state, u))
    )(controls)
    assert batched.object_pose.shape == (8, 3)
    assert jnp.all(jnp.isfinite(batched.wrench))


# ----------------------------------------------------------------------
# ADMM on the 2D backend
# ----------------------------------------------------------------------


def test_admm_2d_optimize_is_finite_under_jit() -> None:
    """The shared ADMM controller must drive the 2D world under jit."""
    task = _task(obstacles=OBSTACLES)
    ctrl, params = build_admm_2d(task, n_admm=3, num_samples=16)
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(0.0, -0.13))

    new_params, rollouts = jax.jit(ctrl.optimize)(state, params)

    assert jnp.all(jnp.isfinite(rollouts.costs))
    assert jnp.all(jnp.isfinite(new_params.mean))
    assert jnp.all(jnp.isfinite(new_params.z))
    assert new_params.z.shape == (15, 3)


def test_admm_2d_object_actions_stay_feasible_through_the_loop() -> None:
    """The nominal contact action must still be realizable after ADMM.

    The whole point of the contact parameterization is that the object
    block cannot converge onto a wrench no point contact could apply.
    """
    task = _task(obstacles=OBSTACLES, contact_actions=True)
    ctrl, params = build_admm_2d(task, n_admm=4, num_samples=16)
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(0.0, -0.13))
    new_params, _ = jax.jit(ctrl.optimize)(state, params)

    p, f_n, f_t = unpack_contact_action(new_params.object_params.mean)
    assert jnp.max(jnp.abs(task.footprint.sdf(p))) < 1e-3
    assert jnp.all(f_n >= 0.0) and jnp.all(f_n <= task.f_max)
    assert jnp.all(jnp.abs(f_t) <= task.mu_c * f_n + 1e-4)


def test_contact_action_object_block_still_works() -> None:
    """The opt-in contact-action parameterization must remain usable."""
    task = _task(contact_actions=True)
    assert task.object_action_dim == 4
    assert task.initial_object_action() is not None

    ctrl, params = build_admm_2d(task, n_admm=2, num_samples=16)
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(0.0, -0.13))
    new_params, rollouts = jax.jit(ctrl.optimize)(state, params)

    assert new_params.object_params.mean.shape == (15, 4)
    assert jnp.all(jnp.isfinite(rollouts.costs))


def test_both_worlds_default_to_the_direct_wrench_object_block() -> None:
    """2D and 3D must agree on the object parameterization by default.

    The object block's job is to say *what motion it wants*; deciding where
    on the object to push is the robot block's. Both worlds therefore
    sample the consensus wrench directly, and the opt-in contact-action
    parameterization changes only the action space -- `z` stays the wrench
    either way.
    """
    task_2d = _task()
    assert task_2d.object_action_dim == task_2d.consensus_dim == 3
    assert task_2d.initial_object_action() is None

    task_3d = PushT(clutter=True, planning_dt=0.05)
    assert task_3d.object_action_dim == task_3d.consensus_dim == 3
    assert task_3d.initial_object_action() is None

    opt_in = _task(contact_actions=True)
    assert opt_in.object_action_dim == 4
    assert opt_in.consensus_dim == 3


def test_closed_loop_makes_progress_toward_the_goal() -> None:
    """A short closed loop must move the object nearer the goal, stably."""
    task = _task(obstacles=OBSTACLES)
    ctrl, params = build_admm_2d(task, n_admm=4, num_samples=32)
    log = run_2d(
        task,
        ctrl,
        params,
        object_pose0=(0.0, 0.0, 0.0),
        robot_pos0=(0.0, -0.13),
        max_steps=25,
        verbose=False,
    )
    poses = log["object_pose"]
    goal = jnp.asarray(task.goal)[:2]
    d0 = float(jnp.linalg.norm(jnp.asarray(poses[0][:2]) - goal))
    d1 = float(jnp.linalg.norm(jnp.asarray(poses[-1][:2]) - goal))

    assert jnp.all(jnp.isfinite(jnp.asarray(poses)))
    assert d1 < d0, f"object got no closer: {d0:.4f} -> {d1:.4f}"


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list_scenarios())
def test_scenario_start_and_goal_are_collision_free(name: str) -> None:
    """A scenario that starts or ends in an obstacle is not solvable.

    Checked on the object's whole footprint, not just its origin -- the T is
    ~0.18 m wide, so an origin-only check would pass poses whose crossbar is
    buried in a wall.
    """
    sc = build_scenario(name)
    field = ObstacleField(sc.obstacles)
    for label, pose in (
        ("start", sc.object_pose0),
        ("goal", sc.goal),
    ):
        world = jnp.asarray(pose[:2]) + rotate(
            pose[2], sc.footprint.sample_boundary(4)
        )
        assert float(jnp.min(field.sdf(world))) > 0.0, (
            f"{name}: object footprint at {label} overlaps an obstacle"
        )


@pytest.mark.parametrize("name", list_scenarios())
def test_scenario_robot_starts_outside_the_object(name: str) -> None:
    """The robot must not begin embedded in the object it is to push."""
    sc = build_scenario(name)
    body = rotate(
        -sc.object_pose0[2],
        jnp.asarray(sc.robot_pos0) - jnp.asarray(sc.object_pose0[:2]),
    )
    assert float(sc.footprint.sdf(body[None, :])[0]) > 0.0


@pytest.mark.parametrize("name", ["corridor", "gate"])
def test_passage_is_wide_enough_for_the_object(name: str) -> None:
    """The passage scenarios must actually be passable.

    These two exist to force the object *through* an opening rather than
    around obstacles, which only tests anything if the opening is wider than
    the object but not by much. Verified by sweeping the object along the
    passage centreline and requiring clearance throughout.
    """
    sc = build_scenario(name)
    field = ObstacleField(sc.obstacles)
    start = jnp.asarray(sc.object_pose0)
    goal = jnp.asarray(sc.goal)

    clearances = []
    for s in jnp.linspace(0.0, 1.0, 25):
        # Translate along the centreline holding the start orientation: the
        # object should not need to rotate to get through.
        xy = (1 - s) * start[:2] + s * goal[:2]
        world = xy + rotate(start[2], sc.footprint.sample_boundary(4))
        clearances.append(float(jnp.min(field.sdf(world))))

    assert min(clearances) > 0.0, (
        f"{name}: object collides while traversing the passage upright "
        f"(min clearance {min(clearances):.4f} m)"
    )
    # ...but the passage should be tight, or it is not testing anything.
    assert min(clearances) < 0.05, (
        f"{name}: passage is too generous (min clearance "
        f"{min(clearances):.4f} m) to exercise non-myopic behaviour"
    )


# ----------------------------------------------------------------------
# The shared-code guarantee that makes the 2D layer worth having
# ----------------------------------------------------------------------


def test_2d_and_mjx_share_the_object_subproblem() -> None:
    """Both worlds must run the *same* object block, not two copies."""
    task = _task()
    ctrl, _ = build_admm_2d(task, n_admm=1, num_samples=8)

    # The object-level dynamics/costs come from the same class the MJX
    # push-T task uses, so validating them in 2D validates them for MJX.
    assert isinstance(task.object_model, PlanarPushingObject)
    # And the consensus math is the shared implementation, not a 2D fork.
    assert isinstance(ctrl.consensus, WrenchConsensus)
    assert ctrl.object_subproblem.consensus is ctrl.consensus
    assert ctrl.robot_subproblem.consensus is ctrl.consensus


def test_default_rollout_backend_is_still_mjx() -> None:
    """Omitting `rollout` must leave the MuJoCo path exactly as it was."""
    task = PushT(clutter=True, planning_dt=0.05)
    robot_opt = MPPI(
        task,
        num_samples=4,
        noise_level=0.4,
        temperature=1.0,
        plan_horizon=15 * 0.05,
        spline_type="linear",
        num_knots=4,
    )
    object_opt = MPPI(
        make_object_shim(task, dt=0.05),
        num_samples=4,
        noise_level=1.0,
        temperature=1.0,
        plan_horizon=15 * 0.05,
        spline_type="zero",
        num_knots=15,
    )
    ctrl = ADMM(
        task,
        robot_opt,
        object_opt,
        WrenchConsensus(max_dual=15.0),
        n_admm=1,
        eps_r=1.0,
        eps_s=1.0,
    )
    assert isinstance(ctrl.robot_subproblem.rollout, MJXRollout)
    # And the MJX task keeps the direct-wrench object block by default.
    assert task.object_action_dim == task.consensus_dim == 3
    assert task.initial_object_action() is None


# ----------------------------------------------------------------------
# Recording and scoring are decoupled: a run file records only what was
# observed, and every metric is re-derived from it.
# ----------------------------------------------------------------------


def test_run_file_holds_no_derived_quantities(tmp_path) -> None:  # noqa: ANN001
    """`save_run` must not persist anything `metrics` can recompute.

    Storing `reached` or `pos_err` would freeze one definition of a metric
    into the evidence, which is exactly what `oim/run_eval.py` exists to
    avoid.
    """
    task = _task()
    ctrl, params = build_admm_2d(task, n_admm=1, num_samples=8, horizon=4)
    log = run_2d(
        task,
        ctrl,
        params,
        max_steps=3,
        verbose=False,
        goal_pos_tol=0.05,
        goal_theta_tol=0.05,
    )
    path = save_run(
        str(tmp_path),
        RunName("t"),
        run=dict(world="2d", task="t", algorithm="admm", seed=0),
        hyperparameters=dict(
            control_dt=float(task.dt), goal_pos_tol=0.05, goal_theta_tol=0.05
        ),
        task=task,
        log=log,
    )
    saved = load_run(path)
    derived = {"pos_err", "theta_err", "reached", "steps_run"}
    assert not derived & set(saved["dynamic"]), (
        "a derived quantity leaked into the run file"
    )
    assert derived & set(log), "the in-memory log should still carry them"


def test_derived_metrics_reproduce_the_runner(tmp_path) -> None:  # noqa: ANN001
    """Post-hoc metrics must equal what the closed loop computed live.

    The whole decoupling rests on this: if re-deriving `pos_err` from the
    recorded poses disagreed with the value the loop used to decide
    success, every table would silently describe a different experiment.
    """
    task = _task()
    ctrl, params = build_admm_2d(task, n_admm=1, num_samples=8, horizon=4)
    log = run_2d(
        task,
        ctrl,
        params,
        max_steps=4,
        verbose=False,
        goal_pos_tol=0.05,
        goal_theta_tol=0.05,
    )
    saved = load_run(
        save_run(
            str(tmp_path),
            RunName("t"),
            run=dict(world="2d", task="t", algorithm="admm", seed=0),
            hyperparameters=dict(
                control_dt=float(task.dt),
                goal_pos_tol=0.05,
                goal_theta_tol=0.05,
            ),
            task=task,
            log=log,
        )
    )
    err = goal_errors(saved)
    assert np.allclose(err["pos_err"], log["pos_err"], atol=1e-9)
    assert np.allclose(err["theta_err"], log["theta_err"], atol=1e-9)
    assert trial_metrics(saved)["reached"] == bool(log["reached"])
    assert trial_metrics(saved)["steps_run"] == len(log["pos_err"])
