"""Object geometry and the contact parameterization.

Split out of the deleted 2D world's suite: these assert on
`oim.objects.sdf` and `oim.objects.contact` -- the SDF, its gradient, the
boundary projection, and the contact action's wrench map, friction cone and
sampler -- none of which the 2D world owned. `PushT` appears only in the
last test, which pins the object block's default rollout backend.
"""
import jax
import jax.numpy as jnp
import pytest

from oim.algs import ADMM, MPPI, MJXRollout, WrenchConsensus, make_object_shim
from oim.objects import (
    Box,
    Capsule,
    Circle,
    Polygon,
    Shape,
    contact_action_to_wrench,
    contact_force_to_com_wrench,
    pack_contact_action,
    project_contact_action,
    sample_contact_actions,
    t_shape_footprint,
    unpack_contact_action,
)
from oim.tasks.pusht import PushT

# A mixed bag of shapes, so the SDF tests below cover every `Shape`
# subclass rather than only the footprint.
OBSTACLES = [
    Circle(center=[0.08, 0.32], radius=0.04),
    Box(center=[0.38, 0.10], half_extents=[0.04, 0.035], angle=0.25),
    Polygon(jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])),
]



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
