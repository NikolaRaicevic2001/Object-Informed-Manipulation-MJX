"""Tests for the trajectory overlay shared by every sampling-based method.

What matters is that one drawing path serves them all: a flat controller
contributes one block, ADMM two, and the sample/chosen distinction reads
identically in both. These check the geometry the overlay writes into an
`MjvScene` -- no rendering, no GPU, no controller.
"""

import mujoco
import numpy as np
import pytest

from oim.runtime.overlay import (
    CONTACT_POINT_HEIGHT,
    CONTACT_SCHEME,
    OBJECT_PLAN_HEIGHT,
    OBJECT_SCHEME,
    ROBOT_OBJECT_PLAN_HEIGHT,
    ROBOT_OBJECT_SCHEME,
    ROBOT_SCHEME,
    BlockTrace,
    PlanOverlay,
    contact_points_world,
    lift_se2,
    traces_for,
)

_H = 8
_N_SAMPLES = 5


@pytest.fixture(scope="module")
def scene() -> mujoco.MjvScene:
    """A scene with room to spare, built off a one-geom model."""
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='sphere' size='.1'/>"
        "</worldbody></mujoco>"
    )
    return mujoco.MjvScene(model, maxgeom=4000)


def _fresh(scene: mujoco.MjvScene) -> mujoco.MjvScene:
    """Reset the scene's geom count between cases."""
    scene.ngeom = 0
    return scene


def _path(n: int = _H) -> np.ndarray:
    return np.linspace(0.0, 1.0, n * 3).reshape(n, 3)


def _samples(n: int = _N_SAMPLES, h: int = _H) -> np.ndarray:
    return np.stack([_path(h) + i for i in range(n)])


# ----------------------------------------------------------------------
# traces_for: what each algorithm contributes
# ----------------------------------------------------------------------


def test_flat_controller_contributes_one_block() -> None:
    """A flat method has one population, in robot space, and no object block."""
    traces = traces_for(robot_chosen=_path(), robot_samples=_samples())
    assert [t.scheme.name for t in traces] == ["robot"]
    assert traces[0].scheme is ROBOT_SCHEME


def test_admm_contributes_object_and_robot() -> None:
    """ADMM's two subproblems are two blocks, object drawn first."""
    traces = traces_for(
        robot_chosen=_path(),
        robot_samples=_samples(),
        object_chosen=_path(),
        object_samples=_samples(),
    )
    assert [t.scheme.name for t in traces] == ["object", "robot"]


def test_admm_draws_both_blocks_predictions_for_the_object() -> None:
    """The robot block's own object plan is a third, separate path.

    It is the diagnostic the other two cannot give: `object_chosen` is what
    the object planner wants and `robot_chosen` is where the tip goes, but
    only this one says what the robot's controls would actually do *to the
    object*, which is the quantity the consensus is arguing about.
    """
    traces = traces_for(
        robot_chosen=_path(),
        robot_samples=_samples(),
        object_chosen=_path(),
        object_samples=_samples(),
        robot_object_chosen=_path(),
    )
    # Object-space paths first, so the robot's thicker end-effector path is
    # drawn over them rather than under.
    assert [t.scheme.name for t in traces] == [
        "object",
        "robot-object",
        "robot",
    ]
    assert traces[1].scheme is ROBOT_OBJECT_SCHEME


def test_robot_object_plan_is_lifted_clear_of_the_object_plan() -> None:
    """The two object-space paths must not be coincident geometry.

    They predict the same object, so at consensus they are the same line --
    which z-fights, and makes "the blocks agree" indistinguishable from
    "one of them was not drawn".
    """
    poses = np.array([[0.1, 0.2, 1.57], [0.3, 0.4, 0.0]])
    traces = traces_for(object_chosen=poses, robot_object_chosen=poses)

    object_path, robot_object_path = traces[0].chosen, traces[1].chosen
    # Same (x, y) -- only the drawing height separates them.
    np.testing.assert_allclose(object_path[:, :2], robot_object_path[:, :2])
    assert np.all(object_path[:, 2] == OBJECT_PLAN_HEIGHT)
    assert np.all(robot_object_path[:, 2] == ROBOT_OBJECT_PLAN_HEIGHT)
    assert ROBOT_OBJECT_PLAN_HEIGHT > OBJECT_PLAN_HEIGHT


def test_all_three_paths_use_different_colors() -> None:
    """No two of the three may share a color, chosen or sample.

    The blue and magenta paths are the pair a viewer actually compares, so
    those two above all must be tellable apart.
    """
    schemes = (OBJECT_SCHEME, ROBOT_OBJECT_SCHEME, ROBOT_SCHEME)
    assert len({s.chosen for s in schemes}) == 3
    assert len({s.sample for s in schemes}) == 3
    assert len({s.name for s in schemes}) == 3


def test_three_paths_fit_the_reserved_geoms(scene) -> None:  # noqa: ANN001
    """A three-path overlay must reserve and stay inside its own budget."""
    overlay = PlanOverlay(horizon=_H, max_blocks=3)
    traces = traces_for(
        robot_chosen=_path(),
        robot_samples=_samples(),
        object_chosen=_path(),
        object_samples=_samples(),
        robot_object_chosen=_path(),
    )
    overlay.draw(_fresh(scene), traces)
    assert scene.ngeom <= overlay.geom_count


def test_the_two_blocks_use_different_colors() -> None:
    """Object and robot must be tellable apart at a glance.

    Within a block the two colors share a hue family, so sample and chosen
    read as one thing; across blocks they must not.
    """
    assert OBJECT_SCHEME.sample != ROBOT_SCHEME.sample
    assert OBJECT_SCHEME.chosen != ROBOT_SCHEME.chosen
    # Cool vs warm: the object's blue channel dominates, the robot's red.
    assert OBJECT_SCHEME.chosen[2] > OBJECT_SCHEME.chosen[0]
    assert ROBOT_SCHEME.chosen[0] > ROBOT_SCHEME.chosen[2]


def test_object_paths_are_lifted_and_robot_paths_are_not() -> None:
    """SE(2) object plans get a drawing height; robot paths are real 3D."""
    poses = np.array([[0.1, 0.2, 1.57], [0.3, 0.4, 0.0]])
    lifted = lift_se2(poses)
    assert lifted.shape == (2, 3)
    np.testing.assert_allclose(lifted[:, :2], poses[:, :2])
    assert np.all(lifted[:, 2] == OBJECT_PLAN_HEIGHT)

    robot = _path()
    traces = traces_for(robot_chosen=robot, object_chosen=poses)
    np.testing.assert_allclose(traces[1].chosen, robot)
    assert np.all(traces[0].chosen[:, 2] == OBJECT_PLAN_HEIGHT)


def test_either_half_alone_still_builds_a_block() -> None:
    """`--show-samples` and `--show-optimal` are independent."""
    only_samples = traces_for(robot_samples=_samples())
    assert only_samples[0].chosen is None
    assert only_samples[0].samples is not None

    only_chosen = traces_for(robot_chosen=_path())
    assert only_chosen[0].samples is None
    assert only_chosen[0].chosen is not None

    assert traces_for() == []


# ----------------------------------------------------------------------
# PlanOverlay: what lands in the scene
# ----------------------------------------------------------------------


def test_chosen_is_drawn_thicker_than_a_sample(scene) -> None:  # noqa: ANN001
    """The whole point: the decision reads through its own sample cloud."""
    overlay = PlanOverlay(horizon=_H, max_blocks=1)
    assert overlay.chosen_width > overlay.sample_width

    _fresh(scene)
    overlay.draw(scene, traces_for(robot_samples=_samples()))
    sample_widths = {scene.geoms[i].size[0] for i in range(scene.ngeom)}

    _fresh(scene)
    overlay.draw(scene, traces_for(robot_chosen=_path()))
    chosen_widths = {scene.geoms[i].size[0] for i in range(scene.ngeom)}

    assert max(sample_widths) < min(chosen_widths)


def test_geom_count_matches_what_was_drawn(scene) -> None:  # noqa: ANN001
    """One segment per pair of consecutive points, per path, per block."""
    overlay = PlanOverlay(horizon=_H, max_blocks=2)
    _fresh(scene)
    overlay.draw(
        scene,
        traces_for(
            robot_chosen=_path(),
            robot_samples=_samples(),
            object_chosen=_path(),
            object_samples=_samples(),
        ),
    )
    per_path = _H - 1
    expected = 2 * (_N_SAMPLES + 1) * per_path
    assert scene.ngeom == expected
    assert expected <= overlay.geom_count


def test_reserved_geoms_cover_the_worst_case(scene) -> None:  # noqa: ANN001
    """`geom_count` is what a caller reserves, so it must not be exceeded."""
    overlay = PlanOverlay(horizon=_H, max_blocks=2, max_samples=3)
    _fresh(scene)
    # More samples than max_samples: it must subsample, not overrun.
    overlay.draw(
        scene,
        traces_for(
            robot_chosen=_path(),
            robot_samples=_samples(n=64),
            object_chosen=_path(),
            object_samples=_samples(n=64),
        ),
    )
    assert scene.ngeom <= overlay.geom_count


def test_samples_are_subsampled_not_truncated(scene) -> None:  # noqa: ANN001
    """Drawing the first k rollouts would misrepresent the population."""
    overlay = PlanOverlay(horizon=_H, max_blocks=1, max_samples=3)
    _fresh(scene)
    overlay.draw(scene, traces_for(robot_samples=_samples(n=64)))
    assert scene.ngeom == 3 * (_H - 1)


def test_a_block_drawn_alone_uses_its_own_colors(scene) -> None:  # noqa: ANN001
    """A flat run's lines are the robot scheme, never the object's."""
    overlay = PlanOverlay(horizon=_H, max_blocks=1)
    _fresh(scene)
    overlay.draw(scene, traces_for(robot_chosen=_path()))
    rgb = np.array([scene.geoms[i].rgba[:3] for i in range(scene.ngeom)])
    expected = np.tile(ROBOT_SCHEME.chosen, (len(rgb), 1))
    np.testing.assert_allclose(rgb, expected, atol=1e-6)


def test_more_blocks_than_reserved_is_rejected(scene) -> None:  # noqa: ANN001
    """A one-block overlay handed ADMM's two would silently overrun."""
    overlay = PlanOverlay(horizon=_H, max_blocks=1)
    traces = traces_for(robot_chosen=_path(), object_chosen=_path())
    with pytest.raises(ValueError, match="reserved geoms for 1 block"):
        overlay.draw(_fresh(scene), traces)


def test_a_path_longer_than_the_horizon_is_rejected(scene) -> None:  # noqa: ANN001
    """H or H+1 points are both normal; anything longer is a mismatch."""
    overlay = PlanOverlay(horizon=_H, max_blocks=1)
    overlay.draw(_fresh(scene), traces_for(robot_chosen=_path(_H)))
    overlay.draw(_fresh(scene), traces_for(robot_chosen=_path(_H + 1)))
    with pytest.raises(ValueError, match="overlay was built for"):
        overlay.draw(_fresh(scene), traces_for(robot_chosen=_path(_H + 5)))


def test_a_full_scene_is_reported_not_overrun(scene) -> None:  # noqa: ANN001
    """Writing past `maxgeom` would corrupt memory rather than raise."""
    overlay = PlanOverlay(horizon=_H, max_blocks=2)
    with pytest.raises(RuntimeError, match="only .* free"):
        overlay.draw(
            _fresh(scene),
            [BlockTrace(scheme=ROBOT_SCHEME, chosen=_path())],
            base=scene.maxgeom - 1,
        )


# ----------------------------------------------------------------------
# Contact-point dots
# ----------------------------------------------------------------------


def test_contact_points_ride_the_object_as_it_turns() -> None:
    """A body-frame point must be placed through each pose's rotation.

    The whole reason the contact parameterization exists is that one fixed
    p tracks the same material point as the object rotates -- so drawing it
    at a fixed world offset would show exactly the thing it is not.
    """
    # Same contact point every step; the object spins in place a quarter turn.
    contacts = np.tile(np.array([0.05, 0.0, 2.0]), (3, 1))
    poses = np.array(
        [[1.0, 2.0, 0.0], [1.0, 2.0, np.pi / 2], [1.0, 2.0, np.pi]]
    )
    pts = contact_points_world(poses, contacts)

    assert pts.shape == (3, 3)
    # +x in the body frame, swung around the (unmoving) object origin.
    assert np.allclose(pts[0, :2], [1.05, 2.0], atol=1e-6)
    assert np.allclose(pts[1, :2], [1.0, 2.05], atol=1e-6)
    assert np.allclose(pts[2, :2], [0.95, 2.0], atol=1e-6)
    # NOT the object plan's height: the dot marks a place the tip has to
    # reach, and `w_z_tip` holds the tip at block mid-height. Drawing it up
    # on the plan would show consensus and height cost disagreeing.
    assert np.allclose(pts[:, 2], CONTACT_POINT_HEIGHT)
    assert CONTACT_POINT_HEIGHT < OBJECT_PLAN_HEIGHT


def test_contact_points_take_the_tasks_tip_height() -> None:
    """The height is the caller's to supply -- `tip_target_z`, per scene."""
    poses = np.zeros((2, 3))
    pts = contact_points_world(poses, np.zeros((2, 3)), height=0.042)
    assert np.allclose(pts[:, 2], 0.042)


def test_contact_points_ignore_lambda_and_extra_steps() -> None:
    """Only p is drawn, and a mismatched length truncates rather than raises."""
    poses = np.zeros((4, 3))
    quiet = contact_points_world(poses, np.tile([0.01, 0.02, 0.0], (4, 1)))
    loud = contact_points_world(poses, np.tile([0.01, 0.02, 99.0], (4, 1)))
    assert np.allclose(quiet, loud), "lambda must not move the dot"

    assert len(contact_points_world(poses, np.zeros((2, 3)))) == 2
    assert len(contact_points_world(poses[:1], np.zeros((9, 3)))) == 1


def test_contact_points_are_their_own_trace_in_red() -> None:
    """A fourth trace, not points bolted onto the object block's."""
    traces = traces_for(
        object_chosen=_path(),
        robot_chosen=_path(),
        robot_object_chosen=_path(),
        contact_points=np.zeros((_H, 3)),
    )
    assert len(traces) == 4
    contact = [t for t in traces if t.scheme is CONTACT_SCHEME]
    assert len(contact) == 1
    # Points only: joining consecutive contacts would draw a line straight
    # through the object whenever the contact changes face.
    assert contact[0].chosen is None and contact[0].samples is None
    assert contact[0].points is not None
    # Red, and distinct from every path colour it is drawn on top of.
    for other in (OBJECT_SCHEME, ROBOT_SCHEME, ROBOT_OBJECT_SCHEME):
        assert CONTACT_SCHEME.chosen != other.chosen


def test_no_contact_points_adds_no_trace() -> None:
    """The wrench path must be untouched: no fourth trace, no empty one."""
    for empty in (None, np.zeros((0, 3))):
        traces = traces_for(object_chosen=_path(), contact_points=empty)
        assert all(t.scheme is not CONTACT_SCHEME for t in traces)


def test_contact_dots_are_spheres_inside_the_reserved_geoms(scene) -> None:  # noqa: ANN001
    """Four traces must fit, and the dots must draw as spheres not lines."""
    overlay = PlanOverlay(horizon=_H, max_blocks=4, max_samples=_N_SAMPLES)
    traces = traces_for(
        object_chosen=_path(),
        object_samples=_samples(),
        robot_chosen=_path(),
        robot_samples=_samples(),
        robot_object_chosen=_path(),
        contact_points=_path(),
    )
    s = _fresh(scene)
    overlay.draw(s, traces)

    assert s.ngeom <= overlay.geom_count
    spheres = [
        s.geoms[i]
        for i in range(s.ngeom)
        if s.geoms[i].type == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    assert len(spheres) == _H
    # Ramped first-to-last in *hue* as well as opacity: dots are separate
    # in space, so alpha alone reads as "some are dimmer", not as an order.
    assert np.allclose(spheres[0].rgba[:3], CONTACT_SCHEME.chosen)
    assert np.allclose(spheres[-1].rgba[:3], CONTACT_SCHEME.sample)
    assert spheres[0].rgba[3] > spheres[-1].rgba[3]
    # Monotone both ways, so any two dots can be ordered by eye.
    reds = [g.rgba[0] for g in spheres]
    greens = [g.rgba[1] for g in spheres]
    alphas = [g.rgba[3] for g in spheres]
    assert greens == sorted(greens), "hue must ramp monotonically"
    assert alphas == sorted(alphas, reverse=True)
    assert reds == sorted(reds)


def test_contact_dots_are_gated_on_the_consensus() -> None:
    """The flag alone must not draw dots; the task has to agree.

    Under a wrench consensus z is [f_x, f_y, tau] -- putting its first two
    entries on the object would draw a force as if it were a place, which
    reads as a real contact point and is not one. Both runners resolve the
    flag against the task before any drawing happens; this pins the shared
    predicate they resolve it with.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from oim.worlds.sim3d.run import _contact_consensus  # noqa: PLC0415

    assert _contact_consensus(
        SimpleNamespace(consensus="contact_point")
    )
    assert not _contact_consensus(SimpleNamespace(consensus="wrench"))
    # A task predating the key at all (2D, flat baselines) is not a
    # contact-point task and must not be treated as one.
    assert not _contact_consensus(SimpleNamespace())
