"""Tests for the trajectory overlay shared by every sampling-based method.

What matters is that one drawing path serves them all: a flat controller
contributes one block, ADMM two, and the sample/chosen distinction reads
identically in both. These check the geometry the overlay writes into an
`MjvScene` -- no rendering, no GPU, no controller.
"""

import mujoco
import numpy as np
import pytest

from oim.sim3d.plan_overlay import (
    OBJECT_PLAN_HEIGHT,
    OBJECT_SCHEME,
    ROBOT_SCHEME,
    BlockTrace,
    PlanOverlay,
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
