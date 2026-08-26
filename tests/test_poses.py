"""Every pose in `examples/poses/` must be a pose the task can actually run.

A pose file is hand-editable, which is the point of it being YAML -- so the
clearance and reach that made the generated set safe are re-derived here
rather than trusted. A pose that grazes an obstacle fails in this file, in
a second, instead of a hundred control steps into a GPU run.

Clearance is checked both ways: the object's boundary against every
obstacle, and every obstacle's boundary against the object polygon. The
one-way check alone would miss an obstacle sitting entirely inside the T's
notch or the C's opening.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from oim.objects.sdf import Polygon, rotate
from oim.utils.poses import RANDOM, load_poses, poses_path
from oim.utils.scenes import SCENES

# The generator required 4.5 cm; the test demands 3 cm so that float
# differences never flip a pose that was deliberately placed at the edge,
# while still catching a genuinely colliding hand-edit.
MIN_CLEARANCE = 0.03

# Arm reach. 0.842 m was measured by sweeping the xArm6's joint limits
# (models/xarm6_pusht_clutter/verify_reach.py); the usable band is smaller,
# since the stick has to get *around* the object to push it.
REACH_BAND = (0.20, 0.86)

# Scenes with a pose file. A scene without one keeps its MJCF's own start
# and goal, which `load_poses` returning None already covers.
POSE_SCENES = sorted(s for s in SCENES if load_poses(s) is not None)


def _world_polygon(spec: object, pose: np.ndarray) -> np.ndarray:
    """The object footprint's vertices at `pose`, in world coordinates."""
    verts = jnp.asarray(spec.footprint().vertices)
    return np.asarray(pose[:2]) + np.asarray(rotate(float(pose[2]), verts))


def _edge_points(verts: np.ndarray, per_edge: int = 24) -> np.ndarray:
    """Points along a closed polygon's edges."""
    out = []
    for i, a in enumerate(verts):
        b = verts[(i + 1) % len(verts)]
        for t in np.linspace(0.0, 1.0, per_edge, endpoint=False):
            out.append(a + t * (b - a))
    return np.array(out)


def clearance(spec: object, pose: np.ndarray) -> float:
    """Smallest signed distance between the object at `pose` and obstacles.

    Args:
        spec: The scene.
        pose: World-frame SE(2) `[x, y, theta]`.

    Returns:
        Metres of separation; negative means overlapping. `inf` when the
        scene has no obstacles.
    """
    field = spec.obstacles
    if not field.shapes:
        return float("inf")

    verts = _world_polygon(spec, pose)
    obj_to_obs = float(
        np.min(np.asarray(field.sdf(jnp.asarray(_edge_points(verts)))))
    )

    poly = Polygon(jnp.asarray(verts))
    angles = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    obs_to_obj = float("inf")
    for shape in field.shapes:
        # `center` is on the Shape base class; `_center` is Circle/Box only.
        probe = jnp.asarray(np.asarray(shape.center) + ring)
        boundary = np.asarray(shape.project_to_boundary(probe))
        d = float(np.min(np.asarray(poly.sdf(jnp.asarray(boundary)))))
        obs_to_obj = min(obs_to_obj, d)
    return min(obj_to_obs, obs_to_obj)


def _reach(spec: object, pose: np.ndarray) -> float:
    base = np.asarray(spec.xarm6_base_pos)
    return float(np.linalg.norm(np.asarray(pose[:2]) - base))


def _all_poses(task: str):
    """`(kind, key, pose)` for every pose in one file."""
    poses = load_poses(task)
    for kind in ("start", "goal"):
        for key in poses.keys(kind):
            yield kind, key, poses.get(kind, key)


# ----------------------------------------------------------------------
# The guarantee the files are supposed to carry
# ----------------------------------------------------------------------


@pytest.mark.parametrize("task", POSE_SCENES)
def test_every_pose_clears_every_obstacle(task: str) -> None:
    """No listed pose puts the object inside, or grazing, an obstacle."""
    spec = SCENES[task]
    for kind, key, pose in _all_poses(task):
        d = clearance(spec, pose)
        assert d >= MIN_CLEARANCE, (
            f"{poses_path(task)}: {kind} {key!r} at "
            f"{np.round(pose, 4).tolist()} "
            f"clears the nearest obstacle by {d * 100:.1f} cm, "
            f"below the {MIN_CLEARANCE * 100:.0f} cm minimum"
        )


@pytest.mark.parametrize("task", POSE_SCENES)
def test_every_pose_is_within_reach(task: str) -> None:
    """A pose the arm cannot get to is not a task, it is a failed run."""
    spec = SCENES[task]
    if spec.xarm6_base_pos is None:
        pytest.skip("no arm in this scene")
    lo, hi = REACH_BAND
    for kind, key, pose in _all_poses(task):
        d = _reach(spec, pose)
        assert lo <= d <= hi, (
            f"{poses_path(task)}: {kind} {key!r} is {d:.3f} m from the arm "
            f"base, outside the usable band {lo}-{hi} m"
        )


@pytest.mark.parametrize("task", POSE_SCENES)
def test_pose_one_is_the_scene_default(task: str) -> None:
    """Pose "1" reproduces the run these files did not exist for.

    Without this, adding pose files would silently change every default
    run and make previously recorded results incomparable.
    """
    spec = SCENES[task]
    poses = load_poses(task)
    np.testing.assert_allclose(
        # `object_start`, not the origin: the block's MJCF *anchor* is the
        # origin, but the pose it starts from lives in the scene's keyframe.
        poses.get("start", "1"),
        np.asarray(spec.object_start, dtype=float),
        atol=1e-9,
        err_msg=f"{poses_path(task)}: start '1' must be "
                f"SCENES[{task!r}].object_start",
    )
    np.testing.assert_allclose(
        poses.get("goal", "1"), np.asarray(spec.goal, dtype=float), atol=1e-4,
        err_msg=f"{poses_path(task)}: goal '1' must be SCENES[{task!r}].goal",
    )


@pytest.mark.parametrize("task", POSE_SCENES)
def test_five_of_each_and_all_distinct(task: str) -> None:
    """Five options per side, none a duplicate of another."""
    poses = load_poses(task)
    for kind in ("start", "goal"):
        keys = poses.keys(kind)
        assert len(keys) == 5, f"{task}: {len(keys)} {kind}s, expected 5"
        stacked = np.stack([poses.get(kind, k) for k in keys])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = np.linalg.norm(stacked[i, :2] - stacked[j, :2])
                assert d > 1e-3, f"{task}: {kind}s {keys[i]}/{keys[j]} coincide"


@pytest.mark.parametrize("task", POSE_SCENES)
def test_variants_stay_in_the_nominal_neighbourhood(task: str) -> None:
    """All five are the same task jittered, not five different tasks.

    A pose set spread across the table would change task difficulty
    between variants, which is exactly what makes them useless as a
    robustness comparison.
    """
    poses = load_poses(task)
    for kind in ("start", "goal"):
        nominal = poses.get(kind, "1")
        for key in poses.keys(kind):
            pose = poses.get(kind, key)
            d = np.linalg.norm(pose[:2] - nominal[:2])
            assert d <= 0.15, (
                f"{task}: {kind} {key!r} is {d:.3f} m from the nominal "
                f"{kind}, outside the intended neighbourhood"
            )
            dtheta = abs(float(np.arctan2(
                np.sin(pose[2] - nominal[2]), np.cos(pose[2] - nominal[2])
            )))
            assert dtheta <= 0.6, (
                f"{task}: {kind} {key!r} is {dtheta:.2f} rad from nominal"
            )


@pytest.mark.parametrize("task", POSE_SCENES)
def test_every_start_goal_pair_is_a_real_push(task: str) -> None:
    """No pairing collapses into a task already solved at t=0."""
    poses = load_poses(task)
    for s in poses.keys("start"):
        for g in poses.keys("goal"):
            d = np.linalg.norm(
                poses.get("start", s)[:2] - poses.get("goal", g)[:2]
            )
            assert d > 0.2, (
                f"{task}: start {s} and goal {g} are {d:.3f} m apart"
            )


# ----------------------------------------------------------------------
# The loader
# ----------------------------------------------------------------------


def test_every_3d_scene_has_a_pose_file() -> None:
    """A scene with a script should have poses to pick from."""
    assert set(POSE_SCENES) == set(SCENES), (
        f"scenes without a pose file: {sorted(set(SCENES) - set(POSE_SCENES))}"
    )


def test_missing_file_is_not_an_error() -> None:
    """A task with no pose file keeps its MJCF start/goal."""
    assert load_poses("definitely_not_a_task") is None


def test_random_selection_reports_which_it_drew() -> None:
    """A random draw has to be recordable, or the run is irreproducible."""
    poses = load_poses("shelf_gap")
    rng = np.random.default_rng(0)
    for _ in range(20):
        key, pose = poses.select("goal", None, rng)
        assert key in poses.keys("goal")
        np.testing.assert_allclose(pose, poses.get("goal", key))
    key, _ = poses.select("start", RANDOM, rng)
    assert key in poses.keys("start")


def test_explicit_selection_is_exact() -> None:
    """`--goal 3` is goal 3, not a draw seeded by it."""
    poses = load_poses("shelf_gap")
    rng = np.random.default_rng(0)
    for key in poses.keys("goal"):
        got_key, pose = poses.select("goal", key, rng)
        assert got_key == key
        np.testing.assert_allclose(pose, poses.get("goal", key))


def test_unknown_key_names_the_available_ones() -> None:
    """A typo says what to type instead."""
    poses = load_poses("shelf_gap")
    with pytest.raises(KeyError, match="available: 1, 2, 3, 4, 5"):
        poses.get("start", "42")
