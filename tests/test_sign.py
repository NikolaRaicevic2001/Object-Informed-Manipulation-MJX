"""`icra_sign_real`: one letter pushed into its slot, the rest standing."""

import numpy as np
import pytest

from oim.objects import sign
from oim.objects.library import PUSH_OBJECTS
from oim.tasks.pusht import PushT
from oim.utils.scenes import SCENES

SCENE = "icra_sign_real"
LETTERS = ["I_block", "C_block", "R_block", "A_block"]


def _task(**kw) -> PushT:
    return PushT(clutter=True, robot="xarm6", env=SCENE, planning_dt=0.05, **kw)


def _standing(task: PushT):
    m = task.mj_model
    return sorted(m.body(i).name[len("obs_"):] for i in range(m.nbody)
                  if m.body(i).name.startswith("obs_"))


def test_unset_object_pushes_the_default_letter() -> None:
    """`--object` left at the scene default resolves to the C, not an error."""
    task = _task()
    assert task.push_object_name == "C_block"
    assert task.push_object is PUSH_OBJECTS["C_block"]
    assert _standing(task) == ["A_block", "I_block", "R_block"]


@pytest.mark.parametrize("letter", LETTERS)
def test_each_letter_is_pushed_into_its_own_slot(letter: str) -> None:
    """Goal is the letter's slot; every other letter stands in its slot."""
    spec = SCENES[SCENE]
    task = _task(push_object=letter)
    np.testing.assert_allclose(np.asarray(task.object_model.goal),
                               spec.letter_slots[letter], atol=1e-9)
    assert _standing(task) == sorted(n for n in LETTERS if n != letter)
    # The goal marker moved with the goal.
    m = task.mj_model
    np.testing.assert_allclose(m.body("goal").pos[:2], spec.letter_slots[letter][:2])
    # The block carries the pushed letter's boxes, nothing else's.
    n_block = sum(1 for i in range(m.ngeom)
                  if m.geom_bodyid[i] == m.body("block").id and m.geom_contype[i])
    assert n_block == len(PUSH_OBJECTS[letter].boxes)


@pytest.mark.parametrize("letter", LETTERS)
def test_standing_letters_are_their_own_boxes_at_their_slots(letter: str) -> None:
    """Planner field and compiled model both hold the standing letters'
    boxes, placed at the slots -- the same boxes those letters are pushed
    with when they are the object."""
    spec = SCENES[SCENE]
    task = _task(push_object=letter)
    m = task.mj_model
    standing = [n for n in LETTERS if n != letter]
    # Planner: base keep-out + one Box per standing-letter box.
    n_boxes = sum(len(PUSH_OBJECTS[n].boxes) for n in standing)
    assert len(task.object_model.obstacles.shapes) == 1 + n_boxes
    # Model: each obs body sits at its slot with that letter's geom count.
    for n in standing:
        body = m.body(f"obs_{n}")
        x, y, _ = spec.letter_slots[n]
        np.testing.assert_allclose(body.pos[:2], [x, y])
        assert float(body.pos[2]) == pytest.approx(PUSH_OBJECTS[n].half_height)
        geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body.id]
        assert len(geoms) == len(PUSH_OBJECTS[n].boxes)
        assert all(m.geom_contype[i] != 0 for i in geoms)
    # And PushT's own obstacle-geom set found them all.
    assert len(task.obstacle_geoms) >= n_boxes


def test_a_letter_with_no_slot_is_refused() -> None:
    with pytest.raises(ValueError, match="no slot in this sign"):
        _task(push_object="T_large_block")


def test_standing_letters_do_not_overlap_each_other_or_the_start() -> None:
    """The layout is physically placeable: no two standing letters share
    ground, and the shared start is clear of every slot."""
    spec = SCENES[SCENE]
    from oim.objects.sdf import Box
    # Sample each letter's boxes and test against every other letter's SDF.
    rng = np.random.default_rng(0)
    for a in LETTERS:
        pts = []
        for centre, half, yaw in sign._placed_boxes(PUSH_OBJECTS[a], spec.letter_slots[a]):
            c, s = np.cos(yaw), np.sin(yaw)
            local = rng.uniform(-1, 1, (64, 2)) * np.asarray(half)
            pts.append(np.asarray(centre) + local @ np.array([[c, s], [-s, c]]))
        pts = np.concatenate(pts)
        for b in LETTERS:
            if b == a:
                continue
            for centre, half, yaw in sign._placed_boxes(PUSH_OBJECTS[b], spec.letter_slots[b]):
                box = Box(center=list(centre), half_extents=list(half), angle=float(yaw))
                assert float(np.min(np.asarray(box.sdf(pts)))) > 0.0, f"{a} overlaps {b}"
    # Start footprint (the largest letter, unrotated) clear of every slot.
    sx, sy, syaw = spec.object_start
    for n in LETTERS:
        for centre, half, yaw in sign._placed_boxes(PUSH_OBJECTS[n], spec.letter_slots[n]):
            box = Box(center=list(centre), half_extents=list(half), angle=float(yaw))
            corners = np.array([[sx + dx, sy + dy] for dx in (-0.075, 0.075) for dy in (-0.1, 0.1)])
            assert float(np.min(np.asarray(box.sdf(corners)))) > 0.0, f"start overlaps {n}"
