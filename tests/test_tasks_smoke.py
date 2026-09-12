"""Every inherited demo task instantiates, steps and scores.

Nine files used to hold a copy of this: build the task on both rollout
backends, make its data, check the two cost functions' shape and sign.
They differed only in the constructor, the state the costs are read at,
and a handful of task-specific accessors -- which is what `_Case` holds.

These tasks are not the ADMM pushing work (`tests/test_pusht.py` and
`tests/test_admm.py` are); they are the single-optimizer demos this repo
inherited from Hydrax, and most are here because an algorithm test uses
one as a cheap fixture. The value of the suite is that a MuJoCo or MJX
upgrade cannot silently break a model, so it stays broad and shallow.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import mujoco
import pytest
from conftest import mjx_forward
from mujoco import mjx

from oim import ROOT
from oim.tasks.bugtrap import BugTrap
from oim.tasks.cart_pole import CartPole
from oim.tasks.crane import Crane
from oim.tasks.cube import CubeRotation
from oim.tasks.double_cart_pole import DoubleCartPole
from oim.tasks.humanoid_mocap import HumanoidMocap
from oim.tasks.humanoid_standup import HumanoidStandup
from oim.tasks.particle import Particle
from oim.tasks.pendulum import Pendulum
from oim.tasks.walker import Walker


@dataclass(frozen=True)
class _Case:
    """One task's smoke test.

    Attributes:
        make: Builds the task on the given rollout backend.
        setup: Places the state the costs are read at -- a mocap target or
            a joint configuration. Forward kinematics runs after it either
            way, so an accessor never reads a stale `site_xpos`.
        check: Task-specific assertions: sensor addresses, geometry, the
            accessors each task adds. Runs on the state after `setup`.
        positive: Costs are strictly positive at that state, not merely
            finite. The three underactuated toys are at their own goal in
            the default configuration, so theirs are only `>= 0`.
        upper: Optional cost ceiling (mocap tracking starts near its
            reference, so a cost above 1 means the reference is misread).
    """

    make: Callable[[str], Any]
    setup: Optional[Callable[[mjx.Data], mjx.Data]] = None
    check: Optional[Callable[[Any, mjx.Data], None]] = None
    positive: bool = True
    upper: Optional[float] = None


def _walker(task: Any, state: mjx.Data) -> None:
    assert task.torso_position_sensor >= 0
    assert task.torso_velocity_sensor >= 0
    assert task.torso_zaxis_sensor >= 0
    assert task._get_torso_height(state) > 0.0
    assert task._get_torso_velocity(state) == 0.0
    assert task._get_torso_deviation_from_upright(state) == 0.0


def _crane(task: Any, state: mjx.Data) -> None:
    assert task.payload_pos_sensor_adr >= 0
    assert task.payload_vel_sensor_adr >= 0
    pos = task._get_payload_position(state)
    assert pos.shape == (3,)
    assert not jnp.all(pos == 0.0)
    assert task._get_payload_velocity(state).shape == (3,)


def _particle(task: Any, state: mjx.Data) -> None:
    assert task.pointmass_id >= 0
    assert state.site_xpos.shape == (1, 3)
    assert not jnp.all(state.site_xpos == 0.0)


def _bugtrap(task: Any, state: mjx.Data) -> None:
    assert task.pointmass_id >= 0
    assert task._wall_pos.shape == (3, 2)
    assert task._wall_size.shape == (3, 2)


def _double_cart_pole(task: Any, state: mjx.Data) -> None:
    tip = state.site_xpos[task.tip_id]
    assert tip[0] != 0.0  # x, driven off centre by the joint angles below
    assert tip[1] == 0.0  # y, the mechanism is planar
    assert tip[2] > 0.0   # z, above the rail


def _cube(task: Any, state: mjx.Data) -> None:
    assert jnp.all(
        task._get_cube_position_err(state) == jnp.array([0.0, 0.0, 0.07])
    )
    orientation = task._get_cube_orientation_err(state)
    assert orientation.shape == (3,)
    assert jnp.allclose(orientation, jnp.array([-jnp.pi, 0.0, 0.0]))


def _standup(task: Any, state: mjx.Data) -> None:
    assert task.orientation_sensor_id >= 0
    assert task.torso_id >= 0
    assert task._get_torso_height(state) > 0.0
    assert task._get_torso_orientation(state).shape == (3,)


def _mocap(task: Any, state: mjx.Data) -> None:
    assert task.reference_qpos is not None


CASES = {
    "pendulum": _Case(Pendulum, positive=False),
    "cart_pole": _Case(CartPole, positive=False),
    "double_cart_pole": _Case(
        DoubleCartPole,
        # x, theta_1, theta_2 -- off the upright the tip check needs.
        setup=lambda s: s.replace(qpos=jnp.array([0.0, 0.1, 0.1])),
        check=_double_cart_pole,
        positive=False,
    ),
    "particle": _Case(
        Particle,
        setup=lambda s: s.replace(mocap_pos=jnp.array([[0.0, 0.1, 0.0]])),
        check=_particle,
    ),
    "bugtrap": _Case(
        BugTrap,
        setup=lambda s: s.replace(mocap_pos=jnp.array([[0.25, 0.0, 0.01]])),
        check=_bugtrap,
    ),
    "walker": _Case(Walker, check=_walker),
    "crane": _Case(
        Crane,
        setup=lambda s: s.replace(
            mocap_pos=jnp.array([[0.1, 0.1, 0.1]]),
            mocap_quat=jnp.array([[1.0, 0.0, 0.0, 0.0]]),
        ),
        check=_crane,
    ),
    "cube": _Case(
        CubeRotation,
        setup=lambda s: s.replace(mocap_quat=jnp.array([[0.0, 1.0, 0.0, 0.0]])),
        check=_cube,
        positive=False,
    ),
    "humanoid_standup": _Case(HumanoidStandup, check=_standup),
    "humanoid_mocap": _Case(HumanoidMocap, check=_mocap, upper=1.0),
}


@pytest.mark.parametrize("impl", ["jax", "warp"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_task(name: str, impl: str) -> None:
    """Build the task, place its state, and score it."""
    case = CASES[name]
    task = case.make(impl=impl)

    state = task.make_data()
    assert isinstance(state, mjx.Data)
    if case.setup is not None:
        state = case.setup(state)
    state = mjx_forward(task.model, state)

    if case.check is not None:
        case.check(task, state)

    ell = task.running_cost(state, jnp.zeros(task.model.nu))
    phi = task.terminal_cost(state)
    for cost in (ell, phi):
        assert cost.shape == ()
        assert cost >= 0.0
    if case.positive:
        assert ell > 0.0
        assert phi > 0.0
    if case.upper is not None:
        assert ell < case.upper
        assert phi < case.upper


# The two contact-rich scenes, straight off their MJCF: the tasks above
# read one state, this integrates a hundred under random torques, which is
# where a model that is subtly broken (bad mass, unstable solver settings)
# shows up as a NaN instead of a plausible number.
@pytest.mark.parametrize(
    "scene,dims",
    [
        ("cube", (16, 16 + 7, 16 + 6)),   # nu, nq (hand + floating cube), nv
        ("g1", None),                     # checked as nu + 6 == nv instead
    ],
)
def test_mjx_model_integrates(scene: str, dims: Optional[tuple]) -> None:
    """A hundred random-input MJX steps stay finite."""
    mj_model = mujoco.MjModel.from_xml_path(f"{ROOT}/models/{scene}/scene.xml")
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)
    assert isinstance(model, mjx.Model)
    assert isinstance(data, mjx.Data)
    if dims is None:
        assert mj_model.nu + 6 == mj_model.nv
    else:
        assert (mj_model.nu, mj_model.nq, mj_model.nv) == dims

    nu = mj_model.nu

    @jax.jit
    def step(data: mjx.Data, rng: jax.Array) -> tuple:
        """One forward step under a random input."""
        rng, sample_rng = jax.random.split(rng)
        u = jax.random.uniform(sample_rng, (nu,), minval=-1.0, maxval=1.0)
        return mjx.step(model, data.replace(ctrl=u)), rng

    rng = jax.random.key(0)
    for _ in range(100):
        data, rng = step(data, rng)

    assert not jnp.any(jnp.isnan(data.qpos))
    assert not jnp.any(jnp.isnan(data.qvel))
