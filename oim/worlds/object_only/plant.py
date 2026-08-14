"""What *executes* the object block's wrench, between control steps.

`oim.worlds.object_only.build` is a receding-horizon loop around
`oim.algs.admm.ObjectSubproblem`. Which plant it drives is the only thing
that changes here: the sampler, the costs, the projection, the warm start
and the ADMM object block itself are the same objects either way, so a
difference between two runs is a difference in *dynamics* and cannot be a
difference in planner.

The planner always predicts with `PushT.object_dynamics` -- the quasi-static
limit surface, paper eq. 5. `AnalyticPlant` executes with it too, which is
the closed loop this world ships with: no model error at all, an upper bound
on what the object block can do. `MujocoPlant` executes the same wrench in
the simulator instead, so the loop is planning with the limit surface and
being graded by MuJoCo, and the per-step gap between the two is a direct
measurement of the modelling assumption.

WHAT THE TWO ACTUALLY DISAGREE ABOUT. The MJCF gives the block three
joints -- two slides and a hinge -- whose `frictionloss` is set to exactly
the friction-cone limit `PlanarPushingObject` derives, 7.848 N and
0.47088 N*m, so the numbers on both sides are the same by construction.
Measured, they are not the same *model*:

1. THE THRESHOLD HAS A DIFFERENT SHAPE, and this dominates everything else.
   `PlanarPushingObject.step` measures friction with the coupled norm
   `||w / D^-1||` -- an ellipsoid. MuJoCo's `frictionloss` is three
   independent per-DoF Coulomb elements, `|w_i|` against `limit_i`
   channel by channel -- a box. A wrench at the limit in two channels at
   once is 1.41x over the ellipsoid and exactly *at* the box, so the two
   disagree completely there. Displacement under a wrench held for 1 s,
   shelf_gap:

       w / limit        ||w/D^-1||   analytic     mujoco
       [1.0, 0,   0]        1.00       0.000 m    0.006 m
       [1.0, 1.0, 0]        1.41       0.414 m    0.007 m   <- total
       [1.5, 0,   0]        1.50       0.500 m    0.984 m      disagreement
       [2.0, 2.0, 0]        2.83       1.828 m    1.290 m

   Where every channel individually clears its own limit the two agree to
   within the inertia term below. Where the *norm* clears but no single
   channel does, the analytic object slides and the simulated one does
   not move. That case is not a corner: the object action box is the unit
   cube, so the optimizer's preferred actions are its corners -- exactly
   the wrenches on which the two models disagree. An ellipsoid is the
   right shape for a block on a table (total friction force is bounded,
   not each axis separately), so the obvious fix is to give the simulator a
   coupled friction cone rather than to change eq. 5.

   THAT FIX WORKS, AND COSTS SOMETHING ELSE. `MujocoPlant(friction=...)`
   implements three laws. `"cone"` applies the coupled ellipsoid through
   `qfrc_applied` with the associated flow rule; `"wrench"` applies eq. 5's
   own force balance `-w/s`; `"box"` is MuJoCo's untouched `frictionloss`.

   Open loop the cone does exactly what it promises. The [1, 1, 0] row above
   goes from 0.007 m (stuck) to 0.506 m against the analytic 0.414 m, and
   every at- or sub-threshold wrench sticks at exactly 0.000 m where
   `frictionloss` used to creep -- so difference 5 below is eliminated too,
   not just this one.

   It also removes the `wrench_fraction` asymmetry. The shipped 2.0 for
   MuJoCo exists *only* to compensate for the box: at 1.0 no channel may
   exceed its own friction, so the net force is ~0. Under the cone, 1.0
   works -- the same value the analytic plant uses -- so the two plants stop
   differing in action scale as well. Comparing the two at equal
   `wrench_fraction` is therefore a mistake: it tests the cone with a
   correction built for the box.

   Each at the fraction it actually needs, 250 steps, 64 samples, seed 0:

       scene              box@2.0                 cone@1.0
                          pred_pos  pred_th  hit  pred_pos  pred_th  hit
       open_table           0.0352   0.2756  yes    0.0270   0.2887   no
       shelf_gap            0.0275   0.1808  yes    0.0137   0.2110  yes
       single_obstacle      0.0277   0.1570  yes    0.0147   0.1445  yes
       ycb_clutter          0.0310   0.3612  yes    0.0113   0.1202   no
       icra_sign            0.0395   0.3730  yes    0.0108   0.1643  yes
       mean                 0.0322   0.2695  5/5    0.0155   0.1857  3/5

   So the cone is the better *instrument*: mean per-step model error drops
   52% in position and 31% in orientation, which is what this plant exists
   to measure. It is the worse *controller*: 3/5 scenes reach the goal
   against 5/5. Raising its action budget does not recover that -- 1.2, 1.5
   and 2.0 are all monotonically worse than 1.0 on both counts -- so it is
   not an authority problem. `open_table`, the 180-degree flip, never
   converges under the cone at any fraction, which fits: it is the most
   rotation-dominated scene, and the flow rule spends one friction budget
   across all three DoFs, so a block sliding fast in translation has little
   left to resist rotation. That is correct limit-surface physics and it is
   exactly what `D`, being diagonal, does not do.

   `"wrench"` marks the other boundary. Porting eq. 5's force balance into a
   second-order integrator diverges outright (final position error 1.2-1.6 m,
   the block coasting away), because that balance is only self-consistent
   with no momentum to dissipate: below threshold it cancels the applied
   wrench exactly and leaves whatever velocity the block already had. No
   friction law fixes inertia from the simulator's side.

   `"box"` remains the default -- least faithful shape, but it converges on
   every scene and it is what the 3D world actually simulates. Use
   `--friction cone --wrench-fraction 1.0` when the question is how good
   eq. 5 is, rather than how well the block does.

2. INERTIA. Eq. 5 is quasi-static: velocity is proportional to the *excess*
   wrench, reached instantly, and gone the instant the wrench is. MuJoCo
   integrates a 2 kg body, so it accelerates into a push and coasts out of
   one. Since `step` began subtracting friction the two agree on *shape*
   -- the ratio is a constant 9.7 across every over-threshold wrench,
   where the old gating form ran from 569 down to 13.7 -- so what is left
   is a single scale factor, the quasi-static velocity scale in `D`
   against real acceleration. Over one 0.05 s step from rest at 1.5x the
   limit that is 0.025 m predicted against 0.003 m realized; held 1 s,
   0.50 m against 0.98 m as the block gets up to speed.

3. THE COUPLING TERMS. `D` is diagonal, so the analytic model cannot turn
   a pure force into a rotation. The block's CoM is offset from its joint
   origin (1 kg crossbar at y = 0, 1 kg stem at y = -0.075), so MuJoCo
   can and does: a pure +x force for 1 s yields -0.021 rad.

4. CONTACT. The analytic model knows obstacles only as a soft cost hinge on
   its footprint and will plan through a shelf if the cost is worth it.
   MuJoCo stops the block dead.

5. THE DEADZONE IS SOFT. `frictionloss` is a solver constraint with
   compliance, not the analytic model's hard `jnp.where`, so a
   sub-threshold wrench creeps instead of sticking -- ~5 mm over 1 s at
   0.9x the limit, against exactly 0 for the analytic model. Small beside
   the rest, and it is why `tests/test_object_plant.py` bounds the held
   case at 1 cm rather than at zero.

The wrench itself needs no conversion: it is written to `qfrc_applied` on
those three DoFs, and for two world-axis slides and a hinge about z that
generalized force *is* the world-frame planar wrench of eq. 5.
"""

from typing import Any, Dict, Optional, Protocol, Sequence

import jax
import mujoco
import numpy as np

from oim.runtime.mjcf import execution_model
from oim.tasks.pusht import PushT

# Bodies that are not the object or its scenery. An object-only study has
# no robot, but the scene it borrows still contains one -- so the arm is
# taken out of collision rather than left standing in the workspace as an
# obstacle the analytic side cannot see. Prefix match: the xArm6 contributes
# `xarm6_link*` and `xarm6_stick`.
_ROBOT_BODY_PREFIXES = ("xarm6", "pusher")

# Below this the block counts as at rest, so friction opposes the *applied
# wrench* rather than the twist. In newtons: `||c * v||` is the wrench that
# would produce the twist `v` quasi-statically, so this is 0.01 N against a
# cone of 7.848 N -- about 1.3 mm/s. Small enough that a stuck block is
# genuinely stuck, large enough that the branch does not flip every substep
# on solver noise.
_REST_WRENCH = 1e-2

# The support surface, taken out of collision for the same reason but with
# a sharper justification: the block's support friction is ALREADY modelled,
# as the `frictionloss` on its two slides and its hinge, set in the MJCF to
# exactly `mu*m*g` and `c*r*mu*m*g` precisely so the simulated block obeys
# the analytic limit surface (see the comment in `tee.xml`). Leaving the
# block resting on the table counts that friction a second time, through the
# contact.
#
# Measured on shelf_gap, force needed to break the block loose:
#
#     analytic limit surface           7.85 N   (= mu*m*g, by construction)
#     MuJoCo, support excluded         7.87 N   <- agrees to 0.3%
#     MuJoCo, block resting on table  11.16 N   <- 1.42x, double counted
#
# The last figure is the same with gravity on and off, so this is the
# contact constraint at zero penetration rather than Coulomb friction from
# the block's weight -- turning gravity off does not avoid it, and only
# taking the surface out of collision does.
_SUPPORT_GEOM_NAMES = ("table", "floor", "ground")


class ObjectPlant(Protocol):
    """Whatever advances the object's SE(2) pose under a wrench.

    Stateful by contract, because the interesting implementation is: a
    quasi-static model is a pure function of `(pose, wrench)`, but a
    second-order one carries velocity between control steps and would be
    misrepresented by a `step(pose, wrench)` signature that lets the caller
    silently reset it.
    """

    name: str

    def reset(self, pose: Sequence[float]) -> np.ndarray:
        """Place the object at `pose` at rest; return the pose realized."""
        ...

    def step(self, wrench: np.ndarray) -> np.ndarray:
        """Hold `wrench` for one control step; return the new SE(2) pose."""
        ...


class AnalyticPlant:
    """The limit surface as its own plant: the model executes itself.

    The plant this world ships with, kept as an explicit object so the
    MuJoCo comparison is a swap rather than a branch, and so the
    prediction-error column has a case that must read exactly zero.
    """

    name = "analytic"

    def __init__(self, task: PushT, jit: bool = True) -> None:
        """Wrap a task's `object_dynamics`.

        Args:
            task: The task whose object model to step.
            jit: Compile the step. False steps through the limit-surface
                dynamics in a debugger.
        """
        self._step = jax.jit(task.object_dynamics) if jit else (
            task.object_dynamics
        )
        self._pose = np.zeros(3)

    def reset(self, pose: Sequence[float]) -> np.ndarray:
        """Set the pose. Exact -- there is no state besides it."""
        self._pose = np.asarray(pose, dtype=float)
        return self._pose

    def step(self, wrench: np.ndarray) -> np.ndarray:
        """One forward-Euler step of eq. 5."""
        self._pose = np.asarray(self._step(self._pose, wrench), dtype=float)
        return self._pose


class MujocoPlant:
    """MuJoCo as the plant, driven by the same wrench the planner chose.

    The wrench goes to `qfrc_applied` on the block's two slides and hinge
    and is held there for the whole control step, which is the zero-order
    hold the object block's `object_spline: zero` already assumes. The
    simulator then substeps at `world3d.exec_timestep`, so the comparison is
    not against the same coarse integration the planner used.

    Gravity is switched off. Nothing in the object's own dynamics needs it
    -- the block's three joints are planar and its friction is
    `frictionloss`, a constant, not a normal-force product -- while leaving
    it on would drop the borrowed scene's unactuated arm onto the table
    during an experiment that has no robot in it.
    """

    name = "mujoco"

    def __init__(
        self,
        task: PushT,
        robot: str,
        cfg: Dict[str, Any],
        *,
        control_dt: float,
        start: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
        keep_robot: bool = False,
        keep_support: bool = False,
        friction: str = "box",
    ) -> None:
        """Build an execution model of the task's scene, object only.

        Args:
            task: The task whose scene and block to simulate.
            robot: Embodiment, selecting the scene variant's start pose.
            cfg: The config's `world3d` block, for the execution timestep
                and solver iteration counts.
            control_dt: Seconds one control step covers -- the planner's
                `planning_dt`. Substepped at `exec_timestep`.
            start: Object start pose, or None for the MJCF keyframe's.
            goal: Goal pose for the marker, or None for the MJCF's.
            keep_robot: Leave the borrowed scene's arm in collision. Off by
                default: an object-only run has no robot, so the arm parked
                at its home pose is an obstacle the analytic side does not
                model and the comparison would charge to MuJoCo's account.
            keep_support: Leave the block resting on the table. Off by
                default, and the more consequential of the two -- it is
                what double-counts the support friction and moves the
                measured breakaway from 7.87 N to 11.16 N. See
                `_SUPPORT_GEOM_NAMES`. On, to reproduce what the 3D runs
                actually simulate rather than what the planner assumes.
            friction: Which shape the simulated support friction has.
                `"box"` (default) is MuJoCo's own three independent per-DoF
                `frictionloss` elements. `"cone"` and `"wrench"` replace
                them with coupled models applied through `qfrc_applied`.

                Box is the default despite being the least faithful *shape*,
                because it is measurably the closest to eq. 5 in closed
                loop. See the module docstring: the coupled cone fixes the
                threshold exactly and still loses, which is the finding.

        Raises:
            ValueError: If `control_dt` is not a whole number of execution
                timesteps. Rounding it silently would put the two plants on
                different control rates while reporting one.
            ValueError: If `friction` is not `"cone"` or `"box"`.
        """
        if friction not in ("cone", "box", "wrench"):
            raise ValueError(
                f"unknown friction {friction!r}; expected 'cone', 'box' or "
                "'wrench'"
            )
        self.mj_model, self.mj_data = execution_model(
            task, robot, cfg, start, goal
        )
        if not keep_robot:
            _disable_collisions(self.mj_model, _ROBOT_BODY_PREFIXES)
        if not keep_support:
            _disable_collisions(self.mj_model, _SUPPORT_GEOM_NAMES, geom=True)
        # See the class docstring: nothing the block does depends on it.
        self.mj_model.opt.gravity[:] = 0.0

        exec_dt = float(self.mj_model.opt.timestep)
        self.substeps = int(round(control_dt / exec_dt))
        if self.substeps < 1 or abs(
            self.substeps * exec_dt - control_dt
        ) > 1e-9:
            raise ValueError(
                f"control_dt {control_dt} is not a whole multiple of "
                f"exec_timestep {exec_dt}"
            )

        self._qpos_adr = np.asarray(task.block_qpos_indices, dtype=int)
        self._dof_adr = np.array(
            [self.mj_model.joint(j).dofadr[0] for j in ("T_x", "T_y", "T_z")],
            dtype=int,
        )
        # The friction-cone limit, `D^-1` -- the same three numbers the MJCF
        # puts in `frictionloss` and `PlanarPushingObject` derives.
        self.friction = friction
        self._limit = np.asarray(task.object_model.wrench_limit, dtype=float)
        if friction in ("cone", "wrench"):
            # Or it would be counted twice: once per DoF by the solver and
            # once, coupled, by `_friction_wrench`.
            self.mj_model.dof_frictionloss[self._dof_adr] = 0.0
        self.recorder: Optional[Any] = None

    def attach_recorder(self, recorder: Any) -> None:
        """Capture an mp4 frame every substep, from the plant's own data.

        The plant is the only thing here that owns a `mujoco.MjData`, so
        video has to be driven from it rather than from the run loop --
        which is also what puts frames at the *physics* rate instead of the
        control rate, so playback is real time. See
        `oim.runtime.video.OffscreenRecorder`, which this shares with the 3D
        runners rather than reimplementing.

        Args:
            recorder: An `OffscreenRecorder`, or None to stop capturing.
        """
        self.recorder = recorder

    def reset(self, pose: Sequence[float]) -> np.ndarray:
        """Teleport the block to `pose` at rest, leaving the scene alone."""
        self.mj_data.qpos[self._qpos_adr] = np.asarray(pose, dtype=float)
        self.mj_data.qvel[:] = 0.0
        self.mj_data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.pose()

    def _friction_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """Coupled ellipsoidal Coulomb friction, opposing motion.

        The limit surface bounds the friction wrench by the *coupled* norm
        `||w_f / c|| <= 1`, where `c` is the friction-cone limit. Two
        regimes, split on whether the block is already moving:

        **Sliding.** The associated flow rule makes the twist normal to the
        limit surface, `v ∝ w_f / c**2`, so the friction wrench on the
        boundary opposing a twist `v` is `-c**2 * v / ||c * v||`. That
        satisfies `||w_f / c|| = 1` exactly.

        **At rest.** Friction opposes the applied wrench, saturating at the
        cone: `-w / max(s, 1)` with `s = ||w / c||`. Below the threshold
        that is `-w`, so the net is zero and the block sticks *hard* -- the
        analytic model's deadzone, which MuJoCo's compliant `frictionloss`
        only approximates. Above it the net is `w (1 - 1/s)`, which is
        exactly the excess term eq. 5 drives the object with. So the two
        models now agree on the instant of breakaway by construction, and
        what remains between them is inertia alone.

        `||c * v||` has units of newtons -- it is the wrench that would
        produce this twist quasi-statically -- so `_REST_WRENCH` is a
        threshold in the same units as the cone itself, not a raw speed.

        Args:
            wrench: The applied world-frame planar wrench.

        Returns:
            The friction wrench to add to it.
        """
        s = float(np.linalg.norm(wrench / self._limit))
        if self.friction == "wrench":
            # Eq. 5's own force balance: friction anti-parallel to the
            # *applied* wrench, so the net is w (1 - 1/s) at every instant.
            return -wrench / max(s, 1.0)
        v = self.mj_data.qvel[self._dof_adr]
        moving = float(np.linalg.norm(self._limit * v))
        if moving > _REST_WRENCH:
            return -(self._limit**2) * v / moving
        return -wrench / max(s, 1.0)

    def step(self, wrench: np.ndarray) -> np.ndarray:
        """Hold `wrench` on the block's DoFs for one control step."""
        wrench = np.asarray(wrench, dtype=float)
        for _ in range(self.substeps):
            # Recomputed per substep, not per control step: the friction
            # direction follows the twist, which changes as the block
            # accelerates. Held constant it would keep pushing along the
            # velocity the step started with.
            applied = wrench
            if self.friction != "box":
                applied = wrench + self._friction_wrench(wrench)
            self.mj_data.qfrc_applied[self._dof_adr] = applied
            mujoco.mj_step(self.mj_model, self.mj_data)
            if self.recorder is not None:
                self.recorder.capture(self.mj_data)
        return self.pose()

    def pose(self) -> np.ndarray:
        """The block's SE(2) pose.

        `qpos` *is* the world pose here: the block's joints are two slides
        along world x/y and a hinge about world z, anchored at the body's
        declared position -- the same invariant `PushT._block_pose` reads
        back on the planning side.
        """
        return np.array(self.mj_data.qpos[self._qpos_adr], dtype=float)

    def twist(self) -> np.ndarray:
        """The block's SE(2) velocity, which the analytic plant cannot have.

        Not used by the control loop -- it is the quantity that explains the
        divergence, since eq. 5 has no state here at all.
        """
        return np.array(self.mj_data.qvel[self._dof_adr], dtype=float)


def _disable_collisions(
    mj_model: mujoco.MjModel,
    names: Sequence[str],
    geom: bool = False,
) -> None:
    """Take matching geoms out of collision, in place.

    Zeroes `contype`/`conaffinity` rather than deleting geoms, so everything
    still renders and every id in the model stays put -- ids that `PushT`
    has already cached against this scene.

    Args:
        mj_model: The execution model to edit (a copy, never the task's).
        names: Name prefixes to match.
        geom: Match the geom's own name rather than its body's. The support
            surface is a bare worldbody geom and so has no body name of its
            own; the robot's links are bodies.
    """
    for geom_id in range(mj_model.ngeom):
        name = (
            mj_model.geom(geom_id).name
            if geom
            else mj_model.body(mj_model.geom_bodyid[geom_id]).name
        )
        if name.startswith(tuple(names)):
            mj_model.geom_contype[geom_id] = 0
            mj_model.geom_conaffinity[geom_id] = 0


def build_plant(
    kind: str,
    task: PushT,
    robot: str,
    cfg: Dict[str, Any],
    *,
    control_dt: float,
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
    jit: bool = True,
    friction: str = "box",
) -> ObjectPlant:
    """An `ObjectPlant` by name.

    Args:
        kind: `"analytic"` or `"mujoco"`.
        task: The task supplying the object model and the scene.
        robot: Embodiment, selecting the scene variant.
        cfg: The config's `world3d` block.
        control_dt: Seconds per control step.
        start: Object start pose, or None for the scene's own.
        goal: Goal pose, or None for the scene's own.
        jit: Compile the analytic step; ignored by MuJoCo, which is not
            traced.
        friction: `"cone"` or `"box"`; MuJoCo only. See `MujocoPlant`.

    Returns:
        The plant.

    Raises:
        ValueError: If `kind` is not a known plant.
    """
    if kind == "analytic":
        return AnalyticPlant(task, jit=jit)
    if kind == "mujoco":
        return MujocoPlant(
            task,
            robot,
            cfg,
            control_dt=control_dt,
            start=start,
            goal=goal,
            friction=friction,
        )
    raise ValueError(f"unknown plant '{kind}' (expected analytic or mujoco)")
