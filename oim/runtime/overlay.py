"""Trajectory overlay for any sampling-based controller.

Every algorithm here plans the same way: draw a population of candidate
rollouts, then reduce it to one trajectory to execute. This draws both --
the candidates as thin lines, the chosen trajectory as a thick one -- so a
frame shows what the controller considered next to what it settled on.

A flat controller (MPPI, predictive sampling, CEM, CBO) has one such
population, in robot space. ADMM has two blocks, and draws *three* paths,
because the two blocks each predict a trajectory for the same object:

    object block, object space   cool: pale cyan samples, strong blue chosen
    robot block, object space    magenta chosen -- what the robot's controls
                                 would actually do to the object
    robot block, robot space     warm: pale amber samples, strong orange
                                 chosen -- where the end-effector goes

The middle one is the diagnostic the other two cannot give. The consensus
is an agreement about a quantity that is hard to read as a number but easy
to read as motion: the blue and magenta paths are the *same object* under
the two blocks' plans, so where they coincide the blocks agree and where
they separate is exactly what the primal residual is measuring -- literally
so under a pose consensus variable, where the residual is that separation.
Without it a frame shows what the object planner wants and where the tip
goes, but not whether the robot can deliver what was asked.

They are told apart by color, not by line style, so the sample/chosen
distinction reads the same way for all three.

`BlockTrace` is what the overlay draws -- one per path, already in world
xyz. `traces_for` builds them from whatever a runner has (lifting the
object-space SE(2) poses to a drawing height), so the runners never touch
colors and the overlay never has to know which algorithm produced what.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np

# Drawing height for the object block's paths. Its plans are SE(2), so they
# carry no z of their own; this clears the block's top face. The robot's
# paths are drawn at their real height -- they are genuine 3D positions.
OBJECT_PLAN_HEIGHT = 0.055

# The robot block's object-space path sits 4 mm above the object block's.
# Both are predictions of the same object's pose, so at consensus they are
# the *same* line -- coincident geometry, which z-fights and flickers, and
# which would also make "agreeing" and "one of them is missing" look
# identical. Offsetting keeps both readable, and the vertical gap closing
# is what agreement looks like.
ROBOT_OBJECT_PLAN_HEIGHT = OBJECT_PLAN_HEIGHT + 0.004


@dataclass(frozen=True)
class ColorScheme:
    """One block's two colors: its candidates, and the one it chose.

    Args:
        sample: RGB for a candidate rollout.
        chosen: RGB for the trajectory the block settled on -- the same
            hue family, deeper, so the pair reads as one block.
        name: What this block is, for error messages.
    """

    sample: Tuple[float, float, float]
    chosen: Tuple[float, float, float]
    name: str


OBJECT_SCHEME = ColorScheme(
    sample=(0.40, 0.82, 1.00),  # pale cyan
    chosen=(0.00, 0.30, 0.95),  # strong blue
    name="object",
)
ROBOT_SCHEME = ColorScheme(
    sample=(1.00, 0.78, 0.35),  # pale amber
    chosen=(0.95, 0.35, 0.00),  # strong orange
    name="robot",
)
# The robot block's prediction for the *object*, against the object block's
# own. Magenta because it has to be legible next to the blue path it is
# meant to be compared with, and every other hue is already spoken for by
# the scene rather than by this overlay: the block is blue, the obstacles
# orange (matching the source repo's rendering), the goal marker green, and
# the table white over a blue-grey floor.
ROBOT_OBJECT_SCHEME = ColorScheme(
    sample=(0.85, 0.60, 1.00),  # pale violet
    chosen=(0.72, 0.00, 0.85),  # strong magenta
    name="robot-object",
)
# The agreed contact point under `consensus_variable: contact_point`, drawn
# as dots on the object rather than as a path -- it is a *place on the
# object*, not a trajectory through the world. Red: the one hue neither the
# scene nor the other three paths use, and it has to read against the blue
# plan it sits on top of.
CONTACT_SCHEME = ColorScheme(
    sample=(1.00, 0.55, 0.55),  # pale red
    chosen=(0.90, 0.05, 0.05),  # strong red
    name="contact",
)

# Radius of a contact dot [m]. Small enough to read as a point on a ~10 cm
# block, large enough to see at the default camera distance.
CONTACT_POINT_RADIUS = 0.006

# Fallback drawing height for a contact dot [m]. Callers pass the task's
# `tip_target_z` instead wherever they have it: the dot marks a place the
# tip is supposed to *be*, so drawing it anywhere else would show the
# consensus and the tip-height cost disagreeing when they do not. Only
# scenes with no `tip_target_z` fall back to this, the usual block
# mid-height.
CONTACT_POINT_HEIGHT = 0.03


@dataclass
class BlockTrace:
    """What one block contributes to a frame, in world coordinates.

    Args:
        scheme: Its two colors.
        chosen: The trajectory it settled on, (H, 3) or (H+1, 3). `None`
            draws no chosen path.
        samples: The candidates it considered, (n, H, 3) or (n, H+1, 3).
            `None` draws no candidates.
        points: World points, (n, 3), drawn as individual spheres instead
            of connected into a path. For a quantity that is a *place*
            rather than a route -- consecutive contact points are not a
            trajectory, and joining them would draw a line through the
            object's interior whenever the contact jumps faces.
    """

    scheme: ColorScheme
    chosen: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None


def lift_se2(
    poses: np.ndarray, height: float = OBJECT_PLAN_HEIGHT
) -> np.ndarray:
    """An SE(2) pose sequence's (x, y), at a fixed drawing height.

    Args:
        poses: SE(2) poses, (n, >=2) -- only (x, y) is read.
        height: World z to place them at.

    Returns:
        World points, (n, 3).
    """
    xy = np.asarray(poses)[:, :2]
    return np.concatenate([xy, np.full((len(xy), 1), height)], axis=1)


def contact_points_world(
    poses: np.ndarray,
    contacts: np.ndarray,
    height: float = CONTACT_POINT_HEIGHT,
) -> np.ndarray:
    """Body-frame contact points, placed on the object at each planned pose.

    The consensus value under `consensus_variable: contact_point` is
    `[p_x, p_y, lambda]` with p in the object's *body* frame -- that is the
    whole point of the parameterization, since one fixed p tracks the same
    material point as the object turns. Drawing it therefore needs the pose
    it belongs to, one per horizon step.

    Args:
        poses: The object's planned SE(2) poses, (H, >=3).
        contacts: Consensus values, (H, >=2) -- only p is read; lambda is
            not drawn (a dot has no magnitude to show).
        height: World z to place the dots at. Pass the task's
            `tip_target_z` -- the contact point is a place the tip is
            meant to reach, and `w_z_tip` holds the tip at exactly that
            height, so drawing the dot anywhere else (the object plan's
            5.5 cm, say) would show the consensus asking for one height
            and the cost pulling to another when they in fact agree.

    Returns:
        World points, (n, 3), with n = min(len(poses), len(contacts)).
    """
    poses = np.asarray(poses)
    contacts = np.asarray(contacts)
    n = min(len(poses), len(contacts))
    poses, p_body = poses[:n], contacts[:n, :2]
    c, sn = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    xy = poses[:, :2] + np.stack(
        [c * p_body[:, 0] - sn * p_body[:, 1],
         sn * p_body[:, 0] + c * p_body[:, 1]],
        axis=-1,
    )
    return np.concatenate([xy, np.full((n, 1), height)], axis=1)


def traces_for(
    robot_chosen: Optional[np.ndarray] = None,
    robot_samples: Optional[np.ndarray] = None,
    object_chosen: Optional[np.ndarray] = None,
    object_samples: Optional[np.ndarray] = None,
    robot_object_chosen: Optional[np.ndarray] = None,
    robot_object_samples: Optional[np.ndarray] = None,
    contact_points: Optional[np.ndarray] = None,
    object_height: float = OBJECT_PLAN_HEIGHT,
    robot_object_height: float = ROBOT_OBJECT_PLAN_HEIGHT,
) -> List[BlockTrace]:
    """Pack a runner's arrays into the paths the overlay draws.

    Object-space arrays are SE(2) and get lifted; the robot's own path is
    already world xyz. A flat controller passes only the robot's and gets a
    one-path list back -- the same call, two paths short.

    Args:
        robot_chosen: The robot block's chosen end-effector path, (H, 3).
        robot_samples: Its candidates' end-effector paths, (n, H, 3).
        object_chosen: The object block's chosen poses, (H, 3) SE(2).
        object_samples: Its candidate pose sequences, (n, H, 3) SE(2).
        robot_object_chosen: The object trajectory the *robot* block's
            chosen controls would produce, (H, 3) SE(2) -- the same object
            as `object_chosen`, predicted by the other block. Their
            separation is the consensus disagreement, made spatial.
        robot_object_samples: Its per-sample counterpart, (n, H, 3) SE(2).
            Currently always `None`: the rollout's own `consensus_values`
            are wrenches or contact points, neither of which is an object
            pose, so there is nothing per-sample to draw.
        contact_points: Already-world contact dots, (n, 3), from
            `contact_points_world`. Only meaningful under
            `consensus_variable: contact_point`; `None` everywhere else.
        object_height: Drawing height for the object block's lifted paths.
        robot_object_height: Drawing height for the robot block's
            object-space paths -- see `ROBOT_OBJECT_PLAN_HEIGHT`.

    Returns:
        One `BlockTrace` per path that has anything to draw, in the order
        the overlay should layer them: object-space paths first, so the
        robot's thicker end-effector path is not hidden under them.
    """
    traces: List[BlockTrace] = []
    if object_chosen is not None or object_samples is not None:
        traces.append(
            BlockTrace(
                scheme=OBJECT_SCHEME,
                chosen=(
                    None
                    if object_chosen is None
                    else lift_se2(object_chosen, object_height)
                ),
                samples=(
                    None
                    if object_samples is None
                    else np.stack(
                        [
                            lift_se2(s, object_height)
                            for s in np.asarray(object_samples)
                        ]
                    )
                ),
            )
        )
    if robot_object_chosen is not None or robot_object_samples is not None:
        traces.append(
            BlockTrace(
                scheme=ROBOT_OBJECT_SCHEME,
                chosen=(
                    None
                    if robot_object_chosen is None
                    else lift_se2(robot_object_chosen, robot_object_height)
                ),
                samples=(
                    None
                    if robot_object_samples is None
                    else np.stack(
                        [
                            lift_se2(s, robot_object_height)
                            for s in np.asarray(robot_object_samples)
                        ]
                    )
                ),
            )
        )
    if contact_points is not None and len(contact_points):
        traces.append(
            BlockTrace(
                scheme=CONTACT_SCHEME,
                points=np.asarray(contact_points),
            )
        )
    if robot_chosen is not None or robot_samples is not None:
        traces.append(
            BlockTrace(
                scheme=ROBOT_SCHEME,
                chosen=(
                    None if robot_chosen is None else np.asarray(robot_chosen)
                ),
                samples=(
                    None if robot_samples is None else np.asarray(robot_samples)
                ),
            )
        )
    return traces


class PlanOverlay:
    """Draws blocks' candidate and chosen trajectories into an `MjvScene`.

    Holds no scene of its own, because the two scenes it has to serve behave
    differently. The passive viewer's `user_scn` persists between frames, so
    the overlay owns a fixed slot in it. An offscreen `mujoco.Renderer`'s
    scene is rebuilt from the model by every `update_scene` call, which
    discards anything added previously -- so there the overlay must be
    appended again for each frame. `draw` covers both: pass a fixed `base`
    for a persistent scene, or omit it to append to a freshly rebuilt one.
    """

    def __init__(
        self,
        horizon: int,
        sample_width: float = 1.5,
        chosen_width: float = 4.0,
        alpha_near: float = 0.95,
        alpha_far: float = 0.35,
        sample_alpha: float = 0.55,
        max_samples: int = 16,
        max_blocks: int = 2,
        point_radius: float = CONTACT_POINT_RADIUS,
    ) -> None:
        """Configure the overlay's geometry.

        Args:
            horizon: Number of predicted poses per plan, H.
            sample_width: Line width of a candidate rollout, in pixels --
                thin, since there are many.
            chosen_width: Line width of the chosen trajectory. Thicker than
                a sample is the whole point: it reads as the decision even
                where it runs through the middle of the cloud.
            alpha_near: Chosen path's opacity at the start of the horizon.
            alpha_far: Its opacity at the end, so time direction reads off
                the image without needing an animation.
            sample_alpha: Flat opacity for candidates. Constant rather than
                faded: `max_samples` overlapping fades turn the cloud into
                a solid wash, and the fade is what makes the *chosen* path
                legible against it.
            max_samples: Draw at most this many candidates per block,
                evenly spaced through the population -- drawing all of them
                would be unreadable and slow.
            max_blocks: Paths one `draw` may be given, for reserving geoms.
                Three for ADMM: the object block's plan, the robot block's
                plan for the same object, and the end-effector's own path.
                A flat controller uses one. A contact-dot trace counts as
                one of these too, so ADMM drawing contact points needs
                four.
            point_radius: Radius of a `BlockTrace.points` sphere [m].
        """
        self.horizon = horizon
        self.sample_width = sample_width
        self.chosen_width = chosen_width
        self.alpha_near = alpha_near
        self.alpha_far = alpha_far
        self.sample_alpha = sample_alpha
        self.max_samples = max_samples
        self.max_blocks = max_blocks
        self.point_radius = point_radius

        # One path: a segment between each pair of consecutive points. Some
        # paths are H points long, some H+1 (a rollout's own final state,
        # appended where it's cheap to); H segments covers either.
        self.per_path = horizon

    @property
    def geom_count(self) -> int:
        """Scene geoms one `draw` call consumes at most, for reserving space."""
        return self.max_blocks * (self.max_samples + 1) * self.per_path

    def _fade(self, k: int, n: int, color: Sequence[float]) -> np.ndarray:
        """Color, faded from `alpha_near` to `alpha_far` over `0..n-1`."""
        frac = k / max(n - 1, 1)
        alpha = self.alpha_near + frac * (self.alpha_far - self.alpha_near)
        return np.array([*color, alpha], dtype=np.float64)

    def _ramp(self, k: int, n: int, scheme: ColorScheme) -> np.ndarray:
        """Color ramped `chosen` -> `sample` *and* faded, over `0..n-1`.

        Dots are separated in space, not connected, so alpha alone reads as
        "some are dimmer" rather than as an ordering. Moving the hue too
        makes first-to-last unambiguous in a still frame: the step to act
        on now is deep and opaque, the end of the horizon pale and faint.
        """
        frac = k / max(n - 1, 1)
        rgb = (1.0 - frac) * np.asarray(scheme.chosen) + frac * np.asarray(
            scheme.sample
        )
        alpha = self.alpha_near + frac * (self.alpha_far - self.alpha_near)
        return np.array([*rgb, alpha], dtype=np.float64)

    @staticmethod
    def _init_line(scene: mujoco.MjvScene, i: int) -> None:
        mujoco.mjv_initGeom(
            scene.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.zeros(4),
        )

    def _draw_path(
        self,
        scene: mujoco.MjvScene,
        start: int,
        points: np.ndarray,
        color: Sequence[float],
        width: float,
        alpha: Optional[float] = None,
    ) -> int:
        """One path, as connected line segments. Returns geoms consumed.

        Args:
            scene: Where to write.
            start: First geom index.
            points: World points, (n, 3).
            color: RGB.
            width: Line width in pixels.
            alpha: Flat opacity, or `None` to fade along the path.

        Returns:
            The number of geoms written.
        """
        n = len(points)
        i = start
        for k in range(n - 1):
            self._init_line(scene, i)
            scene.geoms[i].rgba[:] = (
                self._fade(k, n, color)
                if alpha is None
                else np.array([*color, alpha], dtype=np.float64)
            )
            mujoco.mjv_connector(
                scene.geoms[i],
                mujoco.mjtGeom.mjGEOM_LINE,
                width,
                points[k],
                points[k + 1],
            )
            i += 1
        return i - start

    def _draw_points(
        self,
        scene: mujoco.MjvScene,
        start: int,
        points: Optional[np.ndarray],
        scheme: ColorScheme,
    ) -> int:
        """A block's points, as spheres faded along the horizon; geoms used.

        Ramped in hue as well as opacity along the horizon -- see
        `_ramp`. The dot for the step the robot has to act on *now* is the
        deep, opaque one; the planned relocations trail off pale behind
        it.
        """
        if points is None or len(points) == 0:
            return 0
        points = np.asarray(points)
        n = min(len(points), self.per_path)
        r = self.point_radius
        for k in range(n):
            mujoco.mjv_initGeom(
                scene.geoms[start + k],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.full(3, r),
                pos=points[k, :3].astype(np.float64),
                mat=np.eye(3).flatten(),
                rgba=self._ramp(k, n, scheme),
            )
        return n

    def _draw_samples(
        self,
        scene: mujoco.MjvScene,
        start: int,
        samples: Optional[np.ndarray],
        scheme: ColorScheme,
    ) -> int:
        """Up to `max_samples` of a block's candidates; geoms used."""
        if samples is None or len(samples) == 0:
            return 0
        samples = np.asarray(samples)
        n_show = min(len(samples), self.max_samples)
        idx = np.linspace(0, len(samples) - 1, n_show).astype(int)
        i = start
        for s in idx:
            i += self._draw_path(
                scene,
                i,
                samples[s, :, :3],
                scheme.sample,
                self.sample_width,
                alpha=self.sample_alpha,
            )
        return i - start

    def draw(
        self,
        scene: mujoco.MjvScene,
        traces: Sequence[BlockTrace],
        base: Optional[int] = None,
    ) -> None:
        """Draw every block's candidates and chosen trajectory.

        Candidates first, across all blocks, then the chosen paths -- so a
        thick chosen line is never overdrawn by another block's cloud.

        Args:
            scene: The scene to write into -- a viewer's `user_scn` or a
                `mujoco.Renderer`'s `.scene`.
            traces: One `BlockTrace` per block, from `traces_for`.
            base: First geom index to write. Pass a fixed value for a
                persistent scene, so the overlay keeps its own slot and
                leaves earlier geoms (e.g. `show_traces`) alone. Omit for a
                scene `update_scene` has just rebuilt, to append after the
                model's own geoms.

        Raises:
            ValueError: If more blocks are passed than this overlay
                reserved geoms for, or a path is longer than its horizon.
            RuntimeError: If the scene has too few free geoms.
        """
        if len(traces) > self.max_blocks:
            raise ValueError(
                f"overlay reserved geoms for {self.max_blocks} blocks, "
                f"got {len(traces)}"
            )
        for trace in traces:
            too_long = (
                trace.chosen is not None
                and len(trace.chosen) > self.horizon + 1
            )
            if too_long:
                raise ValueError(
                    f"{trace.scheme.name} chosen path has "
                    f"{len(trace.chosen)} points, overlay was built for "
                    f"{self.horizon}"
                )
        start = scene.ngeom if base is None else base
        if start + self.geom_count > scene.maxgeom:
            raise RuntimeError(
                f"plan overlay needs up to {self.geom_count} scene geoms, "
                f"only {scene.maxgeom - start} free"
            )

        i = start
        for trace in traces:
            i += self._draw_samples(scene, i, trace.samples, trace.scheme)
        # Dots before the chosen paths, so a plan line never hides the
        # contact it belongs to.
        for trace in traces:
            i += self._draw_points(scene, i, trace.points, trace.scheme)
        for trace in traces:
            if trace.chosen is not None:
                i += self._draw_path(
                    scene,
                    i,
                    np.asarray(trace.chosen)[:, :3],
                    trace.scheme.chosen,
                    self.chosen_width,
                )

        scene.ngeom = max(scene.ngeom, i)
