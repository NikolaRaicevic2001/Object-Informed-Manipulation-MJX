"""Trajectory overlay for any sampling-based controller.

Every algorithm here plans the same way: draw a population of candidate
rollouts, then reduce it to one trajectory to execute. This draws both --
the candidates as thin lines, the chosen trajectory as a thick one -- so a
frame shows what the controller considered next to what it settled on.

A flat controller (MPPI, predictive sampling, CEM, CBO) has one such
population, in robot space. ADMM has two, one per block, and they live in
different spaces: the object block's is a sequence of SE(2) object poses,
the robot block's is its end-effector's path. They are told apart by color,
not by line style, so the sample/chosen distinction reads the same way for
both:

    object block   cool: pale cyan samples, strong blue chosen
    robot block    warm: pale amber samples, strong orange chosen

`BlockTrace` is what the overlay draws -- one per block, already in world
xyz. `traces_for` builds them from whatever a runner has (lifting the
object's SE(2) poses to a drawing height), so the runners never touch
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


@dataclass
class BlockTrace:
    """What one block contributes to a frame, in world coordinates.

    Args:
        scheme: Its two colors.
        chosen: The trajectory it settled on, (H, 3) or (H+1, 3). `None`
            draws no chosen path.
        samples: The candidates it considered, (n, H, 3) or (n, H+1, 3).
            `None` draws no candidates.
    """

    scheme: ColorScheme
    chosen: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None


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


def traces_for(
    robot_chosen: Optional[np.ndarray] = None,
    robot_samples: Optional[np.ndarray] = None,
    object_chosen: Optional[np.ndarray] = None,
    object_samples: Optional[np.ndarray] = None,
    object_height: float = OBJECT_PLAN_HEIGHT,
) -> List[BlockTrace]:
    """Pack a runner's arrays into the blocks the overlay draws.

    The object block's arrays are SE(2) and get lifted; the robot block's
    are already world xyz. A flat controller passes only the robot's and
    gets a one-block list back -- the same call, one block short.

    Args:
        robot_chosen: The robot block's chosen end-effector path, (H, 3).
        robot_samples: Its candidates' end-effector paths, (n, H, 3).
        object_chosen: The object block's chosen poses, (H, 3) SE(2).
        object_samples: Its candidate pose sequences, (n, H, 3) SE(2).
        object_height: Drawing height for the lifted object paths.

    Returns:
        One `BlockTrace` per block that has anything to draw, object first
        so the robot's thicker chosen path is not hidden under it.
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
            max_blocks: Blocks one `draw` may be given, for reserving geoms.
                Two: ADMM's object and robot. A flat controller uses one.
        """
        self.horizon = horizon
        self.sample_width = sample_width
        self.chosen_width = chosen_width
        self.alpha_near = alpha_near
        self.alpha_far = alpha_far
        self.sample_alpha = sample_alpha
        self.max_samples = max_samples
        self.max_blocks = max_blocks

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
