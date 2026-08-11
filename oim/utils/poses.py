"""Per-task start and goal pose sets, loaded from `examples/poses/`.

A scene's MJCF fixes one start and one goal. These files give five of each,
so a task can be re-run under a different initial condition rather than
only a different sampler seed -- which redraws the noise but leaves the
problem identical.

Poses live beside the scripts that name them (`examples/poses/<task>.yaml`,
matching `examples/<task>.py`) rather than with the scene registry, so
changing where an object starts is a text edit next to the task it belongs
to. Nothing here validates clearance -- `tests/test_poses.py` re-derives it
for every pose in every file against the scene's own obstacles.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

from oim import ROOT

POSES_DIR = os.path.join(os.path.dirname(ROOT), "examples", "poses")

# What `--start`/`--goal` accept to mean "draw one".
RANDOM = "random"


@dataclass(frozen=True)
class PoseSet:
    """One task's start and goal options.

    Args:
        task: The `examples/` script stem the file was named for.
        starts: Start poses by key, world-frame SE(2) `[x, y, theta]`.
        goals: Goal poses by key, same convention.
    """

    task: str
    starts: Dict[str, np.ndarray]
    goals: Dict[str, np.ndarray]

    def keys(self, kind: str) -> Tuple[str, ...]:
        """The available keys for `"start"` or `"goal"`, in file order."""
        return tuple(self.starts if kind == "start" else self.goals)

    def get(self, kind: str, key: str) -> np.ndarray:
        """One pose by kind and key.

        Args:
            kind: `"start"` or `"goal"`.
            key: A key from `keys`.

        Returns:
            The pose, `[x, y, theta]`.

        Raises:
            KeyError: If the key is not in this set, naming what is.
        """
        table = self.starts if kind == "start" else self.goals
        if key not in table:
            raise KeyError(
                f"{self.task}: no {kind} {key!r} "
                f"(available: {', '.join(table)})"
            )
        return table[key]

    def select(
        self, kind: str, key: Optional[str], rng: np.random.Generator
    ) -> Tuple[str, np.ndarray]:
        """Resolve a `--start`/`--goal` value to a key and a pose.

        `None` or `"random"` draws one, so an unset flag varies the initial
        condition per run. The key comes back so the caller can record
        which was drawn -- a random run stays reproducible.

        Args:
            kind: `"start"` or `"goal"`.
            key: An explicit key, `"random"`, or `None` for random.
            rng: Where a random draw comes from.

        Returns:
            `(key, pose)`.
        """
        if key is None or key == RANDOM:
            options = self.keys(kind)
            key = str(options[int(rng.integers(len(options)))])
        return key, self.get(kind, key)


def poses_path(task: str) -> str:
    """Where a task's pose file would be, whether or not it exists."""
    return os.path.join(POSES_DIR, f"{task}.yaml")


def load_poses(task: str) -> Optional[PoseSet]:
    """Load one task's pose set, or `None` if it has no file.

    Returning `None` rather than raising is deliberate: a scene without a
    pose file keeps its MJCF's own start and goal, so adding these files
    one task at a time never breaks the rest.

    Args:
        task: The `examples/` script stem, e.g. `"shelf_gap"`.

    Returns:
        The pose set, or `None`.

    Raises:
        ValueError: If the file exists but is missing `starts`/`goals`, or
            a pose is not three numbers.
    """
    path = poses_path(task)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for kind in ("starts", "goals"):
        if kind not in raw or not raw[kind]:
            raise ValueError(f"{path}: missing or empty '{kind}'")
        table = {}
        for key, value in raw[kind].items():
            pose = np.asarray(value, dtype=float)
            if pose.shape != (3,):
                raise ValueError(
                    f"{path}: {kind}[{key!r}] must be [x, y, theta], "
                    f"got {value!r}"
                )
            table[str(key)] = pose
        out[kind] = table
    return PoseSet(task=task, starts=out["starts"], goals=out["goals"])
