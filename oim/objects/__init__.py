"""Analytic (simulator-free) object models for ADMM object-level subproblems.

A `ConsensusTask` describes its object-level subproblem by composing these
pieces -- geometry (`sdf`) and a dynamics + cost model (`planar_pushing`) --
rather than re-deriving them in the task file.
"""

from .contact import (
    CONTACT_ACTION_DIM,
    contact_action_to_wrench,
    contact_force_to_com_wrench,
    contact_frame,
    estimate_contact_point,
    pack_contact_action,
    project_contact_action,
    sample_contact_actions,
    unpack_contact_action,
)
from .planar_pushing import (
    PlanarPushingObject,
    c_shape_footprint,
    se2_distance_sq,
    t_shape_footprint,
    wrap_angle,
    wrench_weights,
)
from .sdf import Box, Capsule, Circle, ObstacleField, Polygon, Shape, rotate

__all__ = [
    "Box",
    "CONTACT_ACTION_DIM",
    "Capsule",
    "Circle",
    "ObstacleField",
    "PlanarPushingObject",
    "Polygon",
    "Shape",
    "c_shape_footprint",
    "contact_action_to_wrench",
    "contact_force_to_com_wrench",
    "contact_frame",
    "estimate_contact_point",
    "pack_contact_action",
    "project_contact_action",
    "rotate",
    "sample_contact_actions",
    "se2_distance_sq",
    "t_shape_footprint",
    "unpack_contact_action",
    "wrap_angle",
    "wrench_weights",
]
