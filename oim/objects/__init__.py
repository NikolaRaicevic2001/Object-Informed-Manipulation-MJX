"""Analytic (simulator-free) object models for ADMM object-level subproblems.

A `ConsensusTask` describes its object-level subproblem by composing these
pieces -- geometry (`sdf`) and a dynamics + cost model (`planar_pushing`) --
rather than re-deriving them in the task file.
"""

from .contact import (
    CONTACT_ACTION_DIM,
    CONTACT_POINT_DIM,
    contact_action_to_wrench,
    contact_force_to_com_wrench,
    contact_frame,
    contact_point_to_wrench,
    estimate_contact_point,
    pack_contact_action,
    pack_contact_point,
    project_contact_action,
    project_contact_point,
    sample_contact_actions,
    sample_contact_points,
    unpack_contact_action,
    unpack_contact_point,
    wrench_to_contact_point,
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
    "CONTACT_POINT_DIM",
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
    "contact_point_to_wrench",
    "estimate_contact_point",
    "pack_contact_action",
    "pack_contact_point",
    "project_contact_action",
    "project_contact_point",
    "rotate",
    "sample_contact_actions",
    "sample_contact_points",
    "se2_distance_sq",
    "t_shape_footprint",
    "unpack_contact_action",
    "unpack_contact_point",
    "wrap_angle",
    "wrench_to_contact_point",
    "wrench_weights",
]
