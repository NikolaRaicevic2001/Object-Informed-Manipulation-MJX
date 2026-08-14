#!/usr/bin/env python3
"""Publish the push-T scene (obstacles, goal, block start) as RViz markers.

Lets you place the real obstacles / goal to match the model: open RViz, set the
Fixed Frame to `--frame`, Add -> MarkerArray on `/scene_markers`, and move the
real objects until they sit on the drawn markers.

Geometry is read straight from the scene MJCF (mujoco only, no jax), so the
markers stay in sync with the model. Run in the ROS env:

    python oim/worlds/real3d/scripts/publish_scene_markers.py \
        --scene-xml oim/models/xarm6_pusht_clutter_2/scene.xml \
        --frame xarm_device --start 0.381,0.343,0
"""

import argparse
import os

import mujoco
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker, MarkerArray

# name prefix -> (RGBA, namespace)
STYLE = {
    "obs": ((0.6, 0.6, 0.6, 0.9), "obstacles"),
    "goal_": ((0.0, 1.0, 0.0, 0.4), "goal"),
    "block_": ((0.2, 0.45, 0.85, 0.7), "block_start"),
}


class SceneMarkers(Node):
    def __init__(self, xml: str, frame: str, start) -> None:
        super().__init__("scene_markers")
        self._frame = frame
        # Absolute path: MuJoCo resolves the included meshes relative to the
        # XML's own directory, which only works reliably from an abspath.
        self._model = mujoco.MjModel.from_xml_path(os.path.abspath(xml))
        self._data = mujoco.MjData(self._model)
        if start is not None:  # put the block geoms at the start SE(2)
            adr = [self._model.joint(n).qposadr[0] for n in ("T_x", "T_y", "T_z")]
            self._data.qpos[adr] = start
        mujoco.mj_forward(self._model, self._data)
        self._pub = self.create_publisher(MarkerArray, "scene_markers", 1)
        self.create_timer(0.5, self._publish)  # latched-ish, re-send at 2 Hz

    def _publish(self) -> None:
        arr = MarkerArray()
        for gid in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            style = next((s for p, s in STYLE.items() if name.startswith(p)), None)
            if style is None:
                continue
            rgba, ns = style
            m = Marker()
            m.header.frame_id = self._frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = ns, gid
            m.action = Marker.ADD
            gtype = self._model.geom_type[gid]
            size = self._model.geom_size[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                m.type = Marker.CUBE
                m.scale.x, m.scale.y, m.scale.z = (2 * size[:3]).tolist()
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                m.type = Marker.SPHERE
                m.scale.x = m.scale.y = m.scale.z = float(2 * size[0])
            else:
                continue
            p = self._data.geom_xpos[gid]
            q = Rotation.from_matrix(self._data.geom_xmat[gid].reshape(3, 3)).as_quat()
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p.tolist()
            m.pose.orientation.x, m.pose.orientation.y = float(q[0]), float(q[1])
            m.pose.orientation.z, m.pose.orientation.w = float(q[2]), float(q[3])
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            arr.markers.append(m)
        self._pub.publish(arr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-xml", required=True, help="path to the scene.xml")
    p.add_argument("--frame", default="xarm_device", help="RViz fixed frame")
    p.add_argument("--start", default=None,
                   help="block start 'x,y,yaw' to draw (world/base frame)")
    args = p.parse_args()
    start = ([float(v) for v in args.start.split(",")] if args.start else None)

    rclpy.init()
    node = SceneMarkers(args.scene_xml, args.frame, start)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
