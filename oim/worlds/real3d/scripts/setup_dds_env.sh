#!/usr/bin/env bash
# Source this (`. setup_dds_env.sh`) in EVERY ROS terminal on BOTH machines
# (perception laptop + planner desktop) so their ROS 2 topics + TF connect over
# the LAN. This documents the recurring cross-machine setup so it isn't
# tribal knowledge -- but it does only the necessary part.
#
# On a shared subnet, CycloneDDS discovers peers over multicast on its own;
# you only need these env vars matched on both hosts:
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}   # SAME number on both machines
export ROS_LOCALHOST_ONLY=0                 # allow off-host traffic

echo "[dds] RMW=$RMW_IMPLEMENTATION  ROS_DOMAIN_ID=$ROS_DOMAIN_ID  LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo "[dds] check: 'ros2 topic list' should show the other host's topics"
echo "      (e.g. /object_pose, /joint_states), and 'ros2 topic echo /tf' should"
echo "      carry fp_object_pose."

# OPTIONAL fallback -- only if the above is not enough (multiple NICs so DDS
# picks the wrong interface, or the router blocks multicast). Then edit
# ../config/cyclonedds.xml (interface name + peer IP) and uncomment:
#   HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   export CYCLONEDDS_URI="file://${HERE}/../config/cyclonedds.xml"
