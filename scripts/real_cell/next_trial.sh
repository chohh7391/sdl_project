#!/usr/bin/env bash
# Tell a repeated pour (plan_remote.sh --repeat N) that the cell is ready for
# its next trial: the poured vessel refilled, the receiver emptied or not, the
# operator clear of the arm. The tree holds the arm still until this arrives.
set +u
source /opt/ros/humble/setup.bash
ros2 topic pub --once /real_cell/trial_go std_msgs/msg/Empty
