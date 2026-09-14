#!/usr/bin/env python3
"""Publish the static TF for a camera defined in a MuJoCo scene.

The detector reports a tag in the camera's frame, and `cho_object_pose` asks TF
for `base_link <- tag_<id>`. Something has to connect the two, and on the
physical rig that is the camera mount in the URDF. In the MuJoCo cell the
camera lives in the MJCF instead, so this reads it from there rather than
repeating the numbers: a camera moved in the scene moves here too, with nothing
to keep in step by hand.

Two conversions happen, and both are why this is a script and not a
`static_transform_publisher` line in a launch file:

* MuJoCo's camera frame looks down its own -z with +y up (the OpenGL
  convention). AprilTag reports poses in the ROS optical frame: +z forward,
  +x right, +y down. So the published rotation is the scene's rotated 180
  degrees about the camera x axis, which is what makes a tag come out in front
  of the camera instead of behind it.
* MuJoCo's world origin is the robot's `base_link` here (fr5_ag95.xml puts the
  arm's first geom directly in worldbody), so the camera's scene pose IS its
  pose in the robot frame. That is asserted, not assumed: --base names the
  frame and it is published as the parent.

    python3 TAMP/tamp/scripts/perception/mujoco_camera_tf.py \
        --scene <cho_description_fr5>/xml/scene_ag95_sdl.xml --camera sdl_cam
"""

import argparse
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node


def quat_to_matrix(w, x, y, z):
    """Rotation matrix from a w-first quaternion (MJCF's convention)."""
    n = np.linalg.norm([w, x, y, z])
    if n == 0.0:
        raise ValueError('zero-length quaternion')
    w, x, y, z = np.array([w, x, y, z]) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """w-first quaternion from a rotation matrix."""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


def read_camera(scene_path, camera_name):
    """(position, MuJoCo rotation) of *camera_name* in the scene's world frame.

    Only a camera written directly in `<worldbody>` is supported, which is how
    the SDL cell writes it. A camera nested in a body would need the body chain
    composed, and silently ignoring that would publish a pose that is wrong by
    exactly the parent's transform.
    """
    root = ET.parse(scene_path).getroot()
    for worldbody in root.iter('worldbody'):
        for child in worldbody:
            if child.tag == 'camera' and child.get('name') == camera_name:
                pos = np.array([float(v) for v in child.get('pos', '0 0 0').split()])
                quat = [float(v) for v in child.get('quat', '1 0 0 0').split()]
                return pos, quat_to_matrix(*quat), child
        for nested in worldbody.iter('camera'):
            if nested.get('name') == camera_name:
                raise SystemExit(
                    "camera '%s' is nested inside a body; this publisher only "
                    'handles cameras written directly in <worldbody>, because a '
                    'nested one needs its parent chain composed' % camera_name)
    raise SystemExit("no camera named '%s' in %s" % (camera_name, scene_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scene', required=True, help='path to the MJCF scene')
    parser.add_argument('--camera', default='sdl_cam')
    parser.add_argument('--base', default='base_link',
                        help="robot frame the scene's world origin coincides with")
    parser.add_argument('--frame', default=None,
                        help='published camera frame; defaults to <camera>_frame, '
                             'which is what mujoco_ros2_control stamps its images with')
    parser.add_argument('--print-only', action='store_true')
    args = parser.parse_args(argv)

    position, rotation, element = read_camera(args.scene, args.camera)
    # MuJoCo camera -> ROS optical: 180 degrees about x.
    optical = rotation @ np.diag([1.0, -1.0, -1.0])
    quat = matrix_to_quat(optical)
    frame = args.frame or ('%s_frame' % args.camera)

    print('scene      : %s' % args.scene)
    print('camera     : %s (%s px, fovy %s deg)'
          % (args.camera, element.get('resolution'), element.get('fovy')))
    print('%s -> %s' % (args.base, frame))
    print('  position : %s' % np.array2string(position, precision=6))
    print('  optical z (forward) : %s' % np.array2string(optical[:, 2], precision=6))
    print('  optical y (down)    : %s' % np.array2string(optical[:, 1], precision=6))
    print('  quat (w x y z)      : %s' % np.array2string(quat, precision=9))
    if args.print_only:
        return 0

    rclpy.init()
    node = Node('mujoco_camera_tf')
    broadcaster = tf2_ros.StaticTransformBroadcaster(node)

    transform = TransformStamped()
    transform.header.stamp = node.get_clock().now().to_msg()
    transform.header.frame_id = args.base
    transform.child_frame_id = frame
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    transform.transform.rotation.w = float(quat[0])
    transform.transform.rotation.x = float(quat[1])
    transform.transform.rotation.y = float(quat[2])
    transform.transform.rotation.z = float(quat[3])
    broadcaster.sendTransform(transform)
    node.get_logger().info('publishing %s -> %s' % (args.base, frame))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
