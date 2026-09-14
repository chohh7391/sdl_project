#!/usr/bin/env python3
"""Measure the MuJoCo cell's perception against the scene's own geometry.

The cell exists to exercise the path the physical rig will use, and the only
way that is worth anything is if the answer it produces is checked against
something independent. Here that is the MJCF itself: the scene says where the
vessels and their tag plates are, so this compares

    detector   : TF base_link <- tag_<id>      against the plate in the scene
    end to end : /perception/object_pose/<name> against the vessel in the scene

and, because the object pose is the tag pose plus a tag-frame offset, it also
prints the offset that WOULD make the published pose exact. That number is how
`content/configs/perception/mujoco_cell_objects.yaml` is set: the tag's axes in
the base frame depend on how the renderer maps the texture onto the plate and
on the detector's own convention, and measuring beats assuming either.

    python3 TAMP/tamp/scripts/perception/check_object_pose.py --scene <scene.xml>
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

#: object name -> (scene geom for the vessel, scene geom for its plate, tag id)
DEFAULT_OBJECTS = {
    'beaker': ('beaker', 'beaker_tag', 0),
    'flask': ('flask', 'flask_tag', 1),
}


def scene_geoms(scene_path):
    root = ET.parse(scene_path).getroot()
    out = {}
    for geom in root.iter('geom'):
        name = geom.get('name')
        if name:
            out[name] = np.array([float(v) for v in geom.get('pos', '0 0 0').split()])
    return out


def quat_matrix(x, y, z, w):
    """Rotation matrix from an xyzw quaternion (geometry_msgs order)."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Checker(Node):

    def __init__(self, objects):
        super().__init__('check_object_pose')
        self.objects = objects
        self.poses = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        for name in objects:
            self.create_subscription(
                PoseStamped, '/perception/object_pose/%s' % name,
                lambda msg, key=name: self.poses.__setitem__(key, msg), 10)

    def tag_pose(self, tag_id):
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tag_%d' % tag_id, rclpy.time.Time())
        except Exception as error:                      # noqa: BLE001 - reported, not raised
            return None, str(error)
        p, q = t.transform.translation, t.transform.rotation
        return (np.array([p.x, p.y, p.z]), quat_matrix(q.x, q.y, q.z, q.w)), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scene', required=True)
    parser.add_argument('--timeout', type=float, default=30.0)
    args = parser.parse_args(argv)

    geoms = scene_geoms(args.scene)
    missing = [g for names in DEFAULT_OBJECTS.values() for g in names[:2] if g not in geoms]
    if missing:
        print('scene has no geom named %s' % missing, file=sys.stderr)
        return 2

    rclpy.init()
    node = Checker(DEFAULT_OBJECTS)
    deadline = node.get_clock().now().nanoseconds * 1e-9 + args.timeout
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(node.poses) == len(DEFAULT_OBJECTS):
            break
        if node.get_clock().now().nanoseconds * 1e-9 > deadline:
            break

    status = 0
    for name, (vessel_geom, plate_geom, tag_id) in sorted(DEFAULT_OBJECTS.items()):
        truth_vessel = geoms[vessel_geom]
        truth_plate = geoms[plate_geom]
        print('\n=== %s (tag %d) ===' % (name, tag_id))

        tag, error = node.tag_pose(tag_id)
        if tag is None:
            print('  detector : no TF base_link <- tag_%d (%s)' % (tag_id, error))
            status = 1
        else:
            tag_p, tag_R = tag
            # The plate's top face carries the tag, 1 mm above the plate centre.
            truth_tag = truth_plate + np.array([0.0, 0.0, 0.001])
            delta = tag_p - truth_tag
            print('  detector : %s' % np.array2string(tag_p, precision=4))
            print('    truth  : %s' % np.array2string(truth_tag, precision=4))
            print('    error  : %.1f mm  (%s)'
                  % (np.linalg.norm(delta) * 1e3, np.array2string(delta * 1e3, precision=1)))
            print('    tag x,y,z axes in base:\n      %s'
                  % np.array2string(tag_R, precision=3).replace('\n', '\n      '))
            want = tag_R.T @ (truth_vessel - tag_p)
            print('    grasp_offset.position that would be exact: [%.4f, %.4f, %.4f]'
                  % tuple(want))

        msg = node.poses.get(name)
        if msg is None:
            print('  end2end  : nothing published on /perception/object_pose/%s' % name)
            status = 1
            continue
        got = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        delta = got - truth_vessel
        print('  end2end  : %s' % np.array2string(got, precision=4))
        print('    truth  : %s' % np.array2string(truth_vessel, precision=4))
        print('    error  : %.1f mm  (%s)'
              % (np.linalg.norm(delta) * 1e3, np.array2string(delta * 1e3, precision=1)))

    node.destroy_node()
    rclpy.try_shutdown()
    return status


if __name__ == '__main__':
    sys.exit(main())
