"""Publish the RS-232 scale as the plain number every consumer wants.

The scale driver (``cho_sensor/hansung_scale``) publishes a self-describing
``WeightStamped`` -- value, unit, stability, status -- which is the right thing
for a sensor to publish and the wrong thing for a control loop to subscribe to.
Two consumers want one number in grams:

* the pouring controller, which should not carry a dependency on one vendor's
  message type to read a weight, and
* ``tamp_server``, whose ``scale_cb`` already subscribes to a Float32.

So this relay is the adapter, and it lives here rather than in the scale
package on purpose: that package is deliberately self-contained and must not
grow dependencies on anything else.

**A NaN reading publishes nothing.** ``weight_grams`` is NaN when the indicator
reported a unit the driver does not convert, and a control loop that silently
kept its last value would keep tilting on a weight it no longer knows. Silence
is what a stale-data guard downstream is for.

**Stability is not required by default.** ``stable`` is only true once the
reading has settled, which during a pour is exactly when it is not. Requiring
it would starve the loop of the readings it exists to react to; it is offered
as a parameter for tare/checkpoint uses that do want a settled number.
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64

#: The driver's default node name is `scale_node` in the root namespace, so
#: its `~/weight_stamped` lands here.
DEFAULT_WEIGHT_TOPIC = '/scale_node/weight_stamped'

#: Generic grams, for the pour controller.
DEFAULT_GRAMS_TOPIC = '/scale/grams'

#: What tamp_server's scale_cb already subscribes to.
DEFAULT_LEGACY_TOPIC = '/raw_scale_data'


class ScaleRelay(Node):
    """WeightStamped -> Float64 grams (and the Float32 tamp_server reads)."""

    def __init__(self):
        super().__init__('scale_relay')
        self.declare_parameter('weight_topic', DEFAULT_WEIGHT_TOPIC)
        self.declare_parameter('grams_topic', DEFAULT_GRAMS_TOPIC)
        self.declare_parameter('legacy_topic', DEFAULT_LEGACY_TOPIC)
        self.declare_parameter('publish_legacy', True)
        self.declare_parameter('require_stable', False)

        weight_topic = self.get_parameter('weight_topic').value
        self.require_stable = bool(self.get_parameter('require_stable').value)

        try:
            from hansung_scale_msgs.msg import WeightStamped
        except ImportError as error:
            raise RuntimeError(
                'hansung_scale_msgs is not on the path; source the '
                'cho_robot_project workspace before running the scale relay'
            ) from error

        self.grams_pub = self.create_publisher(
            Float64, self.get_parameter('grams_topic').value, 10)
        self.legacy_pub = None
        if self.get_parameter('publish_legacy').value:
            self.legacy_pub = self.create_publisher(
                Float32, self.get_parameter('legacy_topic').value, 10)

        self.create_subscription(WeightStamped, weight_topic, self._on_weight, 10)
        self.dropped = 0
        self.get_logger().info(
            f'relaying {weight_topic} -> '
            f"{self.get_parameter('grams_topic').value}"
            f"{' + ' + self.get_parameter('legacy_topic').value if self.legacy_pub else ''}"
            f"{' (settled readings only)' if self.require_stable else ''}")

    def _on_weight(self, msg):
        if self.require_stable and not msg.stable:
            return
        grams = float(msg.weight_grams)
        if not math.isfinite(grams):
            self.dropped += 1
            if self.dropped % 25 == 1:
                self.get_logger().warn(
                    f"weight_grams is not a number (unit '{msg.unit}', status "
                    f"'{msg.status}'); publishing nothing. {self.dropped} so far")
            return
        self.grams_pub.publish(Float64(data=grams))
        if self.legacy_pub is not None:
            self.legacy_pub.publish(Float32(data=grams))


def main(args=None):
    rclpy.init(args=args)
    node = ScaleRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
