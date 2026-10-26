# Copyright 2026 Hyunho Cho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tell the pose node the robot took an object, or put it back, or put it down.

A pose node that keeps an object's last measured pose while no camera sees it
(``hold_unseen`` in the object table, ``config/perception/pour_beakers.yaml``)
does so on the rule that nothing has moved it. The replay is what moves it, so
the replay says so, on std_srvs/Trigger services beside the object's pose topic:

  ``<pose topic>/release``  the jaws have closed on it: no pose goes out, and its
                            tag -- which comes with it -- is not believed as
                            where it stands; rviz shows an afterimage where it
                            was, and ``<pose topic>/in_hand`` how it hangs
  ``<pose topic>/restore``  it was set down where it was taken from: the pose
                            held before is held again
  ``<pose topic>/resume``   it was set down somewhere else: the next
                            measurement of it is believed

Those services are the laptop's cho_object_pose (its 6e364e1, never on GitHub),
not cho_robot_project v0.1.0's: the pose node of this workspace's core holds
nothing and serves none of them. The leaf then logs that the service is not
there and fails, which the trees take as nothing to change.

Never a reason to stop the task. A pose node that holds nothing for this object,
or runs without holds at all, has nothing to change, and the grasp it served is
already measured.
"""

import py_trees
from std_srvs.srv import Trigger

from cho_task_manager.behaviors.service.base_service_behavior import BaseServiceBehavior

#: What each call tells the pose node.
VERBS = ('release', 'restore', 'resume')


def release_service_name(pose_topic, verb='release'):
    """The pose node's *verb* service for the object published on *pose_topic*."""
    if verb not in VERBS:
        raise ValueError('verb must be one of %s, got %r' % (VERBS, verb))
    return pose_topic.rstrip('/') + '/' + verb


class HeldPoseServiceBehavior(BaseServiceBehavior):
    """Call one of the pose node's hand-over services for one object."""

    def __init__(self, name, pose_topic, verb, timeout_sec=2.0, response_timeout_sec=3.0):
        if not pose_topic:
            raise ValueError(f'[{name}] pose_topic is required')
        super().__init__(name, Trigger, release_service_name(pose_topic, verb),
                         timeout_sec=timeout_sec, response_timeout_sec=response_timeout_sec)
        self.verb = verb

    def handle_response(self, result):
        log = self.node.get_logger()
        (log.info if result.success else log.warn)(f'[{self.name}] {result.message}')
        return (py_trees.common.Status.SUCCESS if result.success
                else py_trees.common.Status.FAILURE)


class ReleaseHeldPoseBehavior(HeldPoseServiceBehavior):
    """The jaws have closed on the object: ``<pose topic>/release``."""

    def __init__(self, name, pose_topic, **kwargs):
        super().__init__(name, pose_topic, 'release', **kwargs)


__all__ = ['HeldPoseServiceBehavior', 'ReleaseHeldPoseBehavior', 'VERBS', 'release_service_name']
