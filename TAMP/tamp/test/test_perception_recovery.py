"""What the executor does when a tagged vessel is not localized.

The manuscript (III-C2, V-B) says the executor waits up to 5 s for a
re-detection and then halts and reports the affected object. Until
`_localize_with_recovery` existed, the wait did not: the first failed lookup
failed the request. These pin the ladder that implements it, and -- more
importantly -- the three things it must never do.

  * it must never return a pose it did not observe (no ground-truth fallback:
    a substituted pose reports a perception-in-the-loop trial that was not);
  * it must never move an arm that is holding a vessel;
  * it must never move the arm along a path nothing collision-checked.

No ROS and no simulator: the server object is built without `__init__` and its
plant hooks are stubs, so this exercises the ladder's decisions rather than a
robot. Importing the server still pulls in cuTAMP/cuRobo (conda `sdl`), as the
other server tests do.
"""

import ast
import inspect
import os
import re
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'server'))

import tamp_server                                          # noqa: E402

POSE = [0.5, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0]
HOME = [0.0, -1.05, -2.18, -1.57, 1.57, 0.0]


class _Logger:
    def __init__(self):
        self.lines = []

    def _record(self, msg):
        self.lines.append(str(msg))

    info = warn = warning = error = debug = _record


class _Plan:
    """What `motion_plan_js` hands back: joint waypoints to replay."""

    def __init__(self, n_waypoints=3):
        self.position = np.zeros((n_waypoints, 6))
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']


class _Tamp:
    def __init__(self, plan=_Plan, env=object()):
        self.env = env
        self._plan = plan
        self.planned = []

    def motion_plan_js(self, q_init, q_des):
        self.planned.append((list(q_init), list(q_des)))
        return self._plan() if callable(self._plan) else self._plan


class _Server:
    """A TAMPServer whose plant is a notebook.

    Built with __new__ so no ROS node is constructed; the ladder's own methods
    are the real ones, unbound from the class under test.
    """

    def __init__(self, looks=(), available=None, holding=False, tamp=None,
                 plant_ok=True, publish_raises=None):
        self.node = tamp_server.TAMPServer.__new__(tamp_server.TAMPServer)
        self.logger = _Logger()
        self.looks = list(looks)          # scripted _perception_pose results
        # ...or a predicate over this object, for "the pose shows up once X
        # has happened". The rungs are separated by EVENTS, not by a number of
        # polls: how many polls fit in the wait is a function of the clock.
        self.available = available
        self.observed = 0
        self.commands = []                # every configuration sent to the arm
        self.sessions = []                # _start_execution / _finish_execution
        self.plant_ok = plant_ok
        self.publish_raises = publish_raises

        n = self.node
        n.tamp = tamp if tamp is not None else _Tamp()
        n._holding = holding
        n._recovery_attempts = 0
        n.get_logger = lambda: self.logger
        n._perception_pose = self._perception_pose
        n._arm_q = lambda: [0.0] * 6
        n._start_execution = self._start_execution
        n._finish_execution = lambda success: self.sessions.append(
            ('finish', bool(success)))
        n._publish_arm_command = self._publish_arm_command

        # Short enough that the suite runs in under a second; the ladder's
        # order and its refusals are what is under test, not the clock.
        n.RECOVERY_ENABLED = True
        n.RECOVERY_WAIT_S = 0.05
        n.RECOVERY_POLL_S = 0.005
        n.RECOVERY_RETREAT = True
        n.RECOVERY_SCAN = False
        n.RECOVERY_SCAN_S = 1.0
        n.RECOVERY_HOME = list(HOME)
        n.SCAN_J1_RANGE = 1.57
        n.SCAN_J2_OFFSETS = [0.0]

    def _perception_pose(self, entity):
        self.observed += 1
        if self.available is not None:
            return POSE if self.available(self) else None
        return self.looks.pop(0) if self.looks else None

    def _start_execution(self):
        self.sessions.append(('start', self.plant_ok))
        return self.plant_ok

    def _publish_arm_command(self, positions, joint_names=None):
        if self.publish_raises is not None:
            raise self.publish_raises
        self.commands.append(list(positions))

    # -- what the test asks afterwards ---------------------------------
    def localize(self, entity='beaker'):
        return self.node._localize_with_recovery(entity)

    @property
    def summary(self):
        for line in reversed(self.logger.lines):
            if line.startswith('[recovery] entity='):
                return dict(
                    part.split('=', 1) for part in line.split()[1:])
        raise AssertionError('the ladder logged no summary line')

    @property
    def moved(self):
        return bool(self.commands)


@pytest.fixture(autouse=True)
def _fast_replay(monkeypatch):
    # The replay loop sleeps SDL_EXEC_DT per waypoint.
    monkeypatch.setenv('SDL_EXEC_DT', '0')


# -- the happy paths ---------------------------------------------------

def test_a_pose_on_the_first_look_costs_nothing():
    s = _Server(looks=[POSE])
    assert s.localize() == POSE
    assert s.observed == 1
    assert not s.moved
    assert s.summary['outcome'] == 'immediate'


def test_the_wait_recovers_without_moving_the_arm():
    # The rung the manuscript already describes: an intermittent detection
    # comes back on its own, and nothing has to move for it.
    s = _Server(looks=[None, None, POSE])
    assert s.localize() == POSE
    assert not s.moved, 'the wait must not move the arm'
    assert s.summary['outcome'] == 'recovered'
    assert s.summary['stage'] == 'wait'


def test_the_retreat_runs_only_after_the_wait_has_failed():
    # The vessel becomes visible only once the arm has moved off it -- the
    # case the rung exists for, an arm parked over its own workspace.
    s = _Server(available=lambda s: s.moved)
    assert s.localize() == POSE
    assert s.moved, 'the retreat rung never commanded the arm'
    assert s.node.tamp.planned, 'the retreat was not planned'
    assert s.node.tamp.planned[0][1] == HOME
    assert s.summary['stage'] == 'retreat'
    assert s.summary['outcome'] == 'recovered'
    # The plant is taken and released exactly once, whatever the outcome.
    assert s.sessions == [('start', True), ('finish', True)]


def test_the_scan_is_reached_only_when_the_retreat_did_not_help():
    # Visible only from a viewpoint the scan reaches: the retreat plans once
    # (to home), so the second planned motion is the scan's first leg.
    s = _Server(available=lambda s: len(s.node.tamp.planned) >= 2)
    s.node.RECOVERY_SCAN = True
    assert s.localize() == POSE
    assert s.summary['stage'] == 'scan'
    assert s.summary['outcome'] == 'recovered'
    # Both rungs took the plant and handed it back.
    assert s.sessions.count(('start', True)) == 2
    assert [kind for kind, _ in s.sessions] == ['start', 'finish',
                                                'start', 'finish']


# -- the refusals ------------------------------------------------------

def test_a_held_vessel_stops_the_ladder_before_any_motion():
    # Not a tunable. The upright constraint holds along the path the planner
    # planned, and on hardware the vessel has liquid in it, so a recovery
    # sweep with a full beaker trades a lost trial for a spill.
    s = _Server(looks=[], holding=True)
    s.node.RECOVERY_SCAN = True
    assert s.localize() is None
    assert not s.moved
    assert not s.node.tamp.planned, 'a held vessel must not even be planned for'
    assert s.summary['outcome'] == 'skipped'
    assert s.summary['reason'] == 'holding'


def test_an_unplannable_retreat_is_not_interpolated_anyway():
    # motion_plan_js returning None means no collision-free path was found.
    # Ramping the joints there regardless is how a perception failure becomes
    # a collision.
    s = _Server(looks=[], tamp=_Tamp(plan=None))
    assert s.localize() is None
    assert not s.moved
    assert s.summary['outcome'] == 'failed'


def test_recovery_does_not_move_before_a_world_model_exists():
    # The ladder runs BEFORE update_env, so on the first request of a process
    # there is no collision model at all to plan the retreat against.
    s = _Server(looks=[], tamp=_Tamp(env=None))
    s.node.RECOVERY_SCAN = True
    assert s.localize() is None
    assert not s.moved
    assert not s.node.tamp.planned


def test_a_plant_that_refuses_the_arm_is_not_commanded():
    # On hardware _start_execution is the controller switch. If it did not
    # happen, streaming points at the arm is streaming them at nothing --
    # or at whatever else holds the controller.
    s = _Server(looks=[], plant_ok=False)
    assert s.localize() is None
    assert not s.moved


def test_a_plant_that_raises_mid_retreat_fails_closed_and_releases_it():
    # The real plant raises EStopRequested / ChoCommandFailed, which this
    # module cannot import. A refused command is a failed rung, not a dead
    # server and not a swallowed error -- and the arm is still handed back.
    s = _Server(looks=[], publish_raises=RuntimeError('e-stop'))
    assert s.localize() is None
    assert ('finish', False) in s.sessions
    assert any('e-stop' in line for line in s.logger.lines)


def test_disabling_recovery_restores_the_old_single_look():
    # The A/B control: SDL_RECOVERY=0 has to reproduce the behaviour every
    # number measured before this existed was measured under.
    s = _Server(looks=[None, POSE])
    s.node.RECOVERY_ENABLED = False
    assert s.localize() is None
    assert s.observed == 1
    assert not s.moved


# -- the thing that must never happen ----------------------------------

def test_the_ladder_only_ever_returns_what_perception_returned():
    # Every rung returns the value `_perception_pose` gave it, or None. The
    # source-level check is the one that matters: a fallback added later would
    # still satisfy a behavioural test written today, because it would only
    # fire on the path that test does not take.
    for name in ('_localize_with_recovery', '_poll_perception_pose',
                 '_recovery_retreat', '_recovery_scan'):
        src = inspect.getsource(getattr(tamp_server.TAMPServer, name))
        assert 'get_entity_state' not in src, (
            '%s reaches for the simulator; a substituted pose reports a '
            'perception-in-the-loop trial that was not one' % name)
        returns = [
            node for node in ast.walk(ast.parse(textwrap.dedent(src)))
            if isinstance(node, ast.Return) and node.value is not None]
        for node in returns:
            assert isinstance(node.value, (ast.Name, ast.Constant, ast.Call,
                                           ast.Compare)), (
                '%s returns a constructed value; it may only pass on what '
                'perception gave it' % name)


def test_the_summary_line_is_greppable():
    # The run logs are where "how often did recovery rescue a trial, and how
    # long did it take" comes from, so the format is part of the contract.
    s = _Server(looks=[POSE])
    s.localize('flask')
    line = [x for x in s.logger.lines if x.startswith('[recovery] entity=')][-1]
    assert re.match(
        r'^\[recovery\] entity=\S+ outcome=(immediate|recovered|failed|skipped) '
        r'stage=(wait|retreat|scan) elapsed_s=\d+\.\d\d attempts=\d+'
        r'( reason=\S+)?$', line), line
