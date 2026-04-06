"""
Demo: smooth right arc using ``move_with_vectors`` (via the drive actuator).

Combines forward motion with clockwise yaw so the path curves right. Velocity is
ramped up and down with a smoothstep envelope so start/stop are gentle.

Requires ``vex-aim-tools`` and ``vex-aim-librarian`` on ``PYTHONPATH``. From
``simple_cli``::

    runfsm('ArcRightVectorsDemo')

Adjust ``duration_s``, ``forward_pct``, ``clockwise_turn_pct``, and
``right_strafe_pct`` on ``SmoothArcRightVectors`` to tune the arc.
"""

from __future__ import annotations

import time

from aim_fsm import *

from aim_librarian import install_librarian_extensions


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _envelope(t: float, duration: float, ramp_frac: float) -> float:
    """1.0 in the middle, smooth 0->1 and 1->0 at the ends."""
    if duration <= 0.0:
        return 0.0
    tr = max(duration * ramp_frac, 1e-6)
    if t < tr:
        return _smoothstep(t / tr)
    if t > duration - tr:
        return _smoothstep((duration - t) / tr)
    return 1.0


class SmoothArcRightVectors(ActionNode):
    """
    Hold a smooth right-turning arc for ``duration_s`` using repeated
    ``move_with_vectors`` calls.

    Arguments are in *robot* terms (matching ``vex.aim`` docs): forward and
    clockwise rotation percentages, optional right strafe. The actuator maps
    these to ``move_with_vectors(xvel, yvel, rvel)`` as
    ``(forward, -right_strafe, -clockwise_turn)``.
    """

    def __init__(
        self,
        duration_s: float = 3.5,
        forward_pct: float = 42.0,
        clockwise_turn_pct: float = 38.0,
        right_strafe_pct: float = 6.0,
        step_s: float = 0.05,
        ramp_frac: float = 0.18,
    ):
        super().__init__()
        self.duration_s = duration_s
        self.forward_pct = forward_pct
        self.clockwise_turn_pct = clockwise_turn_pct
        self.right_strafe_pct = right_strafe_pct
        self.step_s = step_s
        self.ramp_frac = ramp_frac
        self._tick_handle = None

    def start(self, event=None):
        super().start(event)
        self._t0 = time.monotonic()
        self._tick()  # apply first command immediately; further steps use step_s

    def _schedule_tick(self):
        if self._tick_handle is not None:
            self._tick_handle.cancel()
        self._tick_handle = self.robot.loop.call_later(self.step_s, self._tick)

    def _tick(self):
        self._tick_handle = None
        if not self.running:
            return
        elapsed = time.monotonic() - self._t0
        if elapsed >= self.duration_s:
            self.robot.abort_all_actions()
            self.complete()
            return
        scale = _envelope(elapsed, self.duration_s, self.ramp_frac)
        f = scale * self.forward_pct
        rs = scale * self.right_strafe_pct
        cw = scale * self.clockwise_turn_pct
        # Actuator -> robot: forwards=f, rightwards=-yvel, rotation=-rvel
        self.robot.actuators["drive"].move_with_vectors(self, f, -rs, -cw)
        self._schedule_tick()

    def stop(self):
        if self._tick_handle is not None:
            self._tick_handle.cancel()
            self._tick_handle = None
        try:
            self.robot.abort_all_actions()
        except Exception:
            pass
        super().stop()


class ArcRightVectorsDemo(StateMachineProgram):
    """Run a single smooth right arc, then exit the FSM."""

    def __init__(self, **kwargs):
        opts = dict(
            speech=False,
            launch_path_viewer=False,
            launch_worldmap_viewer=True,
            launch_cam_viewer=True,
        )
        opts.update(kwargs)
        super().__init__(**opts)
        install_librarian_extensions(self.robot)

    class ArcDonePrint(StateNode):
        def start(self, event=None):
            super().start(event)
            print("ArcRightVectorsDemo: arc finished.")
            self.post_completion()

    def setup(self):
        arc = (
            SmoothArcRightVectors(
                duration_s=3.5,
                forward_pct=42.0,
                clockwise_turn_pct=5.0,
                right_strafe_pct=6.0,
            )
            .set_name("arc")
            .set_parent(self)
        )
        done = self.ArcDonePrint().set_name("done").set_parent(self)
        end = ParentCompletes().set_name("end").set_parent(self)

        CompletionTrans().add_sources(arc).add_destinations(done)
        CompletionTrans().add_sources(done).add_destinations(end)

        return self
