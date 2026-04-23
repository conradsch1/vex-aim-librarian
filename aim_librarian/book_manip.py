"""State nodes for magnet-style book attach/detach (world map + ``robot.holding``)."""

from __future__ import annotations

import time

from aim_fsm import StateNode
from aim_fsm.utils import Pose

from aim_librarian.books import BookObj


def _find_book(robot, marker_id: int) -> BookObj | None:
    for obj in robot.world_map.objects.values():
        if isinstance(obj, BookObj) and obj.marker_id == marker_id:
            return obj
    return None


class AttachBook(StateNode):
    """Mark ``robot.holding`` to the :class:`BookObj` after you have driven onto the book with the magnet."""

    def __init__(self, marker_id: int):
        super().__init__()
        self.marker_id = marker_id

    def start(self, event=None):
        super().start(event)
        obj = _find_book(self.robot, self.marker_id)
        if obj is None:
            print(f"AttachBook: no BookObj with marker id {self.marker_id} on the map")
            self.post_failure()
            return
        self.robot.holding = obj
        obj.held_by = self.robot
        print(f"AttachBook: holding Book-{self.marker_id} (map state; magnet should be engaged)")
        self.post_completion()


class DetachBookAtPose(StateNode):
    """Release hold and set the book's map pose to the drop location (after backing off the magnet)."""

    def __init__(self, pose: Pose):
        super().__init__()
        self.pose = pose

    def start(self, event=None):
        super().start(event)
        held = self.robot.holding
        if held is None or not isinstance(held, BookObj):
            print("DetachBookAtPose: not holding a BookObj")
            self.post_failure()
            return
        held.pose.x = self.pose.x
        held.pose.y = self.pose.y
        held.pose.z = self.pose.z
        if self.pose.theta is not None:
            held.pose.theta = self.pose.theta
        held.held_by = None
        self.robot.holding = None
        print(
            f"DetachBookAtPose: released Book-{held.marker_id} at "
            f"({held.pose.x:.1f}, {held.pose.y:.1f}) mm"
        )
        self.post_completion()


class WaitUntilBookRemoved(StateNode):
    """Poll ArUco vision until the book spine id stays out of view; then clear ``robot.holding``.

    Used when a patron removes a book from the magnet: there is no barrel/ball sensor for
    books, so absence of the spine marker (debounced) is the release cue.
    """

    POLL_S = 0.12
    ABSENT_S = 1.0

    def __init__(self, marker_id: int | None = None):
        super().__init__()
        self.marker_id = marker_id
        self._poll_handle = None
        self._absent_since: float | None = None
        self._mid: int | None = None

    def start(self, event=None):
        super().start(event)
        self._absent_since = None
        mid = self.marker_id
        if mid is None:
            mid = getattr(self.parent, "marker_id", None)
        if mid is None:
            print("WaitUntilBookRemoved: no marker_id")
            self.post_failure()
            return
        self._mid = int(mid)
        self._poll_once()

    def _poll_once(self):
        if not self.running:
            return
        det = getattr(self.robot, "aruco_detector", None)
        if det is None:
            print("WaitUntilBookRemoved: no aruco_detector")
            self.post_failure()
            return
        if hasattr(det, "snapshot_seen_markers"):
            seen = det.snapshot_seen_markers()
        else:
            seen = det.seen_marker_objects
        assert self._mid is not None
        if self._mid in seen:
            self._absent_since = None
        else:
            now = time.time()
            if self._absent_since is None:
                self._absent_since = now
            elif now - self._absent_since >= self.ABSENT_S:
                held = self.robot.holding
                if isinstance(held, BookObj) and held.marker_id == self._mid:
                    held.held_by = None
                    self.robot.holding = None
                print(f"WaitUntilBookRemoved: spine {self._mid} absent; holding cleared")
                self.post_completion()
                return
        self._poll_handle = self.robot.loop.call_later(self.POLL_S, self._poll_once)

    def stop(self):
        if self._poll_handle:
            self._poll_handle.cancel()
            self._poll_handle = None
        super().stop()


__all__ = ["AttachBook", "DetachBookAtPose", "WaitUntilBookRemoved"]
