"""State nodes for magnet-style book attach/detach (world map + ``robot.holding``)."""

from __future__ import annotations

from math import degrees

from aim_fsm import StateNode
from aim_fsm.geometry import wrap_angle
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
        # WorldMap ``update_held_object`` only moves x,y with the robot; spine yaw must
        # follow turns. Store offset from robot heading so librarian map can update θ.
        obj._hold_theta_offset_rad = wrap_angle(obj.pose.theta - self.robot.pose.theta)
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
        # θ was updated while held (``LibrarianWorldMap.update_held_object``); keep it so
        # release matches physical spine direction after pivots. Do not snap to template.
        if getattr(held, "_hold_theta_offset_rad", None) is not None:
            delattr(held, "_hold_theta_offset_rad")
        held.held_by = None
        self.robot.holding = None
        th_deg = degrees(held.pose.theta) if held.pose.theta is not None else float("nan")
        print(
            f"DetachBookAtPose: released Book-{held.marker_id} at "
            f"({held.pose.x:.1f}, {held.pose.y:.1f}) mm, theta={th_deg:.1f} deg"
        )
        self.post_completion()


__all__ = ["AttachBook", "DetachBookAtPose"]
