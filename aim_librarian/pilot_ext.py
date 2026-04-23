"""PilotGoal node for navigating to an ArUco marker / librarian Book (book id range)."""

from __future__ import annotations

import numpy as np
from math import atan2, cos, nan, pi, sin, sqrt

from aim_fsm.events import DataEvent
from aim_fsm.geometry import wrap_angle
from aim_fsm.nodes import ActionNode
from aim_fsm.pilot import PilotToPose
from aim_fsm.utils import Pose
from aim_fsm.worldmap import ArucoMarkerObj, WorldObject

from aim_librarian.books import BookObj, is_book_aruco_id


class PilotToArucoMarker(PilotToPose):
    """Same behavior as the former vex-aim-tools implementation; lives in librarian."""

    def __init__(self, marker_id, align_heading=False, **kwargs):
        super().__init__(target_pose=None, **kwargs)
        self.marker_id = marker_id
        self.align_heading = align_heading

    def _world_pose_from_marker(self, marker):
        camera_offset_vector = np.array([0, 0, self.robot.kine.camera_from_origin])
        sensor_coords = np.array(marker.camera_coords) + camera_offset_vector
        sensor_distance = sqrt(sensor_coords[0] ** 2 + sensor_coords[2] ** 2)
        sensor_bearing = np.arctan2(sensor_coords[0], sensor_coords[2])
        sensor_orient = wrap_angle(pi - marker.euler_angles[1])
        theta_robot = self.robot.pose.theta
        x = self.robot.pose.x + sensor_distance * cos(theta_robot + sensor_bearing)
        y = self.robot.pose.y + sensor_distance * sin(theta_robot + sensor_bearing)
        if is_book_aruco_id(self.marker_id):
            z = BookObj.HEIGHT_MM / 2
        else:
            z = marker.aruco_parent.marker_size / 2
        heading = wrap_angle(theta_robot + sensor_orient) if self.align_heading else nan
        return Pose(x, y, z, heading)

    def _world_object_for_marker_id(self):
        """Prefer :class:`BookObj` so path planning uses the book footprint, not a duplicate ArUco."""
        book_obj = None
        aruco_obj = None
        for obj in self.robot.world_map.objects.values():
            if obj.marker_id != self.marker_id:
                continue
            if isinstance(obj, BookObj):
                book_obj = obj
            elif isinstance(obj, ArucoMarkerObj):
                aruco_obj = obj
        return book_obj if book_obj is not None else aruco_obj

    def start(self, event=None):
        if isinstance(event, DataEvent):
            if isinstance(event.data, int):
                self.marker_id = int(event.data)
            elif isinstance(event.data, (WorldObject, Pose)):
                return super().start(event)
            else:
                raise ValueError(
                    "PilotToArucoMarker DataEvent data must be int id, WorldObject, or Pose:",
                    event.data,
                )
        det = getattr(self.robot, "aruco_detector", None)
        if det is None:
            print("PilotToArucoMarker: robot has no aruco_detector")
            self.punt_super_start()
            self.post_failure()
            return
        if hasattr(det, "snapshot_seen_markers"):
            markers = det.snapshot_seen_markers()
        else:
            markers = det.seen_marker_objects.copy()

        if self.marker_id in markers:
            self.target_object = self._world_object_for_marker_id()
            self.target_pose = self._refine_target_pose(
                self._world_pose_from_marker(markers[self.marker_id])
            )
        else:
            # Spine not in the current frame (motion blur, angle, timing) but SLAM may
            # still have a visible BookObj — navigate to the map pose like PilotToPose(WorldObject).
            map_obj = self._world_object_for_marker_id()
            if isinstance(map_obj, BookObj) and map_obj.is_visible:
                print(
                    f"PilotToArucoMarker: marker {self.marker_id} not in snapshot; "
                    "using BookObj map pose"
                )
                self.target_object = map_obj
                bp = map_obj.pose
                z = float(bp.z) if getattr(bp, "z", None) is not None else BookObj.HEIGHT_MM / 2
                self.target_pose = self._refine_target_pose(
                    Pose(float(bp.x), float(bp.y), z, nan)
                )
            else:
                print(
                    f"PilotToArucoMarker: marker id {self.marker_id} not in view "
                    f"and no visible BookObj on map"
                )
                self.punt_super_start()
                self.post_failure()
                return

        self.robot.rrt.max_iter = self.max_iter
        super(PilotToPose, self).start(event)

    def _refine_target_pose(self, pose: Pose) -> Pose:
        """Hook for subclasses (e.g. :class:`PilotToBook`) to shift the navigation goal."""
        return pose


class PilotToBook(PilotToArucoMarker):
    """Navigate using the :class:`BookObj` goal (same spine ArUco id); clearer name for swap demos."""

    def __init__(
        self,
        marker_id,
        align_heading=False,
        book_approach_offset_mm=None,
        **kwargs,
    ):
        super().__init__(marker_id, align_heading=align_heading, **kwargs)
        if book_approach_offset_mm is None:
            self.book_approach_offset_mm = BookObj.SPINE_THICKNESS_MM / 2 + 5.0
        else:
            self.book_approach_offset_mm = float(book_approach_offset_mm)

    def _refine_target_pose(self, pose: Pose) -> Pose:
        """Shift goal from spine marker centroid toward the robot for magnet / front-cover approach.

        The returned pose's heading always points *from the standoff back toward
        the marker centroid* so that a subsequent ``Forward(N)`` engage step
        drives INTO the book, not away from it. (Note: ``align_heading=True``
        on its own sets the heading to the marker's *outward* normal, i.e. the
        direction facing AWAY from the book center. That's the wrong heading
        for engagement, so we override it here whenever an offset is applied.)
        """
        if self.book_approach_offset_mm <= 0:
            return pose
        rx, ry = self.robot.pose.x, self.robot.pose.y
        dx, dy = rx - pose.x, ry - pose.y
        d = sqrt(dx * dx + dy * dy)
        if d < 1e-3:
            return pose
        s = self.book_approach_offset_mm / d
        new_x = pose.x + dx * s
        new_y = pose.y + dy * s
        new_theta = atan2(pose.y - new_y, pose.x - new_x)
        return Pose(new_x, new_y, pose.z, new_theta)


class TurnTowardPose(ActionNode):
    """Rotate in place to face ``(target_pose.x, target_pose.y)`` in the world (cf. ``TurnToward`` for objects)."""

    def __init__(self, target_pose: Pose, turn_speed=None):
        super().__init__()
        self.target_pose = target_pose
        self.turn_speed = turn_speed

    def start(self, event=None):
        if isinstance(event, DataEvent) and isinstance(event.data, Pose):
            self.target_pose = event.data
        super().start(event)
        dx = self.target_pose.x - self.robot.pose.x
        dy = self.target_pose.y - self.robot.pose.y
        angle_deg = wrap_angle(atan2(dy, dx) - self.robot.pose.theta) * 180 / pi
        self.robot.actuators["drive"].turn(self, angle_deg * pi / 180, self.turn_speed)


__all__ = ["PilotToArucoMarker", "PilotToBook", "TurnTowardPose"]
