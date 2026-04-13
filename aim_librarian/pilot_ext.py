"""PilotGoal node for navigating to an ArUco marker / librarian Book (book id range)."""

from __future__ import annotations

import numpy as np
from math import cos, nan, pi, sin, sqrt

from aim_fsm.events import DataEvent
from aim_fsm.geometry import wrap_angle
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
        for obj in self.robot.world_map.objects.values():
            if isinstance(obj, (ArucoMarkerObj, BookObj)) and obj.marker_id == self.marker_id:
                return obj
        return None

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
        if self.marker_id not in markers:
            print(f"PilotToArucoMarker: marker id {self.marker_id} not in view")
            self.punt_super_start()
            self.post_failure()
            return
        self.target_object = self._world_object_for_marker_id()
        self.target_pose = self._world_pose_from_marker(markers[self.marker_id])
        self.robot.rrt.max_iter = self.max_iter
        super(PilotToPose, self).start(event)


__all__ = ["PilotToArucoMarker"]
