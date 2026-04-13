"""Librarian WorldMap: ArUco ids in the book range become BookObj."""

from __future__ import annotations

import math

import numpy as np

from aim_fsm.worldmap import (
    WorldMap,
    ArucoMarkerObj,
    WallObj,
    DoorwayObj,
    AprilTagObj,
)
from aim_fsm.utils import Pose
from aim_fsm.geometry import wrap_angle

from aim_librarian.books import BookObj, is_book_aruco_id


class LibrarianWorldMap(WorldMap):
    def make_new_aruco_objects(self):
        camera_offset_vector = np.array([0, 0, self.robot.kine.camera_from_origin])
        detector = self.robot.aruco_detector
        if hasattr(detector, "snapshot_seen_markers"):
            seen_markers = detector.snapshot_seen_markers()
        else:
            seen_markers = detector.seen_marker_objects.copy()
        for id, marker in seen_markers.items():
            if is_book_aruco_id(id):
                name = f"Book-{id}"
                spec = {"name": name, "id": id, "marker": marker}
                obj = BookObj(spec)
                z_pose = BookObj.HEIGHT_MM / 2
            else:
                name = f"ArucoMarker-{id}"
                spec = {"name": name, "id": id, "marker": marker}
                obj = ArucoMarkerObj(spec)
                z_pose = marker.aruco_parent.marker_size / 2
            sensor_coords = marker.camera_coords + camera_offset_vector
            sensor_distance = math.sqrt(sensor_coords[0] ** 2 + sensor_coords[2] ** 2)
            sensor_bearing = math.atan2(sensor_coords[0], sensor_coords[2])
            sensor_orient = wrap_angle(math.pi - marker.euler_angles[1])
            theta = self.robot.pose.theta
            obj.pose = Pose(
                self.robot.pose.x + sensor_distance * math.cos(theta + sensor_bearing),
                self.robot.pose.y + sensor_distance * math.sin(theta + sensor_bearing),
                z_pose,
                wrap_angle(self.robot.pose.theta + sensor_orient),
            )
            obj.sensor_distance = sensor_distance
            obj.sensor_bearing = sensor_bearing
            obj.sensor_orient = sensor_orient
            obj.is_visible = True
            self.candidates.append(obj)

    def associate_objects_of_type(self, otype):
        import numpy as np

        new = [c for c in self.candidates if type(c) is otype]
        old = [o for o in self.objects.values() if type(o) is otype]
        n_new = len(new)
        n_old = len(old)
        if n_old == 0:
            return
        costs = np.zeros([n_new, n_old])
        if self.robot.particle_filter and self.robot.particle_filter.state != self.robot.particle_filter.LOCALIZED:
            max_acceptable_cost = np.inf
        elif otype in (ArucoMarkerObj, BookObj, WallObj, DoorwayObj):
            max_acceptable_cost = np.inf
        else:
            max_acceptable_cost = 500
        for i in range(n_new):
            for j in range(n_old):
                if otype is ArucoMarkerObj and new[i].marker_id != old[j].marker_id:
                    costs[i, j] = max_acceptable_cost + 1
                elif otype is BookObj and new[i].marker_id != old[j].marker_id:
                    costs[i, j] = max_acceptable_cost + 1
                elif otype is AprilTagObj and new[i].tag_id != old[j].tag_id:
                    costs[i, j] = max_acceptable_cost + 1
                else:
                    costs[i, j] = self.association_cost(new[i], old[j])
        for i in range(n_new):
            bestj = costs[i, :].argmin()
            if costs[i, bestj] < max_acceptable_cost:
                new[i].matched = old[bestj]
                costs[:, bestj] = 1 + max_acceptable_cost

    def detect_missing_objects(self):
        for obj in self.objects.values():
            if not isinstance(obj, (ArucoMarkerObj, BookObj, WallObj, DoorwayObj)) and obj not in self.updated_objects and self.should_be_visible(obj):
                if obj not in self.missing_objects:
                    obj.is_visible = False
                    obj.is_missing = True
                    self.missing_objects.append(obj)


def _migrate_world_map(robot):
    old = robot.world_map
    if isinstance(old, LibrarianWorldMap):
        return
    new_map = LibrarianWorldMap(robot)
    with old._lock:
        new_map.objects = old.objects
        new_map.pending_objects = old.pending_objects
        new_map.missing_objects = list(old.missing_objects)
        new_map.shared_objects = old.shared_objects
        new_map.name_counts = dict(old.name_counts)
        new_map.last_held_time = old.last_held_time
        new_map.visibility_paused = old.visibility_paused
    robot.world_map = new_map


__all__ = ["LibrarianWorldMap", "_migrate_world_map"]
