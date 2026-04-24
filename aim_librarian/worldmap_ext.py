"""Librarian WorldMap: ArUco ids in the book range become BookObj."""

from __future__ import annotations

import math
import time

import numpy as np

from aim_fsm.worldmap import (
    WorldMap,
    ArucoMarkerObj,
    WallObj,
    DoorwayObj,
    AprilTagObj,
)
from aim_fsm.aim_kin import AIMKinematics
from aim_fsm.utils import PoseEstimate
from aim_fsm.geometry import wrap_angle

from aim_librarian.books import BookObj, is_book_aruco_id


def _ensure_bookobj_synthetic_at_robot(robot, marker_id: int) -> BookObj | None:
    """Insert a :class:`BookObj` when the spine is not in the Aruco snapshot (e.g. too close
    to the camera after ``Forward(ENGAGE_MM)``). Pose is the spine centroid from robot
    center along heading, at magnet-touch depth (same components as ``BOOK_APPROACH_OFFSET``
    minus the pre-engage gap).
    """
    wm = robot.world_map
    name = f"Book-{marker_id}"
    spec = {"name": name, "id": marker_id, "marker": None}
    obj = BookObj(spec)
    # After engage, marker centroid ≈ this far ahead of robot origin along +heading.
    d = AIMKinematics.body_diameter / 2.0 + BookObj.SPINE_THICKNESS_MM / 2.0
    th = robot.pose.theta
    obj.pose = PoseEstimate(
        robot.pose.x + d * math.cos(th),
        robot.pose.y + d * math.sin(th),
        BookObj.HEIGHT_MM / 2,
        wrap_angle(th + math.pi),
    )
    obj.is_visible = True

    with wm._lock:
        for o in wm.objects.values():
            if isinstance(o, BookObj) and o.marker_id == marker_id:
                return o
        obj_id = wm.next_in_sequence(name)
        wm.objects[obj_id] = obj
    print(
        f"ensure_bookobj_from_vision: inserted {obj_id} from robot geometry "
        "(spine not in camera FOV)"
    )
    return obj


def ensure_bookobj_from_vision(robot, marker_id: int) -> BookObj | None:
    """Ensure ``world_map.objects`` has a :class:`BookObj` for ``marker_id`` so attach can run.

    Tries, in order:

    1. Return existing map entry.
    2. If the spine is in the current Aruco snapshot, insert using the same pose math as
       :meth:`LibrarianWorldMap.make_new_aruco_objects`.
    3. Otherwise insert a synthetic book at magnet depth along the robot heading — after
       ``Forward(ENGAGE_MM)`` the tag is often **too close** to see, so step 2 fails even
       though the approach succeeded.

    The map normally waits for several frames via ``process_unassociated_objects`` before
    adding a book; :class:`~aim_librarian.book_manip.AttachBook` only looks at ``objects``.
    """
    wm = robot.world_map
    with wm._lock:
        for obj in wm.objects.values():
            if isinstance(obj, BookObj) and obj.marker_id == marker_id:
                return obj

    if not is_book_aruco_id(marker_id):
        return None

    det = getattr(robot, "aruco_detector", None)
    seen_markers: dict = {}
    if det is not None:
        if hasattr(det, "snapshot_seen_markers"):
            seen_markers = det.snapshot_seen_markers()
        else:
            seen_markers = det.seen_marker_objects.copy()

    if marker_id not in seen_markers:
        return _ensure_bookobj_synthetic_at_robot(robot, marker_id)

    marker = seen_markers[marker_id]

    camera_offset_vector = np.array([0, 0, robot.kine.camera_from_origin])
    sensor_coords = marker.camera_coords + camera_offset_vector
    sensor_distance = math.sqrt(sensor_coords[0] ** 2 + sensor_coords[2] ** 2)
    sensor_bearing = math.atan2(sensor_coords[0], sensor_coords[2])
    sensor_orient = wrap_angle(math.pi - marker.euler_angles[1])
    theta = robot.pose.theta
    name = f"Book-{marker_id}"
    spec = {"name": name, "id": marker_id, "marker": marker}
    obj = BookObj(spec)
    obj.pose = PoseEstimate(
        robot.pose.x + sensor_distance * math.cos(theta + sensor_bearing),
        robot.pose.y + sensor_distance * math.sin(theta + sensor_bearing),
        BookObj.HEIGHT_MM / 2,
        wrap_angle(robot.pose.theta + sensor_orient),
    )
    obj.sensor_distance = sensor_distance
    obj.sensor_bearing = sensor_bearing
    obj.sensor_orient = sensor_orient
    obj.is_visible = True

    with wm._lock:
        for o in wm.objects.values():
            if isinstance(o, BookObj) and o.marker_id == marker_id:
                return o
        obj_id = wm.next_in_sequence(name)
        wm.objects[obj_id] = obj
    print(f"ensure_bookobj_from_vision: inserted {obj_id} from snapshot so attach can proceed")
    return obj


def prune_aruco_markers_in_book_id_range(world_map: WorldMap) -> None:
    """Drop legacy :class:`ArucoMarkerObj` entries whose ids are modeled as books.

    Spine markers in ``BOOK_FIRST_ID``…``BOOK_LAST_ID`` are represented by
    :class:`BookObj` only. Old maps may still contain a duplicate ``ArucoMarker``
    for the same id (different Python type), which breaks association and adds a
    second obstacle/goal in the path viewer.
    """
    with world_map._lock:
        remove_keys = [
            key
            for key, obj in world_map.objects.items()
            if isinstance(obj, ArucoMarkerObj) and is_book_aruco_id(obj.marker_id)
        ]
        removed: list = []
        for key in remove_keys:
            removed.append(world_map.objects.pop(key))
        for obj in removed:
            if obj in world_map.missing_objects:
                world_map.missing_objects.remove(obj)


class LibrarianWorldMap(WorldMap):
    def confirm_still_holding(self):
        # Stock logic only tracks kicker-held barrels/balls; magnet-held books would be cleared.
        if isinstance(self.robot.holding, BookObj):
            self.last_held_time = time.time()
            return
        super().confirm_still_holding()

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
            obj.pose = PoseEstimate(
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
        prune_aruco_markers_in_book_id_range(old)
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
    prune_aruco_markers_in_book_id_range(new_map)


__all__ = [
    "LibrarianWorldMap",
    "_migrate_world_map",
    "ensure_bookobj_from_vision",
    "prune_aruco_markers_in_book_id_range",
]
