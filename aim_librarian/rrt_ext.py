"""Extend aim_fsm RRT with book obstacles (no changes inside vex-aim-tools).

Books and fixed ArUco markers use capped RRT inflation so PathPlanner's default
~15 mm margin does not block tight shelf navigation; tune ``BOOK_RRT_MAX_BOOK_INFLATION_MM``.
"""

from __future__ import annotations

from typing import Any, Callable

from math import pi

from aim_fsm.geometry import wrap_angle
from aim_fsm.rrt import RRT
from aim_fsm.worldmap import (
    BarrelObj,
    SportsBallObj,
    AprilTagObj,
    ArucoMarkerObj,
    WallObj,
    DoorwayObj,
    RoomObj,
)

from aim_librarian.books import BookObj

# PathPlanner / Pilot pass ``skinny_obstacle_inflation`` (~15 mm). Book rectangles
# already use spine × cover; adding the full margin makes adjacent slots overlap and
# blocks shelf approaches. Clamp book-only inflation here (mm); raise slightly if the
# planner clips real geometry.
BOOK_RRT_MAX_BOOK_INFLATION_MM = 0

_orig_generate_obstacles: Callable[..., None] | None = None
_orig_compute_bounding_box: Callable[..., Any] | None = None


def generate_book_obstacle(book, inflation=0):
    from aim_fsm import geometry
    from aim_fsm.rrt_shapes import Rectangle

    # Match librarian WorldMapView.qml: books use +90° yaw on the marker delegate (tagOnSpine).
    r = Rectangle(
        center=geometry.point(book.pose.x, book.pose.y),
        dimensions=[
            BookObj.SPINE_THICKNESS_MM + 2 * inflation,
            BookObj.COVER_WIDTH_MM + 2 * inflation,
        ],
        orient=wrap_angle(book.pose.theta + pi / 2),
    )
    r.obstacle_id = book.id
    return r


def _librarian_generate_obstacles(self, goal_object, obstacle_inflation=0, wall_inflation=0, doorway_adjustment=0):
    self.goal_obstacle = None
    obstacles = []
    for obj in self.robot.world_map.objects.values():
        if (not obj.is_obstacle) or obj.is_missing or (self.robot.holding is obj):
            continue
        if isinstance(obj, BarrelObj):
            obst = self.generate_barrel_obstacle(obj, obstacle_inflation)
        elif isinstance(obj, SportsBallObj):
            obst = self.generate_ball_obstacle(obj, obstacle_inflation)
        elif isinstance(obj, AprilTagObj):
            obst = self.generate_apriltag_obstacle(obj, obstacle_inflation)
        elif isinstance(obj, WallObj):
            obsts = self.generate_wall_obstacles(obj, obstacle_inflation, doorway_adjustment)
            obstacles += obsts
            obst = None
        elif isinstance(obj, ArucoMarkerObj):
            # Fixed corner ArUco boxes: PathPlanner/Pilot pass large ``obstacle_inflation``
            # (e.g. 15 mm). Inflated marker rectangles overlap shelf approach lanes and cause
            # GoalCollides when routing to books near the shelf line. Use the physical tag
            # footprint only (no inflation margin).
            obst = self.generate_aruco_obstacle(obj, 0)
        elif isinstance(obj, BookObj):
            book_infl = min(obstacle_inflation, BOOK_RRT_MAX_BOOK_INFLATION_MM)
            obst = generate_book_obstacle(obj, book_infl)
        elif isinstance(obj, DoorwayObj):
            obst = None
        else:
            print("*** Can't generate obstacle for", obj)
            obst = None
        if obj is goal_object:
            self.goal_obstacle = obst
        elif obst is not None:
            obstacles.append(obst)
    self.obstacles = obstacles


def _librarian_compute_bounding_box(self):
    xmin = self.robot.pose.x
    ymin = self.robot.pose.y
    xmax = xmin
    ymax = ymin
    objs = self.robot.world_map.objects.values()
    rooms = [self.generate_room_obstacle(obj) for obj in objs if isinstance(obj, RoomObj)]
    # ArUco markers and books are already represented in self.obstacles; omitting them here
    # avoids double-counting in bbox (and matched duplicate goal shapes).
    non_obstacles = rooms
    goals = [self.goal_obstacle] if self.goal_obstacle else []
    for shape in self.obstacles + non_obstacles + goals:
        ((x0, y0), (x1, y1)) = shape.get_bounding_box()
        xmin = min(xmin, x0)
        ymin = min(ymin, y0)
        xmax = max(xmax, x1)
        ymax = max(ymax, y1)
    self.bbox = ((xmin, ymin), (xmax, ymax))
    return self.bbox


def apply_rrt_extensions() -> None:
    global _orig_generate_obstacles, _orig_compute_bounding_box
    if _orig_generate_obstacles is not None:
        return
    _orig_generate_obstacles = RRT.generate_obstacles
    _orig_compute_bounding_box = RRT.compute_bounding_box
    RRT.generate_book_obstacle = staticmethod(generate_book_obstacle)
    RRT.generate_obstacles = _librarian_generate_obstacles
    RRT.compute_bounding_box = _librarian_compute_bounding_box


__all__ = ["apply_rrt_extensions", "generate_book_obstacle"]
