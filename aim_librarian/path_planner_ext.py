"""Add BookObj goal handling to PathPlanner.setup_problem."""

from __future__ import annotations

from typing import Any, Callable, Tuple

from aim_fsm.path_planner import PathPlanner
from aim_fsm.rrt import RRT, RRTNode
from aim_fsm.worldmap import (
    BarrelObj,
    SportsBallObj,
    AprilTagObj,
    DoorwayObj,
    ArucoMarkerObj,
    RoomObj,
)

from aim_librarian.books import BookObj

_orig_setup_problem: Callable[..., Any] | None = None


def _librarian_setup_problem(goal_object, robot, use_doorways=True):
    robot.rrt.generate_obstacles(
        goal_object,
        PathPlanner.fat_obstacle_inflation,
        PathPlanner.fat_wall_inflation,
        PathPlanner.fat_doorway_adjustment,
    )
    fat_obstacles = robot.rrt.obstacles

    robot.rrt.generate_obstacles(
        goal_object,
        PathPlanner.skinny_obstacle_inflation,
        PathPlanner.skinny_wall_inflation,
        PathPlanner.skinny_doorway_adjustment,
    )
    skinny_obstacles = robot.rrt.obstacles

    start_node = RRTNode(x=robot.pose.x, y=robot.pose.y, q=robot.pose.theta)

    if isinstance(goal_object, BarrelObj):
        goal_shape = RRT.generate_barrel_obstacle(goal_object, 0)
    elif isinstance(goal_object, SportsBallObj):
        goal_shape = RRT.generate_ball_obstacle(goal_object, 0)
    elif isinstance(goal_object, AprilTagObj):
        goal_shape = RRT.generate_apriltag_obstacle(goal_object, 0)
    elif isinstance(goal_object, DoorwayObj):
        goal_shape = RRT.generate_doorway_obstacle(goal_object, 0)
    elif isinstance(goal_object, ArucoMarkerObj):
        goal_shape = RRT.generate_aruco_obstacle(goal_object, 0)
    elif isinstance(goal_object, BookObj):
        goal_shape = RRT.generate_book_obstacle(goal_object, 0)
    elif isinstance(goal_object, RoomObj):
        goal_shape = RRT.generate_room_obstacle(goal_object)
    else:
        raise ValueError("Can't convert path planner goal %s to shape." % goal_object)

    robot.rrt.goal_obstacle = goal_shape
    bbox = robot.rrt.compute_bounding_box()

    robot_parts = robot.rrt.make_robot_parts(robot)

    if use_doorways:
        doorway_list = robot.world_map.generate_doorway_list()
    else:
        doorway_list = []

    need_grid_display = robot.path_viewer is not None

    return (
        start_node,
        goal_shape,
        robot_parts,
        bbox,
        fat_obstacles,
        skinny_obstacles,
        doorway_list,
        need_grid_display,
    )


def apply_path_planner_extensions() -> None:
    global _orig_setup_problem
    if _orig_setup_problem is not None:
        return
    _orig_setup_problem = PathPlanner.__dict__["setup_problem"].__func__
    PathPlanner.setup_problem = staticmethod(_librarian_setup_problem)


__all__ = ["apply_path_planner_extensions"]
