"""
Librarian extensions for vex-aim-tools (books / spine ArUco).

Call :func:`install_librarian_extensions` once after ``StateMachineProgram`` / robot setup
and **before** ``start()`` (e.g. at the end of your program's ``__init__``).

Requires ``vex-aim-tools`` on ``PYTHONPATH`` with unmodified stock ``aim_fsm`` / ``viewer``.
"""

from __future__ import annotations

import sys
from typing import Any

_installed = False


def _patch_worldmap_viewer_aliases(LibrarianWorldMapViewer: type) -> None:
    """Point every live import of WorldMapViewer at the librarian viewer.

    ``runfsm`` starts the FSM on the asyncio thread; the world-map window is then
    created on the CLI thread via ``simple_cli._prime_default_viewers``, which calls
    ``WorldMapViewer(robot)`` using that module's global — often ``__main__.WorldMapViewer``
    when you run ``python simple_cli``. Patching only ``aim_fsm.program`` is not enough
    because ``from aim_fsm import *`` copied the class reference earlier.
    """

    import aim_fsm
    import aim_fsm.program as program_module
    import viewer.worldmap_viewer as worldmap_viewer_module

    program_module.WorldMapViewer = LibrarianWorldMapViewer
    worldmap_viewer_module.WorldMapViewer = LibrarianWorldMapViewer
    aim_fsm.WorldMapViewer = LibrarianWorldMapViewer
    for mod_name in ("simple_cli", "__main__"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "WorldMapViewer"):
            mod.WorldMapViewer = LibrarianWorldMapViewer


def _replace_stock_worldmap_viewer(robot: Any, LibrarianWorldMapViewer: type) -> None:
    """Swap an already-open stock viewer for :class:`LibrarianWorldMapViewer`."""
    from aim_librarian.viewer_ext import LibrarianWorldMapModel

    wv = getattr(robot, "worldmap_viewer", None)
    if not wv or wv is True:
        return
    model = getattr(wv, "model", None)
    if isinstance(model, LibrarianWorldMapModel):
        return
    try:
        wv.stop()
    except Exception:
        pass
    try:
        new_wv = LibrarianWorldMapViewer(robot)
        robot.worldmap_viewer = new_wv
        new_wv.start()
    except Exception as exc:  # pragma: no cover - Qt / headless
        print(f"[install_librarian_extensions] Could not replace world map viewer: {exc}")

__all__ = [
    "BOOK_FIRST_ID",
    "BOOK_LAST_ID",
    "BookObj",
    "is_book_aruco_id",
    "PilotToArucoMarker",
    "PilotToBook",
    "install_librarian_extensions",
]


def install_librarian_extensions(robot: Any, *, skip_viewer_bindings: bool = False) -> None:
    """Wire librarian world map, RRT, planner, pilot, particle, and optionally viewers."""
    global _installed
    from aim_librarian.landmark_ext import apply_landmark_extensions
    from aim_librarian.path_planner_ext import apply_path_planner_extensions
    from aim_librarian.particle_ext import apply_particle_extensions
    from aim_librarian.rrt_ext import apply_rrt_extensions
    from aim_librarian.viewer_ext import LibrarianWorldMapViewer
    from aim_librarian.wall_ext import apply_wall_extensions
    from aim_librarian.worldmap_ext import _migrate_world_map

    apply_rrt_extensions()
    apply_path_planner_extensions()
    apply_wall_extensions()
    apply_particle_extensions()
    apply_landmark_extensions()
    _migrate_world_map(robot)

    if not skip_viewer_bindings:
        _patch_worldmap_viewer_aliases(LibrarianWorldMapViewer)
        _replace_stock_worldmap_viewer(robot, LibrarianWorldMapViewer)

    _installed = True


def __getattr__(name: str):
    if name == "BOOK_FIRST_ID":
        from aim_librarian.books import BOOK_FIRST_ID as v

        return v
    if name == "BOOK_LAST_ID":
        from aim_librarian.books import BOOK_LAST_ID as v

        return v
    if name == "is_book_aruco_id":
        from aim_librarian.books import is_book_aruco_id as v

        return v
    if name == "BookObj":
        from aim_librarian.books import BookObj as v

        return v
    if name == "PilotToArucoMarker":
        from aim_librarian.pilot_ext import PilotToArucoMarker as v

        return v
    if name == "PilotToBook":
        from aim_librarian.pilot_ext import PilotToBook as v

        return v
    raise AttributeError(name)
