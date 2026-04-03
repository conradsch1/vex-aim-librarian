"""Wall alignment check includes book markers."""

from __future__ import annotations

from math import pi
from typing import Any, Callable

from aim_fsm.geometry import wrap_angle
from aim_fsm.worldmap import WallObj, ArucoMarkerObj

from aim_librarian.books import BookObj

_orig_is_wall_aligned: Callable[..., Any] | None = None
_wall_patch_applied = False


def _librarian_is_wall_aligned(self, obj):
    result = abs(wrap_angle(self.sensor_orient - obj.sensor_orient)) < self.ALIGNMENT_THRESHOLD or (
        isinstance(obj, (ArucoMarkerObj, BookObj))
        and abs(wrap_angle(self.sensor_orient + pi - obj.sensor_orient)) < self.ALIGNMENT_THRESHOLD
    )
    return result


def apply_wall_extensions() -> None:
    global _orig_is_wall_aligned, _wall_patch_applied
    if _wall_patch_applied:
        return
    _wall_patch_applied = True
    _orig_is_wall_aligned = WallObj.is_wall_aligned
    WallObj.is_wall_aligned = _librarian_is_wall_aligned


__all__ = ["apply_wall_extensions"]
