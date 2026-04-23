"""
Fixed shelf book-center slots in the world map frame (millimeters).

``slot_0``..``slot_3`` share the same *x* (``tag_x_mm``, typically ``_TAG_X`` in field programs).
*y* is the lateral book-center coordinate in mm: 67.5, 22.5, -22.5, -67.5.
"""

from __future__ import annotations

from math import pi

from aim_fsm.utils import Pose

from aim_librarian.books import BookObj

SHELF_SLOT_Y_MM: tuple[float, float, float, float] = (67.5, 22.5, -22.5, -67.5)

NUM_SHELF_SLOTS = 4

SHELF_SLOT_NAMES: tuple[str, str, str, str] = ("slot_0", "slot_1", "slot_2", "slot_3")


def book_center_pose(slot: int, *, tag_x_mm: float) -> Pose:
    """
    Book-center :class:`Pose` for the given index (``0``=``slot_0`` through ``3``=``slot_3``).
    *tag_x_mm* is the field constant used for the shelf line (e.g. ``_TAG_X``).
    """
    if not 0 <= slot < NUM_SHELF_SLOTS:
        raise ValueError(f"slot must be 0..{NUM_SHELF_SLOTS - 1}, got {slot!r}")
    z = BookObj.HEIGHT_MM / 2.0
    return Pose(float(tag_x_mm), SHELF_SLOT_Y_MM[slot], z, pi)


__all__ = [
    "NUM_SHELF_SLOTS",
    "SHELF_SLOT_NAMES",
    "SHELF_SLOT_Y_MM",
    "book_center_pose",
]
