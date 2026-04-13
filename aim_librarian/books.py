"""Book world objects (ArUco on spine). Kept in vex-aim-librarian; vex-aim-tools unchanged."""

from aim_fsm.worldmap import WorldObject

_MM_PER_INCH = 25.4

# ArUco IDs in [BOOK_FIRST_ID, BOOK_LAST_ID] are modeled as books; see worldmap_ext.
BOOK_FIRST_ID = 9
BOOK_LAST_ID = 15


def is_book_aruco_id(marker_id: int) -> bool:
    return BOOK_FIRST_ID <= marker_id <= BOOK_LAST_ID


class BookObj(WorldObject):
    """1.5\" × 2.5\" × 0.5\" prism; marker on spine (viewer: ±Y faces)."""

    SPINE_THICKNESS_MM = 1 * _MM_PER_INCH
    COVER_WIDTH_MM = 3 * _MM_PER_INCH
    HEIGHT_MM = 4 * _MM_PER_INCH

    def __init__(self, spec, x=0, y=0, z=0, theta=0, **kwargs):
        super().__init__(x=x, y=y, z=z, theta=theta, **kwargs)
        self.name = spec["name"]
        self.marker = spec["marker"]
        self.marker_id = spec["id"]
        self.marker_string = "Book-" + str(spec["id"])
        self.pose_confidence = +1

    def __repr__(self):
        import math

        if self.pose_confidence >= 0:
            vis = "visible" if self.is_visible else "missing" if self.is_missing else "unseen"
            return "<BookObj %s: (%.1f, %.1f, %.1f) @ %d deg. %s>" % (
                self.marker_id,
                self.pose.x,
                self.pose.y,
                self.pose.z,
                self.pose.theta * 180 / math.pi,
                vis,
            )
        return f"<BookObj {self.marker_id}: position unknown>"


__all__ = ["BookObj", "BOOK_FIRST_ID", "BOOK_LAST_ID", "is_book_aruco_id"]
