"""World map viewer: QML + Qt model with book type (vex-aim-tools stays generic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QUrl

from viewer.worldmap_model import Item, WorldMapModel, _pose_attr, _theta_attr, _to_float
from viewer.worldmap_viewer import WorldMapViewer

from aim_librarian.books import BookObj


def _is_book_like(obj: Any) -> bool:
    """True for :class:`BookObj` even if two copies of the class exist (import path)."""
    if isinstance(obj, BookObj):
        return True
    if str(getattr(obj, "name", "") or "").startswith("Book-"):
        return True
    return type(obj).__name__ == "BookObj"


def _book_dimensions_mm(obj: Any) -> tuple[float, float, float]:
    """Cover width, height, spine thickness (mm)."""
    t = type(obj)
    w = getattr(t, "COVER_WIDTH_MM", None)
    h = getattr(t, "HEIGHT_MM", None)
    th = getattr(t, "SPINE_THICKNESS_MM", None)
    if w is not None and h is not None and th is not None:
        return float(w), float(h), float(th)
    return (
        float(BookObj.COVER_WIDTH_MM),
        float(BookObj.HEIGHT_MM),
        float(BookObj.SPINE_THICKNESS_MM),
    )


class LibrarianWorldMapModel(WorldMapModel):
    """Adds ``book`` type for :class:`BookObj` from this package."""

    @staticmethod
    def _resolve_type(obj: Any) -> Optional[str]:
        if _is_book_like(obj):
            return "book"
        return WorldMapModel._resolve_type(obj)

    def _build_object(self, object_id: str, obj: Any) -> Optional[Item]:
        if _is_book_like(obj):
            cover_w, height_mm, spine_t = _book_dimensions_mm(obj)
            entry: Item = {
                "id": object_id,
                "type": "book",
                "x": _pose_attr(obj, "x", 0.0),
                "y": _pose_attr(obj, "y", 0.0),
                "z": _pose_attr(obj, "z", 0.0),
                "theta": _theta_attr(obj),
                "visible": bool(getattr(obj, "is_visible", False)),
                "missing": bool(getattr(obj, "is_missing", False)),
                "diameter_mm": None,
                "height_mm": None,
                "length_mm": None,
                "width_mm": None,
                "thickness_mm": None,
                "size_mm": None,
                "marker_id": None,
                "doorways": [],
                "holding": None,
            }
            entry["width_mm"] = cover_w
            entry["height_mm"] = height_mm
            entry["thickness_mm"] = spine_t
            entry["marker_id"] = getattr(obj, "marker_id", None)
            entry["size_mm"] = max(cover_w, spine_t)
            entry["z"] = height_mm / 2.0
            return entry
        return super()._build_object(object_id, obj)


class LibrarianWorldMapViewer(WorldMapViewer):
    """Uses :class:`LibrarianWorldMapModel` and librarian ``qml/WorldMapView.qml``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model = LibrarianWorldMapModel()
        ctx = self._view.rootContext()
        ctx.setContextProperty("worldModel", self._model)
        qml_path = Path(__file__).resolve().parent / "qml" / "WorldMapView.qml"
        engine = self._view.engine()
        engine.addImportPath(str(qml_path.parent))
        self._view.setSource(QUrl.fromLocalFile(str(qml_path)))
        self.refresh()


__all__ = ["LibrarianWorldMapModel", "LibrarianWorldMapViewer"]
