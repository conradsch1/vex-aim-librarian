"""Particle landmark list: index BookObj for SLAM labels (vex-aim-tools model unchanged)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PyQt6.QtCore import pyqtSlot

from aim_fsm.utils import Pose
from aim_fsm.worldmap import ArucoMarkerObj

from viewer.particle_model import (
    Item,
    LandmarkModel,
    _covariance_components,
    _flatten_vector,
    _kind_from_id as _base_kind_from_id,
    _label_from,
    _pose_components,
    _to_float,
)

from aim_librarian.books import BookObj

_landmark_patch_applied = False


def _librarian_kind_from_id(name: str, world_obj: Any) -> str:
    if isinstance(world_obj, BookObj):
        return "aruco"
    return _base_kind_from_id(name, world_obj)


class LibrarianLandmarkModel(LandmarkModel):
    def sync_from(self, particle_filter: Any, world_map: Any = None) -> None:
        entries: list[Item] = []
        seen_ids: set[str] = set()

        sensor_model = getattr(particle_filter, "sensor_model", None)
        pf_landmarks: Mapping[str, Any] | Iterable[tuple[str, Any]]
        if sensor_model is not None:
            pf_landmarks = getattr(sensor_model, "landmarks", {}) or {}
        else:
            pf_landmarks = {}

        world_objects_by_string = {}
        if world_map is not None:
            snapshot = getattr(world_map, "snapshot_objects", None)
            try:
                if callable(snapshot):
                    world_objects = snapshot() or {}
                else:
                    world_objects = dict(getattr(world_map, "objects", {}) or {})
            except Exception:
                world_objects = {}
        else:
            world_objects = {}

        for obj in world_objects.values():
            if isinstance(obj, ArucoMarkerObj):
                world_objects_by_string[obj.marker_string] = obj
            if isinstance(obj, BookObj):
                world_objects_by_string[obj.marker_string] = obj
            obj_name = getattr(obj, "name", None)
            if isinstance(obj_name, str) and obj_name:
                world_objects_by_string[obj_name] = obj
            obj_id = getattr(obj, "id", None)
            if isinstance(obj_id, str) and obj_id:
                world_objects_by_string[obj_id] = obj

        if hasattr(pf_landmarks, "items"):
            lm_iterable = pf_landmarks.items()
        else:
            lm_iterable = pf_landmarks

        for name, spec in lm_iterable:
            entry = self._build_entry(str(name), spec, world_objects_by_string.get(name), source="slam")
            if entry:
                entries.append(entry)
                seen_ids.add(entry["id"])

        self.beginResetModel()
        self._items = entries
        self.endResetModel()
        self._revision += 1
        self.countChanged.emit()
        self.revisionChanged.emit()

    @pyqtSlot(int, result="QVariant")
    def get(self, row: int):
        if row < 0 or row >= len(self._items):
            return None
        return dict(self._items[row])

    def _build_entry(self, name: str, spec: Any, world_obj: Any, source: str):
        label = _label_from(name, world_obj)
        seen = bool(getattr(world_obj, "is_visible", False)) if world_obj is not None else False
        marker_id = getattr(world_obj, "marker_id", None) if world_obj is not None else None
        length_mm = getattr(world_obj, "length", None) if world_obj is not None else None
        width_mm = getattr(world_obj, "height", None) if world_obj is not None else None

        x = y = theta = 0.0
        sigma_xx = sigma_xy = sigma_yy = 0.0

        if isinstance(spec, Pose):
            x, y, theta = _pose_components(spec)
        elif isinstance(spec, (tuple, list)) and len(spec) >= 3:
            mu, orient, sigma = spec[0], spec[1], spec[2]
            coords = _flatten_vector(mu)
            if len(coords) >= 2:
                x, y = coords[0], coords[1]
            orientation = _flatten_vector(orient)
            if orientation:
                theta = orientation[-1]
            sigma_xx, sigma_xy, sigma_yy = _covariance_components(sigma)
        else:
            x, y, theta = _pose_components(spec)

        if marker_id is None and name.startswith("ArucoMarker-"):
            marker_token = name.split("-", 1)[-1]
            marker_id = marker_token.split(".", 1)[0]
        if marker_id is None and name.startswith("Book-"):
            marker_token = name.split("-", 1)[-1]
            marker_id = marker_token.split(".", 1)[0]

        entry: Item = {
            "id": name,
            "kind": _librarian_kind_from_id(name, world_obj),
            "x": _to_float(x, 0.0),
            "y": _to_float(y, 0.0),
            "theta": _to_float(theta, 0.0),
            "label": label,
            "seen": seen,
            "source": source,
            "sigma_xx": sigma_xx,
            "sigma_xy": sigma_xy,
            "sigma_yy": sigma_yy,
            "length_mm": _to_float(length_mm, 0.0) if length_mm is not None else None,
            "width_mm": _to_float(width_mm, 0.0) if width_mm is not None else None,
            "marker_id": marker_id,
        }
        return entry


def apply_landmark_extensions() -> None:
    global _landmark_patch_applied
    if _landmark_patch_applied:
        return
    _landmark_patch_applied = True
    import viewer.particle_model as pm
    import viewer.particle_viewer as pv

    pm._kind_from_id = _librarian_kind_from_id
    pv.LandmarkModel = LibrarianLandmarkModel


__all__ = ["LibrarianLandmarkModel", "apply_landmark_extensions"]
