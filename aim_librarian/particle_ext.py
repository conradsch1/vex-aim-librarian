"""SLAM sensor: treat BookObj like ArucoMarkerObj for landmark geometry."""

from __future__ import annotations

from typing import Any, Callable

from aim_fsm.particle import SLAMSensorModel, ParticleFilter
from aim_fsm.worldmap import ArucoMarkerObj, WallObj

from aim_librarian.books import BookObj

_orig_process_landmark: Callable[..., Any] | None = None
_particle_patch_applied = False


def _librarian_process_landmark(self, obj, just_looking):
    from math import pi

    import numpy as np

    particles = self.robot.particle_filter.particles
    if not self.landmark_test(obj):
        return False
    if isinstance(obj, (ArucoMarkerObj, BookObj, WallObj)):
        sensor_dist = obj.sensor_distance
        sensor_bearing = obj.sensor_bearing
        sensor_orient = obj.sensor_orient
    else:
        print("Don't know how to process landmark; id =", getattr(obj, "id", obj))
        return False

    lm_id = obj.id
    if lm_id not in self.landmarks:
        if self.pf.state == ParticleFilter.LOCALIZED:
            print(
                "  *** PF ADDING LANDMARK %s at:  distance=%6.1f  bearing=%5.1f deg.  orient=%5.1f deg."
                % (lm_id, sensor_dist, sensor_bearing * 180 / pi, sensor_orient * 180 / pi)
            )
            for p in particles:
                p.add_regular_landmark(lm_id, sensor_dist, sensor_bearing, sensor_orient)
            self.landmarks[lm_id] = self.pf.best_particle.landmarks[lm_id]
        return False

    if just_looking:
        return False
    pp = particles
    evaluated = True

    should_update_landmark = (not obj.is_fixed) and (self.pf.state == ParticleFilter.LOCALIZED)

    for p in pp:
        sensor_direction = p.theta + sensor_bearing
        dx = sensor_dist * np.cos(sensor_direction)
        dy = sensor_dist * np.sin(sensor_direction)
        predicted_lm_x = p.x + dx
        predicted_lm_y = p.y + dy
        (lm_mu, lm_orient, lm_sigma) = p.landmarks[lm_id]
        map_lm_x = lm_mu[0, 0]
        map_lm_y = lm_mu[1, 0]
        error_x = map_lm_x - predicted_lm_x
        error_y = map_lm_y - predicted_lm_y
        error1_sq = error_x**2 + error_y**2
        error2_sq = 0
        p.log_weight -= (error1_sq + error2_sq) / self.distance_variance
        if should_update_landmark:
            p.update_regular_landmark(lm_id, sensor_dist, sensor_bearing, sensor_orient, dx, dy)
    return evaluated


def apply_particle_extensions() -> None:
    global _orig_process_landmark, _particle_patch_applied
    if _particle_patch_applied:
        return
    _particle_patch_applied = True
    _orig_process_landmark = SLAMSensorModel.process_landmark
    SLAMSensorModel.process_landmark = _librarian_process_landmark


__all__ = ["apply_particle_extensions"]
