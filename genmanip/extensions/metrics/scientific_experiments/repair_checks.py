"""Task metric predicates; no custom core-runtime dependency."""

import math
import numpy as np


def valid_titration_completion(state):
    volume, hold = state.get("completion_volume_ml"), state.get(
        "completion_hold_seconds"
    )
    return (
        state.get("pbd_valid") is True
        and not state.get("pbd_error")
        and state.get("task_success") is True
        and isinstance(volume, (int, float))
        and math.isfinite(volume)
        and 15 <= volume <= 3165 / 187
        and isinstance(hold, (int, float))
        and math.isfinite(hold)
        and hold >= 3 - 1e-6
    )


def tube_in_slot(origin, axis, radius, slots):
    """Check a tube against actual aperture/seat geometry in rack-local metres."""
    origin = np.asarray(origin, float)
    axis = np.asarray(axis, float)
    if (
        not np.isfinite(origin).all()
        or not np.isfinite(axis).all()
        or np.linalg.norm(axis) == 0
    ):
        return False
    axis = axis / np.linalg.norm(axis)
    if axis[2] < math.cos(math.radians(20)):
        return False
    for slot in slots:
        seat, top = slot["seat_bottom_z_m"], slot["aperture_top_z_m"]
        if not seat - 0.001 <= origin[2] <= seat + 0.005:
            continue
        center = np.asarray(slot["center_xy_m"])
        points = [origin, origin + axis * ((top - origin[2]) / axis[2])]
        if all(
            np.linalg.norm(p[:2] - center) + radius / axis[2]
            <= slot["aperture_radius_m"] + 0.0005
            for p in points
        ):
            return True
    return False


def tube_inserted_in_rack(scene, tube_uid, rack_uid):
    import json
    from pxr import UsdGeom

    tube = scene.object_list[tube_uid].prim
    rack = scene.object_list[rack_uid].prim
    attr = tube.GetAttribute("repair:rack_geometry")
    if not attr:
        raise ValueError("Missing measured rack-slot geometry")
    geometry = json.loads(attr.Get())
    tm = np.asarray(UsdGeom.Xformable(tube).ComputeLocalToWorldTransform(0))
    rm = np.asarray(UsdGeom.Xformable(rack).ComputeLocalToWorldTransform(0))
    relative = tm @ np.linalg.inv(rm)
    world_up = tm[2, :3] / np.linalg.norm(tm[2, :3])
    if world_up[2] < math.cos(math.radians(20)):
        return False
    radius = 0.009 * max(
        np.linalg.norm(relative[0, :3]), np.linalg.norm(relative[1, :3])
    )
    return tube_in_slot(
        relative[3, :3], relative[2, :3], radius, geometry["rack"]["slots"]
    )


def runtime_ok(scene):
    root = scene.world.stage.GetPrimAtPath(f"/World/{scene.uuid}")
    attr = root.GetAttribute("repair:runtime_error") if root else None
    return bool(root and attr and not attr.Get())
