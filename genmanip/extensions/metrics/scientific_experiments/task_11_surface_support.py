"""Task11 geometric support helper; reads meshes and poses, never physics state writes.

MeshInfo.mesh is a reference WORLD mesh. Canonical vertices follow exactly the
existing get_current_mesh transform: inverse reference pose/scale, then current
rigid pose with unit destination scale. Baked asset scale stays in the geometry.
BVHs live in this canonical frame. World rays are transformed by inverse rigid
pose, preserving ray parameter in metres. Cache ownership is per metric instance;
the metric is reconstructed each episode. Epoch, object/mesh identity and complete
mesh/reference-transform digest additionally invalidate cached entries.
"""

from dataclasses import dataclass
import hashlib

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon


@dataclass
class RigidSurface:
    vertices: np.ndarray
    triangles: np.ndarray
    bvh: object
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def world_vertices(self):
        return self.vertices @ self.rotation.T + self.translation

    def cast_world_rays(self, rays):
        rays = np.asarray(rays, dtype=np.float64)
        local = np.empty_like(rays)
        local[:, :3] = (rays[:, :3] - self.translation) @ self.rotation
        local[:, 3:] = rays[:, 3:] @ self.rotation
        return self.bvh.cast_rays(o3d.core.Tensor(local.astype(np.float32)))[
            "t_hit"
        ].numpy()


class MeshSupportCache:
    def __init__(self):
        self.entries = {}
        self.epoch = None
        self.build_count = 0

    def bind_epoch(self, epoch):
        if self.epoch != epoch:
            self.entries.clear()
            self.epoch = epoch

    def get(self, uid, obj, info):
        vertices = np.asarray(info.mesh.vertices, dtype=np.float64)
        triangles = np.asarray(info.mesh.triangles, dtype=np.int32)
        scale = np.asarray(info.scale, dtype=np.float64)
        reference_quat = np.asarray(
            info.quat, dtype=np.float64
        )  # XYZW, as MeshInfo stores
        reference_trans = np.asarray(info.trans, dtype=np.float64)
        current_scale_array = np.asarray(obj.get_local_scale(), dtype=np.float64)
        if (
            current_scale_array.shape != (3,)
            or not np.isfinite(current_scale_array).all()
            or np.any(current_scale_array == 0)
        ):
            raise ValueError("Invalid current object scale")
        current_scale = tuple(current_scale_array)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 4
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not len(triangles)
            or not np.isfinite(vertices).all()
            or not np.isfinite(scale).all()
            or scale.shape != (3,)
            or np.any(scale == 0)
            or np.any(triangles < 0)
            or np.any(triangles >= len(vertices))
        ):
            raise ValueError("Invalid mesh/reference scale")
        digest = hashlib.sha256()
        for array in (vertices, triangles, scale, reference_quat, reference_trans):
            digest.update(np.ascontiguousarray(array).tobytes())
        identity = (id(obj), id(info), id(info.mesh), digest.hexdigest())
        prior = self.entries.get(uid)
        if prior is not None and prior[0] == identity and prior[1] != current_scale:
            # Existing get_current_mesh assumes scale_B=1; do not guess a new
            # scale transform when geometry has not been refreshed by the loader.
            raise ValueError("Object scale changed without a refreshed mesh cache")
        if prior is None or prior[0] != identity:
            canonical = (
                Rotation.from_quat(reference_quat)
                .inv()
                .apply(vertices - reference_trans)
                / scale
            )
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(canonical),
                o3d.utility.Vector3iVector(triangles),
            )
            bvh = o3d.t.geometry.RaycastingScene()
            bvh.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            self.entries[uid] = (
                identity,
                current_scale,
                canonical,
                triangles.copy(),
                bvh,
            )
            self.build_count += 1
        entry = self.entries[uid]
        position, quaternion_wxyz = obj.get_world_pose()
        position = np.asarray(position, dtype=np.float64)
        quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
        if (
            position.shape != (3,)
            or quaternion_wxyz.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion_wxyz).all()
        ):
            raise ValueError("Invalid current rigid pose")
        rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
        return RigidSurface(entry[2], entry[3], entry[4], rotation, position)


def surface_support_diagnostics(
    upper,
    lower,
    *,
    spacing=0.001,
    grid_shift=(0.0, 0.0),
    gap_tolerance=0.003,
    penetration_tolerance=0.003,
    min_coverage=0.95,
    min_patch_area=3e-6,
    max_grid_rays=250000,
):
    """Geometric near-contact predicate, not a contact-force sensor.

    Tolerances are explicit task parameters requiring runtime calibration.
    No upright requirement. Tray containment currently uses its outer XY hull.
    """
    a, b = upper.world_vertices, lower.world_vertices
    result = {"supported": False}
    try:
        a_outline = Polygon(a[ConvexHull(a[:, :2]).vertices, :2])
        b_outline = Polygon(b[ConvexHull(b[:, :2]).vertices, :2])
    except (QhullError, ValueError):
        result["reason"] = "invalid_projection"
        return result
    if a_outline.area <= 1e-12 or b_outline.area <= 1e-12:
        result["reason"] = "degenerate_projection"
        return result
    coverage = a_outline.intersection(b_outline).area / a_outline.area
    result["coverage"] = float(coverage)
    if coverage < min_coverage:
        result["reason"] = "outside_tray"
        return result
    lo, hi = a.min(axis=0), a.max(axis=0)
    x = (
        np.arange(np.floor(lo[0] / spacing), np.ceil(hi[0] / spacing) + 1)
        + grid_shift[0]
    )
    y = (
        np.arange(np.floor(lo[1] / spacing), np.ceil(hi[1] / spacing) + 1)
        + grid_shift[1]
    )
    if len(x) * len(y) > max_grid_rays:
        result["reason"] = "query_size_limit"
        return result
    xx, yy = np.meshgrid(x * spacing, y * spacing)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    n = len(xy)
    zlow = min(float(lo[2]), float(b[:, 2].min())) - 0.2
    zhigh = max(float(hi[2]), float(b[:, 2].max())) + 0.2
    up = np.column_stack([xy, np.full(n, zlow), np.zeros(n), np.zeros(n), np.ones(n)])
    down = np.column_stack(
        [xy, np.full(n, zhigh), np.zeros(n), np.zeros(n), -np.ones(n)]
    )
    uphit = upper.cast_world_rays(up)
    downhit = lower.cast_world_rays(down)
    valid = np.isfinite(uphit) & np.isfinite(downhit)
    if not valid.any():
        result["reason"] = "no_common_triangle_projection"
        return result
    # Convert hit parameters to float64 before subtracting large world heights.
    gap = (zlow + uphit[valid].astype(np.float64)) - (
        zhigh - downhit[valid].astype(np.float64)
    )
    eps = 1e-6
    contact = (gap >= -penetration_tolerance - eps) & (gap <= gap_tolerance + eps)
    patch = float(contact.sum() * spacing**2)
    worst = float(gap.min())
    penetration = worst < -penetration_tolerance - eps
    result.update(
        supported=bool(not penetration and patch >= min_patch_area),
        reason=(
            "penetration"
            if penetration
            else "supported" if patch >= min_patch_area else "no_contact_patch"
        ),
        closest_gap_m=worst,
        max_gap_m=float(gap.max()),
        max_penetration_m=max(0.0, -worst),
        valid_rays=int(valid.sum()),
        contact_ray_count=int(contact.sum()),
        contact_patch_area_m2=patch,
    )
    return result
