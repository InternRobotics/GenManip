"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Any, Literal

import numpy as np
import open3d as o3d
from pydantic import BaseModel, Field
import scipy
from scipy.spatial import QhullError
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from sklearn.neighbors import NearestNeighbors

from omni.isaac.core.articulations import Articulation  # type: ignore

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.utils.pointcloud.pointcloud import get_current_pointCloud
from genmanip.utils.pointcloud.utils import PointCloudInfo
from genmanip.utils.usd_utils import get_pcd_from_mesh
from genmanip.utils.standalone.object_utils.shelf import extract_shelf_planes, Shelf
from genmanip.utils.standalone.pc_utils import get_xy_contour
from genmanip.utils.standalone.utils import compare_articulation_status

XY_DISTANCE_CLOSE_THRESHOLD = 0.15
MAX_TO_BE_TOUCHING_DISTANCE = 0.1
MIN_ABOVE_BELOW_DISTANCE = 0.05
MAX_TO_BE_SUPPORTING_AREA_RATIO = 1.8
MIN_TO_BE_SUPPORTED_AREA_RATIO = 0.4
MIN_TO_BE_ABOVE_BELOW_AREA_RATIO = 0.1
INSIDE_PROPORTION_THRESH = 0.5
ANGLE_THRESHOLD = 45


class SRBasedGenmanipRelationshipConfig(BaseModel):
    support_mode: Literal["legacy", "contact"] = "legacy"
    support_min_overlap: float = Field(default=0.6, gt=0, le=1, allow_inf_nan=False)
    support_max_gap: float = Field(default=0.015, ge=0, allow_inf_nan=False)
    support_max_penetration: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    support_container_uid: str | None = None
    support_container_min_overlap: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    obj1_uid: str = Field(
        ..., description="UID of the object to measure the relationship"
    )
    obj2_uid: str | None = Field(
        default=None, description="UID of the object to measure the relationship"
    )
    position: str | None = Field(
        default=None, description="Position of the object to measure the relationship"
    )
    another_obj2_uid: str | None = Field(
        default=None,
        description="UID of the another object to measure the relationship",
    )
    status: list | None = Field(
        default=None, description="Status of the object to measure the relationship"
    )


@MetricFactory.register("manip/default/sr_based_genmanip_relationship")
class SRBasedGenmanipRelationship(BaseMetric):
    def __init__(
        self,
        skip_steps: int = 1,
        succ_cnts: int = 0,
        sub_goal_setting: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.goal_setting = SRBasedGenmanipRelationshipConfig(**sub_goal_setting)

    @staticmethod
    def get_target_pc_list(scene, target_uids):
        # OPT: per-frame cache populated by MetricsManager.step keeps the
        # transformed point cloud for each uid so that multiple metrics
        # sharing the same target (e.g. 7 beans vs 1 jar) do not retransform
        # the jar point cloud 7 times per tick.
        frame_cache = getattr(scene, "_frame_pc_cache", None)
        pc_list = {}
        for uid in target_uids:
            if uid not in scene.object_list:
                continue
            if frame_cache is not None and uid in frame_cache:
                pc_list[uid] = frame_cache[uid]
                continue
            if uid not in scene.cache_library.pointcloud_dict:
                if uid in scene.cache_library.mesh_dict:
                    mesh_info = scene.cache_library.mesh_dict[uid]
                    num_points = (
                        100000 if uid == "00000000000000000000000000000000" else 10000
                    )
                    try:
                        pcd = get_pcd_from_mesh(mesh_info.mesh, num_points=num_points)
                        points = np.asarray(pcd.points)
                        pc_info = PointCloudInfo(
                            points=points,
                            scale=mesh_info.scale,
                            trans=mesh_info.trans,
                            quat=mesh_info.quat,
                        )
                        scene.cache_library.pointcloud_dict[uid] = pc_info
                    except (RuntimeError, ValueError, AttributeError) as e:
                        print(f"Failed to generate point cloud for {uid}: {e}")
                        continue
            if uid in scene.cache_library.pointcloud_dict:
                pc_info = scene.cache_library.pointcloud_dict[uid]
                transformed = get_current_pointCloud(scene.object_list[uid], pc_info)
                pc_list[uid] = transformed
                if frame_cache is not None:
                    frame_cache[uid] = transformed
        return pc_list

    def check_status(self, scene):
        articulation_list = scene.articulation_list
        if self.goal_setting.position is not None:
            target_uids = [self.goal_setting.obj1_uid]
            if self.goal_setting.obj2_uid:
                target_uids.append(self.goal_setting.obj2_uid)
            if self.goal_setting.another_obj2_uid:
                target_uids.append(self.goal_setting.another_obj2_uid)
            if self.goal_setting.support_container_uid:
                target_uids.append(self.goal_setting.support_container_uid)

            pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

            if self.goal_setting.another_obj2_uid is not None:
                pcd3 = pclist[self.goal_setting.another_obj2_uid]
            else:
                pcd3 = None
            if self.goal_setting.obj2_uid is None:
                raise ValueError("obj2_uid is required for position relationship")
            if (self.goal_setting.support_mode == "contact"
                    and self.goal_setting.position in ("top", "on")):
                return support_contact_diagnostics(
                    pclist[self.goal_setting.obj1_uid],
                    pclist[self.goal_setting.obj2_uid],
                    min_overlap=self.goal_setting.support_min_overlap,
                    max_gap=self.goal_setting.support_max_gap,
                    max_penetration=self.goal_setting.support_max_penetration,
                    container=pclist[self.goal_setting.support_container_uid]
                    if self.goal_setting.support_container_uid else None,
                    min_container_overlap=self.goal_setting.support_container_min_overlap,
                )["supported"]
            check_status_flag = check_subgoal_finished_rigid(
                self.goal_setting.position,
                pclist[self.goal_setting.obj1_uid],
                pclist[self.goal_setting.obj2_uid],
                pcd3,
                scene=scene,
            )
        elif self.goal_setting.status is not None:
            check_status_flag = check_subgoal_finished_articulation(
                self.goal_setting.status, articulation_list[self.goal_setting.obj1_uid]
            )
        else:
            check_status_flag = False
        return check_status_flag

    def get_info(self):
        suffix = " (contact support)" if self.goal_setting.support_mode == "contact" else ""
        if self.goal_setting.support_mode == "contact" and self.goal_setting.support_container_uid:
            suffix = f" (contact support; footprint in {self.goal_setting.support_container_uid})"
        return f"put {self.goal_setting.obj1_uid} to the {self.goal_setting.position} of {self.goal_setting.obj2_uid}{suffix}"


def support_contact_diagnostics(
    upper: np.ndarray,
    lower: np.ndarray,
    min_overlap: float = 0.6,
    max_gap: float = 0.015,
    max_penetration: float = 0.01,
    container: np.ndarray | None = None,
    min_container_overlap: float = 0.95,
) -> dict:
    """Geometric support proxy for overhanging stacks, not a contact sensor.

    Compare local surfaces in the overlapping footprint, so a plate rim or
    overhanging leaf does not define the support height. Legacy top/in behavior
    remains unchanged unless a goal explicitly selects contact mode.
    """
    result = {"supported": False}
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 4
           or not np.isfinite(p).all() for p in (upper, lower)):
        return result
    try:
        upper_outline, lower_outline = [
            Polygon(p[scipy.spatial.ConvexHull(p[:, :2]).vertices, :2])
            for p in (upper, lower)
        ]
    except (QhullError, ValueError):
        return result
    smaller_area = min(upper_outline.area, lower_outline.area)
    if smaller_area <= 1e-10:
        return result
    if container is not None:
        if (container.ndim != 2 or container.shape[1] != 3 or len(container) < 4
                or not np.isfinite(container).all()):
            return result
        try:
            boundary = Polygon(container[scipy.spatial.ConvexHull(container[:, :2]).vertices, :2])
        except (QhullError, ValueError):
            return result
        coverage = upper_outline.intersection(boundary).area / upper_outline.area
        result["container_coverage"] = float(coverage)
        if coverage < min_container_overlap:
            return result
    overlap = upper_outline.intersection(lower_outline)
    ratio = overlap.area / smaller_area
    result.update(overlap_ratio=float(ratio))
    if ratio < min_overlap:
        return result
    region = overlap.buffer(0.001)
    upper_region = upper[contains(region, upper[:, 0], upper[:, 1])]
    lower_region = lower[contains(region, lower[:, 0], lower[:, 1])]
    if min(len(upper_region), len(lower_region)) < 4:
        return result
    upper_z, lower_z = upper_region[:, 2], lower_region[:, 2]
    center_dz = float((upper_z.min() + upper_z.max() - lower_z.min() - lower_z.max()) / 2)
    # Pair surfaces in 1 cm XY cells; extrema at opposite ends of a tilted
    # stack are not a penetration measurement. Require a robust contact patch.
    columns = []
    for points in (upper_region, lower_region):
        cells = np.floor(points[:, :2] / 0.01).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        order = np.argsort(inverse)
        groups = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
        columns.append({tuple(cells[g[0]]): np.quantile(points[g, 2], [.05, .95])
                        for g in groups if len(g) >= 3})
    common = columns[0].keys() & columns[1].keys()
    if len(common) < 3:
        return result
    gaps = np.array([columns[0][cell][0] - columns[1][cell][1] for cell in common])
    gap = float(np.quantile(gaps, .1))
    result.update(center_dz=center_dz, surface_gap=gap, matched_cells=len(common),
                  supported=bool(center_dz > 0.001 and -max_penetration <= gap <= max_gap))
    return result


def assign_point_to_shelf(point: np.ndarray | list[float], shelves: list[Shelf]) -> int:
    point_xyz = np.array(point)
    point_x, point_y, point_z = point_xyz[:3]
    for i, shelf in enumerate(shelves):
        if shelf.z_min < point_z <= shelf.z_max:
            point_xy = (point_x, point_y)
            if shelf.boundary_polygon.contains(Point(point_xy)):
                return i + 1
    return 0


def _cached_kdtree(points: np.ndarray, scene=None):
    # Per-frame cache keyed on id(points); points come out of the shared
    # _frame_pc_cache so the id is stable across sibling metrics.
    tree_cache = None
    if scene is not None:
        tree_cache = getattr(scene, "_frame_tree_cache", None)
    key = (id(points), points.shape[1])
    if tree_cache is not None and key in tree_cache:
        return tree_cache[key]
    tree = scipy.spatial.cKDTree(points)
    if tree_cache is not None:
        tree_cache[key] = tree
    return tree


def calculate_distance_between_two_point_clouds(
    point_cloud_a: np.ndarray, point_cloud_b: np.ndarray, scene=None
) -> float:
    # OPT: scipy.cKDTree is ~3-5x faster than sklearn NearestNeighbors for
    # this workload, and the tree is cached per-frame so sibling metrics
    # hitting the same target share it.
    tree = _cached_kdtree(point_cloud_a, scene=scene)
    distances, _ = tree.query(point_cloud_b, k=1)
    return float(np.min(distances))


def calculate_xy_distance_between_two_point_clouds(
    point_cloud_a: np.ndarray, point_cloud_b: np.ndarray, scene=None
) -> float:
    pa = np.ascontiguousarray(point_cloud_a[:, :2])
    pb = np.ascontiguousarray(point_cloud_b[:, :2])
    tree = _cached_kdtree(pa, scene=scene)
    distances, _ = tree.query(pb, k=1)
    return float(np.min(distances))


def check_subgoal_finished_articulation(
    status_list: list[list[float]], articulation: Articulation
) -> bool:
    articulation_status = articulation._articulation_view.get_joints_state().positions
    if any(
        compare_articulation_status(articulation_status.tolist(), status)
        for status in status_list
    ):
        return True
    return False


def crop_pcd(pcd1: np.ndarray, pcd2: np.ndarray) -> np.ndarray:
    contour1 = get_xy_contour(pcd1, contour_type="concave_hull").buffer(0.05)
    xy_points = pcd2[:, :2]
    mask = contains(contour1, xy_points[:, 0], xy_points[:, 1])
    return pcd2[mask]


def check_subgoal_finished_rigid(
    position: str,
    pcd1: np.ndarray,
    pcd2: np.ndarray,
    pcd3: np.ndarray | None = None,
    scene=None,
) -> bool:
    relation_list = get_related_position(pcd1, pcd2, pcd3, scene=scene)
    if position == "top" or position == "on":
        croped_pcd2 = crop_pcd(pcd1, pcd2)
        if len(croped_pcd2) > 0:
            relation_list_2 = get_related_position(pcd1, croped_pcd2, scene=scene)
            if "on" in relation_list_2:
                return True
    if position == "top" or position == "on":
        if "on" not in relation_list and "in" not in relation_list:
            return False
    else:
        if position not in relation_list:
            return False
    return True


def get_related_position(
    pcd1: np.ndarray,
    pcd2: np.ndarray,
    pcd3: np.ndarray | None = None,
    scene=None,
) -> list[str]:
    max_pcd1 = np.max(pcd1, axis=0)
    min_pcd1 = np.min(pcd1, axis=0)
    max_pcd2 = np.max(pcd2, axis=0)
    min_pcd2 = np.min(pcd2, axis=0)
    return infer_spatial_relationship(
        pcd1, pcd2, min_pcd1, max_pcd1, min_pcd2, max_pcd2, pcd3, scene=scene
    )


def _check_inside_relationship(
    point_cloud_a: np.ndarray, point_cloud_b: np.ndarray, scene=None
) -> list[str]:
    """检查内外关系"""
    relation_list = []
    if is_inside(
        src_pts=point_cloud_a,
        target_pts=point_cloud_b,
        thresh=INSIDE_PROPORTION_THRESH,
        scene=scene,
    ):
        relation_list.append("in")
    elif is_inside(
        src_pts=point_cloud_b,
        target_pts=point_cloud_a,
        thresh=INSIDE_PROPORTION_THRESH,
        scene=scene,
    ):
        relation_list.append("out of")
    return relation_list


def _check_support_relationship(
    min_points_a: np.ndarray,
    max_points_a: np.ndarray,
    min_points_b: np.ndarray,
    max_points_b: np.ndarray,
    error_margin_percentage: float,
) -> list[str]:
    """检查支撑关系（on/below）"""
    relation_list = []
    b_bottom_to_a_top_dist = min_points_b[2] - max_points_a[2]
    a_bottom_to_b_top_dist = min_points_a[2] - max_points_b[2]

    iou_2d, i_ratios, a_ratios = iou_2d_via_boundaries(
        min_points_a, max_points_a, min_points_b, max_points_b
    )
    i_target_ratio, i_anchor_ratio = i_ratios
    target_anchor_area_ratio, anchor_target_area_ratio = a_ratios

    # a被b支撑（a在b上方）
    a_supported_by_b = (
        i_target_ratio > MIN_TO_BE_SUPPORTED_AREA_RATIO
        and abs(a_bottom_to_b_top_dist)
        <= MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage)
        and target_anchor_area_ratio < MAX_TO_BE_SUPPORTING_AREA_RATIO
    )

    # a支撑b（a在b下方）
    a_supporting_b = (
        i_anchor_ratio > MIN_TO_BE_SUPPORTED_AREA_RATIO
        and abs(b_bottom_to_a_top_dist)
        <= MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage)
        and anchor_target_area_ratio < MAX_TO_BE_SUPPORTING_AREA_RATIO
    )

    if a_supported_by_b:
        relation_list.append("on")
    elif a_supporting_b:
        relation_list.append("below")
    else:
        relation_list.append("near")

    return relation_list


def _check_overlap_relationship(
    min_points_a: np.ndarray,
    max_points_a: np.ndarray,
    min_points_b: np.ndarray,
    max_points_b: np.ndarray,
) -> list[str]:
    """检查重叠关系（left/right/front/back/near）"""
    relation_list = []

    x_overlap = (
        (min_points_a[0] <= max_points_b[0] <= max_points_a[0])
        or (min_points_a[0] <= min_points_b[0] <= max_points_a[0])
        or (min_points_b[0] <= min_points_a[0] <= max_points_b[0])
        or (min_points_b[0] <= max_points_a[0] <= max_points_b[0])
    )
    y_overlap = (
        (min_points_a[1] <= max_points_b[1] <= max_points_a[1])
        or (min_points_a[1] <= min_points_b[1] <= max_points_a[1])
        or (min_points_b[1] <= min_points_a[1] <= max_points_b[1])
        or (min_points_b[1] <= max_points_a[1] <= max_points_b[1])
    )
    if x_overlap and y_overlap:
        relation_list.append("near")
    elif x_overlap:
        if max_points_a[1] < min_points_b[1]:
            relation_list.append("left")
        elif max_points_b[1] < min_points_a[1]:
            relation_list.append("right")
    elif y_overlap:
        if max_points_a[0] < min_points_b[0]:
            relation_list.append("front")
        elif max_points_b[0] < min_points_a[0]:
            relation_list.append("back")
    return relation_list


def _check_between_relationship(
    point_cloud_a: np.ndarray,
    point_cloud_b: np.ndarray,
    point_cloud_c: np.ndarray,
) -> list[str]:
    """检查三点间关系（between）"""

    def compute_centroid(point_cloud):
        return np.mean(point_cloud, axis=0)

    anchor1_center = compute_centroid(point_cloud_b)
    anchor2_center = compute_centroid(point_cloud_c)
    target_center = compute_centroid(point_cloud_a)

    vector1 = target_center - anchor1_center
    vector2 = anchor2_center - target_center

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return []

    vector1_norm = vector1 / norm1
    vector2_norm = vector2 / norm2
    cosine_angle = np.dot(vector1_norm, vector2_norm)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    if angle < ANGLE_THRESHOLD:
        return ["between"]
    return []


def infer_spatial_relationship(
    point_cloud_a: np.ndarray,
    point_cloud_b: np.ndarray,
    min_points_a: np.ndarray,
    max_points_a: np.ndarray,
    min_points_b: np.ndarray,
    max_points_b: np.ndarray,
    point_cloud_c: np.ndarray | None = None,
    error_margin_percentage: float = 0.01,
    scene=None,
) -> list[str]:
    relation_list = []
    if point_cloud_c is None:
        dx = max(
            0, min_points_a[0] - max_points_b[0], min_points_b[0] - max_points_a[0]
        )
        dy = max(
            0, min_points_a[1] - max_points_b[1], min_points_b[1] - max_points_a[1]
        )
        aabb_xy_dist = np.sqrt(dx * dx + dy * dy)

        if aabb_xy_dist > XY_DISTANCE_CLOSE_THRESHOLD * (1 + error_margin_percentage):
            return []

        # 计算距离
        xy_dist = calculate_xy_distance_between_two_point_clouds(
            point_cloud_a, point_cloud_b, scene=scene
        )
        if xy_dist > XY_DISTANCE_CLOSE_THRESHOLD * (1 + error_margin_percentage):
            return []

        dist = calculate_distance_between_two_point_clouds(
            point_cloud_a, point_cloud_b, scene=scene
        )
        # 如果两个物体很接近，检查各种关系
        if dist < MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage):
            # 检查内外关系
            inside_relations = _check_inside_relationship(
                point_cloud_a, point_cloud_b, scene=scene
            )
            relation_list.extend(inside_relations)
            # 如果没有内外关系，检查支撑关系
            if not inside_relations:
                support_relations = _check_support_relationship(
                    min_points_a,
                    max_points_a,
                    min_points_b,
                    max_points_b,
                    error_margin_percentage,
                )
                relation_list.extend(support_relations)
        # 检查重叠关系
        overlap_relations = _check_overlap_relationship(
            min_points_a, max_points_a, min_points_b, max_points_b
        )
        for rel in overlap_relations:
            if rel not in relation_list:
                relation_list.append(rel)
    else:
        # 检查三点间关系
        between_relations = _check_between_relationship(
            point_cloud_a, point_cloud_b, point_cloud_c
        )
        relation_list.extend(between_relations)
    return relation_list


def initialize_shelves(mesh: o3d.geometry.TriangleMesh) -> list[Shelf]:
    pcd = get_pcd_from_mesh(mesh, num_points=100000)
    shelves = extract_shelf_planes(pcd)
    return shelves


def iou_2d_via_boundaries(
    min_points_a: np.ndarray,
    max_points_a: np.ndarray,
    min_points_b: np.ndarray,
    max_points_b: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    a_xmin, a_xmax, a_ymin, a_ymax = (
        min_points_a[0],
        max_points_a[0],
        min_points_a[1],
        max_points_a[1],
    )
    b_xmin, b_xmax, b_ymin, b_ymax = (
        min_points_b[0],
        max_points_b[0],
        min_points_b[1],
        max_points_b[1],
    )

    box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
    box_b = [b_xmin, b_ymin, b_xmax, b_ymax]
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if box_a_area + box_b_area - inter_area == 0:
        iou = 0
    else:
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
    if box_a_area == 0 or box_b_area == 0:
        i_ratios = [0.0, 0.0]
        a_ratios = [0.0, 0.0]
    else:
        i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
        a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]

    return iou, i_ratios, a_ratios


def is_point_in_hull(p: np.ndarray, hull: scipy.spatial.Delaunay) -> np.ndarray:
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p) >= 0


def is_point_in_convex_hull_fast(
    points: np.ndarray, hull_obj: scipy.spatial.ConvexHull, tolerance: float = 1e-12
) -> np.ndarray:
    """
    Faster method using ConvexHull equations directly

    `points` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull_obj` should be a scipy.spatial.ConvexHull object
    """
    return np.all(
        np.add(np.dot(points, hull_obj.equations[:, :-1].T), hull_obj.equations[:, -1])
        <= tolerance,
        axis=1,
    )


def is_point_in_convex_hull_early(
    points: np.ndarray,
    hull: scipy.spatial.ConvexHull,
    thresh_ratio: float = 0.5,
    tol: float = 1e-12,
) -> bool:
    # OPT: batched vectorized replacement for the per-point Python loop.
    # Processes chunks of ~16 MB at a time so peak allocation stays bounded
    # even when ConvexHull yields thousands of faces (seen: ~9600). Keeps
    # the original early-exit: returns as soon as the ratio is reached or
    # provably unreachable.
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    n = len(points)
    required = int(np.ceil(thresh_ratio * n))
    max_bytes = 16 * 1024 * 1024
    chunk = max(1, int(max_bytes / max(1, A.shape[0] * 8)))
    cnt = 0
    for start in range(0, n, chunk):
        block = points[start : start + chunk]
        in_mask = np.all(block @ A.T + b <= tol, axis=1)
        cnt += int(in_mask.sum())
        if cnt >= required:
            return True
        remaining = n - (start + block.shape[0])
        if cnt + remaining < required:
            return False
    return cnt >= required


_HULL_CACHE_KEY = "_is_inside_hull_cache"


def _get_convex_hull_cached(target_pts: np.ndarray, scene=None):
    # Try frame cache first (populated by MetricsManager.step) to dedupe
    # hulls across sibling metrics in the same tick.
    hull_cache = None
    if scene is not None:
        hull_cache = getattr(scene, "_frame_hull_cache", None)
    key = id(target_pts)
    if hull_cache is not None and key in hull_cache:
        return hull_cache[key]
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except (QhullError, ValueError):
        hull = None
    if hull_cache is not None:
        hull_cache[key] = hull
    return hull


def is_inside(
    src_pts: np.ndarray,
    target_pts: np.ndarray,
    thresh: float = 0.5,
    use_fast_method: bool = True,
    scene=None,
) -> bool:
    hull = _get_convex_hull_cached(target_pts, scene=scene)
    if hull is None:
        return False
    num_src_pts = len(src_pts)
    thresh_obj_particles = thresh * num_src_pts
    if use_fast_method:
        is_inside = is_point_in_convex_hull_early(src_pts, hull, thresh_ratio=thresh)
        return is_inside
    else:
        hull_vertices = target_pts[hull.vertices]
        src_points_in_hull = is_point_in_hull(src_pts, hull_vertices)
        return src_points_in_hull.sum() > thresh_obj_particles
