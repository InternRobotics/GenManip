"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import random

from concave_hull import concave_hull
import matplotlib
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
import trimesh

matplotlib.use("Agg")  # Set non-interactive backend
import matplotlib.pyplot as plt


def bbox_to_polygon(x: float, y: float, w: float, h: float) -> Polygon:
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Polygon(points)


def check_mesh_collision(
    mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh
) -> bool | tuple:
    def o3d2trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    tmesh1 = o3d2trimesh(mesh1)
    tmesh2 = o3d2trimesh(mesh2)

    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh1", tmesh1)
    collision_manager.add_object("mesh2", tmesh2)
    result = collision_manager.in_collision_internal()
    return result


def compute_aabb_lwh(
    aabb: o3d.geometry.AxisAlignedBoundingBox,
) -> tuple[float, float, float]:
    # compute the length, width, and height of the aabb
    length = aabb.get_max_bound()[0] - aabb.get_min_bound()[0]
    width = aabb.get_max_bound()[1] - aabb.get_min_bound()[1]
    height = aabb.get_max_bound()[2] - aabb.get_min_bound()[2]
    return length, width, height


def compute_min_distance_between_two_polygons(
    polygon1: BaseGeometry, polygon2: BaseGeometry, num_points: int = 1000
) -> float:
    points1 = sample_points_in_polygon(polygon1, num_points=num_points)
    points2 = sample_points_in_polygon(polygon2, num_points=num_points)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(points1)
    distances, _ = nn.kneighbors(points2)
    res = np.min(distances)
    return res


def sample_points_in_polygon(
    polygon: BaseGeometry, num_points: int = 1000
) -> np.ndarray:
    boundary = polygon.boundary
    boundary_length = boundary.length
    points = []
    for _ in range(num_points):
        point = boundary.interpolate(random.uniform(0, boundary_length))
        points.append(np.array([point.x, point.y]))
    return np.array(points)


def transform_polygon(polygon: BaseGeometry, x: float, y: float) -> BaseGeometry:
    return shapely.affinity.translate(polygon, xoff=x, yoff=y)


def rotate_polygon(
    polygon: BaseGeometry, angle: float, center: tuple[float, float]
) -> BaseGeometry:
    return shapely.affinity.rotate(
        polygon, angle, origin=Point(center), use_radians=True
    )


def compute_near_area(
    mesh1: o3d.geometry.TriangleMesh,
    mesh2: o3d.geometry.TriangleMesh,
    near_distance: float = 0.1,
    angle_steps: int = 36,
) -> BaseGeometry:
    pcd1 = get_pcd_from_mesh(mesh1)
    pcd2 = get_pcd_from_mesh(mesh2)
    polygon1 = get_xy_contour(pcd1, contour_type="concave_hull")
    polygon2 = get_xy_contour(pcd2, contour_type="concave_hull")
    angles = np.linspace(0, 359, angle_steps)
    transformed_polygons_1 = []
    centroid1_x, centroid1_y = polygon1.centroid.x, polygon1.centroid.y
    centroid2_x, centroid2_y = polygon2.centroid.x, polygon2.centroid.y
    angle_rads = np.radians(angles)
    cos_angles = np.cos(angle_rads)
    sin_angles = np.sin(angle_rads)
    for i in range(len(angles)):
        distance = 100
        x = cos_angles[i] * distance + centroid2_x - centroid1_x
        y = sin_angles[i] * distance + centroid2_y - centroid1_y
        transformed_polygon_1 = transform_polygon(polygon1, x, y)
        min_distance = compute_min_distance_between_two_polygons(
            transformed_polygon_1, polygon2, num_points=50
        )
        distance = distance - min_distance + near_distance
        x = cos_angles[i] * distance + centroid2_x - centroid1_x
        y = sin_angles[i] * distance + centroid2_y - centroid1_y
        transformed_polygon_1 = transform_polygon(polygon1, x, y)
        transformed_polygons_1.append(transformed_polygon_1)
    all_points = np.vstack(
        [np.asarray(polygon.exterior.coords) for polygon in transformed_polygons_1]
    )
    near_area = get_xy_contour(all_points, contour_type="convex_hull").difference(
        polygon2
    )
    return near_area


def compute_lrfb_area(
    position: str, mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh
) -> Polygon:
    from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
        XY_DISTANCE_CLOSE_THRESHOLD,
    )

    aabb1 = compute_mesh_bbox(mesh1)
    aabb2 = compute_mesh_bbox(mesh2)
    mesh1_length, mesh1_width, _ = compute_aabb_lwh(aabb1)
    if position == "back":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_length
        polygon = Polygon(
            [
                (
                    aabb2.get_max_bound()[0],
                    min(
                        aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width
                    ),
                ),
                (
                    aabb2.get_max_bound()[0] + distance,
                    min(
                        aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width
                    ),
                ),
                (
                    aabb2.get_max_bound()[0] + distance,
                    max(
                        aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width
                    ),
                ),
                (
                    aabb2.get_max_bound()[0],
                    max(
                        aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width
                    ),
                ),
            ]
        )
    elif position == "front":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_length
        polygon = Polygon(
            [
                (
                    aabb2.get_min_bound()[0],
                    max(
                        aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width
                    ),
                ),
                (
                    aabb2.get_min_bound()[0] - distance,
                    max(
                        aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width
                    ),
                ),
                (
                    aabb2.get_min_bound()[0] - distance,
                    min(
                        aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width
                    ),
                ),
                (
                    aabb2.get_min_bound()[0],
                    min(
                        aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width
                    ),
                ),
            ]
        )
    elif position == "right":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_width
        polygon = Polygon(
            [
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_max_bound()[1],
                ),
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_max_bound()[1] + distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_max_bound()[1] + distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_max_bound()[1],
                ),
            ]
        )
    elif position == "left":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_width
        polygon = Polygon(
            [
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_min_bound()[1],
                ),
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_min_bound()[1] - distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_min_bound()[1] - distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_min_bound()[1],
                ),
            ]
        )
    else:
        polygon = Polygon()
    return polygon


def compute_mesh_xyr(mesh: o3d.geometry.TriangleMesh) -> float:
    bbox = compute_mesh_bbox(mesh)
    l, w, _ = compute_aabb_lwh(bbox)
    xyr = np.sqrt(l**2 + w**2) / 2
    return xyr


def compute_mesh_bbox(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.AxisAlignedBoundingBox:
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_bbox(pcd)


def compute_mesh_center(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_center(pcd)


def compute_pcd_bbox(
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.AxisAlignedBoundingBox:
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb


def compute_pcd_center(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    pointcloud = np.asarray(pcd.points)
    center = np.mean(pointcloud, axis=0)
    return center


def get_max_distance_to_polygon(polygon: BaseGeometry, point: Point) -> float:
    if isinstance(polygon, Polygon):
        return max(
            [point.distance(Point(vertex)) for vertex in list(polygon.exterior.coords)]
        )
    elif isinstance(polygon, MultiPolygon):
        max_distance = 0
        for single_polygon in polygon.geoms:
            _max_distance = max(
                [
                    point.distance(Point(vertex))
                    for vertex in list(single_polygon.exterior.coords)
                ]
            )
            if max_distance < _max_distance:
                max_distance = _max_distance
        return max_distance
    else:
        raise ValueError(f"Invalid polygon type: {type(polygon)}")


def get_mesh_from_points_and_faces(
    points, faceVertexCounts, faceVertexIndices
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        elif count == 4:
            face_indices = faceVertexIndices[idx : idx + 4]
            triangles.append([face_indices[0], face_indices[1], face_indices[2]])
            triangles.append([face_indices[0], face_indices[2], face_indices[3]])
        elif count > 4:
            face_indices = faceVertexIndices[idx : idx + count]
            for i in range(1, count - 1):
                triangles.append(
                    [face_indices[0], face_indices[i], face_indices[i + 1]]
                )
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def get_pcd_from_mesh(
    mesh: o3d.geometry.TriangleMesh, num_points: int = 1000
) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def visualize_polygons(polygons: list, output_path: str = "polygons.png") -> None:
    fig, ax = plt.subplots()
    for polygon in polygons:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            ax.plot(x, y)
        elif isinstance(polygon, MultiPolygon):
            for single_polygon in polygon.geoms:
                x, y = single_polygon.exterior.xy
                ax.plot(x, y)
        else:
            continue
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig(output_path)
    plt.close(fig)


def get_platform_available_area(
    platform_pc: o3d.geometry.PointCloud,
    pc_list: dict[str, np.ndarray],
    filtered_uid: list[str] = [],
    visualize: bool = False,
    buffer_size: float = 0.0,
) -> BaseGeometry:
    platform_polygon = get_xy_contour(platform_pc, contour_type="concave_hull")
    if visualize:
        polygons = []
        for key, pc in pc_list.items():
            polygons.append(get_xy_contour(pc, contour_type="concave_hull"))
        visualize_polygons(polygons)
    for key in pc_list:
        if key not in filtered_uid:
            pc = pc_list[key]
            pc_polygon = get_xy_contour(pc, contour_type="concave_hull").buffer(
                buffer_size
            )
            platform_polygon = platform_polygon.difference(pc_polygon)
    return platform_polygon


def get_random_point_within_polygon(
    polygon: BaseGeometry, attempts: int = 1000
) -> Point | None:
    min_x, min_y, max_x, max_y = polygon.bounds
    for _ in range(attempts):
        rand_x = random.uniform(min_x, max_x)
        rand_y = random.uniform(min_y, max_y)
        point = Point(rand_x, rand_y)
        if polygon.contains(point):
            return point
    return None


def pcd_to_points(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points)


def get_xy_contour(
    points: o3d.geometry.PointCloud | np.ndarray,
    contour_type: str = "convex_hull",
    num_points: float = 1000,
) -> BaseGeometry:
    if isinstance(points, o3d.geometry.PointCloud):
        points = pcd_to_points(points)
    if points.shape[1] == 3:
        points = points[:, :2]
    if contour_type == "convex_hull":
        xy_points = points
        len_points = len(xy_points)
        xy_points = xy_points[:: (len_points // num_points)]
        hull = ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]
        sorted_points = sort_points_clockwise(hull_points)
        polygon = Polygon(sorted_points)
    elif contour_type == "concave_hull":
        xy_points = points
        len_points = len(xy_points)
        xy_points = xy_points[:: (len_points // num_points)]
        concave_hull_points = concave_hull(xy_points)
        polygon = Polygon(concave_hull_points)
    else:
        raise ValueError(f"Invalid contour type: {contour_type}")
    return polygon


def max_distance_to_centroid(polygon: Polygon) -> float:
    centroid = np.array(polygon.centroid.coords[0])
    vertices = np.array(polygon.exterior.coords)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    return np.max(distances)


def sample_point_in_2d_line(
    point1: np.ndarray, point2: np.ndarray, num_samples: int = 100
) -> np.ndarray:
    t = np.linspace(0, 1, num_samples)
    x = point1[0] + (point2[0] - point1[0]) * t
    y = point1[1] + (point2[1] - point1[1]) * t
    return np.stack([x, y], axis=1)


def sample_points_in_aabb(
    aabb: o3d.geometry.AxisAlignedBoundingBox, num_points: int = 1000
) -> np.ndarray:
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    points = np.random.uniform(min_bound, max_bound, size=(num_points, 3))
    return points


def sample_points_in_convex_hull(
    mesh: o3d.geometry.TriangleMesh, num_points: int = 1000
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    points = []
    while len(points) < num_points:
        random_point = np.random.uniform(
            hull_vertices.min(axis=0), hull_vertices.max(axis=0)
        )
        if all(np.dot(eq[:-1], random_point) + eq[-1] <= 0 for eq in hull.equations):
            points.append(random_point)
    points = np.array(points)
    return points


def sort_boundary_points(
    boundary_points: np.ndarray, centroid: np.ndarray
) -> np.ndarray:
    cx, cy = centroid

    def angle(point):
        x, y = point
        return np.arctan2(y - cy, x - cx)

    sorted_points = sorted(boundary_points, key=angle)
    return np.array(sorted_points)


def sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def save_numpy_to_pcd(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    filename: str = "pointcloud.pcd",
) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)


def compute_polygon_iou(polygon1: BaseGeometry, polygon2: BaseGeometry) -> float:
    return polygon1.intersection(polygon2).area / polygon1.union(polygon2).area


def find_fixed_polygon_placement(
    large_polygon: BaseGeometry, small_polygon: BaseGeometry
) -> list[tuple[np.ndarray, float]]:
    large_center = np.array(large_polygon.centroid.coords[0])
    small_center = np.array(small_polygon.centroid.coords[0])
    translation = large_center - small_center
    return [(translation, 0)]


def find_polygon_placement(
    large_polygon: BaseGeometry, small_polygon: BaseGeometry, max_attempts: int = 1000
) -> list[tuple[np.ndarray, float]]:
    if large_polygon.is_empty or small_polygon.is_empty:
        return []
    minx, miny, maxx, maxy = large_polygon.bounds
    valid_placements = []
    for _ in range(max_attempts):
        coords = get_exterior_coords(small_polygon)
        small_centroid = np.mean(coords, axis=0)
        tx = np.random.uniform(minx, maxx)
        ty = np.random.uniform(miny, maxy)
        translation = np.array([tx, ty])
        transformed_polygon = shapely.affinity.translate(
            small_polygon,
            xoff=translation[0] - small_centroid[0],
            yoff=translation[1] - small_centroid[1],
        )
        if large_polygon.contains(transformed_polygon):
            valid_placements.append((translation - small_centroid, 0))
            break
    return valid_placements


def get_exterior_coords(polygon: BaseGeometry) -> np.ndarray:
    if isinstance(polygon, Polygon):
        return np.array(polygon.exterior.coords)
    elif isinstance(polygon, MultiPolygon):
        coords_list = [
            np.asarray(poly.exterior.coords)
            for poly in polygon.geoms
            if poly.exterior is not None
        ]
        if len(coords_list) == 0:
            return np.empty((0, 2), dtype=float)
        return np.vstack(coords_list)
    else:
        raise ValueError(f"Invalid polygon type: {type(polygon)}")


def find_polygon_placement_with_rotation(
    large_polygon: BaseGeometry,
    small_polygon: BaseGeometry,
    object1_center: tuple[float, float],
    max_attempts: int = 1000,
) -> list[tuple[np.ndarray, float]]:
    if large_polygon.is_empty or small_polygon.is_empty:
        return []
    minx, miny, maxx, maxy = large_polygon.bounds
    valid_placements = []
    for _ in range(max_attempts):
        random_angle = np.random.uniform(0, 2 * np.pi)
        rotated_polygon = rotate_polygon(small_polygon, random_angle, object1_center)
        coords = get_exterior_coords(rotated_polygon)
        small_centroid = np.mean(coords, axis=0)
        tx = np.random.uniform(minx, maxx)
        ty = np.random.uniform(miny, maxy)
        translation = np.array([tx, ty])
        transformed_polygon = shapely.affinity.translate(
            rotated_polygon,
            xoff=translation[0] - small_centroid[0],
            yoff=translation[1] - small_centroid[1],
        )
        if large_polygon.contains(transformed_polygon):
            valid_placements.append((translation - small_centroid, random_angle))
            break
    return valid_placements


def get_world_corners_from_bbox3d(extents: dict) -> np.ndarray:
    rdb = np.array([extents["x_max"], extents["y_min"], extents["z_min"]])
    ldb = np.array([extents["x_min"], extents["y_min"], extents["z_min"]])
    lub = np.array([extents["x_min"], extents["y_max"], extents["z_min"]])
    rub = np.array([extents["x_max"], extents["y_max"], extents["z_min"]])
    ldf = np.array([extents["x_min"], extents["y_min"], extents["z_max"]])
    rdf = np.array([extents["x_max"], extents["y_min"], extents["z_max"]])
    luf = np.array([extents["x_min"], extents["y_max"], extents["z_max"]])
    ruf = np.array([extents["x_max"], extents["y_max"], extents["z_max"]])
    transform = np.array(extents["transform"]).T
    points = [ldb, rdb, lub, rub, ldf, rdf, luf, ruf]
    transformed_points = []
    for point in points:
        homo_point = np.concatenate([point, [1]]).reshape(4, 1)
        transformed_point = np.dot(transform, homo_point)
        transformed_points.append(transformed_point[:3])
    return np.array(transformed_points)
