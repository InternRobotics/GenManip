import numpy as np
import open3d as o3d
import random
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import trimesh
import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend
import matplotlib.pyplot as plt
import shapely
from concave_hull import concave_hull


def bbox_to_polygon(x, y, w, h):
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Polygon(points)


def check_mesh_collision(mesh1, mesh2):
    def o3d2trimesh(o3d_mesh):
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    tmesh1 = o3d2trimesh(mesh1)
    tmesh2 = o3d2trimesh(mesh2)

    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh1", tmesh1)
    collision_manager.add_object("mesh2", tmesh2)
    return collision_manager.in_collision_internal()


def compute_aabb_lwh(aabb):
    # compute the length, width, and height of the aabb
    length = aabb.get_max_bound()[0] - aabb.get_min_bound()[0]
    width = aabb.get_max_bound()[1] - aabb.get_min_bound()[1]
    height = aabb.get_max_bound()[2] - aabb.get_min_bound()[2]
    return length, width, height


def compute_min_distance_between_two_polygons(polygon1, polygon2, num_points=1000):
    points1 = sample_points_in_polygon(polygon1, num_points=num_points)
    points2 = sample_points_in_polygon(polygon2, num_points=num_points)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(points1)
    distances, _ = nn.kneighbors(points2)
    res = np.min(distances)
    return res


def sample_points_in_polygon(polygon, num_points=1000):
    boundary = polygon.boundary
    boundary_length = boundary.length
    points = []
    for _ in range(num_points):
        point = boundary.interpolate(random.uniform(0, boundary_length))
        points.append(np.array([point.x, point.y]))
    return np.array(points)


def transform_polygon(polygon, x, y):
    return shapely.affinity.translate(polygon, xoff=x, yoff=y)


def rotate_polygon(polygon, angle, center):
    return shapely.affinity.rotate(
        polygon, angle, origin=tuple(center), use_radians=True
    )


def compute_near_area(mesh1, mesh2, near_distance=0.1, angle_steps=36):
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


def compute_lrfb_area(position, mesh1, mesh2):
    from genmanip.demogen.evaluate.evaluate import XY_DISTANCE_CLOSE_THRESHOLD

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


def compute_mesh_xyr(mesh):
    bbox = compute_mesh_bbox(mesh)
    l, w, _ = compute_aabb_lwh(bbox)
    xyr = np.sqrt(l**2 + w**2) / 2
    return xyr


def compute_mesh_bbox(mesh):
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_bbox(pcd)


def compute_mesh_center(mesh):
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_center(pcd)


def compute_pcd_bbox(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb


def compute_pcd_center(pcd):
    pointcloud = np.asarray(pcd.points)
    center = np.mean(pointcloud, axis=0)
    return center


def get_max_distance_to_polygon(polygon, point):
    return max(
        [point.distance(Point(vertex)) for vertex in list(polygon.exterior.coords)]
    )


def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def get_pcd_from_mesh(mesh, num_points=1000):
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def visualize_polygons(polygons: list[Polygon]):
    fig, ax = plt.subplots()
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig("polygons.png")
    plt.close(fig)


def get_platform_available_area(platform_pc, pc_list, filtered_uid=[], visualize=False):
    platform_polygon = get_xy_contour(platform_pc, contour_type="concave_hull")
    if visualize:
        polygons = []
        for key, pc in pc_list.items():
            if key not in filtered_uid:
                polygons.append(get_xy_contour(pc, contour_type="concave_hull"))
        visualize_polygons(polygons)
    for key in pc_list:
        if key not in filtered_uid:
            platform_polygon = platform_polygon.difference(
                get_xy_contour(pc_list[key], contour_type="concave_hull")
            )
    return platform_polygon


def get_random_point_within_polygon(polygon, attempts=1000):
    min_x, min_y, max_x, max_y = polygon.bounds
    for _ in range(attempts):
        rand_x = random.uniform(min_x, max_x)
        rand_y = random.uniform(min_y, max_y)
        point = Point(rand_x, rand_y)
        if polygon.contains(point):
            return point
    return None


def get_xy_contour(points, contour_type="convex_hull"):
    if type(points) == o3d.geometry.PointCloud:
        points = np.asarray(points.points)
    if points.shape[1] == 3:
        points = points[:, :2]
    if contour_type == "convex_hull":
        xy_points = points
        hull = ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]
        sorted_points = sort_points_clockwise(hull_points)
        polygon = Polygon(sorted_points)
    elif contour_type == "concave_hull":
        xy_points = points
        concave_hull_points = concave_hull(xy_points)
        polygon = Polygon(concave_hull_points)
    return polygon


def max_distance_to_centroid(polygon):
    centroid = np.array(polygon.centroid.coords[0])
    vertices = np.array(polygon.exterior.coords)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    return np.max(distances)


def sample_point_in_2d_line(point1, point2, num_samples=100):
    t = np.linspace(0, 1, num_samples)
    x = point1[0] + (point2[0] - point1[0]) * t
    y = point1[1] + (point2[1] - point1[1]) * t
    return np.stack([x, y], axis=1)


def sample_points_in_aabb(aabb, num_points=1000):
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    points = np.random.uniform(min_bound, max_bound, size=(num_points, 3))
    return points


def sample_points_in_convex_hull(mesh, num_points=1000):
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


def sort_boundary_points(boundary_points, centroid):
    cx, cy = centroid

    def angle(point):
        x, y = point
        return np.arctan2(y - cy, x - cx)

    sorted_points = sorted(boundary_points, key=angle)
    return np.array(sorted_points)


def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def save_numpy_to_pcd(points, colors=None, filename="pointcloud.pcd"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)


def compute_polygon_iou(polygon1, polygon2):
    return polygon1.intersection(polygon2).area / polygon1.union(polygon2).area


def find_polygon_placement(large_polygon, small_polygon, max_attempts=1000):
    if large_polygon.is_empty or small_polygon.is_empty:
        return []
    minx, miny, maxx, maxy = large_polygon.bounds
    valid_placements = []
    for _ in range(max_attempts):
        coords = np.array(small_polygon.exterior.coords)
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


def find_polygon_placement_with_rotation(
    large_polygon, small_polygon, object1_center, max_attempts=1000
):
    if large_polygon.is_empty or small_polygon.is_empty:
        return []
    minx, miny, maxx, maxy = large_polygon.bounds
    valid_placements = []
    for _ in range(max_attempts):
        random_angle = np.random.uniform(0, 2 * np.pi)
        rotated_polygon = rotate_polygon(small_polygon, random_angle, object1_center)
        coords = np.array(rotated_polygon.exterior.coords)
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


def get_world_corners_from_bbox3d(extents):
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
