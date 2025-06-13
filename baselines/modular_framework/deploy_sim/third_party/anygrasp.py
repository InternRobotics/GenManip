import numpy as np
import open3d as o3d
import random
import requests
from scipy.spatial.transform import Rotation as R
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class HostnameIgnoringAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = False
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def anygrasp_response_to_list(grasp_groups):
    grasp_list = []
    for grasp in grasp_groups:
        grasp_list.append(
            {
                "translation": grasp["translation"],
                "orientation": grasp["rotation_matrix"],
                "depth": grasp["depth"],
                "score": grasp["score"],
            }
        )
    return grasp_list


def compute_distance_list_from_point_list_to_mesh(mesh, points):
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=10000)
    mesh_points = np.asarray(pcd_mesh.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_mesh)
    nearest_points = []
    distances = []
    for point in points:
        [_, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
        nearest_point = mesh_points[idx[0]]
        distance = np.sqrt(dist[0])
        nearest_points.append(nearest_point)
        distances.append(distance)
    return nearest_points, distances


def find_closest_grasp_to_point(
    point,
    grasp_list,
    distance_threshold=0.09,
    angle_threshold=None,
    distance_only=False,
):
    grasp_points = np.array([grasp["translation"] for grasp in grasp_list])
    distances = np.linalg.norm(grasp_points - np.array(point), axis=1)
    within_threshold_indices = np.where(distances <= distance_threshold)[0]
    if len(within_threshold_indices) == 0:
        return None
    if distance_only:
        min_index = within_threshold_indices[
            np.argmin(distances[within_threshold_indices])
        ]
        closest_grasp = grasp_list[min_index]
    else:
        filtered_grasps = []
        for i in within_threshold_indices:
            grasp = grasp_list[i]
            grasp_quaternion = grasp["orientation"]
            grasp_rotation = R.from_quat(grasp_quaternion[[1, 2, 3, 0]])
            if angle_threshold is not None:
                relative_rotation = (
                    R.from_quat(np.array([1, 0, 0, 0])).inv() * grasp_rotation
                )
                angle = relative_rotation.magnitude() * (180 / np.pi)
                if angle <= angle_threshold:
                    filtered_grasps.append(grasp)
            else:
                filtered_grasps.append(grasp)
        if not filtered_grasps:
            return None
        sorted_grasps = sorted(filtered_grasps, key=lambda x: x["score"], reverse=True)
        closest_grasp = sorted_grasps[0]
    return closest_grasp


def get_grasp_pose(
    colors: np.ndarray,
    depth: np.ndarray,
    fx: float = 560.0,
    fy: float = 560.0,
    cx: float = 640.0,
    cy: float = 360.0,
    scale: float = 1000.0,
    address: str = "127.0.0.1",
    port: str = "5001",
):
    data = {
        "colors": colors.tolist(),
        "depths": (depth * scale).tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "scale": scale,
    }
    session = requests.Session()
    session.mount("https://", HostnameIgnoringAdapter())
    response = requests.post(
        f"http://{address}:{port}/process", json=data, verify=False
    )
    if response.status_code == 200:
        processed_data = response.json()
        return anygrasp_response_to_list(processed_data["grasp_groups"])
    else:
        print("Failed to process data:", response.text)


def get_init_grasp(colors, depth, intrinsics, address="10.254.30.31", port="5001"):
    # port = random.randint(5001, 5024)
    port = str(port)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    grasp_list = get_grasp_pose(
        colors=colors,
        depth=depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        address=address,
        port=port,
    )
    if grasp_list is None or (isinstance(grasp_list, list) and len(grasp_list) == 0):
        print(f"grasp list has nothing")
        return None
    else:
        return grasp_list


def get_world_grasp_from_camera_coords(
    camera_position: np.ndarray,
    camera_quaternion: np.ndarray,
    point_3d: np.ndarray,
    matrix_grasp: np.ndarray,
):
    rotation_cam_to_world = R.from_quat(camera_quaternion[[1, 2, 3, 0]])
    transform_costum_pose = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    correction_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    point_in_world_frame = (
        rotation_cam_to_world.as_matrix()
        @ correction_matrix
        @ transform_costum_pose
        @ point_3d
        + camera_position
    )
    rotation_quat_cam = R.from_matrix(matrix_grasp)
    rotation_world = (
        rotation_cam_to_world
        * R.from_matrix(correction_matrix @ transform_costum_pose)
        * rotation_quat_cam
        * R.from_matrix(np.linalg.inv(transform_costum_pose))
    ).as_quat()[[3, 0, 1, 2]]
    return point_in_world_frame, rotation_world
