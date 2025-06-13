import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from genmanip.utils.pc_utils import get_world_corners_from_bbox3d
from omni.isaac.sensor import Camera  # type: ignore
from genmanip.utils.transform_utils import pose_to_transform
from omni.isaac.core.prims import XFormPrim  # type: ignore


def get_tcp_3d_trace(tcp_xform_list):
    tcp_3d_trace = []
    for tcp in tcp_xform_list:
        position, orientation = tcp.get_world_pose()
        tcp_3d_trace.append(position)
    return tcp_3d_trace


def get_tcp_2d_trace(camera, tcp_xform_list):
    tcp_3d_trace = get_tcp_3d_trace(tcp_xform_list)
    tcp_2d_trace = []
    for tcp in tcp_3d_trace:
        pixel = get_pixel_from_world_point(camera, tcp.reshape(3, 1))[0]
        tcp_2d_trace.append(pixel)
    return tcp_2d_trace


def collect_camera_info(camera: Camera):
    info = {}
    info["p"], info["q"] = camera.get_world_pose()
    info["rgb"] = get_src(camera, "rgb")
    info["depth"] = get_src(camera, "depth")
    seg_data = get_src(camera, "seg")
    if seg_data is not None:
        info["obj_mask"] = seg_data["mask"]
        info["obj_mask_id2labels"] = seg_data["id2labels"]
    info["bbox2d_tight"], info["bbox2d_tight_id2labels"] = get_src(
        camera, "bbox2d_tight"
    )
    info["bbox2d_loose"], info["bbox2d_loose_id2labels"] = get_src(
        camera, "bbox2d_loose"
    )
    info["bbox3d"], info["bbox3d_id2labels"] = get_src(camera, "bbox3d")
    info["motion_vectors"] = get_src(camera, "motion_vectors")
    info["focal_length"] = camera.get_focal_length()
    info["focus_distance"] = camera.get_focus_distance()
    info["frequency"] = camera.get_frequency()
    info["horizontal_aperture"] = camera.get_horizontal_aperture()
    info["horizontal_fov"] = camera.get_horizontal_fov()
    info["vertical_aperture"] = camera.get_vertical_aperture()
    info["vertical_fov"] = camera.get_vertical_fov()
    info["intrinsics_matrix"] = get_intrinsic_matrix(camera)
    return info


def get_eval_camera_data(camera_list):
    camera_data = {}
    for camera_name, camera in camera_list.items():
        camera_info = collect_camera_info(camera)
        camera_data[camera_name] = {}
        camera_data[camera_name]["rgb"] = camera_info["rgb"]
        camera_data[camera_name]["depth"] = camera_info["depth"]
        camera_data[camera_name]["intrinsics_matrix"] = camera_info["intrinsics_matrix"]
        camera_data[camera_name]["p"] = camera_info["p"]
        camera_data[camera_name]["q"] = camera_info["q"]
        if "obj_mask" in camera_info:
            camera_data[camera_name]["seg_mask"] = camera_info["obj_mask"]
            camera_data[camera_name]["seg_mask_id2labels"] = camera_info[
                "obj_mask_id2labels"
            ]
    return camera_data


def get_depth(camera: Camera):
    depth = camera.get_depth()
    if isinstance(depth, np.ndarray) and depth.size > 0:
        return depth
    else:
        return None


def get_pointcloud(camera: Camera):
    cloud = camera._custom_annotators["pointcloud"].get_data()["data"]
    if isinstance(cloud, np.ndarray) and cloud.size > 0:
        return cloud
    else:
        return None


def get_objectmask(camera: Camera):
    annotator = camera._custom_annotators["semantic_segmentation"]
    annotation_data = annotator.get_data()
    mask = annotation_data["data"]
    idToLabels = annotation_data["info"]["idToLabels"]
    if isinstance(mask, np.ndarray) and mask.size > 0:
        return dict(mask=mask.astype(np.int8), id2labels=idToLabels)
    else:
        return None


def get_rgb(camera: Camera):
    frame = camera.get_rgba()
    if isinstance(frame, np.ndarray) and frame.size > 0:
        frame = frame[:, :, :3]
        return frame
    else:
        return None


def get_bounding_box_2d_tight(camera: Camera):
    annotator = camera._custom_annotators["bounding_box_2d_tight"]
    annotation_data = annotator.get_data()
    bbox = annotation_data["data"]
    info = annotation_data["info"]
    return bbox, info["idToLabels"]


def get_bounding_box_2d_loose(camera: Camera):
    annotator = camera._custom_annotators["bounding_box_2d_loose"]
    annotation_data = annotator.get_data()
    bbox = annotation_data["data"]
    info = annotation_data["info"]
    return bbox, info["idToLabels"]


def get_bounding_box_3d(camera: Camera):
    annotator = camera._custom_annotators["bounding_box_3d"]
    annotation_data = annotator.get_data()
    bbox = annotation_data["data"]
    info = annotation_data["info"]
    bbox_data = []
    for box in bbox:
        extents = {}
        (
            extents["class"],
            extents["x_min"],
            extents["y_min"],
            extents["z_min"],
            extents["x_max"],
            extents["y_max"],
            extents["z_max"],
            extents["transform"],
            _,
        ) = box
        extents["corners"] = get_world_corners_from_bbox3d(extents)
        bbox_data.append(extents)
    return bbox_data, info["idToLabels"]


def get_motion_vectors(camera: Camera):
    annotator = camera._custom_annotators["motion_vectors"]
    annotation_data = annotator.get_data()
    motion_vectors = annotation_data
    return motion_vectors


def get_src(camera: Camera, type: str):
    if type == "rgb":
        return get_rgb(camera)
    if type == "depth":
        return get_depth(camera)
    if type == "cloud":
        return get_pointcloud(camera)
    if type == "seg":
        return get_objectmask(camera)
    if type == "bbox2d_tight":
        return get_bounding_box_2d_tight(camera)
    if type == "bbox2d_loose":
        return get_bounding_box_2d_loose(camera)
    if type == "bbox3d":
        return get_bounding_box_3d(camera)
    if type == "motion_vectors":
        return get_motion_vectors(camera)


def get_world_point_from_pixel_(camera: Camera, point: np.ndarray):
    return camera.get_world_points_from_image_coords(
        np.array([int(point[0]), int(point[1])]).reshape(-1, 2),
        np.array([get_src(camera, "depth")[int(point[1]), int(point[0])]]).reshape(-1),
    )[0]


def get_pixel_from_world_point_(camera: Camera, point: np.ndarray):
    return camera.get_image_coords_from_world_points(point.reshape(-1, 3))


def get_world_point_from_pixel(camera: Camera, point: np.ndarray):
    intrinsic = get_intrinsic_matrix(camera)
    translation, quaternion = camera.get_world_pose()
    depth = get_src(camera, "depth")
    intrinsic = np.array(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    x, y = point[0], point[1]
    Z = depth[int(y)][int(x)]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    point_in_camera_frame = np.array([X, Y, Z])
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    point_in_camera_frame = add_rotation @ point_in_camera_frame
    camera_to_world = pose_to_transform((translation, quaternion))
    point_in_world_frame = camera_to_world @ np.array([*point_in_camera_frame, 1])
    point3d = point_in_world_frame[:3]
    return point3d


def get_pixel_from_world_point(camera: Camera, point: np.ndarray):
    point = point.reshape(-1, 3)
    translation, quaternion = camera.get_world_pose()
    camera_to_world = pose_to_transform((translation, quaternion))
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous_points = np.hstack([point, np.ones((point.shape[0], 1))])
    points_in_camera_frame = np.dot(homogeneous_points, world_to_camera.T)
    point_in_camera_frame = points_in_camera_frame[:, :3]
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    inv_rotation = np.linalg.inv(add_rotation)
    rotated_point = np.dot(point_in_camera_frame, inv_rotation.T)
    intrinsic = get_intrinsic_matrix(camera)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    X, Y, Z = rotated_point[:, 0], rotated_point[:, 1], rotated_point[:, 2]
    x = (fx * X / Z) + cx
    y = (fy * Y / Z) + cy
    return np.column_stack((x, y))


def get_intrinsic_matrix(camera):
    fx, fy = compute_fx_fy(
        camera, camera.get_resolution()[1], camera.get_resolution()[0]
    )
    cx, cy = camera.get_resolution()[0] / 2, camera.get_resolution()[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def compute_fx_fy(camera, height, width):
    focal_length = camera.get_focal_length()
    horiz_aperture = camera.get_horizontal_aperture()
    vert_aperture = camera.get_vertical_aperture()
    near, far = camera.get_clipping_range()
    fov = 2 * np.arctan(0.5 * horiz_aperture / focal_length)
    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    return focal_x, focal_y


def set_camera_rational_polynomial(
    camera,
    fx,
    fy,
    cx,
    cy,
    width,
    height,
    pixel_size=3,
    f_stop=2.0,
    focus_distance=0.3,
    D=None,
) -> Camera:
    if D is None:
        D = np.zeros(8)
    camera.initialize()
    camera.set_resolution([width, height])
    camera.set_clipping_range(0.02, 5)
    horizontal_aperture = pixel_size * 1e-3 * width
    vertical_aperture = pixel_size * 1e-3 * height
    focal_length_x = fx * pixel_size * 1e-3
    focal_length_y = fy * pixel_size * 1e-3
    focal_length = (focal_length_x + focal_length_y) / 2  # in mm
    camera.set_focal_length(focal_length / 10.0)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop * 100.0)
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)
    camera.set_vertical_aperture(vertical_aperture / 10.0)
    camera.set_clipping_range(0.05, 1.0e5)
    diagonal = 2 * math.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)
    diagonal_fov = 2 * math.atan2(diagonal, fx + fy) * 180 / math.pi
    camera.set_projection_type("fisheyePolynomial")
    camera.set_rational_polynomial_properties(width, height, cx, cy, diagonal_fov, D)
    return camera


def set_camera_look_at(camera, target, distance=0.4, elevation=90.0, azimuth=0.0):
    if isinstance(target, XFormPrim):
        target_position, _ = target.get_world_pose()
    else:
        target_position = target
    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)
    offset_x = distance * math.cos(elev_rad) * math.cos(azim_rad)
    offset_y = distance * math.cos(elev_rad) * math.sin(azim_rad)
    offset_z = distance * math.sin(elev_rad)
    camera_position = target_position + np.array([offset_x, offset_y, offset_z])
    rot = R.from_euler("xyz", [0, elevation, azimuth - 180], degrees=True)
    quaternion = rot.as_quat()
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    camera.set_world_pose(position=camera_position, orientation=quaternion)


def setup_camera(
    camera: Camera,
    focal_length: float = 4.5,
    clipping_range_min: float = 0.01,
    clipping_range_max: float = 10000.0,
    vertical_aperture: float = 5.625,
    horizontal_aperture: float = 10.0,
    with_distance: bool = True,
    with_semantic: bool = False,
    with_bbox2d: bool = False,
    with_bbox3d: bool = False,
    with_motion_vector: bool = False,
    camera_params: dict = None,
):
    camera.initialize()
    camera.set_focal_length(focal_length)
    camera.set_clipping_range(clipping_range_min, clipping_range_max)
    camera.set_vertical_aperture(vertical_aperture)
    camera.set_horizontal_aperture(horizontal_aperture)
    if with_distance:
        camera.add_distance_to_image_plane_to_frame()
    if with_semantic:
        camera.add_semantic_segmentation_to_frame()
    if with_bbox2d:
        camera.add_bounding_box_2d_tight_to_frame()
        camera.add_bounding_box_2d_loose_to_frame()
    if with_bbox3d:
        camera.add_bounding_box_3d_to_frame()
    if with_motion_vector:
        camera.add_motion_vectors_to_frame()
    if camera_params is not None:
        set_camera_rational_polynomial(camera, *camera_params)
