from scipy.spatial.transform import Rotation as R
import lmdb
import pickle
import argparse
import cv2
import numpy as np
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm

HAS_AV = True

try:
    import av
except Exception as e:
    HAS_AV = False
    print(e)
    # raise ModuleNotFoundError("please install av: conda install ffmpeg and pip install pyav")

# seed
np.random.seed(42)
random.seed(42)
DEFAULT_RGB_SCALE_FACTOR = 256000.0
COLOR_MAP = []

for i in range(256):
    COLOR_MAP.append(
        (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--camera_name", type=str, required=True)
    parser.add_argument(
        "--annotator_type",
        nargs="+",
        type=str,
        required=True,
        default=["step", "qpos", "arm_action", "ee_pose"],
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_jobs", type=int, default=100)
    parser.add_argument(
        "--video_gen_func", type=str, default="av", help="support av、opencv"
    )
    return parser.parse_args()


def visualize_3d_bbox(
    img,
    pixel_coordinates,
    point_color=(0, 0, 255),
    point_size=5,
    edge_color=(0, 255, 0),
    edge_thickness=1,
    draw_planes=False,
    plane_alpha=0.3,
):
    for pixel_coordinate in pixel_coordinates:
        img = cv2.circle(
            img,
            (int(pixel_coordinate[0]), int(pixel_coordinate[1])),
            point_size,
            point_color,
            -1,
        )
    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (3, 7),
        (2, 6),
    ]
    for edge in edges:
        start_point = (
            int(pixel_coordinates[edge[0]][0]),
            int(pixel_coordinates[edge[0]][1]),
        )
        end_point = (
            int(pixel_coordinates[edge[1]][0]),
            int(pixel_coordinates[edge[1]][1]),
        )
        img = cv2.line(img, start_point, end_point, edge_color, edge_thickness)
    if draw_planes:
        planes = [
            {"corners": [0, 1, 3, 2], "color": edge_color},
            {"corners": [4, 5, 7, 6], "color": edge_color},
            {"corners": [0, 2, 6, 4], "color": edge_color},
            {"corners": [1, 3, 7, 5], "color": edge_color},
            {"corners": [2, 3, 7, 6], "color": edge_color},
            {"corners": [0, 1, 5, 4], "color": edge_color},
        ]
        for plane in planes:
            contour = np.array(
                [
                    [int(pixel_coordinates[i][0]), int(pixel_coordinates[i][1])]
                    for i in plane["corners"]
                ],
                dtype=np.int32,
            )
            overlay = img.copy()
            cv2.fillPoly(overlay, [contour], plane["color"])
            img = cv2.addWeighted(overlay, plane_alpha, img, 1 - plane_alpha, 0)
    return img


def transform_to_pose(transform):
    trans = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return trans, quat


def get_scalar_data_from_lmdb(data_path, key):
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"]["scalar_data"].index(key)
    key_key = meta_info["keys"]["scalar_data"][key_index]
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(key_key))
    return data


def get_json_data_from_lmdb(data_path):
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(b"json_data"))
    return data


def get_color_image_from_lmdb(data_path, key):
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    num_steps = meta_info["num_steps"]
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    color_image = []
    with lmdb_env.begin(write=False) as txn:
        for key in key_index:
            color_image.append(
                cv2.imdecode(pickle.loads(txn.get(key)), cv2.IMREAD_COLOR)
            )
    return color_image


def get_semantic_image_from_lmdb(data_path, key):
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    num_steps = meta_info["num_steps"]
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    semantic_image = []
    with lmdb_env.begin(write=False) as txn:
        for key in key_index:
            semantic_image.append(
                cv2.imdecode(pickle.loads(txn.get(key)), cv2.IMREAD_GRAYSCALE)
            )
    return semantic_image


def uint16_array_to_float_array(uint16_array: np.ndarray) -> np.ndarray:
    float_array = uint16_array.astype(np.float32)
    float_array = float_array / 10000
    return float_array


def get_depth_image_from_lmdb(data_path, key):
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    num_steps = meta_info["num_steps"]
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    depth_image = []
    with lmdb_env.begin(write=False) as txn:
        for key in key_index:
            depth_image.append(
                uint16_array_to_float_array(
                    cv2.imdecode(pickle.loads(txn.get(key)), cv2.IMREAD_UNCHANGED)
                )
            )
    return depth_image


def get_all_data(data_path, camera_name):
    data = {}
    try:
        grasp_point = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))[
            "task_data"
        ]["grasp_point"]["0"][camera_name]
        data["grasp_point"] = grasp_point
    except:
        data["grasp_point"] = None
    data["instruction"] = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))[
        "task_data"
    ]["instruction"]
    data["arm_action"] = get_scalar_data_from_lmdb(data_path, b"arm_action")
    data["qpos"] = get_scalar_data_from_lmdb(data_path, b"observation/robot/qpos")
    data["ee_pose"] = get_scalar_data_from_lmdb(
        data_path, b"observation/robot/ee_pose_state"
    )
    data["bounding_box_3d"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/bbox3d".encode("utf-8")
    )
    data["tcp_trace_2d"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/tcp_2d_trace".encode("utf-8")
    )
    data["bounding_box_2d"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/bbox2d_loose".encode("utf-8")
    )
    data["bounding_box_2d_id2labels"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/bbox2d_loose_id2labels".encode("utf-8")
    )
    data["camera2env_pose"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/camera2env_pose".encode("utf-8")
    )
    data["bbox2d_tight_id2labels"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/bbox2d_tight_id2labels".encode("utf-8")
    )
    data["bbox2d_loose_id2labels"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/bbox2d_loose_id2labels".encode("utf-8")
    )
    data["bbox3d_id2labels"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/bbox3d_id2labels".encode("utf-8")
    )
    data["semantic_mask_id2labels"] = get_scalar_data_from_lmdb(
        data_path, f"observation/{camera_name}/semantic_mask_id2labels".encode("utf-8")
    )
    for i in range(len(data["camera2env_pose"])):
        data["camera2env_pose"][i] = transform_to_pose(data["camera2env_pose"][i])
    data["color_image"] = get_color_image_from_lmdb(
        data_path, f"observation/{camera_name}/color_image"
    )
    data["depth_image"] = get_depth_image_from_lmdb(
        data_path, f"observation/{camera_name}/depth_image"
    )
    data["camera_intrinsic"] = get_json_data_from_lmdb(data_path)[
        f"observation/{camera_name}/camera_params"
    ]
    data["semantic_image"] = get_semantic_image_from_lmdb(
        data_path, f"observation/{camera_name}/semantic_mask"
    )
    try:
        data["depth_range"] = [
            np.min(np.array(data["depth_image"])),
            np.max(np.array(data["depth_image"])),
        ]
    except:
        data["depth_range"] = None
    return data


def annotate_depth_image(data, i):
    depth_image = data["depth_image"][i]
    depth_image = (depth_image - data["depth_range"][0]) / (
        data["depth_range"][1] - data["depth_range"][0]
    )
    depth_image = (depth_image * 255).astype(np.uint8)
    data["color_image"][i] = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return data["color_image"][i]


def annotate_grasp_point(data, i):
    data["color_image"][i] = cv2.circle(
        data["color_image"][i],
        (int(data["grasp_point"][0][0][0]), int(data["grasp_point"][0][0][1])),
        5,
        (0, 0, 255),
        -1,
    )
    data["color_image"][i] = cv2.circle(
        data["color_image"][i],
        (int(data["grasp_point"][1][0][0]), int(data["grasp_point"][1][0][1])),
        5,
        (0, 255, 0),
        -1,
    )
    return data["color_image"][i]


def annotate_semantic_mask(data, i):
    semantic_mask = data["semantic_image"][i]
    colored_mask = np.zeros_like(data["color_image"][i])
    unique_labels = np.unique(semantic_mask)
    for label in unique_labels:
        if label == 0:
            continue
        try:
            data["semantic_mask_id2labels"][i][str(label)]["class"]
        except:
            continue
        if (
            data["semantic_mask_id2labels"][i][str(label)]["class"]
            == "defaultgroundplane"
        ):
            continue
        mask = semantic_mask == label

        colored_mask[mask] = COLOR_MAP[label % len(COLOR_MAP)]
    data["color_image"][i] = cv2.cvtColor(data["color_image"][i], cv2.COLOR_RGB2BGR)
    data["color_image"][i] = cv2.addWeighted(
        data["color_image"][i], 0.7, colored_mask, 0.3, 0
    )
    return data["color_image"][i]


def image_to_float_array(image: np.ndarray, scale_factor: float = None) -> np.ndarray:
    image_array = np.asarray(image)
    if scale_factor is None:
        scale_factor = DEFAULT_RGB_SCALE_FACTOR
    float_array = np.dot(image_array, [65536, 256, 1])
    return float_array / scale_factor


def annotate_all_tcp_trace_2d_depth(data, i):
    for j in range(i, min(i + 20, len(data["tcp_trace_2d"]) - 1)):
        for k in range(len(data["tcp_trace_2d"][j])):
            x1 = int(data["tcp_trace_2d"][j][k][0])
            y1 = int(data["tcp_trace_2d"][j][k][1])
            depth_value1 = data["depth_image"][i][y1, x1]  # in meter
            x2 = int(data["tcp_trace_2d"][j + 1][k][0])
            y2 = int(data["tcp_trace_2d"][j + 1][k][1])
            depth_value2 = data["depth_image"][i][y2, x2]  # in meter
            depth_value = (depth_value1 + depth_value2) / 2
            base_color = COLOR_MAP[-k]
            scale_factor = 2
            darker_color = tuple(
                int(
                    c
                    * (
                        1.0
                        - scale_factor
                        * (depth_value - data["depth_range"][0])
                        / (data["depth_range"][1] - data["depth_range"][0])
                    )
                )
                for c in base_color
            )
            data["color_image"][i] = cv2.line(
                data["color_image"][i],
                (x1, y1),
                (x2, y2),
                darker_color,
                2,
            )
    return data["color_image"][i]


def annotate_all_tcp_trace_2d(data, i):
    for j in range(i, min(i + 20, len(data["tcp_trace_2d"]) - 1)):
        for k in range(len(data["tcp_trace_2d"][j])):
            data["color_image"][i] = cv2.line(
                data["color_image"][i],
                (
                    int(data["tcp_trace_2d"][j][k][0]),
                    int(data["tcp_trace_2d"][j][k][1]),
                ),
                (
                    int(data["tcp_trace_2d"][j + 1][k][0]),
                    int(data["tcp_trace_2d"][j + 1][k][1]),
                ),
                COLOR_MAP[-k],
                2,
            )
    return data["color_image"][i]


def annotate_tcp_trace_2d(data, i):
    for idx, trace in enumerate(data["tcp_trace_2d"][i]):
        data["color_image"][i] = cv2.circle(
            data["color_image"][i],
            (int(trace[0]), int(trace[1])),
            3,
            COLOR_MAP[-idx],
            -1,
        )
    return data["color_image"][i]


def annotate_qpos(data, i):
    cv2.putText(
        data["color_image"][i],
        f"qpos: {[round(x, 2) for x in data['qpos'][i]]}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return data["color_image"][i]


def annotate_arm_action(data, i):
    cv2.putText(
        data["color_image"][i],
        f"arm_action: {[round(x, 2) for x in data['arm_action'][i]]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return data["color_image"][i]


def annotate_ee_pose(data, i):
    cv2.putText(
        data["color_image"][i],
        f"ee_pose: {[round(x, 2) for x in data['ee_pose'][i][0]]}, {[round(x, 2) for x in data['ee_pose'][i][1]]}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return data["color_image"][i]


def pose_to_transform(pose):
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return transform


def get_world_point_from_pixel(point_2d, depth, intrinsic, translation, quaternion):
    intrinsic = np.array(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    x, y = point_2d[0], point_2d[1]
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


def get_pixel_from_world_point(intrinsic, world_point, translation, quaternion):
    world_point = world_point.reshape(-1, 3)
    camera_to_world = pose_to_transform((translation, quaternion))
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous_points = np.hstack([world_point, np.ones((world_point.shape[0], 1))])
    points_in_camera_frame = np.dot(homogeneous_points, world_to_camera.T)
    point_in_camera_frame = points_in_camera_frame[:, :3]
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    inv_rotation = np.linalg.inv(add_rotation)
    rotated_point = np.dot(point_in_camera_frame, inv_rotation.T)
    intrinsic = np.array(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    X, Y, Z = rotated_point[:, 0], rotated_point[:, 1], rotated_point[:, 2]
    x = (fx * X / Z) + cx
    y = (fy * Y / Z) + cy
    return np.column_stack((x, y))


def annotate_bounding_box_3d(data, i):
    pixel_coordinates = []
    for box in data["bounding_box_3d"][i]:
        pixel_coordinates.append(box["corners"])
    pixel_coordinates = np.array(pixel_coordinates)
    pixel_coordinates = get_pixel_from_world_point(
        data["camera_intrinsic"], pixel_coordinates, *data["camera2env_pose"][i]
    )
    for bbox in data["bounding_box_3d"][i]:
        pixel_coordinates = get_pixel_from_world_point(
            data["camera_intrinsic"], bbox["corners"], *data["camera2env_pose"][i]
        )
        data["color_image"][i] = visualize_3d_bbox(
            data["color_image"][i],
            pixel_coordinates,
            edge_color=COLOR_MAP[bbox["class"]],
            draw_planes=True,
            point_size=1,
        )
    return data["color_image"][i]


def annotate_step(data, i):
    length = len(data["color_image"])
    cv2.putText(
        data["color_image"][i],
        f"step: {str(i).zfill(len(str(length)))}/{length}",
        (10, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return data["color_image"][i]


def annotate_bbox2d(data, i):
    for bbox in data["bounding_box_2d"][i]:
        if (
            data["bounding_box_2d_id2labels"][i][str(bbox[0])]["class"]
            == "defaultgroundplane"
        ):
            continue
        overlay = np.zeros_like(data["color_image"][i])
        overlay = cv2.rectangle(
            overlay,
            (bbox[1], bbox[2]),
            (bbox[3], bbox[4]),
            COLOR_MAP[bbox[0]],
            -1,
        )
        alpha = 0.3
        data["color_image"][i] = cv2.addWeighted(
            data["color_image"][i], 1, overlay, alpha, 0
        )
        data["color_image"][i] = cv2.rectangle(
            data["color_image"][i],
            (bbox[1], bbox[2]),
            (bbox[3], bbox[4]),
            COLOR_MAP[bbox[0]],
            1,
        )
    return data["color_image"][i]


def annotate_bbox2d_id2labels(data, i):
    for bbox in data["bounding_box_2d"][i]:
        if (
            data["bounding_box_2d_id2labels"][i][str(bbox[0])]["class"]
            == "defaultgroundplane"
        ):
            continue
        data["color_image"][i] = cv2.putText(
            data["color_image"][i],
            f"class: {data['bounding_box_2d_id2labels'][i][str(bbox[0])]['class'][:8]}",
            (bbox[1], bbox[2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
        )
    return data["color_image"][i]


def annotate_instruction(data, i):
    # put text at the bottom middle of the image, with a white background
    cv2.rectangle(
        data["color_image"][i],
        (10, data["color_image"][i].shape[0] - 20),
        (data["color_image"][i].shape[1] - 10, data["color_image"][i].shape[0] - 5),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        data["color_image"][i],
        data["instruction"],
        (10, data["color_image"][i].shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 0, 0),
        1,
    )
    return data["color_image"][i]


def annotate_all_text(data, i, args):
    if "st" in args.annotator_type:
        annotate_step(data, i)
    if "qp" in args.annotator_type:
        annotate_qpos(data, i)
    if "aa" in args.annotator_type:
        annotate_arm_action(data, i)
    if "ep" in args.annotator_type:
        annotate_ee_pose(data, i)
    if "bb2d" in args.annotator_type:
        annotate_bbox2d_id2labels(data, i)
    if "ins" in args.annotator_type:
        annotate_instruction(data, i)
    return data["color_image"][i]


if __name__ == "__main__":
    args = parse_args()
    cnt = 0

    os.makedirs(args.output_path, exist_ok=True)

    for data_path in os.listdir(args.data_path):
        if cnt >= args.num_jobs:
            break
        cnt += 1
        data = get_all_data(os.path.join(args.data_path, data_path), args.camera_name)

        # ----------------------------------- gen frame ---------------------------
        # 如果只有一帧，则只保存图像，不生成视频

        if len(data["color_image"]) == 1:
            i = 0
            data["color_image"][i] = cv2.cvtColor(
                data["color_image"][i], cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                os.path.join(args.output_path, f"{data_path}_raw.png"),
                data["color_image"][i],
            )
            if "di" in args.annotator_type:
                annotate_depth_image(data, i)
            if "bb2d" in args.annotator_type:
                annotate_bbox2d(data, i)
            if "bb3d" in args.annotator_type:
                annotate_bounding_box_3d(data, i)
            if "tcp2d" in args.annotator_type:
                annotate_tcp_trace_2d(data, i)
            if "atcp2d" in args.annotator_type:
                annotate_all_tcp_trace_2d(data, i)
            if "atcp2dd" in args.annotator_type:
                annotate_all_tcp_trace_2d_depth(data, i)
            if "sm" in args.annotator_type:
                annotate_semantic_mask(data, i)
            if "gp" in args.annotator_type:
                annotate_grasp_point(data, i)
            annotate_all_text(data, i, args)
            cv2.imwrite(
                os.path.join(args.output_path, f"{data_path}.png"),
                data["color_image"][i],
            )
            continue

        # ----------------------------------- gen video ---------------------------
        # 推荐使用python -m http.server xxport, 查看video.

        if args.video_gen_func == "av" and HAS_AV:
            output_container = av.open(
                os.path.join(args.output_path, f"{data_path}.mp4"), "w"
            )
            stream = output_container.add_stream("libx264", 30)
            stream.width = 640
            stream.height = 480
            stream.pix_fmt = "yuv420p"
            for i in tqdm(range(len(data["color_image"]))):
                frame = data["color_image"][i]
                if "di" in args.annotator_type:
                    annotate_depth_image(data, i)
                if "bb2d" in args.annotator_type:
                    annotate_bbox2d(data, i)
                if "bb3d" in args.annotator_type:
                    annotate_bounding_box_3d(data, i)
                if "tcp2d" in args.annotator_type:
                    annotate_tcp_trace_2d(data, i)
                if "atcp2d" in args.annotator_type:
                    annotate_all_tcp_trace_2d(data, i)
                if "atcp2dd" in args.annotator_type:
                    annotate_all_tcp_trace_2d_depth(data, i)
                if "sm" in args.annotator_type:
                    annotate_semantic_mask(data, i)
                if "gp" in args.annotator_type:
                    annotate_grasp_point(data, i)
                annotate_all_text(data, i, args)
                # 将 NumPy 数组转换为 AVFrame
                av_frame = av.VideoFrame.from_ndarray(
                    data["color_image"][i], format="rgb24"
                )  # 将图像转为 rgb24 格式
                av_frame = av_frame.reformat(format="yuv420p")  # 转换为 yuv420p 格式
                # 将 AVFrame 编码并写入视频流
                for packet in stream.encode(av_frame):
                    output_container.mux(packet)
            # 结束编码并写入文件
            for packet in stream.encode():
                output_container.mux(packet)
            # 关闭容器
            output_container.close()
        elif args.video_gen_func == "opencv" or not HAS_AV:
            Path(args.output_path).mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                os.path.join(args.output_path, f"{data_path}.mp4"),
                fourcc,
                30,
                (640, 480),
            )
            for i in range(len(data["color_image"])):
                data["color_image"][i] = cv2.cvtColor(
                    data["color_image"][i], cv2.COLOR_RGB2BGR
                )
                if "di" in args.annotator_type:
                    annotate_depth_image(data, i)
                if "bb2d" in args.annotator_type:
                    annotate_bbox2d(data, i)
                if "bb3d" in args.annotator_type:
                    annotate_bounding_box_3d(data, i)
                if "tcp2d" in args.annotator_type:
                    annotate_tcp_trace_2d(data, i)
                if "atcp2d" in args.annotator_type:
                    annotate_all_tcp_trace_2d(data, i)
                if "atcp2dd" in args.annotator_type:
                    annotate_all_tcp_trace_2d_depth(data, i)
                if "sm" in args.annotator_type:
                    annotate_semantic_mask(data, i)
                if "gp" in args.annotator_type:
                    annotate_grasp_point(data, i)
                annotate_all_text(data, i, args)
                video_writer.write(data["color_image"][i])
            video_writer.release()
