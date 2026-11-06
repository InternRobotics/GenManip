"""
This script is used to create a LeRobot dataset from a directory of LMDB files.
It supports both single-arm and dual-arm robots.

Dependencies:
pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@2b71789e15c35418b1ccecbceb81f4a598bfd883"

Usage:
python genmanip2lerobot.py --data_path /path/to/data --robot franka_robotiq --overwrite
"""

import pickle
import os
from tqdm import tqdm
import lmdb
import cv2
import argparse
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset  # type: ignore
from typing import Literal
from pathlib import Path
import shutil

TARGET_VIDEO_SIZE = (640, 480)  # (width, height)

ROBOT = "lift2"
ROBOT_CONFIGS = {
    "aloha_split": {
        "robot_type": "aloha_split",
        "is_dual_arm": True,
        "motors_state": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "joint_8",
            "joint_9",
            "joint_10",
            "joint_11",
            "gripper_1_left",
            "gripper_1_right",
            "gripper_2_left",
            "gripper_2_right",
        ],
        "motors_action": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "joint_8",
            "joint_9",
            "joint_10",
            "joint_11",
            "gripper_1_left",
            "gripper_1_right",
            "gripper_2_left",
            "gripper_2_right",
        ],
        "joint_indices": [
            12,
            14,
            16,
            18,
            20,
            22,
            13,
            15,
            17,
            19,
            21,
            23,
            24,
            25,
            26,
            27,
        ],
        "base_indices": [0, 1, 2],
        "cameras": [
            "top_camera",
            "left_camera",
            "right_camera",
        ],
    },
    "lift2": {
        "robot_type": "lift2",
        "is_dual_arm": True,
        "motors_state": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "joint_8",
            "joint_9",
            "joint_10",
            "joint_11",
        ],
        "motors_action": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "joint_8",
            "joint_9",
            "joint_10",
            "joint_11",
        ],
        "motors_state_gripper": [
            "gripper_1_left",
            "gripper_1_right",
            "gripper_2_left",
            "gripper_2_right",
        ],
        "motors_action_gripper": [
            "gripper_1_left",
            "gripper_1_right",
            "gripper_2_left",
            "gripper_2_right",
        ],
        "joint_indices": [10, 12, 14, 16, 18, 20, 9, 11, 13, 15, 17, 19],
        "gripper_indices": [23, 24, 21, 22],
        "base_indices": [0, 1, 2],
        "cameras": [
            "top_camera",
            "left_camera",
            "right_camera",
            "overlook_camera",
        ],
    },
    "franka_robotiq": {
        "robot_type": "franka_robotiq",
        "is_dual_arm": False,
        "motors_state": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ],
        "motors_action": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ],
        "motors_state_gripper": [
            "gripper_1",
            "gripper_2",
            "gripper_3",
            "gripper_4",
            "gripper_5",
            "gripper_6",
        ],
        "motors_action_gripper": [
            "gripper_1",
            "gripper_2",
            "gripper_3",
            "gripper_4",
            "gripper_5",
            "gripper_6",
        ],
        "joint_indices": [0, 1, 2, 3, 4, 5, 6],
        "gripper_indices": [7, 8, 9, 10, 11, 12],
        "base_indices": [],
        "cameras": [
            "realsense",
            "obs_camera",
            "obs_camera_2",
        ],
    },
    "franka_pandahand": {
        "robot_type": "franka_pandahand",
        "is_dual_arm": False,
        "motors_state": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ],
        "motors_action": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ],
        "motors_state_gripper": [
            "gripper_1",
            "gripper_2",
        ],
        "motors_action_gripper": [
            "gripper_1",
            "gripper_2",
        ],
        "joint_indices": [0, 1, 2, 3, 4, 5, 6],
        "gripper_indices": [7, 8],
        "base_indices": [],
        "cameras": [
            "realsense",
            "obs_camera",
            "obs_camera_2",
        ],
    },
}


def create_dataset(
    repo_id: str,
    robot_type: str,
    is_dual_arm: bool,
    mode: Literal["video", "image"] = "video",
    overwrite: bool = False,
) -> LeRobotDataset:
    cfg = ROBOT_CONFIGS[robot_type]
    motors_state = cfg["motors_state"]
    motors_action = cfg["motors_action"]
    motors_state_gripper = cfg["motors_state_gripper"]
    motors_action_gripper = cfg["motors_action_gripper"]
    base_motion = cfg["base_indices"]
    cameras = cfg["cameras"]

    features = {
        "state.joints": {
            "dtype": "float32",
            "shape": (len(motors_state),),
            "names": [
                motors_state,
            ],
        },
        "action.joints": {
            "dtype": "float32",
            "shape": (len(motors_action),),
            "names": [
                motors_action,
            ],
        },
        "action.joints_delta": {
            "dtype": "float32",
            "shape": (len(motors_state),),
            "names": [
                motors_action,
            ],
        },
        "state.gripper": {
            "dtype": "float32",
            "shape": (len(motors_state_gripper),),
            "names": [
                motors_state_gripper,
            ],
        },
        "action.gripper": {
            "dtype": "float32",
            "shape": (len(motors_action_gripper),),
            "names": [
                motors_action_gripper,
            ],
        },
        "state.ee_pose": {
            "dtype": "float32",
            "shape": (14,) if is_dual_arm else (7,),
            "names": [
                (
                    [
                        "ee0_x",
                        "ee0_y",
                        "ee0_z",
                        "ee0_qw",
                        "ee0_qx",
                        "ee0_qy",
                        "ee0_qz",
                        "ee1_x",
                        "ee1_y",
                        "ee1_z",
                        "ee1_qw",
                        "ee1_qx",
                        "ee1_qy",
                        "ee1_qz",
                    ]
                    if is_dual_arm
                    else [
                        "ee0_x",
                        "ee0_y",
                        "ee0_z",
                        "ee0_qw",
                        "ee0_qx",
                        "ee0_qy",
                        "ee0_qz",
                    ]
                )
            ],
        },
        "action.ee_pose": {
            "dtype": "float32",
            "shape": (14,) if is_dual_arm else (7,),
            "names": [
                (
                    [
                        "ee0_x",
                        "ee0_y",
                        "ee0_z",
                        "ee0_qw",
                        "ee0_qx",
                        "ee0_qy",
                        "ee0_qz",
                        "ee1_x",
                        "ee1_y",
                        "ee1_z",
                        "ee1_qw",
                        "ee1_qx",
                        "ee1_qy",
                        "ee1_qz",
                    ]
                    if is_dual_arm
                    else [
                        "ee0_x",
                        "ee0_y",
                        "ee0_z",
                        "ee0_qw",
                        "ee0_qx",
                        "ee0_qy",
                        "ee0_qz",
                    ]
                )
            ],
        },
        "action.ee_pose_delta": {
            "dtype": "float32",
            "shape": (14,) if is_dual_arm else (7,),
            "names": [
                (
                    [
                        "ee0_x",
                        "ee0_y",
                        "ee0_z",
                        "ee0_qw",
                        "ee0_qx",
                        "ee0_qy",
                        "ee0_qz",
                        "ee1_x",
                        "ee1_y",
                        "ee1_z",
                        "ee1_qw",
                        "ee1_qx",
                        "ee1_qy",
                        "ee1_qz",
                    ]
                    if is_dual_arm
                    else [
                        "ee0_x",
                        "ee0_y",
                        "ee0_z",
                        "ee0_qw",
                        "ee0_qx",
                        "ee0_qy",
                        "ee0_qz",
                    ]
                )
            ],
        },
    }

    if len(base_motion) > 0:
        features["state.base"] = {  # current base position
            "dtype": "float32",
            "shape": (len(base_motion),),
            "names": [
                ["base_x", "base_y", "base_theta"],
            ],
        }
        features["action.base"] = {
            "dtype": "float32",
            "shape": (len(base_motion),),
            "names": [
                ["base_x", "base_y", "base_theta"],
            ],
        }
        features["action.base_delta"] = {  # base motion delta value
            "dtype": "float32",
            "shape": (len(base_motion),),
            "names": [
                ["base_x", "base_y", "base_theta"],
            ],
        }

    for cam in cameras:
        features[f"video.{cam}_view"] = {
            "dtype": mode,
            "shape": (3, TARGET_VIDEO_SIZE[1], TARGET_VIDEO_SIZE[0]),  # (C, H, W)
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    print(HF_LEROBOT_HOME)
    if Path(HF_LEROBOT_HOME / repo_id).exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists at {HF_LEROBOT_HOME / repo_id}. "
                "Use --overwrite to delete it."
            )
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
        video_backend="ffmpeg",
    )


def process_single_dataset(
    dataset: LeRobotDataset,
    state_joints: np.ndarray,
    action_joints: np.ndarray,
    action_joints_delta: np.ndarray,
    state_gripper: np.ndarray,
    action_gripper: np.ndarray,
    state_base: np.ndarray | None,
    action_base: np.ndarray | None,
    action_base_delta: np.ndarray | None,
    state_ee_pose: np.ndarray,
    action_ee_pose: np.ndarray,
    action_ee_pose_delta: np.ndarray,
    txn,
    camera_key_indices: dict[str, list[bytes]],
    instruction: str,
) -> LeRobotDataset:
    num_frames = state_joints.shape[0]

    for i in tqdm(range(num_frames), desc="Processing frames"):
        frame = {
            "state.joints": state_joints[i],
            "action.joints": action_joints[i],
            "action.joints_delta": action_joints_delta[i],
            "state.gripper": state_gripper[i],
            "action.gripper": action_gripper[i],
            "state.ee_pose": state_ee_pose[i],
            "action.ee_pose": action_ee_pose[i],
            "action.ee_pose_delta": action_ee_pose_delta[i],
        }
        if state_base is not None:
            frame["state.base"] = state_base[i]
        if action_base_delta is not None:
            frame["action.base_delta"] = action_base_delta[i]
        if action_base is not None:
            frame["action.base"] = action_base[i]
        for camera_name, key_index in camera_key_indices.items():
            frame[f"video.{camera_name}_view"] = _decode_color_frame(
                txn, key_index[i], target_size=TARGET_VIDEO_SIZE
            )
        dataset.add_frame(frame, task=instruction)
    dataset.save_episode()

    return dataset


def _load_meta_info(data_path: str) -> dict:
    with open(f"{data_path}/meta_info.pkl", "rb") as f:
        return pickle.load(f)


def _open_lmdb_env(data_path: str):
    return lmdb.open(
        f"{data_path}/lmdb",
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )


def _get_scalar_data(txn, scalar_key_set: set[bytes], key: bytes):
    if key not in scalar_key_set:
        raise ValueError(f"Missing scalar key in meta_info: {key!r}")
    value = txn.get(key)
    if value is None:
        raise KeyError(f"Missing LMDB value for scalar key: {key!r}")
    return pickle.loads(value)


def _decode_color_frame(
    txn, frame_key: bytes, target_size: tuple[int, int]
) -> np.ndarray:
    value = txn.get(frame_key)
    if value is None:
        raise KeyError(f"Missing LMDB value for frame key: {frame_key!r}")

    img = cv2.imdecode(pickle.loads(value), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode color frame for key: {frame_key!r}")

    if (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img


def _compute_absolute_base_action(
    state_base: np.ndarray, action_base_delta: np.ndarray
) -> np.ndarray:
    if state_base.shape != action_base_delta.shape:
        raise ValueError(
            "Base state/action shape mismatch: "
            f"{state_base.shape} != {action_base_delta.shape}"
        )

    action_base_offset = action_base_delta.copy()
    if action_base_offset.shape[1] >= 3:
        # delta_move_to accepts yaw in degrees, while base qpos stores yaw in
        # radians. Keep action.base in the state coordinate units.
        action_base_offset[:, 2] = np.deg2rad(action_base_offset[:, 2])
    return state_base + action_base_offset


def ee_pose_list_to_np(ee_pose_action_list) -> np.ndarray:
    num_frames = len(ee_pose_action_list)
    if num_frames == 0:
        raise ValueError("ee_pose_action_list is empty")

    first = ee_pose_action_list[0]
    if len(first) == 2 and len(first[0]) == 3 and len(first[1]) == 4:
        out = np.empty((num_frames, 7), dtype=np.float32)
        for i, (pos, quat) in enumerate(ee_pose_action_list):
            out[i, :3] = pos
            out[i, 3:] = quat
        return out

    if len(first) == 2:
        out = np.empty((num_frames, 14), dtype=np.float32)
        for i, ((p0, q0), (p1, q1)) in enumerate(ee_pose_action_list):
            out[i, :3] = p0
            out[i, 3:7] = q0
            out[i, 7:10] = p1
            out[i, 10:14] = q1
        return out

    raise ValueError(f"Unsupported ee pose format: {first!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--robot", type=str, default="lift2", choices=list(ROBOT_CONFIGS.keys())
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing LeRobot dataset output if present.",
    )
    return parser.parse_args()


def _validate_episode_data(
    *,
    state_joints: np.ndarray,
    action_joints: np.ndarray,
    action_joints_delta: np.ndarray,
    state_gripper: np.ndarray,
    action_gripper: np.ndarray,
    state_base: np.ndarray | None,
    action_base: np.ndarray | None,
    action_base_delta: np.ndarray | None,
    state_ee_pose: np.ndarray,
    action_ee_pose: np.ndarray,
    action_ee_pose_delta: np.ndarray,
    camera_frame_counts: dict[str, int],
) -> None:
    num_frames = state_joints.shape[0]
    series = {
        "action_joints": action_joints,
        "action_joints_delta": action_joints_delta,
        "state_gripper": state_gripper,
        "action_gripper": action_gripper,
        "state_base": state_base,
        "action_base": action_base,
        "action_base_delta": action_base_delta,
        "state_ee_pose": state_ee_pose,
        "action_ee_pose": action_ee_pose,
        "action_ee_pose_delta": action_ee_pose_delta,
    }
    for name, arr in series.items():
        if arr is not None and arr.shape[0] != num_frames:
            raise ValueError(
                f"Frame count mismatch: state_joints has {num_frames}, "
                f"{name} has {arr.shape[0]}."
            )
    for camera_name, frame_count in camera_frame_counts.items():
        if frame_count != num_frames:
            raise ValueError(
                f"Video frame count mismatch for {camera_name}: "
                f"expected {num_frames}, got {frame_count}."
            )


def add_single_episode_to_dataset(
    dataset: LeRobotDataset,
    lmdb_path: str,
    robot_type: str,
):
    cameras = ROBOT_CONFIGS[robot_type]["cameras"]
    joint_indices = ROBOT_CONFIGS[robot_type]["joint_indices"]
    gripper_indices = ROBOT_CONFIGS[robot_type]["gripper_indices"]
    base_indices = ROBOT_CONFIGS[robot_type]["base_indices"]

    meta_info = _load_meta_info(lmdb_path)
    instruction = meta_info["task_data"]["instruction"]
    camera_key_indices = {
        camera_name: meta_info["keys"][f"observation/{camera_name}/color_image"]
        for camera_name in cameras
    }
    camera_frame_counts = {
        camera_name: len(key_index)
        for camera_name, key_index in camera_key_indices.items()
    }

    lmdb_env = _open_lmdb_env(lmdb_path)
    try:
        with lmdb_env.begin(write=False) as txn:
            scalar_key_set = set(meta_info["keys"]["scalar_data"])

            state_joints_all = np.asarray(
                _get_scalar_data(txn, scalar_key_set, b"observation/robot/qpos"),
                dtype=np.float32,
            )
            action_joints = np.asarray(
                _get_scalar_data(txn, scalar_key_set, b"arm_action"), dtype=np.float32
            )
            action_joints_delta = np.asarray(
                _get_scalar_data(txn, scalar_key_set, b"delta_arm_action"),
                dtype=np.float32,
            )
            action_gripper = np.asarray(
                _get_scalar_data(txn, scalar_key_set, b"gripper_action"),
                dtype=np.float32,
            )

            if action_gripper.ndim == 1:
                action_gripper = action_gripper[:, None]

            state_ee_pose = ee_pose_list_to_np(
                _get_scalar_data(
                    txn, scalar_key_set, b"observation/robot/ee_pose_state"
                )
            )
            action_ee_pose = ee_pose_list_to_np(
                _get_scalar_data(txn, scalar_key_set, b"ee_pose_action")
            )
            action_ee_pose_delta = ee_pose_list_to_np(
                _get_scalar_data(txn, scalar_key_set, b"delta_ee_pose_action")
            )

            state_joints = state_joints_all[:, joint_indices]
            state_gripper = state_joints_all[:, gripper_indices]

            state_base = None
            action_base = None
            action_base_delta = None
            if len(base_indices) > 0:
                state_base = state_joints_all[:, base_indices]
                if b"base_motion" in scalar_key_set:
                    action_base_delta = np.asarray(
                        _get_scalar_data(txn, scalar_key_set, b"base_motion"),
                        dtype=np.float32,
                    )
                else:
                    action_base_delta = np.zeros(
                        (action_joints.shape[0], len(base_indices)), dtype=np.float32
                    )

                action_base = _compute_absolute_base_action(
                    state_base, action_base_delta
                )

            _validate_episode_data(
                state_joints=state_joints,
                action_joints=action_joints,
                action_joints_delta=action_joints_delta,
                state_gripper=state_gripper,
                action_gripper=action_gripper,
                state_base=state_base,
                action_base=action_base,
                action_base_delta=action_base_delta,
                state_ee_pose=state_ee_pose,
                action_ee_pose=action_ee_pose,
                action_ee_pose_delta=action_ee_pose_delta,
                camera_frame_counts=camera_frame_counts,
            )

            process_single_dataset(
                dataset,
                state_joints=state_joints,
                action_joints=action_joints,
                action_joints_delta=action_joints_delta,
                state_gripper=state_gripper,
                action_gripper=action_gripper,
                state_base=state_base,
                action_base=action_base,
                action_base_delta=action_base_delta,
                state_ee_pose=state_ee_pose,
                action_ee_pose=action_ee_pose,
                action_ee_pose_delta=action_ee_pose_delta,
                txn=txn,
                camera_key_indices=camera_key_indices,
                instruction=instruction,
            )
    finally:
        lmdb_env.close()


if __name__ == "__main__":
    args = parse_args()
    ROBOT = args.robot
    is_dual_arm = ROBOT_CONFIGS[args.robot]["is_dual_arm"]
    dataset = create_dataset(
        repo_id=f"genmanip2lerobot/{args.data_path.split('/')[-2]}",
        robot_type=args.robot,
        is_dual_arm=is_dual_arm,
        mode="video",
        overwrite=args.overwrite,
    )
    for lmdb_path in tqdm(os.listdir(args.data_path)):
        if os.path.isdir(os.path.join(args.data_path, lmdb_path)):
            print("=" * 20)
            print(f"Processing episode from {lmdb_path}...")
            print("=" * 20)
            add_single_episode_to_dataset(
                dataset,
                os.path.join(args.data_path, lmdb_path),
                robot_type=args.robot,
            )
