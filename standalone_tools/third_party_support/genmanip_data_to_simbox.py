"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import pickle
from copy import deepcopy
from typing import Any

import lmdb
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta
    return np.array(
        [
            [np.cos(q), -np.sin(q), 0, a],
            [
                np.sin(q) * np.cos(alpha),
                np.cos(q) * np.cos(alpha),
                -np.sin(alpha),
                -np.sin(alpha) * d,
            ],
            [
                np.sin(q) * np.sin(alpha),
                np.cos(q) * np.sin(alpha),
                np.cos(alpha),
                np.cos(alpha) * d,
            ],
            [0, 0, 0, 1],
        ]
    )


def get_fk_solution(joint_angles, target_index=8):
    dh_params = [
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi / 2, joint_angles[1]],
        [0, 0.316, np.pi / 2, joint_angles[2]],
        [0.0825, 0, np.pi / 2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
        [0, 0, np.pi / 2, joint_angles[5]],
        [0.088, 0, np.pi / 2, joint_angles[6]],
        [0, 0.107, 0, 0],
        [0, 0, 0, -np.pi / 4],
        [0.0, 0.1034, 0, 0],
    ]

    T = np.eye(4)
    for i in range(target_index):
        T = T @ get_tf_mat(i, dh_params)
    return T


def get_scalar_data_from_lmdb(data_path: str, key: bytes) -> list[Any]:
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"]["scalar_data"].index(key)
    key_key = meta_info["keys"]["scalar_data"][key_index]
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(key_key))
    return data


def get_color_image_from_lmdb(data_path: str, key: str):
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    with lmdb_env.begin(write=False) as txn:
        color_image = [pickle.loads(txn.get(key)) for key in key_index]
    lmdb_env.close()
    return color_image


def cleanup_lmdb(data_path: str):
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=False, lock=False, readahead=False, meminit=False
    )
    with lmdb_env.begin(write=True) as txn:
        cursor = txn.cursor()
        keys_to_delete = []
        for key, value in cursor:
            keys_to_delete.append(key)
        for key in keys_to_delete:
            txn.delete(key)
    lmdb_env.close()


def save_to_lmdb(data_path: str, key: str, data: Any):
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=False, lock=False, readahead=False, meminit=False
    )
    with lmdb_env.begin(write=True) as txn:
        txn.put(key.encode("utf-8"), pickle.dumps(data))
    lmdb_env.close()


def compact_lmdb(db_path: str):
    print(f"Compacting LMDB at {db_path}...")

    temp_path = db_path + "_compact"

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    env = lmdb.open(db_path, readonly=True, lock=False)

    try:
        env.copy(temp_path, compact=True)
    finally:
        env.close()

    shutil.rmtree(db_path)
    os.rename(temp_path, db_path)

    print(f"Compaction done. Size reduced.")


def delete_depth_images_from_lmdb(data_path: str) -> int:
    """
    Delete all image keys (color_image and depth_image) from lmdb database.

    Args:
        data_path: Path to the data directory containing lmdb folder

    Returns:
        Number of image keys deleted
    """
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb",
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=1 << 40,
    )

    keys_to_delete = []
    with lmdb_env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode("utf-8")
            # Delete keys that contain image data (color_image or depth_image)
            if "observation/" in key_str and (
                "color_image" in key_str or "depth_image" in key_str
            ):
                keys_to_delete.append(key)

    # Delete the keys
    with lmdb_env.begin(write=True) as txn:
        for key in keys_to_delete:
            txn.delete(key)

    lmdb_env.close()

    # Update meta_info.pkl to remove image references
    meta_info_path = f"{data_path}/meta_info.pkl"
    if os.path.exists(meta_info_path):
        meta_info = pickle.load(open(meta_info_path, "rb"))
        if "keys" in meta_info:
            # Remove image related keys from meta_info
            keys_to_remove = []
            for key_name in meta_info["keys"]:
                if "color_image" in key_name or "depth_image" in key_name:
                    keys_to_remove.append(key_name)

            for key_name in keys_to_remove:
                del meta_info["keys"][key_name]

            pickle.dump(meta_info, open(meta_info_path, "wb"))

    print(f"Deleted {len(keys_to_delete)} image keys (RGB and depth) from {data_path}")

    if len(keys_to_delete) > 0:
        compact_lmdb(f"{data_path}/lmdb")
    else:
        print(f"No image keys (RGB and depth) to delete from {data_path}")
    return len(keys_to_delete)


def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


def process_single_data(data_path: str):
    print(f"Processing {data_path}")

    try:
        meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
        print(f"Error loading meta_info.pkl from {data_path}: {e}")
        print(f"Skipping {data_path} due to corrupted or missing pickle file")
        shutil.rmtree(f"{data_path}")
        return None

    if "states.joint.position" in meta_info["keys"]:
        print(f"Skipping {data_path} because it already processed")
        return None

    # 1. get state data
    joint_position = get_scalar_data_from_lmdb(data_path, b"observation/robot/qpos")
    states_joint_position = [pos[:7] for pos in joint_position]
    states_gripper_pose = [
        pose_to_6d(get_fk_solution(pos)) for pos in states_joint_position
    ]

    # 2. get action data
    actions_joint_position = deepcopy(states_joint_position)[1:]
    actions_joint_position.append(actions_joint_position[-1])
    actions_gripper_pose = [
        pose_to_6d(get_fk_solution(pos)) for pos in actions_joint_position
    ]
    actions_gripper_openness = get_scalar_data_from_lmdb(data_path, b"gripper_close")
    actions_gripper_openness = [
        1 if openness == -1 else 0 for openness in actions_gripper_openness
    ]
    actions_gripper_position = deepcopy(actions_gripper_openness)
    actions_gripper_position = [np.array([pos]) for pos in actions_gripper_position]
    states_gripper_position = [actions_gripper_position[0]] + actions_gripper_position[
        :-1
    ]

    # 3. get master action data
    master_actions_joint_position = get_scalar_data_from_lmdb(data_path, b"arm_action")
    master_actions_gripper_openness = deepcopy(actions_gripper_openness)
    master_actions_gripper_pose = [
        pose_to_6d(get_fk_solution(pos)) for pos in master_actions_joint_position
    ]
    master_actions_gripper_position = deepcopy(master_actions_gripper_openness)
    master_actions_gripper_position = [
        np.array([pos]) for pos in master_actions_gripper_position
    ]

    # 4. get observation data
    images_rgb_head = get_color_image_from_lmdb(
        data_path, "observation/obs_camera/color_image"
    )
    images_rgb_hand = get_color_image_from_lmdb(
        data_path, "observation/realsense/color_image"
    )

    # 5. save to lmdb
    _meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    meta_info = {}
    meta_info["language_instruction"] = _meta_info["task_data"]["instruction"]
    meta_info["num_steps"] = len(states_joint_position)
    meta_info["keys"] = {}
    lmdb_env = lmdb.open(f"{data_path}/lmdb", map_size=1 << 40)
    txn = lmdb_env.begin(write=True)

    txn.put(
        "states.joint.position".encode("utf-8"), pickle.dumps(states_joint_position)
    )
    meta_info["keys"]["states.joint.position"] = "states.joint.position".encode("utf-8")
    txn.put(
        "states.gripper.position".encode("utf-8"), pickle.dumps(states_gripper_position)
    )
    meta_info["keys"]["states.gripper.position"] = "states.gripper.position".encode(
        "utf-8"
    )
    txn.put("states.gripper.pose".encode("utf-8"), pickle.dumps(states_gripper_pose))
    meta_info["keys"]["states.gripper.pose"] = "states.gripper.pose".encode("utf-8")
    txn.put(
        "actions.joint.position".encode("utf-8"), pickle.dumps(actions_joint_position)
    )
    meta_info["keys"]["actions.joint.position"] = "actions.joint.position".encode(
        "utf-8"
    )
    txn.put(
        "actions.gripper.position".encode("utf-8"),
        pickle.dumps(actions_gripper_position),
    )
    meta_info["keys"]["actions.gripper.position"] = "actions.gripper.position".encode(
        "utf-8"
    )
    txn.put("actions.gripper.pose".encode("utf-8"), pickle.dumps(actions_gripper_pose))
    meta_info["keys"]["actions.gripper.pose"] = "actions.gripper.pose".encode("utf-8")
    txn.put(
        "actions.gripper.openness".encode("utf-8"),
        pickle.dumps(actions_gripper_openness),
    )
    meta_info["keys"]["actions.gripper.openness"] = "actions.gripper.openness".encode(
        "utf-8"
    )
    txn.put(
        "master_actions.joint.position".encode("utf-8"),
        pickle.dumps(master_actions_joint_position),
    )
    meta_info["keys"]["master_actions.joint.position"] = (
        "master_actions.joint.position".encode("utf-8")
    )
    txn.put(
        "master_actions.gripper.position".encode("utf-8"),
        pickle.dumps(master_actions_gripper_position),
    )
    meta_info["keys"]["master_actions.gripper.position"] = (
        "master_actions.gripper.position".encode("utf-8")
    )
    txn.put(
        "master_actions.gripper.pose".encode("utf-8"),
        pickle.dumps(master_actions_gripper_pose),
    )
    meta_info["keys"]["master_actions.gripper.pose"] = (
        "master_actions.gripper.pose".encode("utf-8")
    )
    txn.put(
        "master_actions.gripper.openness".encode("utf-8"),
        pickle.dumps(master_actions_gripper_openness),
    )
    meta_info["keys"]["master_actions.gripper.openness"] = (
        "master_actions.gripper.openness".encode("utf-8")
    )

    meta_info["keys"]["images.rgb.head"] = []
    meta_info["keys"]["images.rgb.hand"] = []
    for i in range(len(images_rgb_head)):
        txn.put(
            f"images.rgb.head/{i:04d}".encode("utf-8"),
            pickle.dumps(images_rgb_head[i]),
        )
        meta_info["keys"]["images.rgb.head"].append(
            f"images.rgb.head/{i:04d}".encode("utf-8")
        )
        txn.put(
            f"images.rgb.hand/{i:04d}".encode("utf-8"),
            pickle.dumps(images_rgb_hand[i]),
        )
        meta_info["keys"]["images.rgb.hand"].append(
            f"images.rgb.hand/{i:04d}".encode("utf-8")
        )

    txn.commit()
    pickle.dump(meta_info, open(f"{data_path}/meta_info.pkl", "wb"))
    lmdb_env.close()

    # Delete all images (RGB and depth) from lmdb
    num_deleted = delete_depth_images_from_lmdb(data_path)


def process_single_data_dualarm(data_path: str, robot_type: str = "aloha"):
    print(f"Processing {data_path}")
    if robot_type == "aloha":
        left_arm_dof_indices = [12, 14, 16, 18, 20, 22]
        right_arm_dof_indices = [13, 15, 17, 19, 21, 23]
        left_gripper_dof_indices = [24]
        right_gripper_dof_indices = [26]
    elif robot_type == "lift2":
        left_arm_dof_indices = [10, 12, 14, 16, 18, 20]
        right_arm_dof_indices = [9, 11, 13, 15, 17, 19]
        left_gripper_dof_indices = [23]
        right_gripper_dof_indices = [21]
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    try:
        meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
        print(f"Error loading meta_info.pkl from {data_path}: {e}")
        print(f"Skipping {data_path} due to corrupted or missing pickle file")
        shutil.rmtree(f"{data_path}")
        return None

    if "states.left_joint.position" in meta_info["keys"]:
        print(f"Skipping {data_path} because it already processed")
        return None

    # 1. get state data
    joint_position = get_scalar_data_from_lmdb(data_path, b"observation/robot/qpos")
    states_left_joint_position = [pos[left_arm_dof_indices] for pos in joint_position]
    states_right_joint_position = [pos[right_arm_dof_indices] for pos in joint_position]
    states_left_gripper_position = [
        abs(pos[left_gripper_dof_indices]) * 2 for pos in joint_position
    ]
    states_right_gripper_position = [
        abs(pos[right_gripper_dof_indices]) * 2 for pos in joint_position
    ]

    # 2. get action data
    actions_left_joint_position = deepcopy(states_left_joint_position)[1:]
    actions_left_joint_position.append(actions_left_joint_position[-1])
    actions_right_joint_position = deepcopy(states_right_joint_position)[1:]
    actions_right_joint_position.append(actions_right_joint_position[-1])
    actions_gripper_openness = get_scalar_data_from_lmdb(data_path, b"gripper_close")
    actions_left_gripper_openness = [
        1 if openness[0] == -1 else 0 for openness in actions_gripper_openness
    ]
    actions_right_gripper_openness = [
        1 if openness[1] == -1 else 0 for openness in actions_gripper_openness
    ]
    actions_left_gripper_position = deepcopy(states_left_gripper_position)[1:]
    actions_left_gripper_position.append(actions_left_gripper_position[-1])
    actions_right_gripper_position = deepcopy(states_right_gripper_position)[1:]
    actions_right_gripper_position.append(actions_right_gripper_position[-1])

    # 3. get master action data
    master_actions_joint_position = get_scalar_data_from_lmdb(data_path, b"arm_action")
    master_actions_left_joint_position = [
        pos[:6] for pos in master_actions_joint_position
    ]
    master_actions_right_joint_position = [
        pos[6:] for pos in master_actions_joint_position
    ]
    master_actions_left_gripper_openness = deepcopy(actions_left_gripper_openness)
    master_actions_right_gripper_openness = deepcopy(actions_right_gripper_openness)
    master_actions_left_gripper_position = deepcopy(actions_left_gripper_position)
    master_actions_right_gripper_position = deepcopy(actions_right_gripper_position)

    # 4. get observation data
    images_rgb_head = get_color_image_from_lmdb(
        data_path, "observation/top_camera/color_image"
    )
    images_rgb_hand_left = get_color_image_from_lmdb(
        data_path, "observation/left_camera/color_image"
    )
    images_rgb_hand_right = get_color_image_from_lmdb(
        data_path, "observation/right_camera/color_image"
    )
    # 5. save to lmdb
    _meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    meta_info = {}
    meta_info["language_instruction"] = _meta_info["task_data"]["instruction"]
    meta_info["num_steps"] = len(states_left_joint_position)
    meta_info["keys"] = {}
    lmdb_env = lmdb.open(f"{data_path}/lmdb", map_size=1 << 40)
    txn = lmdb_env.begin(write=True)

    txn.put(
        "states.left_joint.position".encode("utf-8"),
        pickle.dumps(states_left_joint_position),
    )
    meta_info["keys"]["states.left_joint.position"] = (
        "states.left_joint.position".encode("utf-8")
    )
    txn.put(
        "states.right_joint.position".encode("utf-8"),
        pickle.dumps(states_right_joint_position),
    )
    meta_info["keys"]["states.right_joint.position"] = (
        "states.right_joint.position".encode("utf-8")
    )
    txn.put(
        "states.left_gripper.position".encode("utf-8"),
        pickle.dumps(states_left_gripper_position),
    )
    meta_info["keys"]["states.left_gripper.position"] = (
        "states.left_gripper.position".encode("utf-8")
    )
    txn.put(
        "states.right_gripper.position".encode("utf-8"),
        pickle.dumps(states_right_gripper_position),
    )
    meta_info["keys"]["states.right_gripper.position"] = (
        "states.right_gripper.position".encode("utf-8")
    )

    txn.put(
        "actions.left_joint.position".encode("utf-8"),
        pickle.dumps(actions_left_joint_position),
    )
    meta_info["keys"]["actions.left_joint.position"] = (
        "actions.left_joint.position".encode("utf-8")
    )
    txn.put(
        "actions.right_joint.position".encode("utf-8"),
        pickle.dumps(actions_right_joint_position),
    )
    meta_info["keys"]["actions.right_joint.position"] = (
        "actions.right_joint.position".encode("utf-8")
    )
    txn.put(
        "actions.left_gripper.position".encode("utf-8"),
        pickle.dumps(actions_left_gripper_position),
    )
    meta_info["keys"]["actions.left_gripper.position"] = (
        "actions.left_gripper.position".encode("utf-8")
    )
    txn.put(
        "actions.right_gripper.position".encode("utf-8"),
        pickle.dumps(actions_right_gripper_position),
    )
    meta_info["keys"]["actions.right_gripper.position"] = (
        "actions.right_gripper.position".encode("utf-8")
    )
    txn.put(
        "actions.left_gripper.openness".encode("utf-8"),
        pickle.dumps(actions_left_gripper_openness),
    )
    meta_info["keys"]["actions.left_gripper.openness"] = (
        "actions.left_gripper.openness".encode("utf-8")
    )
    txn.put(
        "actions.right_gripper.openness".encode("utf-8"),
        pickle.dumps(actions_right_gripper_openness),
    )
    meta_info["keys"]["actions.right_gripper.openness"] = (
        "actions.right_gripper.position".encode("utf-8")
    )

    txn.put(
        "master_actions.left_joint.position".encode("utf-8"),
        pickle.dumps(master_actions_left_joint_position),
    )
    meta_info["keys"]["master_actions.left_joint.position"] = (
        "master_actions.left_joint.position".encode("utf-8")
    )
    txn.put(
        "master_actions.right_joint.position".encode("utf-8"),
        pickle.dumps(master_actions_right_joint_position),
    )
    meta_info["keys"]["master_actions.right_joint.position"] = (
        "master_actions.right_joint.position".encode("utf-8")
    )
    txn.put(
        "master_actions.left_gripper.position".encode("utf-8"),
        pickle.dumps(master_actions_left_gripper_position),
    )
    meta_info["keys"]["master_actions.left_gripper.position"] = (
        "master_actions.left_gripper.position".encode("utf-8")
    )
    txn.put(
        "master_actions.right_gripper.position".encode("utf-8"),
        pickle.dumps(master_actions_right_gripper_position),
    )
    meta_info["keys"]["master_actions.right_gripper.position"] = (
        "master_actions.right_gripper.position".encode("utf-8")
    )
    txn.put(
        "master_actions.left_gripper.openness".encode("utf-8"),
        pickle.dumps(master_actions_left_gripper_openness),
    )
    meta_info["keys"]["master_actions.left_gripper.openness"] = (
        "master_actions.left_gripper.openness".encode("utf-8")
    )
    txn.put(
        "master_actions.right_gripper.openness".encode("utf-8"),
        pickle.dumps(master_actions_right_gripper_openness),
    )
    meta_info["keys"]["master_actions.right_gripper.openness"] = (
        "master_actions.right_gripper.openness".encode("utf-8")
    )

    meta_info["keys"]["images.rgb.head"] = []
    meta_info["keys"]["images.rgb.hand_left"] = []
    meta_info["keys"]["images.rgb.hand_right"] = []
    for i in range(len(images_rgb_head)):
        txn.put(
            f"images.rgb.head/{i:04d}".encode("utf-8"),
            pickle.dumps(images_rgb_head[i]),
        )
        meta_info["keys"]["images.rgb.head"].append(
            f"images.rgb.head/{i:04d}".encode("utf-8")
        )
        txn.put(
            f"images.rgb.hand_left/{i:04d}".encode("utf-8"),
            pickle.dumps(images_rgb_hand_left[i]),
        )
        meta_info["keys"]["images.rgb.hand_left"].append(
            f"images.rgb.hand_left/{i:04d}".encode("utf-8")
        )
        txn.put(
            f"images.rgb.hand_right/{i:04d}".encode("utf-8"),
            pickle.dumps(images_rgb_hand_right[i]),
        )
        meta_info["keys"]["images.rgb.hand_right"].append(
            f"images.rgb.hand_right/{i:04d}".encode("utf-8")
        )

    txn.commit()
    pickle.dump(meta_info, open(f"{data_path}/meta_info.pkl", "wb"))
    lmdb_env.close()

    # Delete all images (RGB and depth) from lmdb
    num_deleted = delete_depth_images_from_lmdb(data_path)


if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--dualarm", action="store_true")
    parser.add_argument("--robot_type", type=str, default="aloha")
    parser.add_argument("--delete_only", action="store_true")
    args = parser.parse_args()
    if args.backup:
        if os.path.exists(f"{args.data_path}_simbox"):
            shutil.rmtree(f"{args.data_path}_simbox")
        shutil.copytree(args.data_path, f"{args.data_path}_simbox")
        args.data_path = f"{args.data_path}_simbox"
    path_list = os.listdir(args.data_path)
    path_list = [
        path
        for path in path_list
        if os.path.isdir(os.path.join(args.data_path, path))
        and os.path.exists(os.path.join(args.data_path, path, "meta_info.pkl"))
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    if args.delete_only:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    delete_depth_images_from_lmdb, os.path.join(args.data_path, path)
                )
                for path in path_list
            ]
            for future in as_completed(futures):
                future.result()
        exit()

    # multi-process

    if args.dualarm:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    process_single_data_dualarm,
                    os.path.join(args.data_path, path),
                    robot_type=args.robot_type,
                )
                for path in path_list
            ]
            for future in as_completed(futures):
                future.result()
    else:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(process_single_data, os.path.join(args.data_path, path))
                for path in path_list
            ]
            for future in as_completed(futures):
                future.result()
