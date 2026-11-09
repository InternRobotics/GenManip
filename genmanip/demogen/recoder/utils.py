"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from copy import deepcopy
import json
import os
import pickle
from typing import Any

import lmdb
import numpy as np

from omni.isaac.core.prims import XFormPrim  # type: ignore

from genmanip.core.scene.scene_config import SceneConfig


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


def _compose_action(
    arm_action: np.ndarray,
    gripper_action: np.ndarray,
    embodiment_name: str | None,
) -> np.ndarray:
    if embodiment_name in ("aloha_split", "lift2"):
        return np.concatenate(
            [arm_action[:6], gripper_action[:2], arm_action[6:], gripper_action[2:]]
        )
    return np.concatenate([arm_action, gripper_action])


def downsample_planning_result(data_list: list[dict], factor: int) -> list[dict]:
    """Downsample states while preserving each sampled transition target."""
    if factor < 1:
        raise ValueError(f"Downsample factor must be positive, got {factor}")
    if factor == 1:
        return data_list

    sampled_data = []
    for start in range(0, len(data_list), factor):
        stop = min(start + factor, len(data_list))
        state_frame = data_list[start].copy()
        action_frame = data_list[stop - 1]

        # action[stop - 1] is the target that produces state[stop], which is
        # the next retained state for a complete downsampling window.
        state_frame["action"] = np.array(action_frame["action"], copy=True)
        state_frame["gripper_close"] = deepcopy(action_frame["gripper_close"])
        state_frame["name"] = action_frame["name"]
        state_frame["base_motion"] = np.sum(
            [np.asarray(frame["base_motion"]) for frame in data_list[start:stop]],
            axis=0,
        )
        sampled_data.append(state_frame)

    return sampled_data


def parse_planning_result(
    dir: str,
    default_config: dict,
    task_name: str,
    object_list: dict[str, XFormPrim],
    embodiment_name: str | None = None,
) -> list[dict]:
    data_list = []
    data_dir = os.path.join(
        default_config["DEMONSTRATION_DIR"],
        task_name,
        "trajectory",
        dir,
    )
    json_data = json.load(open(os.path.join(data_dir, "lmdb", "info.json"), "r"))
    if "observation/articulation_mapping" in json_data:
        articulation_mapping = json_data["observation/articulation_mapping"]
    else:
        articulation_mapping = {}
    points_mapping = json_data.get("observation/points_mapping", {})

    qpos_data = get_scalar_data_from_lmdb(data_dir, b"observation/robot/qpos")
    qvel_data = get_scalar_data_from_lmdb(data_dir, b"observation/robot/qvel")
    arm_action_data = get_scalar_data_from_lmdb(data_dir, b"arm_action")
    gripper_action_data = get_scalar_data_from_lmdb(data_dir, b"gripper_action")
    gripper_close_data = get_scalar_data_from_lmdb(data_dir, b"gripper_close")
    try:
        base_motion_data = get_scalar_data_from_lmdb(data_dir, b"base_motion")
    except Exception:
        base_motion_data = [np.array([0.0, 0.0, 0.0])] * len(arm_action_data)
    name_data = get_scalar_data_from_lmdb(data_dir, b"name")
    meta_info = pickle.load(open(os.path.join(data_dir, "meta_info.pkl"), "rb"))
    scalar_keys = set(meta_info["keys"]["scalar_data"])

    prim_attributes_data = None
    prim_attributes_key = b"observation/prim_attributes"
    if prim_attributes_key in scalar_keys:
        prim_attributes_data = get_scalar_data_from_lmdb(data_dir, prim_attributes_key)

    try:
        joint_world_pose_data = get_scalar_data_from_lmdb(
            data_dir, b"observation/robot/joint_world_pose"
        )
    except Exception:
        joint_world_pose_data = None

    for frame_index, (
        qpos,
        qvel,
        arm_action,
        gripper_action,
        gripper_close,
        base_motion,
        name,
        joint_world_pose,
    ) in enumerate(
        zip(
            qpos_data,
            qvel_data,
            arm_action_data,
            gripper_action_data,
            gripper_close_data,
            base_motion_data,
            name_data,
            (
                joint_world_pose_data
                if joint_world_pose_data is not None
                else [None] * len(qpos_data)
            ),
        )
    ):
        data = {}
        data["qpos"] = qpos
        data["qvel"] = qvel
        data["action"] = _compose_action(arm_action, gripper_action, embodiment_name)
        data["gripper_close"] = gripper_close
        data["base_motion"] = base_motion
        data["name"] = name
        data["obj_info"] = {}
        data["articulation_info"] = {}
        data["points_info"] = {}
        data["prim_attributes"] = (
            prim_attributes_data[frame_index]
            if prim_attributes_data is not None
            and frame_index < len(prim_attributes_data)
            else {}
        )
        if joint_world_pose is not None:
            data["joint_world_pose"] = joint_world_pose
        data_list.append(data)

    for key, mapping in points_mapping.items():
        points_key = f"observation/points/{key}/points".encode("utf-8")
        if points_key not in scalar_keys:
            continue
        points_data = get_scalar_data_from_lmdb(data_dir, points_key)
        for data, points in zip(data_list, points_data):
            data["points_info"][key] = {
                "prim_path": mapping["prim_path"],
                "points": points,
            }

    for key in object_list:
        if not f"observation/obj_pose/{key}/position".encode("utf-8") in scalar_keys:
            continue
        position_data = get_scalar_data_from_lmdb(
            data_dir, f"observation/obj_pose/{key}/position".encode("utf-8")
        )
        for data, position in zip(data_list, position_data):
            data["obj_info"][key] = {}
            data["obj_info"][key]["position"] = position
        orientation_data = get_scalar_data_from_lmdb(
            data_dir, f"observation/obj_pose/{key}/orientation".encode("utf-8")
        )
        for data, orientation in zip(data_list, orientation_data):
            data["obj_info"][key]["orientation"] = orientation
        scale_data = get_scalar_data_from_lmdb(
            data_dir, f"observation/obj_pose/{key}/scale".encode("utf-8")
        )
        for data, scale in zip(data_list, scale_data):
            data["obj_info"][key]["scale"] = scale

    for key in articulation_mapping.keys():
        articulation_id = articulation_mapping[key]["articulation_id"]
        part_name = articulation_mapping[key]["part_name"]
        position_data = get_scalar_data_from_lmdb(
            data_dir,
            f"observation/articulation_part_pose/{articulation_id}/{part_name}/position".encode(
                "utf-8"
            ),
        )
        for data, position in zip(data_list, position_data):
            data["articulation_info"][articulation_id + "/" + part_name] = {}
            data["articulation_info"][articulation_id + "/" + part_name][
                "position"
            ] = position
        orientation_data = get_scalar_data_from_lmdb(
            data_dir,
            f"observation/articulation_part_pose/{articulation_id}/{part_name}/orientation".encode(
                "utf-8"
            ),
        )
        for data, orientation in zip(data_list, orientation_data):
            data["articulation_info"][articulation_id + "/" + part_name][
                "orientation"
            ] = orientation
        scale_data = get_scalar_data_from_lmdb(
            data_dir,
            f"observation/articulation_part_pose/{articulation_id}/{part_name}/scale".encode(
                "utf-8"
            ),
        )
        for data, scale in zip(data_list, scale_data):
            data["articulation_info"][articulation_id + "/" + part_name][
                "scale"
            ] = scale
        for data in data_list:
            data["articulation_info"][articulation_id + "/" + part_name][
                "prim_path"
            ] = articulation_mapping[articulation_id + "/" + part_name]["prim_path"]
    return data_list
