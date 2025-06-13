from datetime import datetime
import os
import json
import pickle
import numpy as np
import lmdb
from pathlib import Path
from typing import Any
from genmanip.core.sensor.camera import get_pixel_from_world_point, get_intrinsic_matrix
from genmanip.core.robot.franka import create_joint_xform_list
from omni.isaac.franka import Franka  # type: ignore
import shutil
from genmanip.core.robot.franka import create_tcp_xform_list
from genmanip.utils.transform_utils import (
    compute_delta_eepose,
    pose_to_transform,
    transform_to_pose,
    compute_pose2,
)

DEFAULT_RGB_SCALE_FACTOR = 256000.0


def collect_task_data(object_list, franka_list, camera_data, task_data, usd_path_list):
    task_data["initial_scene_graph"] = None
    task_data["initial_layout"] = {}
    for key in object_list:
        task_data["initial_layout"][key] = {}
        task_data["initial_layout"][key]["position"] = object_list[
            key
        ].get_world_pose()[0]
        task_data["initial_layout"][key]["orientation"] = object_list[
            key
        ].get_world_pose()[1]
        task_data["initial_layout"][key]["scale"] = object_list[key].get_local_scale()
        if key in usd_path_list:
            task_data["initial_layout"][key]["path"] = usd_path_list[key]
        else:
            task_data["initial_layout"][key]["path"] = ""
        task_data["initial_layout"][key]["prim_path"] = object_list[key].prim_path
    for embodiment in franka_list:
        task_data["initial_layout"][embodiment.robot.name] = {}
        task_data["initial_layout"][embodiment.robot.name]["position"] = embodiment.robot.get_world_pose()[
            0
        ]
        task_data["initial_layout"][embodiment.robot.name][
            "orientation"
        ] = embodiment.robot.get_world_pose()[1]
        task_data["initial_layout"][embodiment.robot.name][
            "joint_positions"
        ] = embodiment.robot.get_joint_positions()
    task_data["camera_data"] = camera_data
    return task_data


def clip_float_values(
    float_array: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    return np.clip(float_array, min_value, max_value)


def encode_seg_mask(seg_mask: np.ndarray):
    assert seg_mask.shape[0] % 2 == 0
    assert seg_mask.shape[1] % 2 == 0
    h, w = seg_mask.shape[0], seg_mask.shape[1]
    reshaped = seg_mask.reshape(h // 2, 2, w // 2, 2)
    reshaped = reshaped.transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    encoded = np.zeros((h // 2, w // 2, 4), dtype=np.uint8)
    encoded[..., 0] = reshaped[..., 0]
    encoded[..., 1] = reshaped[..., 1]
    encoded[..., 2] = reshaped[..., 2]
    encoded[..., 3] = reshaped[..., 3]
    return encoded


def decode_seg_mask(seg_mask: np.ndarray):
    assert seg_mask.shape[2] == 4
    h, w = seg_mask.shape[0], seg_mask.shape[1]
    decoded = np.zeros((h * 2, w * 2), dtype=np.uint8)
    decoded[0::2, 0::2] = seg_mask[..., 0]
    decoded[0::2, 1::2] = seg_mask[..., 1]
    decoded[1::2, 0::2] = seg_mask[..., 2]
    decoded[1::2, 1::2] = seg_mask[..., 3]
    return decoded


def float_array_to_rgb_image(
    float_array: np.ndarray,
    scale_factor: float = DEFAULT_RGB_SCALE_FACTOR,
    drop_blue: bool = False,
    min_inttype: int = 0,
    max_inttype: int = 2**24 - 1,
) -> np.ndarray:
    scaled_array = np.round(float_array * scale_factor)
    scaled_array = clip_float_values(scaled_array, min_inttype, max_inttype)
    int_array = scaled_array.astype(np.uint32)
    r = int_array // (256 * 256)
    g = (int_array // 256) % 256
    b = int_array % 256 if not drop_blue else np.zeros_like(int_array)
    rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb_array


def image_to_float_array(image: np.ndarray, scale_factor: float = None) -> np.ndarray:
    image_array = np.asarray(image)
    if scale_factor is None:
        scale_factor = DEFAULT_RGB_SCALE_FACTOR
    float_array = np.dot(image_array, [65536, 256, 1])
    return float_array / scale_factor


class Logger:
    def __init__(
        self,
        cameras,
        franka: Franka,
        object_list,
        instruction: str,
        log_dir: str = "logs",
        max_size: int = 1,  # Size in TB
        name: str = None,
        task_data: dict = None,
        tcp_config: dict = None,
    ):
        if name is None:
            self.name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        else:
            self.name = name
        self.instruction = instruction
        self.log_dir = Path(f"{log_dir}/{self.name}")
        print(self.log_dir)
        self.max_size = int(max_size * 1024**4)
        self.json_data_logger = {}
        self.scalar_data_logger = {}
        self.color_image_logger = {}
        self.depth_image_logger = {}
        self.obj_mask_logger = {}
        self.cameras = cameras
        self.franka = franka
        self.log_num_steps = 0
        self.task_data = task_data
        self.tcp_config = tcp_config
        # self.tcp_xform_list = create_tcp_xform_list(franka, self.tcp_config)
        self.frame_status = {}
        self.load_static_info()
        # self.joint_xform_list = create_joint_xform_list(franka)
        self.object_list = object_list

    def load_static_info(self):
        for camera_name, camera in self.cameras.items():
            intrinsics_matrix = get_intrinsic_matrix(camera)
            self.add_json_data(
                f"observation/{camera_name}/camera_params",
                intrinsics_matrix.tolist(),
            )
        self.add_json_data(
            f"observation/tcp_config",
            self.tcp_config,
        )

    def load_dynamic_info(self, action, gripper_close, name=None):
        self.add_name_frame(name)
        self.add_scalar_data(f"arm_action", action[:7])
        self.add_scalar_data(f"gripper_action", action[7:])
        self.add_scalar_data(f"gripper_close", gripper_close)
        self.add_scalar_data(f"name", name)
        for key in self.object_list:
            self.add_scalar_data(
                f"observation/obj_pose/{key}/position",
                self.object_list[key].get_world_pose()[0],
            )
            self.add_scalar_data(
                f"observation/obj_pose/{key}/orientation",
                self.object_list[key].get_world_pose()[1],
            )
            self.add_scalar_data(
                f"observation/obj_pose/{key}/scale",
                self.object_list[key].get_local_scale(),
            )
        self.add_scalar_data(
            f"observation/robot/qpos", self.franka.get_joint_positions()
        )
        # joint_world_pose = {}
        # for joint_name, joint_xform in self.joint_xform_list.items():
        #     joint_world_pose[joint_name] = joint_xform.get_world_pose()
        # self.add_scalar_data(f"observation/robot/joint_world_pose", joint_world_pose)
        self.add_scalar_data(
            f"observation/robot/qvel", self.franka.get_joint_velocities()
        )
        self.add_scalar_data(
            f"observation/robot/robot2env_pose",
            pose_to_transform(self.franka.get_world_pose()),
        )
        self.log_num_steps += 1

    def add_name_frame(self, name):
        if name not in self.frame_status:
            self.frame_status[name] = self.log_num_steps

    def add_scalar_data(self, key: str, value: Any):
        if key not in self.scalar_data_logger:
            self.scalar_data_logger[key] = []
        self.scalar_data_logger[key].append(value)

    def add_color_image(self, key: str, value: np.ndarray):
        if key not in self.color_image_logger:
            self.color_image_logger[key] = []
        self.color_image_logger[key].append(value)

    def add_obj_mask(self, key: str, value: np.ndarray):
        if key not in self.obj_mask_logger:
            self.obj_mask_logger[key] = []
        self.obj_mask_logger[key].append(value)

    def add_depth_image(self, key: str, value: np.ndarray):
        if key not in self.depth_image_logger:
            self.depth_image_logger[key] = []
        self.depth_image_logger[key].append(value)

    def add_json_data(self, key, data):
        self.json_data_logger[key] = data

    def save(self, task_name=None, config_path=None):
        # if os.path.exists(self.log_dir):
        #     return False
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_dir_lmdb = self.log_dir / "lmdb"
        meta_info = {}
        meta_info["max_size"] = self.max_size
        meta_info["num_steps"] = self.log_num_steps
        meta_info["language_instruction"] = self.instruction
        meta_info["task_data"] = self.task_data
        meta_info["task_data"]["frame_status"] = self.frame_status
        meta_info["keys"] = {}
        if task_name is not None:
            meta_info["task_name"] = task_name
        else:
            meta_info["task_name"] = ""
        meta_info["episode_name"] = self.name
        if (
            "arm_action" not in self.scalar_data_logger
            or len(self.scalar_data_logger["arm_action"]) == 0
        ):
            pickle.dump(
                meta_info, open(os.path.join(self.log_dir, "meta_info.pkl"), "wb")
            )
            if config_path is not None:
                shutil.copy(config_path, self.log_dir / "config.yaml")
            self.set_permissions(self.log_dir)
            return True
        self.env = lmdb.open(str(log_dir_lmdb), map_size=self.max_size)
        txn = self.env.begin(write=True)
        with open(log_dir_lmdb / "info.json", "w") as f:
            json.dump(self.json_data_logger, f)
        txn.put("json_data".encode("utf-8"), pickle.dumps(self.json_data_logger))
        meta_info["keys"]["json_data"] = ["json_data".encode("utf-8")]
        meta_info["keys"]["scalar_data"] = []
        for key, value in self.scalar_data_logger.items():
            txn.put(key.encode("utf-8"), pickle.dumps(value))
            meta_info["keys"]["scalar_data"].append(key.encode("utf-8"))
        txn.commit()
        self.env.close()
        pickle.dump(meta_info, open(os.path.join(self.log_dir, "meta_info.pkl"), "wb"))
        if config_path is not None:
            shutil.copy(config_path, self.log_dir / "config.yaml")
        self.set_permissions(self.log_dir)
        print(f"save data with length {self.log_num_steps} to {self.log_dir}")
        return True

    def set_permissions(self, path):
        os.chmod(path, 0o777)
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                os.chmod(os.path.join(root, dir_name), 0o777)
            for file_name in files:
                os.chmod(os.path.join(root, file_name), 0o777)
