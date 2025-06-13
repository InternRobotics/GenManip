import numpy as np
import os
import pickle
from genmanip.core.sensor.camera import get_src
from genmanip.utils.frame_utils import create_video_from_image_folder, save_image
from genmanip.utils.robot_utils import joint_positions_to_ee_pose_translation_euler
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path
from genmanip.core.pointcloud.pointcloud import meshlist_to_pclist, get_current_meshList
from genmanip.core.loading.loading import reset_object_xyz, collect_world_pose_list
from genmanip.core.random_place.random_place import place_object_to_object_by_relation
from genmanip.core.sensor.camera import collect_camera_info, get_eval_camera_data
from genmanip_bench.request_model.detect_target import detect_target_is_grasped
from genmanip_bench.request_model.request_model import request_action
from genmanip.utils.file_utils import make_dir, save_dict_to_json
from genmanip.core.robot.franka import joint_position_to_end_effector_pose
from genmanip.thirdparty.mplib_planner import get_mplib_planner

from mplib import Pose
import json
import lmdb
import pickle


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


def parse_lmdb_data(lmdb_path):
    data = {}
    meta_info = pickle.load(open(f"{lmdb_path}/meta_info.pkl", "rb"))
    arm_action = get_scalar_data_from_lmdb(lmdb_path, b"arm_action")
    key_action = []
    idx_dict = {}
    for frame_status in meta_info["task_data"]["frame_status"]:
        idx = int(frame_status.split("/")[0])
        if idx not in idx_dict:
            idx_dict[idx] = []
        idx_dict[idx].append(frame_status)
    for idx in idx_dict.keys():
        if (
            f"{idx}/pre_grasp" in meta_info["task_data"]["frame_status"]
            and f"{idx}/post_grasp" in meta_info["task_data"]["frame_status"]
            and f"{idx}/post_place" in meta_info["task_data"]["frame_status"]
        ):
            key_action.append(
                [
                    joint_position_to_end_effector_pose(
                        arm_action[
                            meta_info["task_data"]["frame_status"][f"{idx}/pre_grasp"]
                        ]
                    ),
                    joint_position_to_end_effector_pose(
                        arm_action[
                            meta_info["task_data"]["frame_status"][f"{idx}/post_grasp"]
                        ]
                    ),
                    joint_position_to_end_effector_pose(
                        arm_action[
                            meta_info["task_data"]["frame_status"][f"{idx}/post_place"]
                        ]
                    ),
                ]
            )
        else:
            for action_name in idx_dict[idx]:
                key_action.append(
                    [
                        joint_position_to_end_effector_pose(
                            arm_action[
                                meta_info["task_data"]["frame_status"][action_name]
                            ]
                        )
                    ]
                )
    data["key_action"] = key_action
    return data


class Evaluator:
    def __init__(
        self,
        scene,
        instruction,
        log_dir,
        current_dir,
        is_relative_action=False,
        send_port=None,
        receive_port=None,
    ):
        self.scene = scene
        self.obs_camera = scene["camera_list"]["obs_camera"]
        self.realsense_camera = scene["camera_list"]["realsense"]
        self.franka = scene["robot_info"]["robot_list"][0].robot
        self.instruction = instruction
        self.success_cnt = 0
        self.total_cnt = 0
        self.log_dir = log_dir
        self.send_port = send_port
        self.receive_port = receive_port
        self.is_relative_action = is_relative_action
        self.current_joint_position = self.franka.get_joint_positions()[:7]
        self.last_joint_position = self.franka.get_joint_positions()[:7]
        self.meta_record = {}
        self.task_data = []
        self.oracle_camera_data = {}
        self.grasp_cnt = 0
        self.planning_data = {}
        self.planner = get_mplib_planner(self.franka, "franka", current_dir)
        make_dir(self.log_dir)

    def update_task_data(self, task_data, planning_data):
        self.task_data = task_data
        self.planning_data = planning_data

    def finish(self, success, success_rate):
        if len(os.listdir(os.path.join(self.traj_log_dir, "realsense"))) > 0:
            create_video_from_image_folder(
                os.path.join(self.traj_log_dir, "realsense"),
                os.path.join(self.traj_log_dir, "realsense.mp4"),
            )
        if len(os.listdir(os.path.join(self.traj_log_dir, "obs"))) > 0:
            create_video_from_image_folder(
                os.path.join(self.traj_log_dir, "obs"),
                os.path.join(self.traj_log_dir, "obs.mp4"),
            )
        self.save_meta_record()
        new_traj_log_dir = self.traj_log_dir + (
            "_success" if success != 0 else "_failure"
        )
        os.rename(self.traj_log_dir, new_traj_log_dir)
        sr_info = {"success_rate": success_rate}
        save_dict_to_json(sr_info, os.path.join(new_traj_log_dir, "sr_info.json"))
        if success != 0:
            self.success_cnt += 1
        self.total_cnt += 1

    def initialize(self, seed):
        self.grasp_cnt = 0
        self.steps = 0
        self.oracle_camera_data = {}
        self.last_joint_position = self.franka.get_joint_positions()[:7]
        self.traj_log_dir = os.path.join(self.log_dir, str(seed))
        self.current_joint_position = self.franka.get_joint_positions()[:7]
        self.meta_record = {}
        self.meta_record["joint_positions"] = []
        self.meta_record["joint_velocities"] = []
        self.meta_record["tcp"] = []
        self.meta_record["instruction"] = self.instruction
        self.meta_record["model_output"] = []
        make_dir(self.traj_log_dir)
        make_dir(os.path.join(self.traj_log_dir, "realsense"))
        make_dir(os.path.join(self.traj_log_dir, "obs"))
        self.record_config()

    def record_config(self):
        with open(os.path.join(self.traj_log_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "instruction": self.instruction,
                },
                f,
            )

    def record(self, is_save_image=True):
        if is_save_image:
            save_image(
                get_src(self.realsense_camera, "rgb"),
                os.path.join(
                    self.traj_log_dir, "realsense", f"{str(self.steps).zfill(5)}.png"
                ),
            )
            save_image(
                get_src(self.obs_camera, "rgb"),
                os.path.join(
                    self.traj_log_dir, "obs", f"{str(self.steps).zfill(5)}.png"
                ),
            )
        self.meta_record["joint_positions"].append(self.franka.get_joint_positions())
        self.meta_record["joint_velocities"].append(self.franka.get_joint_velocities())
        self.meta_record["tcp"].append(
            joint_positions_to_ee_pose_translation_euler(
                self.franka.get_joint_positions()
            )
        )
        self.steps += 1
        return self.steps

    def request_action(self, without_render=False):
        franka_hand_pose = get_world_pose_by_prim_path(
            self.franka.prim_path + "/panda_hand"
        )
        franka_pose = self.franka.get_world_pose()
        self.current_joint_position = self.franka.get_joint_positions()
        meshlist = get_current_meshList(
            self.scene["object_list"], self.scene["cacheDict"]["meshDict"]
        )
        is_grasped = detect_target_is_grasped(
            self.franka, meshlist[self.task_data["goal"][0][0]["obj1_uid"]]
        )
        if is_grasped:
            self.grasp_cnt = 10
        else:
            if self.grasp_cnt > 0:
                self.grasp_cnt -= 1
            else:
                self.grasp_cnt = 0
        camera_data = {}
        if not without_render:
            camera_data = get_eval_camera_data(
                {"realsense": self.realsense_camera, "obs_camera": self.obs_camera}
            )
            for key in camera_data.keys():
                camera_data[key]["obj_mask"] = self.oracle_camera_data[key]["obj_mask"]
                camera_data[key]["bbox2d"] = self.oracle_camera_data[key]["bbox2d"]
        action = request_action(
            camera_data,
            franka_hand_pose,
            franka_pose,
            self.instruction,
            self.current_joint_position,
            self.planning_data["key_action"][0],
            self.steps,
            self.grasp_cnt != 0,
            send_port=self.send_port,
            receive_port=self.receive_port,
        )
        self.meta_record["model_output"].append(action)
        if isinstance(action, list):
            if self.is_relative_action:
                action[:7] += self.last_joint_position[:7]
            self.last_joint_position = np.array(action[:7])
        elif isinstance(action, tuple):
            position = action[0]
            orientation = action[1]
            gripper_width = action[2]
            pose = Pose(p=position, q=orientation)
            ik_result = self.planner.IK(
                pose,
                self.franka.get_joint_positions()[:9],
                return_closest=True,
            )
            if ik_result[0] != "Success":
                print("IK failed")
                action = self.franka.get_joint_positions()
            else:
                action = ik_result[1]
            action = np.concatenate([action[:7], gripper_width])
            self.last_joint_position = np.array(action[:7])
        return action

    def set_permissions(self, path):
        os.chmod(path, 0o777)
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                os.chmod(os.path.join(root, dir_name), 0o777)
            for file_name in files:
                os.chmod(os.path.join(root, file_name), 0o777)

    def save_meta_record(self):
        with open(os.path.join(self.traj_log_dir, "meta_record.pkl"), "wb") as f:
            pickle.dump(self.meta_record, f)
