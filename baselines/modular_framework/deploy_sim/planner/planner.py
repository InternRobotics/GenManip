import sys
import os
import requests

projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if projectroot not in sys.path:
    sys.path.append(projectroot)
import numpy as np
from scipy.spatial.transform import Rotation as R
from deploy_sim.third_party.anygrasp import (
    get_init_grasp,
    get_world_grasp_from_camera_coords,
)
from deploy_sim.third_party.anygrasp import find_closest_grasp_to_point


def adjust_translation_along_quaternion(
    translation, quaternion, distance, aug_distance=0.0
):
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    arbitrary_vector = (
        np.array([1, 0, 0]) if direction_vector[0] == 0 else np.array([0, 1, 0])
    )
    perp_vector1 = np.cross(direction_vector, arbitrary_vector)
    perp_vector2 = np.cross(direction_vector, perp_vector1)
    perp_vector1 /= np.linalg.norm(perp_vector1)
    perp_vector2 /= np.linalg.norm(perp_vector2)
    random_shift = np.random.uniform(-aug_distance, aug_distance, size=2)
    new_translation += random_shift[0] * perp_vector1 + random_shift[1] * perp_vector2
    return new_translation


def pose_to_transform(pose):
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return transform


def get_world_point_from_pixel_by_server(point_2d, depth):
    depth_point = depth[int(point_2d[1]), int(point_2d[0])]
    url = "http://localhost:5000/convert"
    if isinstance(point_2d, np.ndarray):
        point_2d = point_2d.tolist()
    data = {
        "point": point_2d,  # 图像坐标
        "depth": float(depth_point),
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        world_point = response.json().get("world_point")
    return world_point


def get_world_point_from_pixel(point_2d, depth, intrinsic, translation, quaternion):
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
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


def motion_plan(planner, grasp_pose, joint_position):
    grasp_trans = grasp_pose[:3, 3]
    grasp_oriet = R.from_matrix(grasp_pose[:3, :3]).as_quat(scalar_first=True)
    target_pose = Pose(p=grasp_trans, q=grasp_oriet)
    result = planner.plan_pose(
        target_pose, joint_position, time_step=1 / 30, rrt_range=0.01
    )
    return result


from mplib import Planner, Pose


class SimPlanner:
    def __init__(
        self,
        franka_pose,
        franka_hand_pose,
        prompt_url="127.0.0.1",
        anygrasp_url="127.0.0.1",
        intrinsics=None,
        camera_pose=None,
    ) -> None:
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.update_cnt = 0
        self.franka_pose = franka_pose
        self.franka_hand_pose = (
            franka_hand_pose[0] + np.array([0, 0, 0.1]),
            franka_hand_pose[1],
        )
        self.status = "finished"
        self.prompt_url = prompt_url
        self.anygrasp_url = anygrasp_url
        self.instruction = ""
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        self.camera_position = camera_pose[0]
        self.camera_quaternion = camera_pose[1]
        self.planner = Planner(
            urdf=os.path.join(self.current_dir, "../urdf/franka/panda_v2.urdf"),
            srdf=os.path.join(self.current_dir, "../urdf/franka/panda_v2.srdf"),
            move_group="panda_hand",
        )
        self.planner.set_base_pose(Pose(p=self.franka_pose[0], q=self.franka_pose[1]))
        self.action_list = []

    def initialize(self, prompt_response, colors, depth):
        print(colors.shape)
        self.status = "start"
        self.result = prompt_response
        self.process_prompt_response(depth)
        self.request_grasp_pose(colors, depth)

    def process_prompt_response(self, depth):
        self.grasp_point = get_world_point_from_pixel(
            self.result["selected_grab_point"]["coordinates"],
            depth,
            (self.fx, self.fy, self.cx, self.cy),
            self.camera_position,
            self.camera_quaternion,
        )
        self.release_point = get_world_point_from_pixel(
            self.result["path_planning_instructions"][-1]["point"],
            depth,
            (self.fx, self.fy, self.cx, self.cy),
            self.camera_position,
            self.camera_quaternion,
        )
        self.waypoint = []
        for idx, info in enumerate(self.result["path_planning_instructions"]):
            if idx == 0 or info != self.result["path_planning_instructions"][idx - 1]:
                waypoint = get_world_point_from_pixel(
                    info["point"],
                    depth,
                    (self.fx, self.fy, self.cx, self.cy),
                    self.camera_position,
                    self.camera_quaternion,
                )
                waypoint[2] = self.release_point[2] + 0.3
                self.waypoint.append(waypoint)

    def request_grasp_pose(self, colors, depth):
        grasp_list = None
        while grasp_list is None:
            grasp_list = get_init_grasp(
                colors, depth, self.intrinsics, address=self.anygrasp_url
            )
        world_grasp_list = []
        for grasp in grasp_list:
            point, pose = get_world_grasp_from_camera_coords(
                self.camera_position,
                self.camera_quaternion,
                grasp["translation"],
                grasp["orientation"],
            )
            world_grasp_list.append(
                {
                    "translation": point,
                    "orientation": pose,
                    "depth": grasp["depth"],
                    "score": grasp["score"],
                }
            )
        self.grasp_pose = find_closest_grasp_to_point(
            self.grasp_point,
            world_grasp_list,
            distance_only=True,
            distance_threshold=0.2,
        )

    def update_status(self):
        if self.status == "start":
            self.status = "pre_grasp"
        elif self.status == "pre_grasp":
            self.status = "grasp"
        elif self.status == "grasp":
            self.status = "post_grasp"
        elif self.status == "post_grasp":
            self.status = "path0"
        elif self.status[:4] == "path":
            idx = int(self.status[4:])
            if idx < len(self.waypoint) - 1:
                self.status = f"path{idx+1}"
            else:
                self.status = "pre_release"
        elif self.status == "pre_release":
            self.status = "release"
        elif self.status == "release":
            self.status = "post_release"
        elif self.status == "post_release":
            self.status = "finished"

    def get_next_stage(self, joint_position):
        if len(self.action_list) == 0:
            self.action_list = self.get_next_action_list(joint_position)[
                "position"
            ].tolist()
        if len(self.action_list) == 0:
            return joint_position
        return self.action_list.pop(0)

    def get_gripper_status(self):
        if (
            self.status == "post_grasp"
            or self.status[:4] == "path"
            or self.status == "pre_release"
            or self.status == "release"
        ):
            return True
        else:
            return False

    def pq2mat(self, p, q):
        target_pose = np.eye(4)
        target_pose[:3, 3] = p
        target_pose[:3, :3] = R.from_quat(q, scalar_first=True).as_matrix()
        return target_pose

    def get_next_action_list(self, joint_position):
        self.update_status()
        if self.status == "pre_grasp":
            grasp_trans = adjust_translation_along_quaternion(
                self.grasp_pose["translation"], self.grasp_pose["orientation"], 0.2
            )
            # grasp_trans = self.grasp_pose["translation"] + np.array([0, 0, 0.2])
            actions = motion_plan(
                self.planner,
                self.pq2mat(grasp_trans, self.grasp_pose["orientation"]),
                joint_position,
            )
        elif self.status == "grasp":
            grasp_trans = adjust_translation_along_quaternion(
                self.grasp_pose["translation"], self.grasp_pose["orientation"], 0.08
            )
            # grasp_trans = self.grasp_pose["translation"]
            actions = motion_plan(
                self.planner,
                self.pq2mat(grasp_trans, self.grasp_pose["orientation"]),
                joint_position,
            )
        elif self.status == "post_grasp":
            grasp_trans = self.grasp_pose["translation"] + np.array([0, 0, 0.3])
            actions = motion_plan(
                self.planner,
                self.pq2mat(grasp_trans, self.grasp_pose["orientation"]),
                joint_position,
            )
        elif self.status[:4] == "path":
            idx = int(self.status[4:])
            actions = motion_plan(
                self.planner,
                self.pq2mat(self.waypoint[idx], self.grasp_pose["orientation"]),
                joint_position,
            )
        elif self.status == "pre_release":
            ori = self.grasp_pose["orientation"]
            trans = self.release_point + np.array([0, 0, 0.30])
            # trans = adjust_translation_along_quaternion(
            #     self.release_point, ori, 0.16
            # )
            actions = motion_plan(
                self.planner,
                self.pq2mat(trans, ori),
                joint_position,
            )
        elif self.status == "release":
            ori = self.grasp_pose["orientation"]
            trans = self.release_point + np.array([0, 0, 0.20])
            # trans = adjust_translation_along_quaternion(
            #     self.release_point, ori, 0.20
            # )
            actions = motion_plan(
                self.planner,
                self.pq2mat(trans, ori),
                joint_position,
            )
        elif self.status == "post_release":
            ori = self.grasp_pose["orientation"]
            trans = self.release_point + np.array([0, 0, 0.30])
            # trans = adjust_translation_along_quaternion(
            #     self.release_point, ori, 0.30
            # )
            actions = motion_plan(
                self.planner,
                self.pq2mat(trans, ori),
                joint_position,
            )
        else:
            actions = []
            self.update_cnt += 1
            actions = motion_plan(
                self.planner,
                self.pq2mat(self.franka_hand_pose[0], self.franka_hand_pose[1]),
                joint_position,
            )
            if self.update_cnt >= 5:
                actions = [joint_position]
            return actions
        self.update_cnt = 0
        return actions
