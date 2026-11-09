"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import copy
import os
from typing import TYPE_CHECKING

from curobo.util.trajectory import get_smooth_trajectory
import numpy as np
from pydantic import Field, BaseModel, field_validator
import torch
from tqdm import tqdm

from omni.isaac.core.objects import VisualCuboid  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
from omni.isaac.core.utils.types import JointsState  # type: ignore

from genmanip.core.robot.base import BaseEmbodiment
from genmanip.core.robot.dualarm_manip import DualArmEmbodiment
from genmanip.core.skill.base import BaseSkill, SkillConfig
from genmanip.core.skill.utils import SkillFactory
from genmanip.demogen.recoder.planning_recorder import PlanningRecorder
from genmanip.utils.standalone.transform_utils import (
    adjust_translation_along_quaternion,
    pose_frame_to_world,
    pose_world_to_frame,
    rot_orientation_by_axis,
    rot_orientation_by_z_axis,
)

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


class SingleArmMotion(BaseModel):
    name: str = Field(..., description="Name of the skill")
    translation: list[float] = Field([0.0, 0.0, 0.0], description="Translation")
    orientation: list[float] = Field(
        [0.707, 0.0, 0.707, 0.0], description="Orientation"
    )
    grasp: bool = Field(..., description="Grasp")
    type: str = Field(..., description="Type")
    rel_object_uid: str | None = Field(default=None, description="Relative object UID")
    rel_object_point: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Object-position anchor in rotation-only local meters; no USD scale",
    )

    @field_validator("rel_object_point")
    @classmethod
    def validate_object_point(cls, value):
        if not np.isfinite(value).all():
            raise ValueError("rel_object_point must contain finite values")
        return value
    rel_arm: str | None = Field(default=None, description="Relative arm")
    debug_rel_object_uid: str | None = Field(
        default=None,
        description="Additional object UID whose frame will be printed in debug mode",
    )
    cnt: int = Field(default=1, description="Count")
    downsample_rate: int = Field(default=1, description="Downsample rate")
    reset_pos: list[float] | None = Field(default=None, description="Reset position")


class CustomMotionConfig(SkillConfig):
    name: str = "custom_motion"
    motion_list: dict[str, list[SingleArmMotion]] = Field(
        ..., description="Motion list"
    )


@SkillFactory.register("custom_motion")
class CustomMotionSkill(BaseSkill):
    def __init__(self, config_dict: dict, demogen_config: dict):
        self._raw_config_dict = copy.deepcopy(config_dict)
        self.config = CustomMotionConfig(**copy.deepcopy(self._raw_config_dict))
        self.demogen_config = demogen_config

    def _plan(self, embodiment: BaseEmbodiment, arm: str) -> list[dict]:
        action_list = []
        sim_js = embodiment.robot.get_joints_state()
        for motion in self.config.motion_list[arm]:
            if motion.type == "pending":
                action_list.extend(
                    self._create_pending_data(
                        embodiment, motion, sim_js, motion.grasp, arm, cnt=motion.cnt
                    )
                )
            elif motion.type == "reset":
                action_list.extend(
                    self._create_reset_data(
                        embodiment, motion, sim_js, motion.grasp, arm, cnt=motion.cnt
                    )
                )
            else:
                action_list.extend(self._plan_pose(embodiment, motion, sim_js, arm))
        return action_list

    def _plan_pose(
        self,
        embodiment: BaseEmbodiment,
        target: SingleArmMotion,
        sim_js: JointsState,
        arm: str,
    ) -> list[dict]:
        action_list = []
        trajectory_points = self._plan_step(
            embodiment,
            target,
            arm,
            sim_js,
            smooth=False,
        )
        for point in trajectory_points:
            action_data = self._create_action_data(
                embodiment,
                point,
                target,
                target.grasp,
                arm,
            )
            action_list.append(action_data)
            sim_js.positions = embodiment.convert_action_to_joint_state(
                action_data["action"], arm
            )
        return action_list

    def _merge_dual_arm_action_list(
        self,
        embodiment: BaseEmbodiment,
        action_list_left: list[dict],
        action_list_right: list[dict],
    ) -> list[dict]:
        if len(action_list_left) > len(action_list_right):
            for _ in range(len(action_list_left) - len(action_list_right)):
                action_list_right.append(action_list_right[-1])
        elif len(action_list_left) < len(action_list_right):
            for _ in range(len(action_list_right) - len(action_list_left)):
                action_list_left.append(action_list_left[-1])
        action_list = []
        for left_action, right_action in zip(action_list_left, action_list_right):
            action_list.append(
                {
                    "action": left_action["action"][
                        : (embodiment.arm_dof_num + embodiment.gripper_dof_num)
                    ]
                    + right_action["action"][
                        (embodiment.arm_dof_num + embodiment.gripper_dof_num) :
                    ],
                    "name": "default",
                    "grasp": [left_action["grasp"], right_action["grasp"]],
                },
            )
        return action_list

    def _process_motion_target(self, scene: "Scene"):
        embodiment = scene.robot_list[0]

        for arm in self.config.motion_list.keys():
            for info in self.config.motion_list[arm]:
                if info.type == "world_frame":
                    info.translation = info.translation
                    info.orientation = info.orientation
                elif info.type == "object_position":
                    if info.rel_object_uid is None:
                        raise ValueError("Relative object is required for object_position")
                    frame_in_world = scene.object_list[info.rel_object_uid].get_world_pose()
                    anchor, _ = pose_frame_to_world(
                        (np.array(info.rel_object_point), np.array([1.0, 0.0, 0.0, 0.0])),
                        frame_in_world,
                    )
                    # Follow the object's point without inheriting its yaw/tilt.
                    # Translation offset and orientation remain in world axes.
                    info.translation = (anchor + np.array(info.translation)).tolist()
                elif info.type == "object_frame":
                    if info.rel_object_uid is not None:
                        rel_object = scene.object_list[info.rel_object_uid]
                        frame_in_world = rel_object.get_world_pose()
                        translation, orientation = pose_frame_to_world(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        info.translation = translation.tolist()
                        info.orientation = orientation.tolist()
                    else:
                        raise ValueError(
                            f"Relative object is not provided: {info.rel_object_uid}"
                        )
                elif info.type == "robot_frame":
                    if info.rel_arm is not None:
                        if info.rel_arm == "left":
                            if not isinstance(embodiment, DualArmEmbodiment):
                                raise ValueError(
                                    f"Embodiment {embodiment.embodiment_name} is not a dual arm robot"
                                )
                            frame_in_world = embodiment.robot_base_left.get_world_pose()
                        elif info.rel_arm == "right":
                            if not isinstance(embodiment, DualArmEmbodiment):
                                raise ValueError(
                                    f"Embodiment {embodiment.embodiment_name} is not a dual arm robot"
                                )
                            frame_in_world = (
                                embodiment.robot_base_right.get_world_pose()
                            )
                        elif info.rel_arm == "default":
                            frame_in_world = embodiment.robot.get_world_pose()
                        else:
                            raise ValueError(f"Invalid relative arm: {info.rel_arm}")
                        translation, orientation = pose_frame_to_world(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        info.translation = translation.tolist()
                        info.orientation = orientation.tolist()
                    else:
                        raise ValueError(
                            f"Relative arm is not provided: {info.rel_arm}"
                        )
                elif info.type == "pending" or info.type == "reset":
                    pass
                elif info.type == "debug":
                    # get the target pose in world frame
                    if info.rel_object_uid is not None:
                        rel_object = scene.object_list[info.rel_object_uid]
                        frame_in_world = rel_object.get_world_pose()
                        translation, orientation = pose_frame_to_world(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        info.translation = translation.tolist()
                        info.orientation = orientation.tolist()
                    elif info.rel_arm is not None:
                        if info.rel_arm == "left":
                            if not isinstance(embodiment, DualArmEmbodiment):
                                raise ValueError(
                                    f"Embodiment {embodiment.embodiment_name} is not a dual arm robot"
                                )
                            frame_in_world = embodiment.robot_base_left.get_world_pose()
                        elif info.rel_arm == "right":
                            if not isinstance(embodiment, DualArmEmbodiment):
                                raise ValueError(
                                    f"Embodiment {embodiment.embodiment_name} is not a dual arm robot"
                                )
                            frame_in_world = (
                                embodiment.robot_base_right.get_world_pose()
                            )
                        elif info.rel_arm == "default":
                            frame_in_world = embodiment.robot.get_world_pose()
                        else:
                            raise ValueError(f"Invalid relative arm: {info.rel_arm}")
                        translation, orientation = pose_frame_to_world(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        info.translation = translation.tolist()
                        info.orientation = orientation.tolist()

                    # create debugger cube if not exists
                    prim = get_prim_at_path(f"/World/{scene.uuid}/debugger")
                    if not prim.IsValid():
                        self._create_debugger_cube(
                            scene, info.translation, info.orientation
                        )
                    cube = XFormPrim(f"/World/{scene.uuid}/debugger", name="debugger")
                    # cube.set_world_pose(
                    #     position=np.array(info.translation),
                    #     orientation=np.array(info.orientation),
                    # )
                    for _ in tqdm(range(info.cnt), desc="Debugging cube"):
                        scene.world.render()
                    info.translation = cube.get_world_pose()[0].tolist()
                    info.orientation = cube.get_world_pose()[1].tolist()
                    if info.rel_object_uid is not None:
                        rel_object = scene.object_list[info.rel_object_uid]
                        frame_in_world = rel_object.get_world_pose()
                        translation, orientation = pose_world_to_frame(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        print(
                            f"In Frame of {info.rel_object_uid}, pose translation: {translation.tolist()}, orientation: {orientation.tolist()}"
                        )
                    elif info.rel_arm is not None:
                        if info.rel_arm == "left":
                            frame_in_world = embodiment.robot_base_left.get_world_pose()
                        elif info.rel_arm == "right":
                            frame_in_world = (
                                embodiment.robot_base_right.get_world_pose()
                            )
                        elif info.rel_arm == "default":
                            frame_in_world = embodiment.robot.get_world_pose()
                        else:
                            raise ValueError(f"Invalid relative arm: {info.rel_arm}")
                        translation, orientation = pose_world_to_frame(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        print(
                            f"In Frame of {info.rel_arm}, pose translation: {translation.tolist()}, orientation: {orientation.tolist()}"
                        )
                    if info.debug_rel_object_uid is not None:
                        if info.debug_rel_object_uid not in scene.object_list:
                            raise ValueError(
                                f"Debug relative object {info.debug_rel_object_uid} not found"
                            )
                        rel_object = scene.object_list[info.debug_rel_object_uid]
                        frame_in_world = rel_object.get_world_pose()
                        translation, orientation = pose_world_to_frame(
                            (np.array(info.translation), np.array(info.orientation)),
                            frame_in_world,
                        )
                        print(
                            f"In Frame of {info.debug_rel_object_uid}, pose translation: {translation.tolist()}, orientation: {orientation.tolist()}"
                        )
                else:
                    raise ValueError(f"Invalid motion type: {info.type}")
                target_pose = adjust_target_pose_by_embodiment(
                    {"translation": info.translation, "orientation": info.orientation},
                    embodiment,
                )
                info.translation = target_pose["translation"].tolist()
                info.orientation = target_pose["orientation"].tolist()

    def _create_debugger_cube(
        self, scene: "Scene", translation: list[float], orientation: list[float]
    ):
        prim = XFormPrim(f"/World/{scene.uuid}/debugger", name="debugger")
        prim.set_world_pose(
            position=np.array(translation), orientation=np.array(orientation)
        )
        visual_cube = VisualCuboid(
            prim_path=f"/World/{scene.uuid}/debugger/cube",
            name="debugger",
        )
        visual_cube.set_local_pose(
            translation=[0.10725, 0.0, 0.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
        )
        visual_cube.set_local_scale([0.1, 0.05, 0.02])

        visual_cube_head = VisualCuboid(
            prim_path=f"/World/{scene.uuid}/debugger/cube_head",
            name="debugger_head",
            color=np.array([1.0, 0.0, 0.0]),
        )
        visual_cube_head.set_local_pose(
            translation=[0.15225, 0.0, 0.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
        )
        visual_cube_head.set_local_scale([0.01, 0.07, 0.04])

    def execute(
        self, scene: "Scene", recorder: PlanningRecorder, idx: str
    ) -> tuple[bool, str]:
        # Rebuild config every execute to keep relative motion definitions immutable
        # across episodes (object_frame / robot_frame / debug are converted in-place).
        self.config = CustomMotionConfig(**copy.deepcopy(self._raw_config_dict))
        self._process_motion_target(scene)

        embodiment = scene.robot_list[0]

        if "left" in self.config.motion_list and "right" not in self.config.motion_list:
            action_list = self._plan(embodiment, "left")
            self._excute_and_record(
                scene, recorder, embodiment, action_list, idx, "left"
            )
        elif (
            "right" in self.config.motion_list and "left" not in self.config.motion_list
        ):
            action_list = self._plan(embodiment, "right")
            self._excute_and_record(
                scene, recorder, embodiment, action_list, idx, "right"
            )
        elif "left" in self.config.motion_list and "right" in self.config.motion_list:
            action_list_left = self._plan(embodiment, "left")
            action_list_right = self._plan(embodiment, "right")
            action_list = self._merge_dual_arm_action_list(
                embodiment, action_list_left, action_list_right
            )
            self._excute_and_record(
                scene, recorder, embodiment, action_list, idx, "default"
            )
        elif "default" in self.config.motion_list:
            action_list = self._plan(embodiment, "default")
            self._excute_and_record(
                scene, recorder, embodiment, action_list, idx, "default"
            )
        else:
            raise ValueError(f"Invalid motion list: {self.config.motion_list}")

        return True, "default"

    def _plan_step(
        self, embodiment, target: SingleArmMotion, arm, sim_js, smooth: bool
    ) -> list[list[float]]:
        results = embodiment.plan_pose(
            (np.array(target.translation), np.array(target.orientation)),
            sim_js,
            arm=arm,
        )
        if results is None:
            eepose = embodiment._transform_goal_pose(
                (np.array(target.translation), np.array(target.orientation)), arm
            )
            raise RuntimeError(
                f"Motion planning failed for target: {target.name}, target pose might be out of reach: {eepose}"
            )

        if smooth:
            results = get_smooth_trajectory(
                torch.from_numpy(np.array(results)), 5
            ).numpy()
        if target.downsample_rate > 1:
            results = results[:: target.downsample_rate]
        return [np.asarray(res).tolist() for res in results]

    def _create_reset_data(
        self,
        embodiment: BaseEmbodiment,
        target: SingleArmMotion,
        sim_js: JointsState,
        grasp_state: bool,
        arm: str,
        cnt: int = 1,
    ) -> list[dict]:
        joint_pos = sim_js.positions
        if arm == "left":
            joint_pos = joint_pos[embodiment.left_arm_dof_indices]
        elif arm == "right":
            joint_pos = joint_pos[embodiment.right_arm_dof_indices]
        elif arm == "default":
            joint_pos = joint_pos[
                embodiment.left_arm_dof_indices + embodiment.right_arm_dof_indices
            ]
        else:
            raise ValueError(f"Invalid arm: {arm}")
        if target.reset_pos is None:
            reset_pos = np.zeros_like(joint_pos)
        else:
            reset_pos = np.array(target.reset_pos)
            if len(reset_pos) != len(joint_pos):
                raise ValueError(
                    f"Invalid reset position length: reset_pos: {reset_pos} != joint_pos: {joint_pos}"
                )
        action_list = []
        if cnt == 1:
            print(
                f"[Warning] cnt is 1, consider set a value larger than 30 for better result"
            )
        for i in range(1, cnt + 1):
            action_list.append(
                {
                    "action": embodiment.convert_curobo_result_to_action(
                        (reset_pos - joint_pos) * i / cnt + joint_pos, grasp_state, arm
                    ).tolist(),
                    "name": target.name,
                    "grasp": grasp_state,
                }
            )
        return action_list

    def _create_pending_data(
        self,
        embodiment: BaseEmbodiment,
        target: SingleArmMotion,
        sim_js: JointsState,
        grasp_state: bool,
        arm: str,
        cnt: int = 1,
    ) -> list[dict]:
        if not isinstance(embodiment, DualArmEmbodiment):
            raise ValueError(
                f"Embodiment {embodiment.embodiment_name} is not a dual arm robot"
            )
        joint_pos = sim_js.positions
        if arm == "left":
            joint_pos = joint_pos[embodiment.left_arm_dof_indices]
        elif arm == "right":
            joint_pos = joint_pos[embodiment.right_arm_dof_indices]
        elif arm == "default":
            joint_pos = joint_pos[
                embodiment.left_arm_dof_indices + embodiment.right_arm_dof_indices
            ]
        else:
            raise ValueError(f"Invalid arm: {arm}")
        joint_pos = joint_pos[: embodiment.arm_dof_num]
        return [
            {
                "action": embodiment.convert_curobo_result_to_action(
                    joint_pos, grasp_state, arm
                ).tolist(),
                "name": target.name,
                "grasp": grasp_state,
            }
            for _ in range(cnt)
        ]

    def _create_action_data(
        self,
        embodiment: BaseEmbodiment,
        joint_pos: list[float],
        target_info: SingleArmMotion,
        grasp_state: bool,
        arm: str,
    ) -> dict:
        return {
            "action": embodiment.convert_curobo_result_to_action(
                joint_pos, grasp_state, arm
            ).tolist(),
            "name": target_info.name,
            "grasp": grasp_state,
        }

    def _excute_and_record(
        self,
        scene: "Scene",
        recorder: PlanningRecorder,
        embodiment: BaseEmbodiment,
        data_list: list[dict],
        idx_name: str,
        arm: str,
    ) -> None:
        for action in tqdm(data_list, desc=f"Arm action executing {idx_name}"):
            embodiment.robot_view.set_joint_position_targets(
                action["action"], joint_indices=embodiment.default_dof_indices
            )
            if isinstance(action["grasp"], list):
                grasp_val = [1.0 if grasp else -1.0 for grasp in action["grasp"]]
            else:
                grasp_val = 1.0 if action["grasp"] else -1.0
            recorder.load_dynamic_info(
                action["action"],
                grasp_val,
                arm=arm,
                name=f"{idx_name}/{action['name']}",
            )
            scene.step(render=False)
            if os.environ.get("GENMANIP_DEBUG", "0") == "1":
                scene.world.render()
        data_list.clear()


def adjust_target_pose_by_embodiment(
    target_pose: dict,
    embodiment: BaseEmbodiment,
) -> dict:
    # convert list to numpy array
    if isinstance(target_pose["orientation"], list):
        target_pose["orientation"] = np.array(target_pose["orientation"])
    if isinstance(target_pose["translation"], list):
        target_pose["translation"] = np.array(target_pose["translation"])

    # adjust target pose to grasp axis
    target_pose["orientation"] = rot_orientation_by_axis(
        target_pose["orientation"], "z", -180
    )
    target_pose["orientation"] = rot_orientation_by_axis(
        target_pose["orientation"], "y", -90
    )
    target_pose["translation"] = adjust_translation_along_quaternion(
        target_pose["translation"],
        target_pose["orientation"],
        -0.135,
        aug_distance=0.0,
    )

    # adjust target pose by embodiment
    if embodiment.embodiment_name == "franka":
        if embodiment.gripper_name == "panda_hand":
            target_pose["translation"] = adjust_translation_along_quaternion(
                target_pose["translation"],
                target_pose["orientation"],
                0.09,
                aug_distance=0.0,
            )
        elif embodiment.gripper_name == "robotiq":
            # robotiq 的 grasp pose 需要绕 z 轴旋转 45 度
            target_pose["orientation"] = rot_orientation_by_z_axis(
                target_pose["orientation"], -45
            )
            target_pose["translation"] = adjust_translation_along_quaternion(
                target_pose["translation"],
                target_pose["orientation"],
                0.135,
                aug_distance=0.0,
            )
    elif embodiment.embodiment_name == "aloha_split":
        if embodiment.gripper_name == "piper":
            target_pose["orientation"] = rot_orientation_by_z_axis(
                target_pose["orientation"], -90
            )
            target_pose["translation"] = adjust_translation_along_quaternion(
                target_pose["translation"],
                target_pose["orientation"],
                0.12,
                aug_distance=0.0,
            )
    elif embodiment.embodiment_name == "lift2":
        if embodiment.gripper_name == "lift2":
            target_pose["translation"] = adjust_translation_along_quaternion(
                target_pose["translation"],
                target_pose["orientation"],
                0.135,
                aug_distance=0.0,
            )
            target_pose["orientation"] = rot_orientation_by_axis(
                target_pose["orientation"], "y", 90
            )
            target_pose["orientation"] = rot_orientation_by_axis(
                target_pose["orientation"], "z", 180
            )
    return target_pose
