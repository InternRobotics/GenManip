"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import random
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from tqdm import tqdm

from genmanip.core.robot.dualarm_manip import DualArmEmbodiment
from genmanip.core.skill.base import BaseSkill, SkillConfig
from genmanip.core.skill.utils import SkillFactory
from genmanip.demogen.recoder.planning_recorder import PlanningRecorder

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


class MoveToConfig(SkillConfig):
    name: str = "move_to"
    move_type: str = Field(..., description="Type of the move to skill")
    grasp: bool = Field(default=False, description="Grasp")
    linear_step: float = Field(
        default=0.01, gt=0.0, le=0.015,
        description="Maximum base translation per axis per physics step, in meters",
    )
    rel_object_uid: str | None = Field(
        default=None,
        description="UID of the relative object",
    )
    rel_position: dict | None = Field(
        default=None,
        description="Position of the relative object",
    )
    arm: str = Field(
        default="default",
        description="Arm to use for the move to skill",
    )
    delta_position: dict = Field(
        default={"x": 0.0, "y": 0.0, "yaw": 0.0},
        description="Delta position of the relative object",
    )


@SkillFactory.register("move_to")
class MoveToSkill(BaseSkill):
    def __init__(self, config_dict: dict, demogen_config: dict):
        self.config = MoveToConfig(**config_dict)
        self.demogen_config = demogen_config

    def execute(
        self, scene: "Scene", recorder: PlanningRecorder, idx: str
    ) -> tuple[bool, str]:
        self._process_delta_position()
        embodiment = scene.robot_list[0]
        if not isinstance(embodiment, DualArmEmbodiment):
            raise ValueError(
                f"Embodiment {embodiment.embodiment_name} is not a dual arm robot"
            )
        if self.config.move_type == "delta" or self.config.move_type == "deltav2":
            if self.config.move_type == "deltav2":
                self.config.delta_position["y"] = -self.config.delta_position["y"]
            self._handle_dual_arm_base_move(
                scene,
                recorder,
                embodiment,
                idx_name=str(idx),
                delta_move_config=self.config.delta_position,
            )
        elif self.config.move_type == "align" or self.config.move_type == "alignv2":
            if self.config.move_type == "alignv2":
                if self.config.delta_position["x"] is not None:
                    self.config.delta_position["x"] = -self.config.delta_position["x"]
                if self.config.delta_position["y"] is not None:
                    self.config.delta_position["y"] = -self.config.delta_position["y"]
            if self.config.rel_object_uid is not None:
                rel_object = scene.object_list[self.config.rel_object_uid]
                target = rel_object.get_world_pose()
            elif self.config.rel_position is not None:
                target = (
                    np.array(self.config.rel_position["translation"]),
                    np.array(self.config.rel_position["orientation"]),
                )
            else:
                raise ValueError("rel_object_uid or rel_position must be provided")
            pose = embodiment._transform_goal_pose(
                (target[0], target[1]), self.config.arm
            )
            if self.config.delta_position["x"] is not None:
                x_diff = pose[0][0] - self.config.delta_position["x"]
            else:
                x_diff = 0.0
            if self.config.delta_position["y"] is not None:
                y_diff = pose[0][1] - self.config.delta_position["y"]
            else:
                y_diff = 0.0
            if self.config.move_type == "alignv2":
                x_diff = -x_diff
            self._handle_dual_arm_base_move(
                scene,
                recorder,
                embodiment,
                idx_name=str(idx),
                delta_move_config={"x": -x_diff, "y": -y_diff, "yaw": None},
            )
        return True, "default"

    def _process_delta_position(self):
        if isinstance(self.config.delta_position["x"], list):
            x_diff = random.uniform(
                self.config.delta_position["x"][0], self.config.delta_position["x"][1]
            )
        else:
            x_diff = self.config.delta_position["x"]
        if isinstance(self.config.delta_position["y"], list):
            y_diff = random.uniform(
                self.config.delta_position["y"][0], self.config.delta_position["y"][1]
            )
        else:
            y_diff = self.config.delta_position["y"]
        if isinstance(self.config.delta_position["yaw"], list):
            yaw_diff = random.uniform(
                self.config.delta_position["yaw"][0],
                self.config.delta_position["yaw"][1],
            )
        else:
            yaw_diff = self.config.delta_position["yaw"]
        self.config.delta_position["x"] = x_diff
        self.config.delta_position["y"] = y_diff
        self.config.delta_position["yaw"] = yaw_diff

    def _handle_dual_arm_base_move(
        self,
        scene: "Scene",
        recorder: PlanningRecorder,
        embodiment: DualArmEmbodiment,
        idx_name: str,
        delta_move_config: dict = {"x": 0.0, "y": 0.0, "yaw": 0.0},
    ) -> None:
        max_xy = max(abs(delta_move_config["x"]), abs(delta_move_config["y"]))
        if max_xy == 0.0:
            x_step = 0.0
            y_step = 0.0
        else:
            x_step = delta_move_config["x"] / max_xy * self.config.linear_step
            y_step = delta_move_config["y"] / max_xy * self.config.linear_step
        x_remaining = delta_move_config["x"]
        y_remaining = delta_move_config["y"]
        if delta_move_config["yaw"] is not None:
            yaw_remaining = delta_move_config["yaw"]
        else:
            yaw_remaining = 0.0
        yaw_step = 1.0 * np.sign(yaw_remaining)
        motion_list = []

        while (
            x_remaining != 0.0
            or y_remaining != 0.0
            or (yaw_remaining is not None and yaw_remaining != 0.0)
        ):
            if abs(x_step) > abs(x_remaining):
                x_step = x_remaining
            if abs(y_step) > abs(y_remaining):
                y_step = y_remaining
            if yaw_remaining is not None and abs(yaw_step) > abs(yaw_remaining):
                yaw_step = yaw_remaining
            elif yaw_remaining is None:
                yaw_step = 0.0
            x_remaining -= x_step
            y_remaining -= y_step
            if yaw_remaining is not None:
                yaw_remaining -= yaw_step
            motion_list.append((x_step, y_step, yaw_step))

        for motion in tqdm(motion_list, desc=f"Base motion executing {idx_name}"):
            action = embodiment.robot.get_joint_positions()
            action = action[embodiment.default_dof_indices]

            if self.config.arm == "default":
                if self.config.grasp:
                    action[embodiment.default_gripper_dof_indices] = (
                        embodiment.gripper_close * 2
                    )
                gripper_action = [1.0 if self.config.grasp else -1.0] * 2
            elif self.config.arm == "left":
                if self.config.grasp:
                    action[embodiment.default_gripper_dof_indices[:2]] = (
                        embodiment.gripper_close
                    )
                gripper_action = [1.0 if self.config.grasp else -1.0, -1.0]
            elif self.config.arm == "right":
                if self.config.grasp:
                    action[embodiment.default_gripper_dof_indices[2:]] = (
                        embodiment.gripper_close
                    )
                gripper_action = [-1.0, 1.0 if self.config.grasp else -1.0]
            else:
                raise ValueError(f"Invalid arm: {self.config.arm}")

            embodiment.robot_view.set_joint_position_targets(
                action,
                joint_indices=embodiment.default_dof_indices,
            )
            embodiment.delta_move_to(motion[0], motion[1], motion[2])

            recorder.load_dynamic_info(
                action,
                gripper_action,
                arm="default",
                base_motion=np.array([*motion]),
                name=f"{idx_name}/move",
            )
            scene.step(render=False)
            if os.environ.get("GENMANIP_DEBUG", "0") == "1":
                scene.world.render()
