"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import numpy as np
from typing import List

from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path  # type: ignore

from genmanip.core.robot.dualarm_manip import DualArmEmbodiment, ManipDualArmRobotConfig
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.planner.curobo.base import CuroboPlanner
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.core.scene.scene_config import RobotConfig


@RobotFactory.register("manip/lift2/R5a")
class Lift2Embodiment(DualArmEmbodiment):
    def __init__(self, *args, **kwargs) -> None:
        config = ManipDualArmRobotConfig(
            embodiment_name="lift2",
            arm_name="R5a",
            gripper_name="lift2",
            arm_dof_num=6,
            gripper_dof_num=2,
            gripper_open=[0.044, 0.044],
            gripper_close=[0.0, 0.0],
            default_arm_dof_indices=[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
            default_gripper_dof_indices=[6, 7, 14, 15],
            left_arm_dof_indices=[10, 12, 14, 16, 18, 20, 23, 24],
            right_arm_dof_indices=[9, 11, 13, 15, 17, 19, 21, 22],
            body_dof_indices=[6],
            base_dof_indices=[0, 1, 2],
            default_lift_joint_position=0.46,
            default_lift_joint_path="/lift2/lift2/base_link/joint4",
            robot_base_left="/lift2/lift2/fl/base_link",
            robot_base_right="/lift2/lift2/fr/base_link",
            base_joint_path_x="/lift2/lift2/dummy_base_x/mobile_translate_x",
            base_joint_path_y="/lift2/lift2/dummy_base_y/mobile_translate_y",
            base_joint_path_rotate="/lift2/lift2/dummy_base_rotate/mobile_rotate",
        )
        super().__init__(config, *args, **kwargs)

    def _initialize(self, default_joint_positions: list[float] | None = None) -> None:
        super()._initialize(default_joint_positions)

        # The articulation view is valid only after Robot.initialize().  Apply
        # the intended velocity cap here and preserve stricter USD limits.
        max_joint_velocities = self.robot_view.get_joint_max_velocities()
        self.robot_view.set_max_joint_velocities(np.minimum(max_joint_velocities, 2.0))

    def create_robot(
        self, scene_uid: str, default_config: dict, robot_config: RobotConfig
    ) -> Robot:
        # Deactivate the franka robot
        prim = get_prim_at_path(f"/World/{scene_uid}/franka")
        if prim.IsValid() and prim.IsActive():
            prim.SetActive(False)

        # Create the lift2 robot
        if robot_config.position is not None:
            position = robot_config.position
        else:
            position = np.array([-0.45, 0.0, 0.5])
        if robot_config.orientation is not None:
            orientation = robot_config.orientation
        else:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
        prim = create_prim(
            prim_path=f"/World/{scene_uid}/lift2",
            prim_type="Xform",
            usd_path=os.path.join(
                default_config["ASSETS_DIR"], "robot_usds/lift2/robot.usd"
            ),
            position=position,
            orientation=orientation,
        )

        # Create the lift2 robot object
        lift2 = Robot(
            prim_path=f"/World/{scene_uid}/lift2",
            name="lift2",
        )

        # Set default parameters for the lift2 robot
        lift2.set_solver_position_iteration_count(128)
        lift2.set_stabilization_threshold(0.005)
        lift2.set_solver_velocity_iteration_count(4)
        return lift2

    def set_planner(self, current_dir: str) -> List[CuroboPlanner]:
        lift2_cfg = load_yaml(
            os.path.join(current_dir, "saved/assets/miscs/curobo/R5a/r5a_left_arm.yml")
        )
        lift2_cfg["robot_cfg"]["kinematics"]["usd_path"] = lift2_cfg["robot_cfg"][
            "kinematics"
        ]["usd_path"].replace("${ASSETS_DIR}", current_dir + "/saved/assets")
        lift2_cfg["robot_cfg"]["kinematics"]["urdf_path"] = lift2_cfg["robot_cfg"][
            "kinematics"
        ]["urdf_path"].replace("${ASSETS_DIR}", current_dir + "/saved/assets")

        left_planner = CuroboPlanner(lift2_cfg, self.robot.prim_path)
        right_planner = CuroboPlanner(lift2_cfg, self.robot.prim_path)

        left_planner.ordered_js_names = [
            "fl_joint1",
            "fl_joint2",
            "fl_joint3",
            "fl_joint4",
            "fl_joint5",
            "fl_joint6",
        ]
        right_planner.ordered_js_names = [
            "fr_joint1",
            "fr_joint2",
            "fr_joint3",
            "fr_joint4",
            "fr_joint5",
            "fr_joint6",
        ]
        raw_js_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        left_planner.raw_js_names = raw_js_names
        right_planner.raw_js_names = raw_js_names

        return [left_planner, right_planner]
