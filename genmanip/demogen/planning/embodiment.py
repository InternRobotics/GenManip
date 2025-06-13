from omni.isaac.core.utils.prims import delete_prim, get_prim_at_path  # type: ignore
import math


class BaseEmbodiment:
    def __init__(self, robot):
        self.embodiment_name = "franka"
        self.gripper_name = "default"
        self.arm_dof_num = 7
        self.gripper_dof_num = 2
        self.gripper_open = [0.04, 0.04]
        self.gripper_close = [0.0, 0.0]
        self.robot = robot
        self.robot_view = robot._articulation_view

    def get_joint_position(self, joint_position, gripper_action):
        if gripper_action:
            return joint_position.tolist() + self.gripper_open
        else:
            return joint_position.tolist() + self.gripper_close


class FrankaNormalEmbodiment(BaseEmbodiment):
    def __init__(self, scene):
        super().__init__(scene)
        self.embodiment_name = "franka"
        self.gripper_name = "panda_hand"
        self.arm_dof_num = 7
        self.gripper_dof_num = 2
        self.gripper_open = [0.04, 0.04]
        self.gripper_close = [0.0, 0.0]
        self.robot_view.set_max_joint_velocities([2.0] * 9)

class FrankaRobotiqEmbodiment(BaseEmbodiment):
    def __init__(self, scene):
        super().__init__(scene)
        self.embodiment_name = "franka"
        self.gripper_name = "robotiq"
        self.arm_dof_num = 7
        self.gripper_dof_num = 6
        self.gripper_open = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper_close = [math.pi, math.pi, 0.0, 0.0, 0.0, 0.0]
        self.robot_view.set_max_joint_velocities([2.0] * 13)