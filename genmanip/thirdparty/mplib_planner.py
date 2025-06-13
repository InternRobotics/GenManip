from mplib import Planner, Pose
from omni.isaac.franka import Franka  # type: ignore
import os
from typing import Sequence
import numpy as np


def add_panda_planner(urdf_path, srdf_path, move_group="panda_hand"):
    planner = Planner(
        urdf=urdf_path,
        srdf=srdf_path,
        move_group=move_group,
    )
    return planner


def get_target(
    robot,
    target_p: Sequence[float],
    target_q: Sequence[float],
    planner: Planner,
    combined_cloud=[],
    grasp: bool = False,
    steps: int = 60,
):
    planner.remove_point_cloud()
    pose = Pose(p=target_p, q=target_q)
    if len(combined_cloud) > 0:
        combined_cloud = np.vstack(combined_cloud)
        planner.update_point_cloud(combined_cloud)
    paths = planner.plan_pose(
        pose, robot.robot.get_joint_positions(), time_step=1 / steps, rrt_range=0.01
    )
    if "position" not in paths:
        return []
    grasp_action = robot.gripper_close if grasp else robot.gripper_open
    actions = [
        paths["position"][i].tolist() + grasp_action
        for i in range(paths["position"].shape[0])
    ]
    return actions


def relate_planner_with_franka(franka: Franka, planner: Planner):
    franka_p, franka_q = franka.get_world_pose()
    planner.set_base_pose(Pose(p=franka_p, q=franka_q))
    return planner


def get_mplib_planner(robot, robot_type, current_dir):
    if robot_type == "franka":
        planner = add_panda_planner(
            urdf_path=os.path.join(
                current_dir, "assets", "robots", "panda", "panda_v2.urdf"
            ),
            srdf_path=os.path.join(
                current_dir, "assets", "robots", "panda", "panda_v2.srdf"
            ),
        )
        planner = relate_planner_with_franka(robot, planner)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    return planner
