from mplib import Planner, Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Sequence  # type: ignore
from genmanip.thirdparty.mplib_planner import relate_planner_with_franka
from omni.isaac.core.articulations import ArticulationView  # type: ignore
from omni.isaac.franka import Franka  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
import roboticstoolbox as rtb


def get_franka_PD_controller(
    franka: Franka, max_joint_velocities: Optional[Sequence[float]] = [1.0] * 9
):
    franka_view = ArticulationView(franka.prim_path)
    franka_view.initialize()
    franka_view.set_max_joint_velocities(max_joint_velocities)
    return franka_view


def joint_positions_action_to_joint_positions_state(
    joint_positions: np.ndarray, franka: Franka
):
    grasp_action = franka.gripper.forward(
        action=("close" if joint_positions[7] < 0 else "open")
    ).joint_positions[7:]
    return np.concatenate([joint_positions[:7], grasp_action])


def replay_skill(object_to_franka, franka, planner, skill_data):
    pose_data = []
    gripper_data = []
    # set planner base to [0, 0, 0] in robot frame
    planner.set_base_pose(Pose(p=np.array([0, 0, 0]), q=np.array([1, 0, 0, 0])))
    for action in skill_data:
        hand_to_franka = np.dot(object_to_franka, action["hand_to_object"])
        p_transformed, rot_mat = hand_to_franka[:3, 3], hand_to_franka[:3, :3]
        q_transformed = R.from_matrix(rot_mat).as_quat()[[3, 0, 1, 2]]
        pose_data.append(Pose(p=p_transformed, q=q_transformed))
        gripper_data.append(action["gripper_open"])

    paths = planner.plan_pose(
        pose_data[0], franka.get_joint_positions(), time_step=1 / 30.0, rrt_range=0.01
    )
    actions = [
        np.array(paths["position"][i].tolist() + [0.04, 0.04])
        for i in range(paths["position"].shape[0])
    ]

    start_joint_positions = actions[-1]

    # actions = []
    # start_joint_positions = franka.get_joint_positions()

    for pose, gripper in zip(pose_data, gripper_data):
        ik_result = planner.IK(
            pose,
            start_joint_positions,
            return_closest=True,
        )
        if ik_result[0] != "Success":
            continue
        start_joint_positions = ik_result[1]
        gripper_positions = [0.04, 0.04] if gripper else [0.0, 0.0]
        actions.append(np.array(start_joint_positions.tolist()[:7] + gripper_positions))
    # set planner back to robot pose in world frame
    planner = relate_planner_with_franka(franka, planner)
    return actions


def replay_skill_curobo(object_to_franka, franka, curobo_planner, skill_data):
    pose_data = []
    gripper_data = []
    actions = []

    for action in skill_data:
        hand_to_franka = np.dot(object_to_franka, action["hand_to_object"])
        p_transformed, rot_mat = hand_to_franka[:3, 3], hand_to_franka[:3, :3]
        q_transformed = R.from_matrix(rot_mat).as_quat()[[3, 0, 1, 2]]
        pose_data.append(p_transformed.tolist() + q_transformed.tolist())
        gripper_data.append(action["gripper_open"])
    cur_joint_positions = pose_data[0]
    for pose, gripper in zip(pose_data, gripper_data):
        ik_result = curobo_planner.ik_single(pose, np.array(cur_joint_positions))
        if ik_result is None:
            continue
        gripper_positions = [0.04, 0.04] if gripper else [0.0, 0.0]
        actions.append(np.concatenate([ik_result[:7], gripper_positions]).tolist())
        cur_joint_positions = actions[-1][:7]

    print("action len: ", len(actions))
    return actions


def create_joint_xform_list(robot):
    joint_name_list = [
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    ]
    joint_xform_list = {}
    for joint_name in joint_name_list:
        joint_xform_list[joint_name] = XFormPrim(f"{robot.prim_path}/{joint_name}")
    return joint_xform_list


def create_tcp_xform_list(robot, tcp_config):
    tcp_xform_list = []
    for tcp_info in tcp_config:
        tcp = XFormPrim(
            f"{robot.prim_path}/{tcp_info['parent_prim_path']}/{tcp_info['name']}"
        )
        tcp.set_local_pose(tcp_info["position"], tcp_info["orientation"])
        tcp_xform_list.append(tcp)
    return tcp_xform_list


def joint_position_to_end_effector_pose(joint_position, panda=None):
    if panda is None:
        panda = rtb.models.Panda()
    hand_pose = panda.fkine(q=joint_position, end="panda_hand").A
    position = hand_pose[:3, 3]
    rotation = hand_pose[:3, :3]
    orientation = R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return position, orientation
