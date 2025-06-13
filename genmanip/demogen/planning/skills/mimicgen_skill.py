import numpy as np
from scipy.spatial.transform import Rotation as R
from genmanip.core.robot.franka import replay_skill, replay_skill_curobo
from genmanip.thirdparty.mplib_planner import relate_planner_with_franka
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
from omni.isaac.core.utils.transformations import get_relative_transform  # type: ignore


def get_pose_matrix(pose):
    translation = pose[0]
    rotation = R.from_quat(pose[1][[1, 2, 3, 0]]).as_matrix()
    prim_matrix = np.eye(4)
    prim_matrix[:3, :3] = rotation
    prim_matrix[:3, 3] = translation
    return prim_matrix


def get_relative_transform_matrix(object_pose, franka_pose):
    object_matrix = get_pose_matrix(object_pose)
    franka_matrix = get_pose_matrix(franka_pose)
    return np.linalg.inv(franka_matrix) @ object_matrix


def replay_mimicgen_skill(scene, recorder, demogen_config, action_info, idx):
    franka = scene["robot_info"]["robot_list"][0].robot
    franka_prim = franka.prim
    object_prim_path = scene["articulation_list"][action_info["obj1_uid"]].prim_path
    object_prim = get_prim_at_path(object_prim_path)
    object_to_franka = get_relative_transform(object_prim, franka_prim)
    skill_data = scene["articulation_data"][action_info["obj1_uid"]]["skills"][
        action_info["skill_name"]
    ]["skill_trajectory"]
    # todo: support mplib and curobo
    if demogen_config["generation_config"]["planner"] == "mplib":
        actions = replay_skill(
            object_to_franka, franka, scene["planner_list"][0], skill_data
        )
        scene["planner_list"][0] = relate_planner_with_franka(
            franka, scene["planner_list"][0]
        )
    elif demogen_config["generation_config"]["planner"] == "curobo":
        # scene["planner_list"][0].update()
        actions = replay_skill_curobo(
            object_to_franka, franka, scene["planner_list"][0], skill_data
        )
    print("mimicgen skill actions: ", len(actions))
    if len(actions) == 0:
        return False
    while actions:
        action = actions.pop(0)
        scene["robot_info"]["robot_view_list"][0].set_joint_position_targets(action)
        if np.allclose(action[7:], [0.04, 0.04]):
            grasp = 1
        elif np.allclose(action[7:], [0, 0]):
            grasp = -1
        else:
            raise ValueError("mimicgen skill action gripper invalid.")
        recorder.load_dynamic_info(
            action, grasp, name=f"{idx}/{action_info['skill_name']}"
        )
        scene["world"].step(render=True)
        if (
            get_world_pose_by_prim_path(franka.prim_path + "/panda_hand")[0][0] + 0.1
            < get_world_pose_by_prim_path(franka.prim_path + "/panda_link0")[0][0]
        ):
            return False
    return True
