import numpy as np
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path


def reach_target_and_record(scene, recorder, translation, orientation, grasp, idx):
    franka_pose = scene["robot_info"]["robot_list"][0].robot.get_world_pose()
    franka = scene["robot_info"]["robot_list"][0].robot
    position = translation - franka_pose[0]
    orientation = orientation
    results = scene["planner_list"][0].plan(
        position,
        orientation,
        franka.get_joints_state(),
    )
    actions = []
    if results is not None:
        for res in results:
            actions.append(
                np.concatenate([res, [0.00, 0.00] if grasp else [0.04, 0.04]]).tolist()
            )
    if len(actions) == 0:
        return False
    while actions:
        action = actions.pop(0)
        scene["robot_info"]["robot_view_list"][0].set_joint_position_targets(action)
        recorder.load_dynamic_info(
            action,
            1 if grasp else -1,
            name=f"{idx}/move_to_banana",
        )
        scene["world"].step(render=False)
        if (
            get_world_pose_by_prim_path(franka.prim_path + "/panda_hand")[0][0] + 0.1
            < get_world_pose_by_prim_path(franka.prim_path + "/panda_link0")[0][0]
        ):
            return False
    return True


def pick_up_banana(scene, recorder, demogen_config, action_info, idx):
    if (
        "fb1b6fc41f7e49adbf467e5e5988d190" in scene["object_list"]
        and demogen_config["generation_config"]["planner"] == "curobo"
    ):
        grasp_t = scene["object_list"][
            "fb1b6fc41f7e49adbf467e5e5988d190"
        ].get_world_pose()[0]
        grasp_o = np.array([0, 1, 0, 0])
        if not reach_target_and_record(
            scene, recorder, grasp_t + np.array([0, 0, 0.3]), grasp_o, False, idx
        ):
            return False
        if not reach_target_and_record(
            scene, recorder, grasp_t + np.array([0, 0, 0.1]), grasp_o, False, idx
        ):
            return False
        if not reach_target_and_record(
            scene, recorder, grasp_t + np.array([0, 0, 0.3]), grasp_o, True, idx
        ):
            return False
        return True
    else:
        return False
