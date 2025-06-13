import copy
import numpy as np
from tqdm import tqdm
from genmanip.demogen.planning.skill_lib import record_code_skill, record_mimicgen_skill
from genmanip.core.usd_utils.prim_utils import set_mass
from genmanip.core.loading.loading import collect_world_pose_list, reset_object_xyz
from genmanip.core.pointcloud.pointcloud import get_current_meshList, meshlist_to_pclist
from genmanip.core.random_place.random_place import place_object_to_object_by_relation
from genmanip.core.robot.franka import get_franka_PD_controller
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path
from genmanip.demogen.evaluate.evaluate import check_subgoal_finished_rigid
from genmanip.thirdparty.anygrasp import get_init_grasp
from genmanip.thirdparty.mplib_planner import get_target
from genmanip.utils.transform_utils import (
    adjust_orientation,
    adjust_translation_along_quaternion,
)
from genmanip.demogen.planning.pick_and_place import (
    prepare_motion_planning_payload,
    get_action_meta_info,
    record_planning,
)
from omni.isaac.core import World  # type: ignore


def feasibility_analysis(
    init_grasp,
    grasp_tar_t,
    grasp_tar_o,
    embodiment,
    planner,
    combined_cloud,
    steps=30,
    aug_distance=0.0,
):
    action_list = prepare_motion_planning_payload(
        init_grasp, grasp_tar_t, grasp_tar_o, steps=steps, aug_distance=aug_distance
    )
    planner.remove_point_cloud()
    if len(combined_cloud) > 0:
        combined_cloud = np.vstack(combined_cloud)
        planner.update_point_cloud(combined_cloud)
    world = World()
    franka_view = embodiment.robot_view
    paths = []
    for idx, target in tqdm(enumerate(action_list)):
        actions = get_target(
            embodiment,
            target["translation"],
            target["orientation"],
            planner,
            combined_cloud,
            steps=target["steps"],
            grasp=target["grasp"],
        )
        if len(actions) == 0:
            return paths
        if idx != len(action_list) - 1:
            if target["grasp"] != action_list[idx + 1]["grasp"]:
                action = actions[-1][:7] + (
                    embodiment.gripper_open
                    if not action_list[idx + 1]["grasp"]
                    else embodiment.gripper_close
                )
                for _ in range(5):
                    actions.append(action)
        paths.append(
            {
                "paths": copy.deepcopy(actions),
                "grasp": target["grasp"],
                "name": target["name"],
            }
        )
        while actions:
            action = actions.pop(0)
            franka_view.set_joint_position_targets(action)
            world.step(render=False)
            if (
                get_world_pose_by_prim_path(embodiment.robot.prim_path + "/panda_hand")[
                    0
                ][0]
                + 0.1
                < get_world_pose_by_prim_path(
                    embodiment.robot.prim_path + "/panda_link0"
                )[0][0]
            ):
                return []
    return paths


def feasibility_analysis_curobo(
    init_grasp, grasp_tar_t, grasp_tar_o, embodiment, planner, aug_distance=0.0
):
    action_list = prepare_motion_planning_payload(
        init_grasp, grasp_tar_t, grasp_tar_o, aug_distance=aug_distance
    )
    world = World()
    franka_view = embodiment.robot_view
    paths = []
    for idx, target in tqdm(enumerate(action_list)):
        position = target["translation"]
        franka_p, _ = embodiment.robot.get_world_pose()
        position = position - franka_p
        orientation = target["orientation"]
        results = planner.plan(
            position,
            orientation,
            embodiment.robot.get_joints_state(),
        )
        actions = []
        if results is not None:
            for res in results:
                if target["grasp"]:
                    actions.append(
                        np.concatenate([res, embodiment.gripper_close]).tolist()
                    )
                else:
                    actions.append(
                        np.concatenate([res, embodiment.gripper_open]).tolist()
                    )
        if len(actions) == 0:
            return paths
        if idx != len(action_list) - 1:
            if target["grasp"] != action_list[idx + 1]["grasp"]:
                action = actions[-1][:7] + (
                    embodiment.gripper_open
                    if not action_list[idx + 1]["grasp"]
                    else embodiment.gripper_close
                )
                for _ in range(5):
                    actions.append(action)
        paths.append(
            {
                "paths": copy.deepcopy(actions),
                "grasp": target["grasp"],
                "name": target["name"],
            }
        )
        while actions:
            action = actions.pop(0)
            franka_view.set_joint_position_targets(action)
            world.step(render=False)
            if (
                get_world_pose_by_prim_path(embodiment.robot.prim_path + "/panda_hand")[
                    0
                ][0]
                + 0.1
                < get_world_pose_by_prim_path(
                    embodiment.robot.prim_path + "/panda_link0"
                )[0][0]
            ):
                return []
    return paths


def grasp_analysis(init_grasp, franka, planner, combined_cloud, steps=30):
    action_list = prepare_grasp_motion_planning_payload(init_grasp, steps=steps)
    planner.remove_point_cloud()
    if len(combined_cloud) > 0:
        combined_cloud = np.vstack(combined_cloud)
        planner.update_point_cloud(combined_cloud)
    world = World()
    franka_view = get_franka_PD_controller(franka, [2.0] * 9)
    paths = []
    for idx, target in tqdm(enumerate(action_list)):
        actions = get_target(
            franka,
            target["translation"],
            target["orientation"],
            planner,
            combined_cloud,
            steps=target["steps"],
            grasp=target["grasp"],
        )
        if len(actions) == 0:
            return []
        paths.append({"paths": copy.deepcopy(actions), "grasp": target["grasp"]})
        while actions:
            action = actions.pop(0)
            franka_view.set_joint_position_targets(action)
            world.step(render=False)
    return paths


def grasp_analysis_curobo(init_grasp, franka, planner, steps=30):
    action_list = prepare_grasp_motion_planning_payload(init_grasp, steps=steps)
    world = World()
    franka_view = get_franka_PD_controller(franka, [2.0] * 9)
    paths = []
    for idx, target in tqdm(enumerate(action_list)):
        position = target["translation"]
        franka_p, _ = franka.get_world_pose()
        position = position - franka_p
        orientation = target["orientation"]
        results = planner.plan(
            position,
            orientation,
            franka.get_joints_state(),
        )
        actions = []
        if results is not None:
            for res in results:
                if target["grasp"]:
                    actions.append(np.concatenate([res, [0.00, 0.00]]).tolist())
                else:
                    actions.append(np.concatenate([res, [0.04, 0.04]]).tolist())
        if len(actions) == 0:
            return []
        paths.append({"paths": copy.deepcopy(actions), "grasp": target["grasp"]})
        while actions:
            action = actions.pop(0)
            franka_view.set_joint_position_targets(action)
            world.step(render=False)
    return paths


def prepare_grasp_motion_planning_payload(init_grasp, steps=30):
    init_grasp["orientation"] = adjust_orientation(init_grasp["orientation"])
    action_list = []
    action_list.append(
        {
            "name": "pre_grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"], init_grasp["orientation"], 0.14
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": False,
        }
    )
    action_list.append(
        {
            "name": "grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"], init_grasp["orientation"], 0.08
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": False,
        }
    )
    action_list.append(
        {
            "name": "post_grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"], init_grasp["orientation"], 0.30
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": True,
        }
    )
    return action_list


def apply_action_by_config(
    scene, action_info, default_config, demogen_config, recorder, idx
):
    # if is pick and place action
    if (
        "position" in action_info
        and "obj1_uid" in action_info
        and "obj2_uid" in action_info
    ):
        set_mass(scene["object_list"][action_info["obj1_uid"]].prim_path, 0.1)
        set_mass(scene["object_list"][action_info["obj2_uid"]].prim_path, 10.0)
        action_meta_info = get_action_meta_info(scene, action_info, default_config)
        record_planning(
            scene, recorder, demogen_config, action_meta_info, action_info, idx
        )
        pclist = meshlist_to_pclist(
            get_current_meshList(scene["object_list"], scene["cacheDict"]["meshDict"])
        )
        is_success = check_subgoal_finished_rigid(
            action_info,
            pclist[action_info["obj1_uid"]],
            pclist[action_info["obj2_uid"]],
        )
        return is_success
    elif "type" in action_info and action_info["type"] == "code_skill":
        is_success = record_code_skill(
            scene, recorder, demogen_config, action_info, idx
        )
        return is_success
    elif "type" in action_info and action_info["type"] == "mimicgen_skill":
        is_success = record_mimicgen_skill(
            scene, recorder, demogen_config, action_info, idx
        )
        return is_success
    else:
        raise ValueError("Unsupported action")


def get_action_sequence(
    scene, demogen_config, action_meta_info, action_info, max_try=1
):
    cnt = 0
    while True:
        current_pose_list = collect_world_pose_list(scene["object_list"])
        current_joint_positions = scene["robot_info"]["robot_list"][
            0
        ].robot.get_joint_positions()
        combined_cloud = []
        meshlist = get_current_meshList(
            scene["object_list"], scene["cacheDict"]["meshDict"]
        )
        pclist = meshlist_to_pclist(meshlist)
        for key in pclist:
            if key != action_info["obj1_uid"]:
                combined_cloud.append(pclist[key])
        if demogen_config["generation_config"]["planner"] == "mplib":
            combined_cloud = np.vstack(combined_cloud)
            paths = feasibility_analysis(
                action_meta_info["init_grasp"],
                action_meta_info["grasp_tar_t"],
                action_meta_info["grasp_tar_o"],
                scene["robot_info"]["robot_list"][0],
                scene["planner_list"][0],
                combined_cloud,
                aug_distance=demogen_config["generation_config"].get(
                    "aug_distance", 0.0
                ),
            )
        elif demogen_config["generation_config"]["planner"] == "curobo":
            paths = feasibility_analysis_curobo(
                action_meta_info["init_grasp"],
                action_meta_info["grasp_tar_t"],
                action_meta_info["grasp_tar_o"],
                scene["robot_info"]["robot_list"][0],
                scene["planner_list"][0],
                aug_distance=demogen_config["generation_config"].get(
                    "aug_distance", 0.0
                ),
            )
        if len(paths) != 6:
            raise ValueError("Task planning failed")
        for _ in tqdm(range(30)):
            scene["world"].step(render=False)
        meshlist = get_current_meshList(
            scene["object_list"], scene["cacheDict"]["meshDict"]
        )
        pclist = meshlist_to_pclist(meshlist)
        finished = check_subgoal_finished_rigid(
            action_info,
            pclist[action_info["obj1_uid"]],
            pclist[action_info["obj2_uid"]],
        )
        reset_object_xyz(scene["object_list"], current_pose_list)
        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
            current_joint_positions
        )
        if finished:
            break
        cnt += 1
        if cnt >= max_try:
            raise ValueError("Task planning failed")
    return paths
