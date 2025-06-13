import math
import roboticstoolbox as rtb
from genmanip.core.pointcloud.pointcloud import (
    get_current_meshList,
    meshlist_to_pclist,
)
from genmanip.core.random_place.random_place import place_object_to_object_by_relation
from genmanip.core.robot.franka import joint_position_to_end_effector_pose
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path
from genmanip.thirdparty.mplib_planner import get_target
from genmanip.thirdparty.anygrasp import get_init_grasp
from genmanip.core.sensor.camera import set_camera_look_at
from genmanip.utils.transform_utils import (
    adjust_orientation,
    compute_final_pose,
    adjust_translation_along_quaternion,
)
import numpy as np
from tqdm import tqdm
from omni.isaac.core import World  # type: ignore
from genmanip.core.loading.loading import collect_world_pose_list, reset_object_xyz
from genmanip.utils.transform_utils import rot_orientation_by_z_axis


def get_action_init_grasp(scene, action_info, default_config, action_meta_info):
    set_camera_look_at(
        scene["camera_list"]["camera1"],
        scene["object_list"][action_info["obj1_uid"]],
        azimuth=180.0,
    )
    current_pose_list = collect_world_pose_list(scene["object_list"])
    current_joint_positions = scene["robot_info"]["robot_list"][
        0
    ].robot.get_joint_positions()
    robot_world_pose = scene["robot_info"]["robot_list"][0].robot.get_world_pose()
    scene["robot_info"]["robot_list"][0].robot.set_world_pose(
        robot_world_pose[0] + np.array([1000.0, 0.0, 0.0]), robot_world_pose[1]
    )
    for _ in range(5):
        scene["world"].step(render=True)
    reset_object_xyz(scene["object_list"], current_pose_list)
    scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
        current_joint_positions
    )
    scene["robot_info"]["robot_list"][0].robot.set_world_pose(*robot_world_pose)
    meshlist = get_current_meshList(
        scene["object_list"], scene["cacheDict"]["meshDict"]
    )
    mesh = meshlist[action_info["obj1_uid"]]
    action_meta_info["init_grasp"] = get_init_grasp(
        scene["camera_list"]["camera1"],
        mesh,
        address=default_config["ANYGRASP_ADDR"],
        port=default_config["ANYGRASP_PORT"],
    )
    action_meta_info["obj_init_t"], action_meta_info["obj_init_o"] = scene[
        "object_list"
    ][action_info["obj1_uid"]].get_world_pose()
    return action_meta_info


def compute_final_grasp(
    object_list, action, meshDict, ignored_uid=[], extra_erosion=0.05
):
    obj_init_t, obj_init_o = object_list[action["obj1_uid"]].get_world_pose()
    if action["position"] == "top" or action["position"] == "in":
        IS_OK = place_object_to_object_by_relation(
            action["obj1_uid"],
            action["obj2_uid"],
            object_list,
            meshDict,
            "on",
            platform_uid="00000000000000000000000000000000",
            ignored_uid=ignored_uid,
            extra_erosion=extra_erosion,
        )
    elif action["position"] == "near":
        IS_OK = place_object_to_object_by_relation(
            action["obj1_uid"],
            action["obj2_uid"],
            object_list,
            meshDict,
            "near",
            platform_uid="00000000000000000000000000000000",
            ignored_uid=ignored_uid,
            extra_erosion=extra_erosion,
        )
    else:
        if "another_obj2_uid" in action:
            IS_OK = place_object_to_object_by_relation(
                action["obj1_uid"],
                action["obj2_uid"],
                object_list,
                meshDict,
                action["position"],
                platform_uid="00000000000000000000000000000000",
                ignored_uid=ignored_uid,
                extra_erosion=extra_erosion,
                another_object2_uid=action["another_obj2_uid"],
            )
        else:
            IS_OK = place_object_to_object_by_relation(
                action["obj1_uid"],
                action["obj2_uid"],
                object_list,
                meshDict,
                action["position"],
                platform_uid="00000000000000000000000000000000",
                ignored_uid=ignored_uid,
                extra_erosion=extra_erosion,
            )
    if IS_OK == -1:
        return None, None
    obj_tar_t, obj_tar_o = object_list[action["obj1_uid"]].get_world_pose()
    object_list[action["obj1_uid"]].set_world_pose(
        position=obj_init_t, orientation=obj_init_o
    )
    return obj_tar_t, obj_tar_o


def prepare_motion_planning_payload(
    init_grasp, grasp_tar_t, grasp_tar_o, steps=30, aug_distance=0.0
):
    action_list = []
    action_list.append(
        {
            "name": "pre_grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"],
                init_grasp["orientation"],
                0.08,
                aug_distance=aug_distance,
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
                init_grasp["translation"], init_grasp["orientation"], 0.0
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
                init_grasp["translation"],
                init_grasp["orientation"],
                0.16,
                aug_distance=aug_distance,
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": True,
        }
    )
    action_list.append(
        {
            "name": "pre_place",
            "translation": adjust_translation_along_quaternion(
                grasp_tar_t, grasp_tar_o, 0.16, aug_distance=aug_distance
            ),
            "orientation": grasp_tar_o,
            "steps": steps,
            "grasp": True,
        }
    )
    action_list.append(
        {
            "name": "place",
            "translation": adjust_translation_along_quaternion(
                grasp_tar_t, grasp_tar_o, 0.02
            ),
            "orientation": grasp_tar_o,
            "steps": steps,
            "grasp": True,
        }
    )
    action_list.append(
        {
            "name": "post_place",
            "translation": adjust_translation_along_quaternion(
                grasp_tar_t, grasp_tar_o, 0.08, aug_distance=aug_distance
            ),
            "orientation": grasp_tar_o,
            "steps": steps,
            "grasp": False,
        }
    )
    return action_list


def record_planning_result(
    init_grasp,
    grasp_tar_t,
    grasp_tar_o,
    embodiment,
    planner,
    recorder,
    combined_cloud,
    idx_name,
    steps=30,
    aug_distance=0.0,
):
    panda = rtb.models.Panda()
    action_list = prepare_motion_planning_payload(
        init_grasp, grasp_tar_t, grasp_tar_o, steps=steps, aug_distance=aug_distance
    )
    planner.remove_point_cloud()
    if len(combined_cloud) > 0:
        combined_cloud = np.vstack(combined_cloud)
        planner.update_point_cloud(combined_cloud)
    world = World()
    franka_view = embodiment.robot_view
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
            raise Exception("motion planning failed at step: " + str(idx))
        if idx != len(action_list) - 1:
            if target["grasp"] != action_list[idx + 1]["grasp"]:
                action = actions[-1][:7] + (
                    embodiment.gripper_open
                    if not action_list[idx + 1]["grasp"]
                    else embodiment.gripper_close
                )
                for _ in range(5):
                    actions.append(action)
        while actions:
            action = actions.pop(0)
            franka_view.set_joint_position_targets(action)
            recorder.load_dynamic_info(
                action,
                1 if target["grasp"] else -1,
                name=f"{idx_name}/{target['name']}",
            )
            world.step(render=False)
            if joint_position_to_end_effector_pose(action, panda)[0][0] < -0.1:
                raise Exception("tcp out of bound")
    return True


def record_planning_result_curobo(
    init_grasp,
    grasp_tar_t,
    grasp_tar_o,
    embodiment,
    planner,
    recorder,
    idx_name,
    aug_distance=0.0,
):
    panda = rtb.models.Panda()
    action_list = prepare_motion_planning_payload(
        init_grasp, grasp_tar_t, grasp_tar_o, aug_distance=aug_distance
    )
    world = World()
    franka_view = embodiment.robot_view
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
            raise Exception("motion planning failed at step: " + str(idx))
        if idx != len(action_list) - 1:
            if target["grasp"] != action_list[idx + 1]["grasp"]:
                action = actions[-1][:7] + (
                    embodiment.gripper_open
                    if not action_list[idx + 1]["grasp"]
                    else embodiment.gripper_close
                )
                for _ in range(5):
                    actions.append(action)
        while actions:
            action = actions.pop(0)
            franka_view.set_joint_position_targets(action)
            recorder.load_dynamic_info(
                action,
                1 if target["grasp"] else -1,
                name=f"{idx_name}/{target['name']}",
            )
            world.step(render=True)
            if joint_position_to_end_effector_pose(action, panda)[0][0] < -0.1:
                raise Exception("tcp out of bound")
    return True


def adjust_grasp_by_embodiment(grasp, embodiment):
    grasp["orientation"] = adjust_orientation(grasp["orientation"])
    if embodiment.embodiment_name == "franka":
        if embodiment.gripper_name == "panda_hand":
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.08,
                aug_distance=0.0,
            )
        elif embodiment.gripper_name == "robotiq":
            grasp["orientation"] = rot_orientation_by_z_axis(grasp["orientation"], 45)
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.15,
                aug_distance=0.0,
            )
    return grasp


def get_action_meta_info(scene, action_info, default_config):
    action_meta_info = {}
    action_meta_info["obj_tar_t"], action_meta_info["obj_tar_o"] = compute_final_grasp(
        scene["object_list"],
        action_info,
        scene["cacheDict"]["meshDict"],
        ignored_uid=action_info.get("ignored_uid", []),
    )
    if action_meta_info["obj_tar_t"] is None or action_meta_info["obj_tar_o"] is None:
        raise Exception("can't create target position, retry......")
    action_meta_info = get_action_init_grasp(
        scene, action_info, default_config, action_meta_info
    )
    action_meta_info["init_grasp"] = adjust_grasp_by_embodiment(
        action_meta_info["init_grasp"],
        scene["robot_info"]["robot_list"][0],
    )
    action_meta_info["grasp_tar_t"], action_meta_info["grasp_tar_o"] = (
        compute_final_pose(
            action_meta_info["obj_init_t"],
            action_meta_info["obj_init_o"],
            action_meta_info["init_grasp"]["translation"],
            action_meta_info["init_grasp"]["orientation"],
            action_meta_info["obj_tar_t"],
            action_meta_info["obj_tar_o"],
        )
    )
    return action_meta_info


def record_planning(
    scene, recorder, demogen_config, action_meta_info, action_info, idx
):
    if demogen_config["generation_config"]["planner"] == "mplib":
        combined_cloud = []
        meshlist = get_current_meshList(
            scene["object_list"], scene["cacheDict"]["meshDict"]
        )
        pclist = meshlist_to_pclist(meshlist)
        for key in pclist:
            if key != action_info["obj1_uid"]:
                combined_cloud.append(pclist[key])
        combined_cloud = np.vstack(combined_cloud)
        is_success = record_planning_result(
            action_meta_info["init_grasp"],
            action_meta_info["grasp_tar_t"],
            action_meta_info["grasp_tar_o"],
            scene["robot_info"]["robot_list"][0],
            scene["planner_list"][0],
            recorder,
            combined_cloud,
            idx_name=idx,
            aug_distance=demogen_config["generation_config"].get("aug_distance", 0.0),
        )
    elif demogen_config["generation_config"]["planner"] == "curobo":
        # ignore_list = [
        #     f"obj_{action_info['obj1_uid']}",
        #     f"obj_{demogen_config['table_uid']}",
        # ]
        # ignore_list.extend(action_info.get("plan_ignored_list", []))
        # scene["planner_list"][0].update(ignore_list=ignore_list)
        is_success = record_planning_result_curobo(
            action_meta_info["init_grasp"],
            action_meta_info["grasp_tar_t"],
            action_meta_info["grasp_tar_o"],
            scene["robot_info"]["robot_list"][0],
            scene["planner_list"][0],
            recorder,
            idx_name=idx,
            aug_distance=demogen_config["generation_config"].get("aug_distance", 0.0),
        )
    if not is_success:
        raise ValueError("Task planning failed")
    return is_success
