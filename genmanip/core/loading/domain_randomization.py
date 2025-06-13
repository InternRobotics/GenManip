import cv2
import numpy as np
import os
import random
from mplib import Pose
from genmanip.utils.pc_utils import compute_aabb_lwh, compute_mesh_bbox
from scipy.spatial.transform import Rotation as R
from genmanip.core.random_place.random_place import (
    setup_random_tableset,
    setup_random_tableset_buffered,
    setup_random_tableset_by_centric_range,
    meshlist_to_pclist,
    verify_placement,
    setup_random_obj1_range,
    setup_random_custom_tableset,
    setup_random_all_range,
    setup_random_all_range_buffered,
    setup_scene_graph_placement,
)
from genmanip.demogen.evaluate.evaluate import check_finished
from genmanip.core.loading.loading import reset_object_xyz, reset_articulation_positions
from genmanip.core.pointcloud.pointcloud import (
    get_current_meshList,
    get_mesh_info_by_load,
    meshlist_to_pclist,
)
from genmanip.core.sensor.camera import get_src
from genmanip.core.usd_utils.light_utils import create_dome_light
from genmanip.core.usd_utils.material_utils import (
    change_material_info,
    change_table_mdl,
)
from genmanip.core.usd_utils.prim_utils import (
    add_usd_to_world,
    get_prim_bbox,
    resize_object,
    resize_object_by_lwh,
    set_colliders,
)
from genmanip.thirdparty.mplib_planner import get_mplib_planner
from genmanip.utils.robot_utils import joint_positions_to_position_and_orientation
from omni.isaac.core.utils.prims import delete_prim  # type: ignore


def remove_object_from_scene_by_preload(uid, scene):
    if scene["object_list"][uid].prim.IsActive():
        scene["object_list"][uid].prim.SetActive(False)
    scene["cacheDict"]["meshDict"].pop(uid)
    scene["object_list"].pop(uid)


def add_object_to_scene_by_preload(uid, scene, default_config, demogen_config):
    scene["object_list"][uid] = scene["cacheDict"]["preloaded_object_list"][uid]
    if not scene["object_list"][uid].prim.IsActive():
        scene["object_list"][uid].prim.SetActive(True)
    if uid in scene["cacheDict"]["preloaded_object_path_list"]:
        scene["cacheDict"]["meshDict"][uid] = get_mesh_info_by_load(
            scene["object_list"][uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                demogen_config["task_name"],
                os.path.dirname(scene["cacheDict"]["preloaded_object_path_list"][uid]),
                f"{uid}.obj",
            ),
        )
    else:
        scene["cacheDict"]["meshDict"][uid] = get_mesh_info_by_load(
            scene["object_list"][uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                demogen_config["task_name"],
                f"{uid}.obj",
            ),
        )


def replace_object_in_scene_by_uid(
    previous_uid, replaced_uid, scene, default_config, demogen_config
):
    if previous_uid in scene["object_list"]:
        remove_object_from_scene_by_preload(previous_uid, scene)
    add_object_to_scene_by_preload(replaced_uid, scene, default_config, demogen_config)


def resize_object_in_scene_by_uid(uid, scene, default_config, scale, demogen_config):
    meshlist = get_current_meshList(
        scene["object_list"], scene["cacheDict"]["meshDict"]
    )
    resize_object(
        scene["object_list"][uid],
        scale,
        meshlist[uid],
    )
    if uid in scene["cacheDict"]["preloaded_object_path_list"]:
        scene["cacheDict"]["meshDict"][uid] = get_mesh_info_by_load(
            scene["object_list"][uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                demogen_config["task_name"],
                os.path.dirname(scene["cacheDict"]["preloaded_object_path_list"][uid]),
                f"{uid}.obj",
            ),
        )
    else:
        scene["cacheDict"]["meshDict"][uid] = get_mesh_info_by_load(
            scene["object_list"][uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                demogen_config["task_name"],
                f"{uid}.obj",
            ),
        )


def replace_object_in_scene_by_uid_and_resize(
    previous_uid, replaced_uid, scene, default_config, scale, demogen_config
):
    replace_object_in_scene_by_uid(
        previous_uid, replaced_uid, scene, default_config, demogen_config
    )
    if scale != None:
        resize_object_in_scene_by_uid(
            replaced_uid, scene, default_config, scale, demogen_config
        )


def random_robot_pose(robot, random_range):
    position, _ = robot.get_world_pose()
    robot.set_world_pose(
        position=(
            random.uniform(position[0] - random_range, position[0] + random_range),
            random.uniform(position[1] - random_range, position[1] + random_range),
            position[2],
        ),
    )


def random_robot_eepose(robot, current_dir):
    franka_robot = robot.robot
    planner = get_mplib_planner(
        franka_robot, robot_type="franka", current_dir=current_dir
    )
    position, orientation = joint_positions_to_position_and_orientation(
        franka_robot.get_joint_positions()
    )
    position += np.array(
        [
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
        ]
    )
    rot = R.from_quat(orientation[[1, 2, 3, 0]])
    rot = rot * R.from_euler(
        "zyx",
        [
            random.uniform(-np.pi / 6, np.pi / 6),
            random.uniform(-np.pi / 6, np.pi / 6),
            random.uniform(-np.pi / 6, np.pi / 6),
        ],
        degrees=False,
    )
    orientation = rot.as_quat()[[3, 0, 1, 2]]
    joint_positions = planner.IK(
        Pose(p=position, q=orientation),
        franka_robot.get_joint_positions()[:9],
        return_closest=True,
    )
    if joint_positions[0] != "Success":
        return -1
    else:
        franka_robot.set_joint_positions(
            np.concatenate([joint_positions[1][:7], robot.gripper_open])
        )
        return 0


def replace_table(object_list, table_path, uuid):
    object = object_list["00000000000000000000000000000000"]
    try:
        object_list["00000000000000000000000000000000"].prim.SetActive(False)
    except:
        pass
    table_uid = table_path.split("/")[-2]
    object_list["00000000000000000000000000000000"] = add_usd_to_world(
        asset_path=table_path,
        prim_path=f"/World/{uuid}/obj_{table_uid}",
        name=f"obj_{table_uid}",
        orientation=R.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()[
            [3, 0, 1, 2]
        ],
    )
    try:
        resize_object_by_lwh(
            object_list["00000000000000000000000000000000"], l=1.0, w=1.50, h=1.002
        )
        aabb = get_prim_bbox(object_list["00000000000000000000000000000000"].prim)
        position, _ = object_list["00000000000000000000000000000000"].get_world_pose()
        position[2] -= aabb.get_min_bound()[2]
        object_list["00000000000000000000000000000000"].set_world_pose(
            position=position
        )
        set_colliders(object_list["00000000000000000000000000000000"].prim_path)
    except:
        delete_prim(object_list["00000000000000000000000000000000"].prim_path)
        object_list["00000000000000000000000000000000"] = object
        object_list["00000000000000000000000000000000"].prim.SetActive(True)


def random_texture_once(scene, default_config, demogen_config):
    if demogen_config["domain_randomization"]["random_environment"]["hdr"]:
        light_path = random.choice(scene["assets_list"]["domelight"])
        create_dome_light(
            f"/World/{scene['uuid']}/obj_defaultGroundPlane/GroundPlane/DomeLight",
            f"{default_config['ASSETS_DIR']}/miscs/hdrs/{light_path}",
        )
    if demogen_config["domain_randomization"]["random_environment"][
        "table_texture"
    ] and not (
        "table_type" in demogen_config["domain_randomization"]["random_environment"]
        and demogen_config["domain_randomization"]["random_environment"]["table_type"]
    ):
        light_intensity = 0
        while light_intensity < 80:
            texture_path = random.choice(scene["assets_list"]["wall_texture"])
            image = cv2.imread(
                os.path.abspath(
                    f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
                )
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            light_intensity = np.mean(np.array(image))
        change_material_info(
            f"{scene['object_list']['00000000000000000000000000000000'].prim_path}",
            texture_path=os.path.abspath(
                f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
            ),
            translation=(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)),
            rotation=0,
            scale=[0.4, 0.4],
        )
    if demogen_config["domain_randomization"]["random_environment"]["wall_texture"]:
        for i in range(5):
            light_intensity = 0
            while light_intensity < 80:
                texture_path = random.choice(scene["assets_list"]["wall_texture"])
                image = cv2.imread(
                    os.path.abspath(
                        f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
                    )
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                light_intensity = np.mean(np.array(image))
            scene["background"]["wall_textures"][i].set_texture(
                os.path.abspath(
                    f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
                )
            )
    if (
        "table_type" in demogen_config["domain_randomization"]["random_environment"]
        and demogen_config["domain_randomization"]["random_environment"]["table_type"]
    ):
        table_path = random.choice(scene["assets_list"]["table"])
        replace_table(scene["object_list"], table_path, scene["uuid"])
        if demogen_config["domain_randomization"]["random_environment"][
            "table_texture"
        ]:
            change_table_mdl(
                scene["object_list"]["00000000000000000000000000000000"].prim_path,
                texture_path_list=[
                    os.path.abspath(
                        f"{default_config['ASSETS_DIR']}/object_usds/grutopia_usd/Table/Materials/{table_mdl_path}"
                    )
                    for table_mdl_path in scene["assets_list"]["table_mdl"]
                ],
            )
    for _ in range(10):
        scene["world"].render()
    image = get_src(scene["camera_list"]["obs_camera"], "rgb")
    light_intensity = np.mean(image)
    return light_intensity


def random_texture(scene, default_config, demogen_config):
    cnt = 0
    while cnt < 10:
        light_intensity = random_texture_once(scene, default_config, demogen_config)
        if light_intensity >= 80:
            break
        cnt += 1
    if cnt == 10:
        print("random texture failed")
        return -1
    return 0


def domain_randomization(
    scene, default_config, demogen_config, task_data, mode="default"
):
    if demogen_config["domain_randomization"]["random_environment"][
        "robot_base_position"
    ]:
        for robot in scene["robot_info"]["robot_list"]:
            if isinstance(
                demogen_config["domain_randomization"]["random_environment"][
                    "robot_base_position"
                ],
                dict,
            ):
                random_robot_pose(
                    robot.robot,
                    demogen_config["domain_randomization"]["random_environment"][
                        "robot_base_position"
                    ]["random_range"],
                )
            else:
                random_robot_pose(robot.robot, 0.1)
    if demogen_config["domain_randomization"]["random_environment"].get(
        "robot_eepose", False
    ):
        for robot in scene["robot_info"]["robot_list"]:
            random_robot_eepose(
                robot,
                default_config["current_dir"],
            )
    if demogen_config["layout_config"]["type"] == "random_all":
        IS_OK = setup_random_tableset(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config["layout_config"]["ignored_objects"],
        )
    elif demogen_config["layout_config"]["type"] == "random_all_buffered":
        IS_OK = setup_random_tableset_buffered(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config["layout_config"]["ignored_objects"],
            task_data["goal"][0][0]["obj1_uid"],
            task_data["goal"][0][0]["obj2_uid"],
        )
    elif demogen_config["layout_config"]["type"] == "centric_random_range":
        IS_OK = setup_random_tableset_by_centric_range(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config["layout_config"],
            demogen_config["layout_config"]["ignored_objects"],
            demogen_config["layout_config"].get("partial_ignore", {}),
        )
    elif demogen_config["layout_config"]["type"] == "random_obj1_range":
        IS_OK = setup_random_obj1_range(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            task_data,
            demogen_config["layout_config"],
            scene["meta_infos"]["world_pose_list"],
        )
    elif demogen_config["layout_config"]["type"] == "random_custom_tableset":
        IS_OK = setup_random_custom_tableset(
            scene["object_list"],
            scene["articulation_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config["layout_config"]["custom_tableset"],
        )
    elif demogen_config["layout_config"]["type"] == "random_all_range":
        IS_OK = setup_random_all_range(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config["layout_config"],
            demogen_config["layout_config"]["ignored_objects"],
        )
    elif demogen_config["layout_config"]["type"] == "random_all_range_buffered":
        IS_OK = setup_random_all_range_buffered(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config["layout_config"],
            demogen_config["layout_config"]["ignored_objects"],
            task_data,
        )
    elif demogen_config["layout_config"]["type"] == "scene_graph_placement":
        IS_OK = setup_scene_graph_placement(
            scene["object_list"],
            scene["cacheDict"]["meshDict"],
            demogen_config,
        )
    else:
        IS_OK = 0
    if IS_OK == -1:
        print("random position failed")
        return IS_OK
    if (
        len(task_data["goal"]) > 0
        and len(task_data["goal"][0]) > 0
        and "obj1_uid" in task_data["goal"][0][0]
    ):
        if task_data["goal"][0][0]["obj1_uid"] in scene["object_list"].keys():
            IS_OK = verify_placement(
                scene["object_list"][task_data["goal"][0][0]["obj1_uid"]],
                scene["world"],
            )
            if not IS_OK:
                print("verify placement failed")
                return -1
    meshlist = get_current_meshList(
        scene["object_list"], scene["cacheDict"]["meshDict"]
    )
    pclist = meshlist_to_pclist(meshlist)
    finished = (
        check_finished(task_data["goal"], pclist, scene["articulation_list"]) == 1
    )
    if finished:
        print("check finished failed")
        return -1
    if mode == "demogen":
        return 0
    random_texture(scene, default_config, demogen_config)
    return 0


def reset_scene(scene):
    reset_object_xyz(scene["object_list"], scene["meta_infos"]["world_pose_list"])
    reset_articulation_positions(scene)
    for robot, joint_positions, joint_velocities, robot_pose in zip(
        scene["robot_info"]["robot_list"],
        scene["meta_infos"]["joint_positions"],
        scene["meta_infos"]["joint_velocities"],
        scene["meta_infos"]["robot_pose_list"],
    ):
        robot.robot.set_joint_positions(joint_positions)
        robot.robot.set_joint_velocities(joint_velocities)
        robot.robot.set_world_pose(*robot_pose)


def adjust_object_scale_by_thickness(
    scene, uid, default_config, demogen_config, object_scale, min_thickness
):
    meshlist = get_current_meshList(
        scene["object_list"], scene["cacheDict"]["meshDict"]
    )
    mesh = meshlist[uid]
    aabb = compute_mesh_bbox(mesh)
    if np.min(compute_aabb_lwh(aabb)) > min_thickness:
        l, w, h = compute_aabb_lwh(aabb)
        min_thickness_ratio = min_thickness / np.min(compute_aabb_lwh(aabb))
        min_thickness_ratio = max(min_thickness_ratio, 0.04 / h)
        l *= min_thickness_ratio
        w *= min_thickness_ratio
        h *= min_thickness_ratio
        resize_object_by_lwh(
            scene["object_list"][uid],
            l=l,
            w=w,
            h=h,
            mesh=mesh,
        )
        scene["cacheDict"]["meshDict"][uid] = get_mesh_info_by_load(
            scene["object_list"][uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                demogen_config["task_name"],
                os.path.dirname(scene["cacheDict"]["preloaded_object_path_list"][uid]),
                f"{uid}.obj",
            ),
        )


def get_object_scale(replace_object_config, key, replaced_uid, object_pool):
    if (
        "option" in replace_object_config[key]
        and "plain_replace" in replace_object_config[key]["option"]
    ):
        scale = None
    else:
        object_info = object_pool.get_object_info(replaced_uid)
        if object_info is None:
            scale = random.uniform(0.08, 0.12)
        else:
            scale = random.uniform(
                object_info["scale"][0],
                object_info["scale"][1],
            )
    if "clip_range" in replace_object_config[key]:
        scale = np.clip(
            scale,
            replace_object_config[key]["clip_range"]["min"],
            replace_object_config[key]["clip_range"]["max"],
        )
    return scale


def build_up_scene(scene, demogen_config, default_config, task_data):
    if "meta_to_fine_projection" not in scene["cacheDict"]:
        scene["cacheDict"]["meta_to_fine_projection"] = {}
        for key in scene["cacheDict"]["preload_hash_feature"]:
            scene["cacheDict"]["meta_to_fine_projection"][key] = ""
    replace_object_config = demogen_config["object_config"]
    added_uid_list = []
    object_config_key_list = list(replace_object_config.keys())
    object_config_key_list.sort()
    for key in replace_object_config:
        if replace_object_config[key]["type"] == "add_additional_object_from_path":
            object_config_key_list.remove(key)
    for key in object_config_key_list:
        while True:
            replaced_uid = random.choice(
                scene["cacheDict"]["preloaded_object_uid_list"][
                    scene["cacheDict"]["preload_hash_feature"][key]
                ]
            )
            if replaced_uid not in added_uid_list:
                break
        added_uid_list.append(replaced_uid)
    for key, replaced_uid in scene["cacheDict"]["meta_to_fine_projection"].items():
        if (
            replaced_uid in scene["object_list"]
            and replace_object_config[key]["type"] == "load_object_from_path"
        ):
            remove_object_from_scene_by_preload(replaced_uid, scene)
    for key, replaced_uid in zip(object_config_key_list, added_uid_list):
        if replace_object_config[key]["type"] == "existed_object":
            if not scene["cacheDict"]["preloaded_object_list"][
                replaced_uid
            ].prim.IsActive():
                scene["cacheDict"]["preloaded_object_list"][
                    replaced_uid
                ].prim.SetActive(True)
            if replaced_uid not in scene["object_list"]:
                scene["object_list"][replaced_uid] = scene["cacheDict"][
                    "preloaded_object_list"
                ][replaced_uid]
                scene["cacheDict"]["meshDict"][replaced_uid] = get_mesh_info_by_load(
                    scene["object_list"][replaced_uid],
                    os.path.join(
                        default_config["ASSETS_DIR"],
                        "mesh_data",
                        demogen_config["task_name"],
                        f"{replaced_uid}.obj",
                    ),
                )
            scene["cacheDict"]["meta_to_fine_projection"][key] = replaced_uid
        elif replace_object_config[key]["type"] == "load_object_from_path":
            scale = get_object_scale(
                replace_object_config, key, replaced_uid, scene["object_pool"]
            )
            add_object_to_scene_by_preload(
                replaced_uid, scene, default_config, demogen_config
            )
            if scale is not None:
                resize_object_in_scene_by_uid(
                    replaced_uid, scene, default_config, scale, demogen_config
                )
            scene["cacheDict"]["meta_to_fine_projection"][key] = replaced_uid
            if (
                "option" in replace_object_config[key]
                and "adjust_thickness" in replace_object_config[key]["option"]
            ):
                adjust_object_scale_by_thickness(
                    scene,
                    replaced_uid,
                    default_config,
                    demogen_config,
                    scale,
                    0.06,
                )
    for i in range(len(task_data["goal"])):
        for j in range(len(task_data["goal"][i])):
            if (
                "obj1_uid" in task_data["goal"][i][j]
                and task_data["goal"][i][j]["obj1_uid"]
                in scene["cacheDict"]["meta_to_fine_projection"]
            ):
                task_data["goal"][i][j]["obj1_uid"] = scene["cacheDict"][
                    "meta_to_fine_projection"
                ][task_data["goal"][i][j]["obj1_uid"]]
            if (
                "obj2_uid" in task_data["goal"][i][j]
                and task_data["goal"][i][j]["obj2_uid"]
                in scene["cacheDict"]["meta_to_fine_projection"]
            ):
                task_data["goal"][i][j]["obj2_uid"] = scene["cacheDict"][
                    "meta_to_fine_projection"
                ][task_data["goal"][i][j]["obj2_uid"]]
            if "ignored_uid" in task_data["goal"][i][j]:
                for k in range(len(task_data["goal"][i][j]["ignored_uid"])):
                    if (
                        task_data["goal"][i][j]["ignored_uid"][k]
                        in scene["cacheDict"]["meta_to_fine_projection"]
                    ):
                        task_data["goal"][i][j]["ignored_uid"][k] = scene["cacheDict"][
                            "meta_to_fine_projection"
                        ][task_data["goal"][i][j]["ignored_uid"][k]]
    task_data["object_infos"] = {}
    for obj_uid in scene["object_list"]:
        obj_info = scene["object_pool"].get_object_info(obj_uid)
        if obj_info is not None and obj_uid not in task_data["object_infos"]:
            task_data["object_infos"][obj_uid] = obj_info
    for obj_uid in scene["articulation_list"]:
        obj_info = scene["object_pool"].get_object_info(obj_uid)
        if obj_info is not None and obj_uid not in task_data["object_infos"]:
            task_data["object_infos"][obj_uid] = obj_info
    for obj_uid in scene["cacheDict"]["preloaded_object_list"]:
        obj_info = scene["object_pool"].get_object_info(obj_uid)
        if obj_info is not None and obj_uid not in task_data["object_infos"]:
            task_data["object_infos"][obj_uid] = obj_info
