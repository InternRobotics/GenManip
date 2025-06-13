import copy
import numpy as np
import os
import random
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import omni.usd  # type: ignore
from omni.isaac.core import World  # type: ignore
from omni.isaac.core.articulations import Articulation  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.utils.prims import create_prim, delete_prim, is_prim_path_valid  # type: ignore
from omni.isaac.franka import Franka  # type: ignore
from omni.isaac.sensor import Camera  # type: ignore
from pxr import UsdGeom  # type: ignore

from genmanip.core.loading.preload_rules import (
    apply_rule,
    generate_long_horizon_by_shape,
    generate_long_horizon_by_materials,
    generate_long_horizon_by_color,
    generate_long_horizon_by_category,
    collect_all_colors,
    collect_all_shapes,
    collect_all_materials,
)
from genmanip.core.pointcloud.pointcloud import objectList2meshList
from genmanip.core.robot.franka import get_franka_PD_controller
from genmanip.core.sensor.camera import setup_camera, get_src
from genmanip.core.usd_utils.material_utils import create_omni_pbr
from genmanip.core.usd_utils.prim_utils import (
    add_usd_to_world,
    set_colliders,
    set_gravity,
    set_mass,
    set_rigid_body_CCD,
    set_semantic_label,
    setup_physics_scene,
    remove_contact_offset,
    add_contact_offset,
)
from genmanip.thirdparty.curobo_planner import get_curobo_planner
from genmanip.thirdparty.mplib_planner import get_mplib_planner
from genmanip.utils.utils import generate_hash
from genmanip.utils.file_utils import load_yaml, load_json
from object_utils.object_pool import ObjectPool
from genmanip.demogen.planning.embodiment import (
    FrankaNormalEmbodiment,
    FrankaRobotiqEmbodiment,
)
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path
from omni.isaac.core.utils.prims import create_prim, delete_prim, is_prim_path_valid, get_prim_at_path  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.robots.robot import Robot  # type: ignore
import omni.replicator.core as rep  # type: ignore


def setup_walls_and_materials(uuid, world, object_list):
    walls = []
    wall_position_list = [[0, -25, 10], [25, 0, 10], [0, 25, 10], [-25, 0, 10]]
    wall_textures = []
    for i in range(5):
        mat = create_omni_pbr(
            f"/World/{uuid}/obj_defaultGroundPlane/GroundPlane/Wall{i}_Material"
        )
        wall_textures.append(mat)
    for i in range(4):
        prim_path = f"/World/{uuid}/obj_defaultGroundPlane/GroundPlane/Wall{i}"
        plane_geom = UsdGeom.Plane.Define(world.stage, prim_path)
        plane_geom.CreateLengthAttr().Set(20)
        plane_geom.CreateWidthAttr().Set(50)
        plane_xform = XFormPrim(
            prim_path,
            scale=[1, 1, 1],
            translation=wall_position_list[i],
            orientation=R.from_euler("xyz", [90, 0, 90 * i], degrees=True).as_quat()[
                [3, 0, 1, 2]
            ],
        )
        walls.append(plane_xform)
        plane_xform.apply_visual_material(wall_textures[i])
    default_ground_plane_xform = XFormPrim(
        object_list["defaultGroundPlane"].prim_path + "/GroundPlane/CollisionMesh"
    )
    default_ground_plane_xform.apply_visual_material(wall_textures[4])
    return walls, wall_textures


def collect_world_pose_list(object_list):
    world_pose_list = {}
    for key in object_list:
        if key == "00000000000000000000000000000000" or key == "defaultGroundPlane":
            continue
        world_pose_list[key] = object_list[key].get_world_pose()
    return world_pose_list


def collect_articulation_list(scene, articulation_list):
    init_positions_list = {}
    for articulation_id, articulation in articulation_list.items():
        if scene["articulation_data"][articulation_id]["is_articulated"]:
            init_positions_list[articulation_id] = articulation.get_joint_positions()
    return init_positions_list


def create_camera_list(camera_data: dict, uuid, rendering_dt=1 / 60.0):
    camera_list = {}
    for key in camera_data:
        rp = rep.create.render_product(
            rep.create.camera(),
            (
                camera_data[key]["resolution"][0],
                camera_data[key]["resolution"][1],
            ),
        )
        if camera_data[key]["exists"]:
            camera_list[key] = Camera(
                prim_path=f'/World/{uuid}{camera_data[key]["prim_path"]}',
                name=camera_data[key]["name"],
                frequency=1 / rendering_dt,
                resolution=(
                    camera_data[key]["resolution"][0],
                    camera_data[key]["resolution"][1],
                ),
                render_product_path=rp.path,
            )
            if "position" in camera_data[key] and "orientation" in camera_data[key]:
                camera_list[key].set_local_pose(
                    camera_data[key]["position"], camera_data[key]["orientation"]
                )
        else:
            camera_list[key] = Camera(
                prim_path=f'/Camera{camera_data[key]["prim_path"]}',
                name=camera_data[key]["name"],
                frequency=1 / rendering_dt,
                resolution=(
                    camera_data[key]["resolution"][0],
                    camera_data[key]["resolution"][1],
                ),
                position=camera_data[key]["position"],
                orientation=camera_data[key]["orientation"],
                render_product_path=rp.path,
            )
        if "camera_params" not in camera_data[key]:
            camera_data[key]["camera_params"] = None
        setup_camera(
            camera_list[key],
            focal_length=camera_data[key]["focal_length"],
            clipping_range_min=camera_data[key]["clipping_range_min"],
            clipping_range_max=camera_data[key]["clipping_range_max"],
            vertical_aperture=camera_data[key]["vertical_aperture"],
            horizontal_aperture=camera_data[key]["horizontal_aperture"],
            with_distance=camera_data[key]["with_distance"],
            with_semantic=camera_data[key]["with_semantic"],
            with_bbox2d=camera_data[key]["with_bbox2d"],
            with_bbox3d=camera_data[key]["with_bbox3d"],
            with_motion_vector=camera_data[key]["with_motion_vector"],
            camera_params=camera_data[key]["camera_params"],
        )
    return camera_list


def add_background_scene(scene, usd_path, position, scale=[0.01, 0.01, 0.01]):
    background_xform, background_uuid = load_world_xform_prim(
        os.path.join(usd_path),
        scene_prim_path=f"/World/background_{usd_path.split('/')[-1].split('.')[0]}",
    )
    background_xform.set_world_pose(position=position)
    background_xform.set_local_scale(scale)
    if (
        "defaultGroundPlane" in scene["object_list"]
        and scene["object_list"]["defaultGroundPlane"].prim.IsActive()
    ):
        scene["object_list"]["defaultGroundPlane"].prim.SetActive(False)


def get_object_list(uuid, scene_xform, table_uid):
    object_list = {}
    for scene in scene_xform.prim.GetAllChildren():
        for object in scene.GetAllChildren():
            if str(object.GetPath()).split("/")[-1] == "franka":
                set_semantic_label(
                    str(object.GetPath()), str(object.GetPath()).split("/")[-1]
                )
                continue
            if "camera" in str(object.GetPath()).split("/")[-1]:
                continue
            if str(object.GetPath()).split("/")[-1][:4] != "obj_":
                pass
            elif str(object.GetPath()).split("/")[-1][4:] != table_uid:
                object_list[str(object.GetPath()).split("/")[-1][4:]] = (
                    relate_object_from_data(
                        scene_uid=uuid,
                        uid=str(object.GetPath()).split("/")[-1][4:],
                    )
                )
            else:
                object_list["00000000000000000000000000000000"] = (
                    relate_object_from_data(
                        scene_uid=uuid, uid=str(object.GetPath()).split("/")[-1][4:]
                    )
                )
            set_semantic_label(
                str(object.GetPath()), str(object.GetPath()).split("/")[-1][4:]
            )
    return object_list


def load_camera_from_data(camera_data: dict, uuid):
    camera = Camera(
        prim_path=f"/World/{uuid}/" + camera_data["name"],
        name=camera_data["name"],
        position=camera_data["position"],
        frequency=camera_data["frequency"],
        resolution=(camera_data["resolution_width"], camera_data["resolution_height"]),
        orientation=camera_data["orientation"],
    )
    setup_camera(
        camera,
        focal_length=camera_data["focal_length"],
        clipping_range_min=camera_data["clipping_range_min"],
        clipping_range_max=camera_data["clipping_range_max"],
        vertical_aperture=camera_data["vertical_aperture"],
        horizontal_aperture=camera_data["horizontal_aperture"],
        with_distance=camera_data["with_distance"],
    )
    return camera


def load_world_xform_prim(scene_path, scene_prim_path="/World"):
    scene_path = os.path.abspath(scene_path)
    prim = create_prim(
        prim_path=scene_prim_path,
        prim_type="Xform",
        usd_path=scene_path,
    )
    prim_path = str(prim.GetPath())
    scene_xform = XFormPrim(
        prim_path,
        name="World",
    )
    uuid = str(scene_xform.prim.GetAllChildren()[0].GetPath()).split("/")[-1]
    return scene_xform, uuid


def relate_franka_from_data(scene_uid):
    robot = Franka(
        prim_path=f"/World/{scene_uid}/franka",
    )
    robot.set_solver_position_iteration_count(32)
    robot.set_enabled_self_collisions(True)
    robot.set_stabilization_threshold(0.005)
    robot.set_solver_velocity_iteration_count(16)
    return robot


def relate_franka_robotiq_from_data(scene_uid, scene, default_config):
    position, orientation = get_world_pose_by_prim_path(f"/World/{scene_uid}/franka")
    prim = get_prim_at_path(f"/World/{scene['uuid']}/franka")
    if prim.IsActive():
        prim.SetActive(False)
    prim = create_prim(
        prim_path=f"/World/{scene['uuid']}/robotiq",
        prim_type="Xform",
        usd_path=os.path.join(
            default_config["ASSETS_DIR"], "robot_usds/robotiq/robot.usd"
        ),
    )
    franka_robotiq = Robot(
        prim_path=f"/World/{scene['uuid']}/robotiq",
        name="franka_robotiq",
    )
    franka_robotiq.set_solver_position_iteration_count(124)
    franka_robotiq.set_stabilization_threshold(0.005)
    franka_robotiq.set_solver_velocity_iteration_count(4)
    franka_robotiq.set_world_pose(position, orientation)
    return franka_robotiq


def relate_franka_from_mimicgen(scene_uid):
    return Franka(
        prim_path=f"/World/{scene_uid}/_base/franka",
    )


def relate_object_from_data(scene_uid, uid):
    return XFormPrim(f"/World/{scene_uid}/obj_{uid}")


def reset_object_xyz(object_list, xyz):
    for key in object_list:
        if key == "00000000000000000000000000000000" or key == "defaultGroundPlane":
            continue
        if key in xyz:
            object_list[key].set_world_pose(*xyz[key])


def reset_articulation_positions(scene):
    if (
        "meta_infos" not in scene
        or "articulation_pose_list" not in scene["meta_infos"]
        or "articulation_list" not in scene
        or "articulation_data" not in scene
    ):
        print("No articulation data found")
        return
    articulation_list = scene["articulation_list"]
    articulation_pose_list = scene["meta_infos"]["articulation_pose_list"]
    for articulation_id, articulation in articulation_list.items():
        if scene["articulation_data"][articulation_id]["is_articulated"]:
            articulation.set_joint_positions(articulation_pose_list[articulation_id])


def get_embodiment(robot_config, robot):
    if robot_config["type"] == "franka":
        if robot_config["config"]["gripper_type"] == "panda_hand":
            return FrankaNormalEmbodiment(robot)
        elif robot_config["config"]["gripper_type"] == "robotiq":
            return FrankaRobotiqEmbodiment(robot)
    else:
        raise ValueError(f"Unsupported robot type: {robot_config['type']}")


def add_robot_to_scene(scene, robot_config, default_config):
    if robot_config["type"] == "franka":
        if robot_config["config"]["gripper_type"] == "panda_hand":
            return relate_franka_from_data(scene["uuid"])
        elif robot_config["config"]["gripper_type"] == "robotiq":
            return relate_franka_robotiq_from_data(scene["uuid"], scene, default_config)
        else:
            raise ValueError(f"Unsupported robot config: {robot_config}")
    else:
        raise ValueError(f"Unsupported robot type: {robot_config['type']}")


def add_robot_view(robot_config, robot):
    if robot_config["type"] == "franka":
        if robot_config["config"]["gripper_type"] == "panda_hand":
            return get_franka_PD_controller(robot, max_joint_velocities=[2.0] * 9)
        elif robot_config["config"]["gripper_type"] == "robotiq":
            return get_franka_PD_controller(robot, max_joint_velocities=[2.0] * 13)
        else:
            raise ValueError(f"Unsupported robot config: {robot_config}")
    else:
        raise ValueError(f"Unsupported robot type: {robot_config['type']}")


def preload_object(
    object_path, uuid, uid, world, add_rigid_body=True, add_colliders=True
):
    if not os.path.exists(object_path):
        raise ValueError(f"Object path {object_path} does not exist")
    obj_xform = add_usd_to_world(
        asset_path=object_path,
        prim_path=f"/World/{uuid}/obj_{uid}",
        name=f"obj_{uid}",
        translation=[1000.0, 0.0, 0.0],
        orientation=[0.5, 0.5, 0.5, 0.5],
        scale=None,
        add_rigid_body=add_rigid_body,
        add_colliders=add_colliders,
        collision_approximation="convexDecomposition",
    )
    if obj_xform is not None:
        world.step()
        if obj_xform.prim.IsActive():
            obj_xform.prim.SetActive(False)
        print(f"Object {uid} loaded")
    else:
        delete_prim(f"/World/{uuid}/obj_{uid}")
    return obj_xform


def process_long_horizon_replacement(scene, default_config, demogen_config):
    replacement_config = demogen_config["domain_randomization"]["replace_object"][
        "replacement"
    ]
    folder_path = os.path.join(
        default_config["ASSETS_DIR"],
        replacement_config["random_long_horizon"]["folder_path"],
    )
    usd_list = os.listdir(folder_path)
    usd_list = [
        usd for usd in usd_list if not os.path.isdir(os.path.join(folder_path, usd))
    ]
    types = ["category", "materials", "color", "shape"]
    while True:
        type = random.choice(types)
        if type == "category":
            replacement_config, meta_info = generate_long_horizon_by_category(
                scene,
                usd_list,
                demogen_config["domain_randomization"]["replace_object"]["replacement"][
                    "random_long_horizon"
                ]["folder_path"],
            )
        elif type == "materials":
            replacement_config, meta_info = generate_long_horizon_by_materials(
                scene,
                usd_list,
                demogen_config["domain_randomization"]["replace_object"]["replacement"][
                    "random_long_horizon"
                ]["folder_path"],
            )
        elif type == "color":
            replacement_config, meta_info = generate_long_horizon_by_color(
                scene,
                usd_list,
                demogen_config["domain_randomization"]["replace_object"]["replacement"][
                    "random_long_horizon"
                ]["folder_path"],
            )
        elif type == "shape":
            replacement_config, meta_info = generate_long_horizon_by_shape(
                scene,
                usd_list,
                demogen_config["domain_randomization"]["replace_object"]["replacement"][
                    "random_long_horizon"
                ]["folder_path"],
            )
        if replacement_config is None:
            continue
        obj1_list = copy.deepcopy(usd_list)
        for rule in replacement_config["obj1"]["rule"]:
            obj1_list = apply_rule(rule, obj1_list, scene["object_pool"])
        obj2_list = copy.deepcopy(usd_list)
        for rule in replacement_config["obj2"]["rule"]:
            obj2_list = apply_rule(rule, obj2_list, scene["object_pool"])
        background_list = copy.deepcopy(usd_list)
        for rule in replacement_config["background"]["rule"]:
            background_list = apply_rule(rule, background_list, scene["object_pool"])
        if len(obj1_list) > 5 and len(obj2_list) > 5 and len(background_list) > 5:
            break
    return replacement_config, meta_info


def preprocess_object_config(scene, default_config, demogen_config):
    object_config_backup = copy.deepcopy(demogen_config["object_config"])
    while True:
        object_config = copy.deepcopy(object_config_backup)
        color_list = collect_all_colors(scene["object_pool"])
        shape_list = collect_all_shapes(scene["object_pool"])
        material_list = collect_all_materials(scene["object_pool"])
        object_config_keys = list(object_config.keys())
        object_config_keys.sort()
        color_project_dict = {}
        shape_project_dict = {}
        material_project_dict = {}
        for key in object_config_keys:
            if object_config[key]["type"] == "rule":
                for rule in object_config[key]["filter_rule"]:
                    if "retrieve_color_[" in rule:
                        color_index = rule.split("retrieve_color_[")[1].split("]")[0]
                        if color_index not in color_project_dict:
                            color_project_dict[color_index] = random.choice(color_list)
                    elif "retrieve_shape_[" in rule:
                        shape_index = rule.split("retrieve_shape_[")[1].split("]")[0]
                        if shape_index not in shape_project_dict:
                            shape_project_dict[shape_index] = random.choice(shape_list)
                    elif "retrieve_material_[" in rule:
                        material_index = rule.split("retrieve_material_[")[1].split(
                            "]"
                        )[0]
                        if material_index not in material_project_dict:
                            material_project_dict[material_index] = random.choice(
                                material_list
                            )
                    elif "retrieve_not_color_[" in rule:
                        color_index = rule.split("retrieve_not_color_[")[1].split("]")[
                            0
                        ]
                        if color_index not in color_project_dict:
                            color_project_dict[color_index] = random.choice(color_list)
                    elif "retrieve_not_shape_[" in rule:
                        shape_index = rule.split("retrieve_not_shape_[")[1].split("]")[
                            0
                        ]
                        if shape_index not in shape_project_dict:
                            shape_project_dict[shape_index] = random.choice(shape_list)
                    elif "retrieve_not_material_[" in rule:
                        material_index = rule.split("retrieve_not_material_[")[1].split(
                            "]"
                        )[0]
                        if material_index not in material_project_dict:
                            material_project_dict[material_index] = random.choice(
                                material_list
                            )
        for key in object_config_keys:
            if object_config[key]["type"] == "rule":
                for rule_idx in range(len(object_config[key]["filter_rule"])):
                    rule = object_config[key]["filter_rule"][rule_idx]
                    if "retrieve_color_[" in rule:
                        color_index = rule.split("retrieve_color_[")[1].split("]")[0]
                        rule = f"retrieve_color_{color_project_dict[color_index]}"
                    elif "retrieve_shape_[" in rule:
                        shape_index = rule.split("retrieve_shape_[")[1].split("]")[0]
                        rule = f"retrieve_shape_{shape_project_dict[shape_index]}"
                    elif "retrieve_material_[" in rule:
                        material_index = rule.split("retrieve_material_[")[1].split(
                            "]"
                        )[0]
                        rule = (
                            f"retrieve_material_{material_project_dict[material_index]}"
                        )
                    elif "retrieve_not_color_[" in rule:
                        color_index = rule.split("retrieve_not_color_[")[1].split("]")[
                            0
                        ]
                        rule = f"retrieve_not_color_{color_project_dict[color_index]}"
                    elif "retrieve_not_shape_[" in rule:
                        shape_index = rule.split("retrieve_not_shape_[")[1].split("]")[
                            0
                        ]
                        rule = f"retrieve_not_shape_{shape_project_dict[shape_index]}"
                    elif "retrieve_not_material_[" in rule:
                        material_index = rule.split("retrieve_not_material_[")[1].split(
                            "]"
                        )[0]
                        rule = f"retrieve_not_material_{material_project_dict[material_index]}"
                    object_config[key]["filter_rule"][rule_idx] = rule
        is_vaild = True
        for key in object_config_keys:
            if object_config[key]["type"] == "load_object_from_path":
                folder_path = os.path.join(
                    default_config["ASSETS_DIR"],
                    object_config[key]["path"],
                )
                usd_list = os.listdir(folder_path)
                usd_list = [
                    usd
                    for usd in usd_list
                    if usd.endswith(".usd")
                    and not os.path.isdir(os.path.join(folder_path, usd))
                ]
                usd_list_len = len(usd_list)
                for rule in object_config[key]["filter_rule"]:
                    usd_list = apply_rule(rule, usd_list, scene["object_pool"])
                if len(usd_list) < 5 and len(usd_list) < usd_list_len:
                    is_vaild = False
                    break
        if is_vaild:
            break
    return object_config


def preload_objects(scene, default_config, demogen_config):
    demogen_config["object_config"] = preprocess_object_config(
        scene, default_config, demogen_config
    )
    scene["cacheDict"]["replacement"] = {}
    scene["cacheDict"]["preloaded_object_list"] = {}
    scene["cacheDict"]["preloaded_object_path_list"] = {}
    scene["cacheDict"]["preloaded_object_uid_list"] = {}
    scene["cacheDict"]["preload_hash_feature"] = {}
    object_config = demogen_config["object_config"]
    object_config_keys = list(object_config.keys())
    object_config_keys.sort()
    for key in object_config_keys:
        if object_config[key]["type"] == "load_object_from_path":
            origin_text = object_config[key]["path"]
            rules = object_config[key]["filter_rule"]
            rules.sort()
            for rule in rules:
                origin_text += rule
            scene["cacheDict"]["preload_hash_feature"][key] = generate_hash(origin_text)
        elif object_config[key]["type"] == "existed_object":
            origin_text = object_config[key]["uid_list"]
            if not isinstance(origin_text, list):
                origin_text = [origin_text]
            origin_text.sort()
            concat_text = ""
            for uid in origin_text:
                concat_text += uid
            scene["cacheDict"]["preload_hash_feature"][key] = generate_hash(concat_text)
        elif object_config[key]["type"] == "add_additional_object_from_path":
            scene["cacheDict"]["preload_hash_feature"][key] = generate_hash(
                object_config[key]["path"]
            )
            object_config[key]["max_cached_num"] = len(
                os.listdir(
                    os.path.join(
                        default_config["ASSETS_DIR"],
                        object_config[key]["path"],
                    )
                )
            )
    max_cached_num_dict = {}
    for key in object_config_keys:
        if object_config[key]["type"] == "existed_object":
            continue
        if scene["cacheDict"]["preload_hash_feature"][key] not in max_cached_num_dict:
            max_cached_num_dict[scene["cacheDict"]["preload_hash_feature"][key]] = (
                object_config[key]["max_cached_num"]
            )
        else:
            max_cached_num_dict[scene["cacheDict"]["preload_hash_feature"][key]] = max(
                max_cached_num_dict[scene["cacheDict"]["preload_hash_feature"][key]],
                object_config[key]["max_cached_num"],
            )
    sorted_object_config_keys = sorted(
        object_config_keys,
        key=lambda x: object_config[x]["type"] == "add_additional_object_from_path",
        reverse=True,
    )
    for key in sorted_object_config_keys:
        if (
            scene["cacheDict"]["preload_hash_feature"][key]
            in scene["cacheDict"]["preloaded_object_uid_list"]
        ):
            continue
        else:
            if object_config[key]["type"] == "existed_object":
                scene["cacheDict"]["preloaded_object_uid_list"][
                    scene["cacheDict"]["preload_hash_feature"][key]
                ] = object_config[key]["uid_list"]
                for uid in object_config[key]["uid_list"]:
                    if uid in scene["object_list"]:
                        scene["cacheDict"]["preloaded_object_list"][uid] = scene[
                            "object_list"
                        ][uid]
                    else:
                        scene["cacheDict"]["preloaded_object_list"][uid] = scene[
                            "articulation_list"
                        ][uid]


def create_planner(scene, demogen_config, current_dir):
    planner_name = demogen_config["generation_config"]["planner"]
    if planner_name == "mplib":
        return [
            get_mplib_planner(robot.robot, robot_config["type"], current_dir)
            for robot, robot_config in zip(
                scene["robot_info"]["robot_list"],
                demogen_config["robots"],
            )
        ]
    elif planner_name == "curobo":
        return [
            get_curobo_planner(robot.robot, robot_config["type"], scene, current_dir)
            for robot, robot_config in zip(
                scene["robot_info"]["robot_list"],
                demogen_config["robots"],
            )
        ]
    else:
        raise ValueError(f"Unsupported planner type: {planner_name}")


def add_articulation_to_scene(uid, uuid, world):
    articulation = Articulation(
        prim_path=f"/World/{uuid}/obj_{uid}",
        name=f"obj_{uid}",
    )
    world.scene.add(articulation)
    return articulation


def load_articulation_data(scene, demogen_config, current_dir):
    if "articulation_data_path" in demogen_config["domain_randomization"]:
        scene["articulation_data"] = load_json(
            os.path.join(
                current_dir,
                "assets/objects",
                f"{demogen_config['domain_randomization']['articulation_data_path']}.json",
            )
        )
    else:
        scene["articulation_data"] = load_json(
            os.path.join(current_dir, "assets/objects/articulation_data.json")
        )


def load_object_pool(scene, demogen_config, current_dir):
    if "object_data_path" in demogen_config["domain_randomization"]:
        scene["object_pool"] = ObjectPool(
            os.path.join(
                current_dir,
                "assets/objects",
                f"{demogen_config['domain_randomization']['object_data_path']}.pickle",
            )
        )
    else:
        scene["object_pool"] = ObjectPool(
            os.path.join(current_dir, "assets/objects/object_v7.pickle")
        )


def set_articulation(scene, demogen_config, world):
    for key in demogen_config["generation_config"]["articulation"]:
        # todo: remove "pan" from "is_articulated", or change key to "have_joint"
        if scene["articulation_data"][key]["is_articulated"]:
            scene["articulation_list"][key] = add_articulation_to_scene(
                key, scene["uuid"], world
            )
        else:
            scene["articulation_list"][key] = scene["object_list"][key]
    world.reset()
    for articulation in scene["articulation_list"].values():
        if scene["articulation_data"][key]["is_articulated"]:
            articulation._articulation_view.initialize()
    world.initialize_physics()
    for _ in range(10):
        world.step()
    for arti_id, articulation in scene["articulation_list"].items():
        if scene["articulation_data"][key]["is_articulated"]:
            if (
                arti_id in demogen_config["generation_config"]["articulation"]
                and "target_positions"
                in demogen_config["generation_config"]["articulation"][arti_id]
            ):
                articulation._articulation_view.set_joint_positions(
                    demogen_config["generation_config"]["articulation"][arti_id][
                        "target_positions"
                    ]
                )
    for _ in range(10):
        world.step()


def parse_articulation(scene, demogen_config):
    for arti_id, articulation in scene["articulation_list"].items():
        arti_parts = scene["articulation_data"][arti_id]["part"]
        arti_prim = scene["object_list"][arti_id]
        arti_prim_path = arti_prim.prim_path
        for part_name, part_group in arti_parts.items():
            part_prim_path = arti_prim_path + f"/Instance/{part_group}"
            arti_part = f"{arti_id}_{part_name}"
            scene["object_list"][arti_part] = XFormPrim(part_prim_path)
            scene["articulation_part_list"][arti_part] = XFormPrim(part_prim_path)
        scene["object_list"].pop(arti_id)


def parse_articulation_from_object_config(demogen_config):
    demogen_config["generation_config"]["articulation"] = {}
    for key in demogen_config["object_config"]:
        if demogen_config["object_config"][key][
            "type"
        ] == "existed_object" and demogen_config["object_config"][key].get(
            "is_articulated", False
        ):
            for uid in demogen_config["object_config"][key]["uid_list"]:
                info = {}
                info["target_positions"] = demogen_config["object_config"][key][
                    "target_positions"
                ]
                info["is_articulated"] = demogen_config["object_config"][key][
                    "is_articulated"
                ]
                demogen_config["generation_config"]["articulation"][uid] = info


def build_scene_from_config(
    demogen_config,
    default_config,
    current_dir,
    physics_dt=1 / 60.0,
    rendering_dt=1 / 60.0,
    is_eval=False,
    is_render=False,
):
    scene = {}
    world = World(physics_dt=physics_dt, rendering_dt=rendering_dt)
    setup_physics_scene()
    usd_name = demogen_config["usd_name"]
    scene["world"] = world
    scene["scene_xform"], scene["uuid"] = load_world_xform_prim(
        os.path.join(default_config["ASSETS_DIR"], f"{usd_name}.usda")
    )
    scene["object_list"] = get_object_list(
        scene["uuid"], scene["scene_xform"], demogen_config["table_uid"]
    )
    camera_info = demogen_config["domain_randomization"]["cameras"]
    if camera_info["type"] == "fixed":
        camera_data = load_yaml(os.path.join(current_dir, camera_info["config_path"]))
    else:
        raise ValueError(f"Unsupported camera type: {camera_info['type']}")
    if is_render:
        if "camera1" in camera_data:
            camera_data["camera1"]["resolution"] = [640, 480]
    scene["robot_info"] = {}
    scene["robot_info"]["robot_list"] = [
        get_embodiment(
            robot_config, add_robot_to_scene(scene, robot_config, default_config)
        )
        for robot_config in demogen_config["robots"]
    ]
    scene["camera_list"] = create_camera_list(camera_data, scene["uuid"], rendering_dt)
    for robot in scene["robot_info"]["robot_list"]:
        robot = world.scene.add(robot.robot)
    load_articulation_data(scene, demogen_config, current_dir)
    scene["articulation_list"] = {}
    scene["articulation_part_list"] = {}
    parse_articulation_from_object_config(demogen_config)
    if demogen_config["generation_config"]["articulation"]:
        set_articulation(scene, demogen_config, world)
        parse_articulation(scene, demogen_config)
    world.reset()
    for robot in scene["robot_info"]["robot_list"]:
        robot.robot.initialize()
    for idx, robot_cfg in enumerate(demogen_config["robots"]):
        if "default_joint_positions" in robot_cfg:
            scene["robot_info"]["robot_list"][idx].robot.set_joint_positions(
                np.array(robot_cfg["default_joint_positions"])
            )
    for key, articulation in scene["articulation_list"].items():
        if scene["articulation_data"][key]["is_articulated"]:
            articulation._articulation_view.initialize()
    scene["cacheDict"] = {}
    if not is_render:
        scene["cacheDict"]["meshDict"] = objectList2meshList(
            scene["object_list"],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                demogen_config["task_name"],
            ),
        )
    scene["robot_info"]["robot_view_list"] = [
        add_robot_view(robot_config, robot.robot)
        for robot_config, robot in zip(
            demogen_config["robots"], scene["robot_info"]["robot_list"]
        )
    ]
    for robot in scene["robot_info"]["robot_list"]:
        if robot.robot.get_joints_default_state() is not None:
            robot.robot.set_joint_positions(
                robot.robot.get_joints_default_state().positions
            )
    scene["background"] = {}
    if demogen_config["domain_randomization"]["random_environment"]["has_wall"]:
        scene["background"]["wall"], scene["background"]["wall_textures"] = (
            setup_walls_and_materials(scene["uuid"], world, scene["object_list"])
        )
    else:
        scene["background"]["wall"] = None
        scene["background"]["wall_textures"] = None
    scene["assets_list"] = {}
    if os.path.exists(os.path.join(default_config["ASSETS_DIR"], "miscs/hdrs")):
        scene["assets_list"]["domelight"] = os.listdir(
            f"{default_config['ASSETS_DIR']}/miscs/hdrs"
        )
    else:
        scene["assets_list"]["domelight"] = []
    if os.path.exists(os.path.join(default_config["ASSETS_DIR"], "miscs/textures")):
        scene["assets_list"]["wall_texture"] = os.listdir(
            f"{default_config['ASSETS_DIR']}/miscs/textures"
        )
    else:
        scene["assets_list"]["wall_texture"] = []
    if os.path.exists(
        os.path.join(
            default_config["ASSETS_DIR"], "object_usds/grutopia_usd/Table/Materials"
        )
    ):
        scene["assets_list"]["table_mdl"] = os.listdir(
            f"{default_config['ASSETS_DIR']}/object_usds/grutopia_usd/Table/Materials"
        )
    else:
        scene["assets_list"]["table_mdl"] = []
    if os.path.exists(
        os.path.join(
            default_config["ASSETS_DIR"], "object_usds/grutopia_usd/Table/table"
        )
    ):
        scene["assets_list"]["table"] = os.listdir(
            f"{default_config['ASSETS_DIR']}/object_usds/grutopia_usd/Table/table"
        )
        scene["assets_list"]["table"] = [
            os.path.join(
                default_config["ASSETS_DIR"],
                "object_usds/grutopia_usd/Table/table",
                table_path,
                "instance.usd",
            )
            for table_path in scene["assets_list"]["table"]
        ]
    else:
        scene["assets_list"]["table"] = []
    if not is_eval:
        scene["planner_list"] = create_planner(scene, demogen_config, current_dir)
    scene["tcp_configs"] = {}
    scene["tcp_configs"]["franka"] = load_yaml(
        os.path.join(current_dir, "configs/robots/franka_tcp.yaml")
    )
    return scene


def preprocess_scene(scene, demogen_config):
    preprocess_config = demogen_config["preprocess_config"]
    for preprocess_info in preprocess_config:
        if preprocess_info["type"] == "disable_contact_offset":
            for object in scene["object_list"].values():
                remove_contact_offset(object.prim_path)
        if preprocess_info["type"] == "enable_contact_offset":
            for object in scene["object_list"].values():
                add_contact_offset(object.prim_path, 0.1)
        if preprocess_info["type"] == "disable_gravity":
            for name, object in scene["object_list"].items():
                if (
                    name != "defaultGroundPlane"
                    and name not in scene["articulation_part_list"].keys()
                    and name not in scene["articulation_list"]
                ):
                    set_gravity(object.prim_path, preprocess_info["config"]["value"])
        if preprocess_info["type"] == "ccd":
            for name, object in scene["object_list"].items():
                if (
                    name != "defaultGroundPlane"
                    and name != "00000000000000000000000000000000"
                    and name
                    not in demogen_config["layout_config"].get("ignored_objects", [])
                    and name not in scene["articulation_part_list"].keys()
                    and name not in scene["articulation_list"]
                ):
                    set_rigid_body_CCD(object.prim_path, True)
        if preprocess_info["type"] == "collider":
            for name, object in scene["object_list"].items():
                if (
                    name != "defaultGroundPlane"
                    and name not in scene["articulation_part_list"].keys()
                    and name not in scene["articulation_list"]
                ):
                    if name == "00000000000000000000000000000000":
                        set_colliders(
                            object.prim_path,
                            collision_approximation=preprocess_info["config"]["type"],
                            convex_hulls=2048,
                        )
                    else:
                        set_colliders(
                            object.prim_path,
                            collision_approximation=preprocess_info["config"]["type"],
                        )
        if preprocess_info["type"] == "remove_all_object":
            object_list_key = list(scene["object_list"].keys())
            for key in object_list_key:
                if (
                    key != "defaultGroundPlane"
                    and key != "00000000000000000000000000000000"
                ):
                    if scene["object_list"][key].prim.IsActive():
                        scene["object_list"][key].prim.SetActive(False)
                    scene["object_list"].pop(key)
                    if (
                        "meshDict" in scene["cacheDict"]
                        and key in scene["cacheDict"]["meshDict"]
                    ):
                        scene["cacheDict"]["meshDict"].pop(key)


def cleanup_camera(camera_data, camera):
    if camera_data["with_bbox2d"]:
        camera.remove_bounding_box_2d_tight_from_frame()
        camera.remove_bounding_box_2d_loose_from_frame()
    if camera_data["with_bbox3d"]:
        camera.remove_bounding_box_3d_from_frame()
    if camera_data["with_motion_vector"]:
        camera.remove_motion_vectors_from_frame()
    if camera_data["with_semantic"]:
        camera.remove_semantic_segmentation_from_frame()
    if camera_data["with_distance"]:
        camera.remove_distance_to_image_plane_from_frame()
    if camera._render_product is not None:
        camera._render_product.destroy()
    del camera
    # delete_prim(camera.prim_path)


def clear_scene(scene, demogen_config, current_dir):
    camera_info = demogen_config["domain_randomization"]["cameras"]
    if camera_info["type"] == "fixed":
        camera_data = load_yaml(os.path.join(current_dir, camera_info["config_path"]))
    else:
        raise ValueError(f"Unsupported camera type: {camera_info['type']}")
    for camera_name, camera_info in camera_data.items():
        cleanup_camera(camera_info, scene["camera_list"][camera_name])
    scene["world"].scene.clear()
    del scene
    delete_prim("/World")
    delete_prim("/Camera")
    delete_prim("/Replicator")
    omni.usd.get_context().close_stage()
    omni.usd.get_context().new_stage()
    # action_registry = omni.kit.actions.core.get_action_registry()
    # action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    # action.execute()


def warmup_world(scene, physics_steps=100, without_depth=False):
    if not without_depth:
        while any(
            get_src(camera, "depth") is None for camera in scene["camera_list"].values()
        ):
            scene["world"].step()
    for _ in range(physics_steps):
        scene["world"].step(render=False)


def collect_meta_infos(scene):
    scene["meta_infos"] = {}
    scene["meta_infos"]["world_pose_list"] = collect_world_pose_list(
        scene["object_list"]
    )
    scene["meta_infos"]["articulation_pose_list"] = collect_articulation_list(
        scene, scene["articulation_list"]
    )
    scene["meta_infos"]["robot_pose_list"] = [
        robot.robot.get_world_pose() for robot in scene["robot_info"]["robot_list"]
    ]
    scene["meta_infos"]["joint_positions"] = [
        robot.robot.get_joint_positions() for robot in scene["robot_info"]["robot_list"]
    ]
    scene["meta_infos"]["joint_velocities"] = [
        robot.robot.get_joint_velocities()
        for robot in scene["robot_info"]["robot_list"]
    ]


def update_meta_infos(scene):
    articulation_part_list = scene["articulation_part_list"]
    for part_id, part_xform in articulation_part_list.items():
        scene["meta_infos"]["world_pose_list"][part_id] = part_xform.get_world_pose()


def recovery_scene(scene, evaluator, task_data, eval_config, default_config):
    layout = copy.deepcopy(task_data["initial_layout"])
    if scene["robot_info"]["robot_list"][0].robot.name in layout:
        layout.pop(scene["robot_info"]["robot_list"][0].robot.name)
    object_list_keys = list(scene["object_list"].keys())
    for key in object_list_keys:
        if key not in layout:
            if scene["object_list"][key].prim.IsActive():
                scene["object_list"][key].prim.SetActive(False)
            scene["object_list"].pop(key)
        else:
            if not scene["object_list"][key].prim.IsActive():
                scene["object_list"][key].prim.SetActive(True)
    for key in layout:
        if key not in scene["object_list"]:
            scene["object_list"][key] = preload_object(
                os.path.join(default_config["ASSETS_DIR"], layout[key]["path"]),
                scene["uuid"],
                layout[key]["prim_path"].split("/")[-1][4:],
                scene["world"],
            )
            if not scene["object_list"][key].prim.IsActive():
                scene["object_list"][key].prim.SetActive(True)
    for key, object_info in layout.items():
        if is_prim_path_valid(object_info["prim_path"]):
            scene["object_list"][key].set_world_pose(
                object_info["position"], object_info["orientation"]
            )
            scene["object_list"][key].set_local_scale(object_info["scale"])
    scene["cacheDict"]["meshDict"] = objectList2meshList(
        scene["object_list"],
        os.path.join(
            default_config["ASSETS_DIR"],
            "mesh_data",
            eval_config["task_name"],
        ),
    )
    for goal in task_data["goal"]:
        for subgoal in goal:
            if "another_obj2_uid" in subgoal:
                set_mass(
                    scene["object_list"][subgoal["another_obj2_uid"]].prim_path, 10.0
                )
            if "status" in subgoal:
                continue
            if "obj1_uid" in subgoal:
                set_mass(scene["object_list"][subgoal["obj1_uid"]].prim_path, 0.1)
            if "obj2_uid" in subgoal:
                set_mass(scene["object_list"][subgoal["obj2_uid"]].prim_path, 10.0)
    if scene["robot_info"]["robot_list"][0].robot.name in task_data["initial_layout"]:
        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
            task_data["initial_layout"][
                scene["robot_info"]["robot_list"][0].robot.name
            ]["joint_positions"]
        )
    else:
        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
            np.array([0.012, -0.57, 0.0, -2.81, 0.0, 3.37, 0.741, 0.04, 0.04])
        )
    evaluator.instruction = task_data["instruction"]


def recovery_scene_render(scene, task_data, eval_config, default_config):
    layout = copy.deepcopy(task_data["initial_layout"])
    scene["robot_info"]["robot_list"][0].robot.set_world_pose(
        layout[scene["robot_info"]["robot_list"][0].robot.name]["position"],
        layout[scene["robot_info"]["robot_list"][0].robot.name]["orientation"],
    )
    if scene["robot_info"]["robot_list"][0].robot.name in layout:
        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
            layout[scene["robot_info"]["robot_list"][0].robot.name]["joint_positions"]
        )
    else:
        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
            np.array([0.012, -0.57, 0.0, -2.81, 0.0, 3.37, 0.741, 0.04, 0.04])
        )
    if scene["robot_info"]["robot_list"][0].robot.name in layout:
        layout.pop(scene["robot_info"]["robot_list"][0].robot.name)
    object_list_keys = list(scene["object_list"].keys())
    for key in object_list_keys:
        if key not in layout:
            if scene["object_list"][key].prim.IsActive():
                scene["object_list"][key].prim.SetActive(False)
            scene["object_list"].pop(key)
        else:
            if not scene["object_list"][key].prim.IsActive():
                scene["object_list"][key].prim.SetActive(True)
    for key in layout:
        if key not in scene["object_list"]:
            scene["object_list"][key] = preload_object(
                os.path.join(default_config["ASSETS_DIR"], layout[key]["path"]),
                scene["uuid"],
                layout[key]["prim_path"].split("/")[-1][4:],
                scene["world"],
                add_colliders=False,
                add_rigid_body=False,
            )
            if not scene["object_list"][key].prim.IsActive():
                scene["object_list"][key].prim.SetActive(True)
    for key, object_info in layout.items():
        if is_prim_path_valid(object_info["prim_path"]):
            scene["object_list"][key].set_world_pose(
                object_info["position"], object_info["orientation"]
            )
            scene["object_list"][key].set_local_scale(object_info["scale"])
    # scene["cacheDict"]["meshDict"] = objectList2meshList(
    #     scene["object_list"],
    #     os.path.join(
    #         default_config["ASSETS_DIR"],
    #         "mesh_data",
    #         eval_config["task_name"],
    #     ),
    # )
    for goal in task_data["goal"]:
        for subgoal in goal:
            if "is_articulated" in subgoal:
                continue
            if "another_obj2_uid" in subgoal:
                set_mass(
                    scene["object_list"][subgoal["another_obj2_uid"]].prim_path, 10.0
                )
            if "obj1_uid" in subgoal:
                set_mass(scene["object_list"][subgoal["obj1_uid"]].prim_path, 0.1)
            if "obj2_uid" in subgoal:
                set_mass(scene["object_list"][subgoal["obj2_uid"]].prim_path, 10.0)
