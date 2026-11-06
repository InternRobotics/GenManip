"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import copy
import os
import random
import re
from typing import TYPE_CHECKING
from mplib.planner import Planner as MplibPlanner
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import omni.replicator.core as rep  # type: ignore
import omni.usd  # type: ignore
from omni.isaac.core import World  # type: ignore
from omni.isaac.core.articulations import Articulation  # type: ignore
from omni.isaac.core.materials.omni_pbr import OmniPBR  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid  # type: ignore
from omni.isaac.sensor import Camera  # type: ignore

from pxr import Gf, UsdGeom, Usd, UsdPhysics, Vt  # type: ignore

from genmanip.core.scene.scene_config import SceneConfig, RobotConfig
from genmanip.utils.loader.asset_search import (
    extract_asset_search_pattern,
    get_asset_search_root,
)
from genmanip.utils.loader.preload_rules import (
    apply_rule,
    collect_all_colors,
    collect_all_materials,
    collect_all_shapes,
    generate_long_horizon_by_category,
    generate_long_horizon_by_color,
    generate_long_horizon_by_materials,
    generate_long_horizon_by_shape,
)
from genmanip.utils.usd_utils import set_mdl, create_dome_light, set_texture
from genmanip.utils.pointcloud.pointcloud import (
    objectList2meshList,
    get_mesh_info_by_load,
)
from genmanip.core.robot.base import BaseEmbodiment  # type: ignore
from genmanip.utils.usd_utils.camera_utils import setup_camera, get_src
from genmanip.utils.usd_utils import (
    add_physics_material,
    add_usd_to_world,
    create_omni_pbr,
    clean_prim_velocity,
    has_collision_api,
    remove_contact_offset,
    remove_colliders,
    set_colliders,
    set_contact_offset,
    set_drive_max_force,
    set_gravity,
    set_mass,
    set_rigid_body,
    set_rigid_body_CCD,
    set_robot_physics_material,
    set_robot_contact_offset,
    set_robot_rest_offset,
    set_semantic_label,
    get_world_pose_by_prim_path,
    get_local_scale_by_prim_path,
)
from genmanip.utils.standalone.utils import generate_hash
from genmanip.utils.standalone.file_utils import load_yaml, load_json
from genmanip.utils.annotation.object_pool import ObjectPool
import genmanip.extensions.robots
import genmanip.extensions.skills

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


def _get_uid_from_usd_path(usd_path: str) -> str:
    return os.path.splitext(os.path.basename(usd_path))[0]


def _list_usd_candidates(assets_dir: str, path_expr: str) -> list[str]:
    asset_path = os.path.join(assets_dir, path_expr)
    if os.path.isdir(asset_path):
        usd_list = [
            os.path.join(path_expr, usd)
            for usd in os.listdir(asset_path)
            if usd.endswith(".usd") and not os.path.isdir(os.path.join(asset_path, usd))
        ]
        usd_list.sort()
        return usd_list
    if os.path.isfile(asset_path):
        if not asset_path.endswith(".usd"):
            raise ValueError(f"Object path {path_expr} is not a usd file")
        return [path_expr]
    pattern = extract_asset_search_pattern(path_expr)
    if pattern is None:
        raise ValueError(
            f"Object path {path_expr} does not exist under assets dir {assets_dir}"
        )
    try:
        compiled_pattern = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid grep pattern {path_expr}: {exc}") from exc
    usd_list = []
    _, search_root = get_asset_search_root(assets_dir, path_expr)
    for root, _, files in os.walk(search_root):
        relative_root = os.path.relpath(root, assets_dir)
        for usd in files:
            if not usd.endswith(".usd"):
                continue
            relative_path = (
                usd if relative_root == "." else os.path.join(relative_root, usd)
            )
            normalized_relative_path = relative_path.replace(os.sep, "/")
            if compiled_pattern.search(normalized_relative_path):
                usd_list.append(relative_path)
    usd_list.sort()
    return usd_list


def setup_walls_and_materials(
    uuid: str, world: World, object_list: dict
) -> tuple[list[XFormPrim], list[OmniPBR]]:
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


def create_camera_list(
    camera_data: dict[str, dict],
    uuid: str,
    rendering_dt: float = 1 / 60.0,
    only_depth_rep_for_camera: bool = False,
    only_color_rep_for_camera: bool = False,
) -> dict[str, Camera]:
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
            camera_cfg=camera_data[key],
            only_depth_rep_for_camera=only_depth_rep_for_camera,
            only_color_rep_for_camera=only_color_rep_for_camera,
        )
    return camera_list


def add_background_scene(
    scene: "Scene",
    usd_path: str,
    position: np.ndarray,
    scale: list[float] = [0.01, 0.01, 0.01],
) -> None:
    background_xform, background_uuid = load_world_xform_prim(
        os.path.join(usd_path),
        scene_prim_path=f"/World/background_{usd_path.split('/')[-1].split('.')[0]}",
    )
    background_xform.set_world_pose(position=position)
    background_xform.set_local_scale(scale)
    if (
        "defaultGroundPlane" in scene.object_list
        and scene.object_list["defaultGroundPlane"].prim.IsActive()
    ):
        scene.object_list["defaultGroundPlane"].prim.SetActive(False)


def get_object_list(
    uuid: str, scene_xform: XFormPrim, table_uid: str
) -> dict[str, XFormPrim]:
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
                continue
            elif str(object.GetPath()).split("/")[-1][4:] != table_uid:
                object_uid = str(object.GetPath()).split("/")[-1][4:]
                object_list[object_uid] = relate_object_from_data(
                    f"/World/{uuid}/obj_{object_uid}"
                )
            else:
                object_list["00000000000000000000000000000000"] = (
                    relate_object_from_data(f"/World/{uuid}/obj_{table_uid}")
                )
            set_semantic_label(
                str(object.GetPath()), str(object.GetPath()).split("/")[-1][4:]
            )
    return object_list


def load_camera_from_data(camera_data: dict[str, dict], uuid: str) -> Camera:
    if not isinstance(camera_data["name"], str):
        raise ValueError(f"Camera name {camera_data['name']} is not a string")
    camera = Camera(
        prim_path=f"/World/{uuid}/" + camera_data["name"],
        name=camera_data["name"],
        position=camera_data["position"],
        frequency=camera_data["frequency"],
        resolution=(camera_data["resolution_width"], camera_data["resolution_height"]),
        orientation=camera_data["orientation"],
    )
    setup_camera(camera, camera_cfg=camera_data)
    return camera


def load_world_xform_prim(
    scene_path: str, scene_prim_path: str = "/World"
) -> tuple[XFormPrim, str]:
    scene_path = os.path.abspath(scene_path)
    omni.usd.get_context().new_stage()
    omni.usd.get_context().open_stage(scene_path)
    scene_xform = XFormPrim(
        "/World",
        name="World",
    )
    uuid = str(scene_xform.prim.GetAllChildren()[0].GetPath()).split("/")[-1]
    return scene_xform, uuid


def relate_object_from_data(prim_path) -> XFormPrim:
    position, orientation = get_world_pose_by_prim_path(prim_path)
    scale = get_local_scale_by_prim_path(prim_path)

    return XFormPrim(
        prim_path=prim_path, position=position, orientation=orientation, scale=scale
    )


def preload_object(
    object_path: str,
    uuid: str,
    uid: str,
    world: World,
    add_rigid_body: bool = True,
    add_colliders: bool = True,
    remove_collider: bool = False,
    mass: float | None = None,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
) -> XFormPrim:
    if not os.path.exists(object_path):
        raise ValueError(f"Object path {object_path} does not exist")
    print(f"Object {uid} loading")
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
        mass=mass,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
    )
    if remove_collider:
        remove_colliders(obj_xform.prim_path)
    if obj_xform is not None:
        world.step()
        if obj_xform.prim.IsActive():
            obj_xform.prim.SetActive(False)
        print(f"Object {uid} loaded")
    else:
        delete_prim(f"/World/{uuid}/obj_{uid}")
    return obj_xform


def process_long_horizon_replacement(
    scene: dict, default_config: dict, demogen_config: dict
) -> tuple[dict, dict]:
    replacement_config = demogen_config["domain_randomization"]["replace_object"][
        "replacement"
    ]
    folder_path = os.path.join(
        default_config["ASSETS_DIR"],
        replacement_config["random_long_horizon"]["folder_path"],
    )
    usd_list = os.listdir(folder_path)
    usd_list = [
        str(usd)
        for usd in usd_list
        if not os.path.isdir(os.path.join(folder_path, usd))
    ]
    types = ["category", "materials", "color", "shape"]
    meta_info = None
    while True:
        type = random.choice(types)
        random_long_horizon_folder_path = demogen_config["domain_randomization"][
            "replace_object"
        ]["replacement"]["random_long_horizon"]["folder_path"]
        if not isinstance(random_long_horizon_folder_path, str):
            raise ValueError(
                f"Random long horizon folder path {random_long_horizon_folder_path} is not a string"
            )
        if type == "category":
            replacement_config, meta_info = generate_long_horizon_by_category(
                scene,
                usd_list,
                random_long_horizon_folder_path,
            )
        elif type == "materials":
            replacement_config, meta_info = generate_long_horizon_by_materials(
                scene,
                usd_list,
                random_long_horizon_folder_path,
            )
        elif type == "color":
            replacement_config, meta_info = generate_long_horizon_by_color(
                scene,
                usd_list,
                random_long_horizon_folder_path,
            )
        elif type == "shape":
            replacement_config, meta_info = generate_long_horizon_by_shape(
                scene,
                usd_list,
                random_long_horizon_folder_path,
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
    if meta_info is None:
        raise ValueError("Meta info is None")
    return replacement_config, meta_info


def preprocess_object_config(
    scene: "Scene", default_config: dict, scene_config: SceneConfig
) -> dict:
    object_config_backup = copy.deepcopy(scene_config.object_config)
    while True:
        object_config = copy.deepcopy(object_config_backup)
        color_list = collect_all_colors(scene.object_pool)
        shape_list = collect_all_shapes(scene.object_pool)
        material_list = collect_all_materials(scene.object_pool)
        object_config_keys = list(object_config.keys())
        object_config_keys.sort()
        color_project_dict = {}
        shape_project_dict = {}
        material_project_dict = {}
        for key in object_config_keys:
            if object_config[key].type == "rule":
                for rule in object_config[key].filter_rule:
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
            if object_config[key].type == "rule":
                for rule_idx in range(len(object_config[key].filter_rule)):
                    rule = object_config[key].filter_rule[rule_idx]
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
                    object_config[key].filter_rule[rule_idx] = rule
        is_vaild = True
        for key in object_config_keys:
            if object_config[key].type == "load_object_from_path":
                usd_list = _list_usd_candidates(
                    default_config["ASSETS_DIR"], object_config[key].path
                )
                usd_list_len = len(usd_list)
                for rule in object_config[key].filter_rule:
                    usd_list = apply_rule(rule, usd_list, scene.object_pool)
                if len(usd_list) < 5 and len(usd_list) < usd_list_len:
                    is_vaild = False
                    break
        if is_vaild:
            break
    return object_config


def preload_objects(
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
    without_planning: bool = False,
) -> None:
    scene_config.object_config = preprocess_object_config(
        scene, default_config, scene_config
    )
    object_config = scene_config.object_config
    object_config_keys = list(object_config.keys())
    object_config_keys.sort()
    for key in object_config_keys:
        if object_config[key].type == "load_object_from_path":
            origin_text = object_config[key].path
            rules = object_config[key].filter_rule
            rules.sort()
            for rule in rules:
                origin_text += rule
            scene.cache_library.preload_hash_feature[key] = generate_hash(origin_text)
        elif object_config[key].type == "existed_object":
            origin_text = object_config[key].uid_list
            if not isinstance(origin_text, list):
                origin_text = [origin_text]
            origin_text.sort()
            concat_text = ""
            for uid in origin_text:
                concat_text += uid
            scene.cache_library.preload_hash_feature[key] = generate_hash(concat_text)
        elif object_config[key].type == "add_additional_object_from_path":
            scene.cache_library.preload_hash_feature[key] = generate_hash(
                object_config[key].path
            )
            if not object_config[key].path.endswith(".usd"):
                object_config[key].max_cached_num = len(
                    os.listdir(
                        os.path.join(
                            default_config["ASSETS_DIR"],
                            object_config[key].path,
                        )
                    )
                )
            else:
                object_config[key].max_cached_num = 1
    max_cached_num_dict = {}
    for key in object_config_keys:
        if object_config[key].type == "existed_object":
            continue
        if scene.cache_library.preload_hash_feature[key] not in max_cached_num_dict:
            max_cached_num_dict[scene.cache_library.preload_hash_feature[key]] = (
                object_config[key].max_cached_num
            )
        else:
            max_cached_num_dict[scene.cache_library.preload_hash_feature[key]] = max(
                max_cached_num_dict[scene.cache_library.preload_hash_feature[key]],
                object_config[key].max_cached_num,
            )
    sorted_object_config_keys = sorted(
        object_config_keys,
        key=lambda x: object_config[x].type == "add_additional_object_from_path",
        reverse=True,
    )
    for key in sorted_object_config_keys:
        if (
            scene.cache_library.preload_hash_feature[key]
            in scene.cache_library.preloaded_object_uid_list
        ):
            continue
        else:
            if object_config[key].type == "add_additional_object_from_path":
                folder_path = os.path.join(
                    default_config["ASSETS_DIR"],
                    object_config[key].path,
                )
                if folder_path.endswith(".usd"):
                    usd_list = [folder_path.split("/")[-1]]
                    folder_path = os.path.dirname(folder_path)
                    object_config[key].uid = (
                        object_config[key].path.split("/")[-1].split(".")[0]
                    )
                    object_config[key].path = os.path.dirname(object_config[key].path)
                else:
                    usd_list = os.listdir(folder_path)
                    usd_list = [
                        usd
                        for usd in usd_list
                        if not os.path.isdir(os.path.join(folder_path, usd))
                    ]
                max_cached_num = np.clip(
                    max_cached_num_dict[scene.cache_library.preload_hash_feature[key]],
                    0,
                    len(usd_list),
                )
                usd_list = random.sample(usd_list, max_cached_num)
                scene.cache_library.preloaded_object_uid_list[
                    scene.cache_library.preload_hash_feature[key]
                ] = []
                for obj in tqdm(usd_list):
                    uid = obj.split(".")[0]
                    if (
                        os.path.isdir(os.path.join(folder_path, obj))
                        or uid not in scene.object_pool.uids
                    ):
                        continue
                    scene.cache_library.preloaded_object_uid_list[
                        scene.cache_library.preload_hash_feature[key]
                    ].append(uid)
                    if (
                        uid in scene.cache_library.preloaded_object_list
                        or uid in scene.object_list
                    ):
                        continue
                    obj_xform = preload_object(
                        os.path.join(folder_path, obj),
                        scene.uuid,
                        uid,
                        scene.world,
                        add_colliders=not without_planning
                        and not object_config[key].without_colliders,
                        add_rigid_body=not without_planning
                        and not object_config[key].is_not_rigid,
                    )
                    scene.cache_library.preload_object_meta_info[uid] = {
                        "add_colliders": not without_planning
                        and not object_config[key].without_colliders,
                        "add_rigid_body": not without_planning
                        and not object_config[key].is_not_rigid,
                        "mass": object_config[key].mass,
                    }
                    joints_default_state_list = [
                        robot.robot.get_joints_default_state()
                        for robot in scene.robot_list
                    ]
                    for robot, joints_default_state in zip(
                        scene.robot_list, joints_default_state_list
                    ):
                        robot.robot.set_joint_positions(joints_default_state.positions)
                        robot.robot.set_joint_velocities(
                            joints_default_state.velocities
                        )
                    scene.cache_library.preloaded_object_list[uid] = obj_xform
                    scene.cache_library.preloaded_object_path_list[uid] = os.path.join(
                        object_config[key].path,
                        obj,
                    )
                    scene.object_list[uid] = obj_xform
                    if not scene.object_list[uid].prim.IsActive():
                        scene.object_list[uid].prim.SetActive(True)
                    mesh_info = get_mesh_info_by_load(
                        scene.object_list[uid],
                        os.path.join(
                            default_config["ASSETS_DIR"],
                            "mesh_data",
                            scene_config.task_name,
                            os.path.dirname(
                                scene.cache_library.preloaded_object_path_list[uid]
                            ),
                            f"{uid}.obj",
                        ),
                    )
                    if mesh_info is not None:
                        scene.cache_library.mesh_dict[uid] = mesh_info
            elif object_config[key].type == "existed_object":
                scene.cache_library.preloaded_object_uid_list[
                    scene.cache_library.preload_hash_feature[key]
                ] = object_config[key].uid_list
                for uid in object_config[key].uid_list:
                    if uid in scene.object_list:
                        scene.cache_library.preloaded_object_list[uid] = (
                            scene.object_list[uid]
                        )
                    elif uid in scene.articulation_list:
                        scene.cache_library.preloaded_object_list[uid] = (
                            scene.articulation_list[uid]
                        )
                    else:
                        raise ValueError(f"Object {uid} not found in scene")
            elif object_config[key].type == "load_object_from_path":
                usd_list = _list_usd_candidates(
                    default_config["ASSETS_DIR"], object_config[key].path
                )
                for rule in object_config[key].filter_rule:
                    usd_list = apply_rule(rule, usd_list, scene.object_pool)
                max_cached_num = np.clip(
                    max_cached_num_dict[scene.cache_library.preload_hash_feature[key]],
                    0,
                    len(usd_list),
                )
                usd_list = random.sample(usd_list, max_cached_num)
                scene.cache_library.preloaded_object_uid_list[
                    scene.cache_library.preload_hash_feature[key]
                ] = []
                for uid in object_config[key].replace_existed_object:
                    # if not uid in scene["cacheDict"]["preloaded_object_list"]:
                    scene.cache_library.preloaded_object_list[uid] = scene.object_list[
                        uid
                    ]
                    if scene.object_list[uid].prim.IsActive():
                        scene.object_list[uid].prim.SetActive(False)
                    scene.cache_library.mesh_dict.pop(uid)
                    scene.object_list.pop(uid)
                if "remove_existed_object" not in object_config[key].option:
                    scene.cache_library.preloaded_object_uid_list[
                        scene.cache_library.preload_hash_feature[key]
                    ].extend(object_config[key].replace_existed_object)
                for obj in tqdm(usd_list):
                    uid = _get_uid_from_usd_path(obj)
                    obj_path = os.path.join(default_config["ASSETS_DIR"], obj)
                    if (
                        not os.path.isfile(obj_path)
                        or uid not in scene.object_pool.uids
                    ) and not ("plain_replace" in object_config[key].option):
                        continue
                    scene.cache_library.preloaded_object_uid_list[
                        scene.cache_library.preload_hash_feature[key]
                    ].append(uid)
                    if (
                        uid in scene.cache_library.preloaded_object_list
                        or uid in scene.object_list
                    ):
                        continue
                    obj_xform = preload_object(
                        obj_path,
                        scene.uuid,
                        uid,
                        scene.world,
                        add_colliders=not without_planning
                        and not object_config[key].without_colliders,
                        add_rigid_body=not without_planning
                        and not object_config[key].is_not_rigid,
                    )
                    scene.cache_library.preload_object_meta_info[uid] = {
                        "add_colliders": not without_planning
                        and not object_config[key].without_colliders,
                        "add_rigid_body": not without_planning
                        and not object_config[key].is_not_rigid,
                    }
                    joints_default_state_list = [
                        robot.robot.get_joints_default_state()
                        for robot in scene.robot_list
                    ]
                    for robot, joints_default_state in zip(
                        scene.robot_list, joints_default_state_list
                    ):
                        robot.robot.set_joint_positions(joints_default_state.positions)
                        robot.robot.set_joint_velocities(
                            joints_default_state.velocities
                        )
                    scene.cache_library.preloaded_object_list[uid] = obj_xform
                    scene.cache_library.preloaded_object_path_list[uid] = obj


def add_articulation_to_scene(uid: str, uuid: str, world: World) -> Articulation:
    articulation = Articulation(
        prim_path=f"/World/{uuid}/obj_{uid}",
        name=f"obj_{uid}",
    )
    world.scene.add(articulation)
    return articulation


def load_articulation_data(demogen_config: SceneConfig, current_dir: str) -> dict:
    if demogen_config.domain_randomization.articulation_data_path is not None:
        return load_json(
            os.path.join(
                current_dir,
                "assets/objects",
                f"{demogen_config.domain_randomization.articulation_data_path}.json",
            )
        )
    else:
        return load_json(
            os.path.join(current_dir, "assets/objects/articulation_data.json")
        )


def load_object_pool(demogen_config: SceneConfig, current_dir: str) -> ObjectPool:
    if demogen_config.domain_randomization.object_data_path is not None:
        return ObjectPool(
            os.path.join(
                current_dir,
                "assets/objects",
                f"{demogen_config.domain_randomization.object_data_path}.pickle",
            )
        )
    else:
        return ObjectPool(
            os.path.join(
                current_dir, "assets/objects/objaverse_annotation_refined_pnp.pickle"
            )
        )


def set_articulation(scene: "Scene", demogen_config: dict, world: World) -> None:
    for key in demogen_config["generation_config"]["articulation"]:
        # todo: remove "pan" from "is_articulated", or change key to "have_joint"
        if scene.articulation_data[key]["is_articulated"]:
            scene.articulation_list[key] = add_articulation_to_scene(
                key, scene.uuid, world
            )
        else:
            scene.articulation_list[key] = scene.object_list[key]
    world.reset()
    for key, articulation in scene.articulation_list.items():
        if scene.articulation_data[key]["is_articulated"]:
            articulation._articulation_view.initialize()
    world.initialize_physics()
    for _ in range(10):
        world.step()
    for arti_id, articulation in scene.articulation_list.items():
        if scene.articulation_data[arti_id]["is_articulated"]:
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


def parse_articulation_from_object_config(demogen_config: dict) -> None:
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


def setup_robot_joint_positions(
    embodiment: BaseEmbodiment, robot_config: RobotConfig
) -> BaseEmbodiment:
    if robot_config.default_joint_positions is not None:
        embodiment.robot.set_joint_positions(
            np.array(robot_config.default_joint_positions)
        )
    return embodiment


def collect_assets(assets_dir: str) -> dict:
    assets_list = {}
    if os.path.exists(os.path.join(assets_dir, "miscs/hdrs")):
        assets_list["domelight"] = os.listdir(f"{assets_dir}/miscs/hdrs")
    else:
        assets_list["domelight"] = []
    if os.path.exists(os.path.join(assets_dir, "miscs/textures")):
        assets_list["wall_texture"] = os.listdir(f"{assets_dir}/miscs/textures")
    else:
        assets_list["wall_texture"] = []
    if os.path.exists(
        os.path.join(assets_dir, "object_usds/grutopia_usd/Table/Materials")
    ):
        assets_list["table_mdl"] = os.listdir(
            f"{assets_dir}/object_usds/grutopia_usd/Table/Materials"
        )
    else:
        assets_list["table_mdl"] = []
    if os.path.exists(os.path.join(assets_dir, "object_usds/grutopia_usd/Table/table")):
        assets_list["table"] = os.listdir(
            f"{assets_dir}/object_usds/grutopia_usd/Table/table"
        )
        assets_list["table"] = [
            os.path.join(
                assets_dir,
                "object_usds/grutopia_usd/Table/table",
                table_path,
                "instance.usd",
            )
            for table_path in assets_list["table"]
        ]
    else:
        assets_list["table"] = []
    return assets_list


def preprocess_scene(scene: "Scene", demogen_config: SceneConfig) -> None:
    preprocess_config = demogen_config.preprocess_config
    for preprocess_info in preprocess_config:
        if preprocess_info["type"] == "recompute_inertia":
            for uid in preprocess_info["config"]["uids"]:
                prim = scene.object_list[uid].prim
                # Zero selects PhysX's automatic inertia calculation from the
                # current mass and collision geometry, instead of stale values
                # authored for a different mass in the asset.
                UsdPhysics.MassAPI.Apply(prim).CreateDiagonalInertiaAttr().Set(
                    Gf.Vec3f(0.0)
                )
        if preprocess_info["type"] == "disable_contact_offset":
            for object in scene.object_list.values():
                remove_contact_offset(object.prim_path)
        if preprocess_info["type"] == "enable_contact_offset":
            for object in scene.object_list.values():
                set_contact_offset(object.prim_path, 0.1)
        if preprocess_info["type"] == "disable_gravity":
            for name, object in scene.object_list.items():
                if (
                    name != "defaultGroundPlane"
                    and name not in scene.articulation_part_list.keys()
                    and name not in scene.articulation_list
                ):
                    set_gravity(object.prim_path, preprocess_info["config"]["value"])
        if preprocess_info["type"] == "ccd":
            for name, object in scene.object_list.items():
                if (
                    name != "defaultGroundPlane"
                    and name != "00000000000000000000000000000000"
                    and name not in demogen_config.layout_config.ignored_objects
                    and name not in scene.articulation_part_list.keys()
                    and name not in scene.articulation_list
                ):
                    set_rigid_body_CCD(object.prim_path, True)
        if preprocess_info["type"] == "collider":
            target_uids = preprocess_info.get("config", {}).get("uids")
            if target_uids is not None:
                if not isinstance(target_uids, list) or any(
                    not isinstance(uid, str) for uid in target_uids
                ):
                    raise ValueError("collider.config.uids must be a list of strings")
                missing = set(target_uids) - scene.object_list.keys()
                if missing:
                    raise ValueError(f"Collider objects not found: {sorted(missing)}")
            for name, object in scene.object_list.items():
                if (
                    name != "defaultGroundPlane"
                    and (target_uids is None or name in target_uids)
                    and name not in scene.articulation_part_list.keys()
                    and name not in scene.articulation_list
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
            object_list_key = list(scene.object_list.keys())
            for key in object_list_key:
                if (
                    key != "defaultGroundPlane"
                    and key != "00000000000000000000000000000000"
                ):
                    if scene.object_list[key].prim.IsActive():
                        scene.object_list[key].prim.SetActive(False)
                    scene.object_list.pop(key)
                    if key in scene.cache_library.mesh_dict:
                        scene.cache_library.mesh_dict.pop(key)
        if preprocess_info["type"] == "remove_objects":
            object_uids = preprocess_info.get("config", {}).get("uids", [])
            if not isinstance(object_uids, list):
                raise ValueError("remove_objects.config.uids must be a list")
            for uid in object_uids:
                if uid not in scene.object_list:
                    print(f"Object {uid} not found in scene, skipping removal")
                    continue
                if scene.object_list[uid].prim.IsActive():
                    scene.object_list[uid].prim.SetActive(False)
                scene.object_list.pop(uid)
                scene.cache_library.mesh_dict.pop(uid, None)
        if preprocess_info["type"] == "apply_rigid_body_solution":
            for name, object in scene.object_list.items():
                schema_list = object.prim.GetAppliedSchemas()
                if (
                    "PhysicsRigidBodyAPI" in schema_list
                    and "PhysxRigidBodyAPI" in schema_list
                ):
                    set_rigid_body(object.prim_path)
        if preprocess_info["type"] == "rigid_body":
            for name, object in scene.object_list.items():
                if (
                    name != "defaultGroundPlane"
                    and name not in scene.articulation_part_list.keys()
                    and name not in scene.articulation_list
                    and name != "00000000000000000000000000000000"
                ):
                    set_rigid_body(object.prim_path)
        if preprocess_info["type"] == "apply_default_physics_material":
            for name, object in scene.object_list.items():
                if has_collision_api(object.prim):
                    add_physics_material(object.prim_path)
        if preprocess_info["type"] == "set_robot_joint_max_force":
            for robot in scene.robot_list:
                if robot.robot.name != preprocess_info["robot_type"]:
                    continue
                for joint_name, max_force in preprocess_info["config"].items():
                    max_force = float(max_force)
                    if not np.isfinite(max_force) or max_force <= 0:
                        raise ValueError("Joint max force must be finite and positive")
                    joint = robot.joint_dict[joint_name]
                    set_drive_max_force(str(joint.GetPath()), max_force)
        if preprocess_info["type"] == "set_robot_physics_material":
            prim_path = None
            if preprocess_info["robot_type"] == "lift2":
                prim_path = "/World/_scene/lift2/lift2/lift2/PhysicsMaterial"

            if prim_path is not None:
                set_robot_physics_material(prim_path, preprocess_info["config"])
        if preprocess_info["type"] == "set_robot_contact_offset":
            prim_paths = None
            if preprocess_info["robot_type"] == "lift2":
                prim_paths = [
                    "/World/_scene/lift2/lift2/lift2/fl/link7/mesh",
                    "/World/_scene/lift2/lift2/lift2/fl/link8/mesh",
                    "/World/_scene/lift2/lift2/lift2/fr/link7/mesh",
                    "/World/_scene/lift2/lift2/lift2/fr/link8/mesh",
                ]
            if prim_paths is not None:
                set_robot_contact_offset(prim_paths, preprocess_info["config"])
        if preprocess_info["type"] == "set_robot_rest_offset":
            prim_paths = None
            if preprocess_info["robot_type"] == "lift2":
                prim_paths = [
                    "/World/_scene/lift2/lift2/lift2/fl/link7/mesh",
                    "/World/_scene/lift2/lift2/lift2/fl/link8/mesh",
                    "/World/_scene/lift2/lift2/lift2/fr/link7/mesh",
                    "/World/_scene/lift2/lift2/lift2/fr/link8/mesh",
                ]
            if prim_paths is not None:
                set_robot_rest_offset(prim_paths, preprocess_info["config"])


def cleanup_camera(camera_data: dict, camera: Camera) -> None:
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


def clear_scene(scene: "Scene", scene_config: SceneConfig, current_dir: str) -> None:
    camera_info = scene_config.domain_randomization.cameras
    if camera_info.type == "fixed":
        camera_data = load_yaml(os.path.join(current_dir, camera_info.config_path))
    else:
        raise ValueError(f"Unsupported camera type: {camera_info.type}")

    world = getattr(scene, "world", None)
    if world is not None:
        stop_fn = getattr(world, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass

    for camera_name, camera_info in camera_data.items():
        if camera_name in scene.camera_list:
            cleanup_camera(camera_info, scene.camera_list[camera_name])
    scene.camera_list = {}

    if world is not None:
        try:
            world.scene.clear()
        except Exception:
            pass

    scene.object_list = {}
    scene.articulation_list = {}
    scene.articulation_part_list = {}
    scene.robot_list = []

    for prim_path in ("/Replicator", "/Camera", "/World"):
        try:
            delete_prim(prim_path)
        except Exception:
            pass

    usd_context = omni.usd.get_context()
    usd_context.close_stage()
    usd_context.new_stage()
    # action_registry = omni.kit.actions.core.get_action_registry()
    # action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    # action.execute()


def warmup_world(
    scene: "Scene",
    physics_steps: int = 100,
    without_depth: bool = False,
) -> None:
    if not without_depth:
        while any(
            camera._custom_annotators["distance_to_image_plane"] is not None
            and get_src(camera, "depth") is None
            for camera in scene.camera_list.values()
        ):
            scene.world.step()
    for _ in range(physics_steps):
        scene.world.step(render=False)


def update_meta_infos(scene: "Scene") -> None:
    articulation_part_list = scene.articulation_part_list
    for part_id, part_xform in articulation_part_list.items():
        scene.meta_infos["world_pose_list"][part_id] = part_xform.get_world_pose()


def capture_particle_layout_references(scene: "Scene") -> None:
    """Save the asset's particle frames before physics or episode recovery."""
    references = {}
    layout_references = scene.scene_config.particle_layout_references
    for parent_path, object_id in layout_references.items():
        parent_path = parent_path.replace("/World/_scene", f"/World/{scene.uuid}", 1)
        if not scene.world.stage.GetPrimAtPath(parent_path).IsValid():
            raise ValueError(f"Particle layout parent not found: {parent_path}")
        parent_position, _ = XFormPrim(parent_path).get_world_pose()
        reference_position, _ = scene.object_list[object_id].get_world_pose()
        references[parent_path] = {
            "position": np.array(parent_position, dtype=np.float64, copy=True),
            "reference_position": np.array(
                reference_position, dtype=np.float64, copy=True
            ),
            "object_id": object_id,
        }
    scene.meta_infos["particle_layout_references"] = references


def restore_particle_layout(scene: "Scene", task_data: dict) -> None:
    """Restore the fixed particle frame's initial XY layout randomization.

    Particle arrays contain local coordinates. Their parent shares the reference
    object's initial XY layout offset, but must not follow its subsequent motion.
    Z settling is excluded because it is not part of XY layout randomization.
    Always use the saved asset layout so repeated resets cannot accumulate offsets.
    """
    if not scene.scene_config.particle_layout_references:
        return
    references = scene.meta_infos.get("particle_layout_references")
    if references is None:
        raise RuntimeError(
            "Particle layout references were not captured at scene initialization"
        )
    for parent_path, info in references.items():
        reference = task_data["initial_layout"][info["object_id"]]
        offset = np.zeros(3, dtype=np.float64)
        offset[:2] = (
            np.asarray(reference["position"])[:2] - info["reference_position"][:2]
        )
        XFormPrim(parent_path).set_world_pose(position=info["position"] + offset)
        print(f"[OK] Restored particle layout {parent_path}: XY offset {offset[:2]}")


def restore_initial_particle_states(scene: "Scene", task_data: dict) -> None:
    restore_particle_layout(scene, task_data)
    particle_states = task_data.get("initial_particle_states")
    if not particle_states:
        return

    stage = scene.world.stage
    for particle_id, particle_info in particle_states.items():
        original_prim_path = particle_info.get("prim_path")
        points = particle_info.get("points")
        if not original_prim_path or points is None:
            print(
                f"[WARN] Skip initial particle state {particle_id}: "
                "missing prim_path or points"
            )
            continue

        current_prim_path = original_prim_path.replace(
            "/World/_scene", f"/World/{scene.uuid}", 1
        )
        particle_prim = stage.GetPrimAtPath(current_prim_path)
        if particle_prim is None or not particle_prim.IsValid():
            print(
                f"[WARN] Skip initial particle state {particle_id}: "
                f"invalid prim path {current_prim_path}"
            )
            continue

        points_array = np.ascontiguousarray(points, dtype=np.float32)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            print(
                f"[WARN] Skip initial particle state {particle_id}: "
                f"expected Nx3 points, got shape {points_array.shape}"
            )
            continue

        usd_points = Vt.Vec3fArray.FromNumpy(points_array)
        restored_attributes = 0
        for attribute_name in ("points", "physxParticle:simulationPoints"):
            attribute = particle_prim.GetAttribute(attribute_name)
            if attribute and attribute.IsValid():
                attribute.Set(usd_points)
                restored_attributes += 1

        velocity_attribute = particle_prim.GetAttribute("velocities")
        if velocity_attribute and velocity_attribute.IsValid():
            zero_velocities = np.zeros_like(points_array, dtype=np.float32)
            velocity_attribute.Set(Vt.Vec3fArray.FromNumpy(zero_velocities))

        widths_attribute = particle_prim.GetAttribute("widths")
        if widths_attribute and widths_attribute.IsValid():
            current_widths = widths_attribute.Get()
            if current_widths and len(current_widths) != len(points_array):
                particle_widths = np.full(
                    len(points_array), float(current_widths[0]), dtype=np.float32
                )
                widths_attribute.Set(Vt.FloatArray.FromNumpy(particle_widths))

        if restored_attributes == 0:
            print(
                f"[WARN] Initial particle state {particle_id} has no writable "
                "points attributes"
            )


def restore_initial_prim_attributes(scene: "Scene", task_data: dict) -> None:
    """Restore optional <prim>::<attribute> snapshots from the recording's first frame."""
    stage = scene.world.stage
    for path, value in task_data.get("initial_prim_attributes", {}).items():
        prim_path, attribute_name = path.rsplit("::", 1)
        # Match recorded object roots, including children such as material shaders.
        for object_id, info in task_data.get("initial_layout", {}).items():
            original = info.get("prim_path")
            obj = scene.object_list.get(object_id) or scene.articulation_list.get(
                object_id
            )
            if (
                original
                and obj is not None
                and (prim_path == original or prim_path.startswith(original + "/"))
            ):
                prim_path = str(obj.prim_path) + prim_path[len(original) :]
                break
        prim_path = prim_path.replace("/World/_scene/", f"/World/{scene.uuid}/", 1)
        prim = stage.GetPrimAtPath(prim_path)
        attribute = prim.GetAttribute(attribute_name) if prim else None
        if not attribute:
            raise ValueError(
                f"Initial USD attribute not found: {prim_path}::{attribute_name}"
            )
        if isinstance(value, np.ndarray):
            value = value.tolist()
        usd_type = attribute.GetTypeName().type.pythonClass
        if usd_type is not None:
            value = usd_type(value)
        attribute.Set(value)


def recovery_scene(
    scene: "Scene",
    task_data: dict,
    task_name: str,
    default_config: dict,
) -> dict:
    def _set_robot_joint_positions(joint_positions) -> None:
        embodiment = scene.robot_list[0]
        joint_positions_arr = np.asarray(joint_positions)
        if joint_positions_arr.shape[0] == len(embodiment.default_dof_indices):
            embodiment.set_joint_positions(joint_positions_arr)
            return
        embodiment.robot.set_joint_positions(joint_positions_arr)

    layout = copy.deepcopy(task_data["initial_layout"])

    # Remove objects that are not in the layout
    object_list_keys = list(scene.object_list.keys())
    for key in object_list_keys:
        if key not in layout:
            object_prim_path = str(scene.object_list[key].prim_path).rstrip("/")
            has_layout_descendant = any(
                str(object_info.get("prim_path", "")).startswith(object_prim_path + "/")
                for object_info in layout.values()
            )
            if has_layout_descendant:
                continue
            if scene.object_list[key].prim.IsActive():
                scene.object_list[key].prim.SetActive(False)
            scene.object_list.pop(key)
        else:
            if not scene.object_list[key].prim.IsActive():
                scene.object_list[key].prim.SetActive(True)

    # Add objects that are not in the scene
    for key in layout:
        if layout[key].get("type", "object") != "object":
            continue
        if scene.robot_list[0].robot.name == key:
            continue
        if key not in scene.object_list:
            path = layout[key]["path"].split("GenManip-Sim/saved/assets/")[-1]
            if not is_valid_object_path(
                os.path.join(default_config["ASSETS_DIR"], path)
            ):
                continue
            scene.object_list[key] = preload_object(
                os.path.join(default_config["ASSETS_DIR"], path),
                scene.uuid,
                layout[key]["prim_path"].split("/")[-1][4:],
                scene.world,
                add_colliders=layout[key].get("add_colliders", True),
                add_rigid_body=layout[key].get("add_rigid_body", True),
                mass=layout[key].get("mass", None),
                static_friction=layout[key].get("static_friction", 1.0),
                dynamic_friction=layout[key].get("dynamic_friction", 1.0),
            )
            if not scene.object_list[key].prim.IsActive():
                scene.object_list[key].prim.SetActive(True)

    # Set objects' world pose and scale
    for key, object_info in layout.items():
        if object_info.get("type", "object") != "object":
            continue
        if scene.robot_list[0].robot.name == key:
            continue

        original_prim_path = object_info["prim_path"]
        original_uuid = original_prim_path.split("/")[2]
        current_prim_path = original_prim_path.replace(
            f"/World/{original_uuid}", f"/World/{scene.uuid}"
        )
        if not is_prim_path_valid(current_prim_path):
            continue

        if object_info.get("is_articulation_part", False):
            continue
        scene.object_list[key].set_world_pose(
            object_info["position"], object_info["orientation"]
        )
        scene.object_list[key].set_local_scale(object_info["scale"])
        clean_prim_velocity(scene.object_list[key].prim_path)

    for key, articulation_info in layout.items():
        if articulation_info.get("type", "object") != "articulation":
            continue
        if scene.robot_list[0].robot.name == key:
            continue

        original_prim_path = articulation_info["prim_path"]
        original_uuid = original_prim_path.split("/")[2]
        current_prim_path = original_prim_path.replace(
            f"/World/{original_uuid}", f"/World/{scene.uuid}"
        )
        if not is_prim_path_valid(current_prim_path):
            continue

        articulation = scene.articulation_list[key]
        root_pose = articulation_info
        physics_view = getattr(
            getattr(articulation, "_articulation_view", None), "_physics_view", None
        )
        if physics_view is not None:
            # PhysX poses address the root rigid body, which can have a different
            # origin from the enclosing USD prim stored in articulation_info.
            root_path = physics_view.link_paths[0][0]
            for part_info in layout.values():
                part_path = part_info.get("prim_path", "").replace(
                    f"/World/{original_uuid}", f"/World/{scene.uuid}", 1
                )
                if part_path == root_path:
                    root_pose = part_info
                    break
        articulation.set_world_pose(root_pose["position"], root_pose["orientation"])
        articulation.set_local_scale(articulation_info["scale"])
        if "joint_positions" in articulation_info:
            scene.articulation_list[key].set_joint_positions(
                articulation_info["joint_positions"]
            )

    restore_initial_prim_attributes(scene, task_data)
    restore_initial_particle_states(scene, task_data)

    for key, robot_info in layout.items():
        if robot_info.get("type", "object") != "robot":
            continue
        if scene.robot_list[0].robot.name != key:
            continue

        scene.robot_list[0].robot.set_world_pose(
            robot_info["position"], robot_info["orientation"]
        )
        _set_robot_joint_positions(robot_info["joint_positions"])

    scene.cache_library.mesh_dict = objectList2meshList(
        scene.object_list,
        os.path.join(
            default_config["ASSETS_DIR"],
            "mesh_data",
            task_name,
        ),
    )

    def recursive_set_mass(goal) -> None:
        for subgoal in goal:
            if isinstance(subgoal, list):
                recursive_set_mass(subgoal)
            elif isinstance(subgoal, dict):
                if subgoal.get("not_set_mass", False):
                    continue
                obj1_uid = subgoal.get("obj1_uid", None)
                obj2_uid = subgoal.get("obj2_uid", None)
                another_obj2_uid = subgoal.get("another_obj2_uid", None)
                if obj1_uid is None and obj2_uid is None:
                    continue
                if obj1_uid is not None:
                    if obj1_uid in scene.object_list:
                        set_mass(scene.object_list[obj1_uid].prim_path, 0.1)
                elif obj2_uid is not None:
                    if obj2_uid in scene.object_list:
                        set_mass(scene.object_list[obj2_uid].prim_path, 10.0)
                elif another_obj2_uid is not None:
                    if another_obj2_uid in scene.object_list:
                        set_mass(scene.object_list[another_obj2_uid].prim_path, 10.0)

    recursive_set_mass(task_data["goal"])

    if scene.robot_list[0].robot.name in task_data["initial_layout"]:
        _set_robot_joint_positions(
            task_data["initial_layout"][scene.robot_list[0].robot.name][
                "joint_positions"
            ]
        )
    else:
        _set_robot_joint_positions(
            np.array([0.012, -0.57, 0.0, -2.81, 0.0, 3.37, 0.741, 0.04, 0.04])
        )

    if "random_visuals" in task_data:
        for prim_path, visual_info in task_data["random_visuals"].items():
            if visual_info["type"] == "mdl":
                if not is_prim_path_valid(prim_path):
                    print(
                        f"[WARN] Skip random_visual mdl for invalid prim path: {prim_path}"
                    )
                    continue
                set_mdl(
                    prim_path,
                    os.path.join(
                        default_config["ASSETS_DIR"], visual_info["asset_path"]
                    ),
                )
            elif visual_info["type"] == "dome_light":
                create_dome_light(
                    prim_path,
                    os.path.join(
                        default_config["ASSETS_DIR"], visual_info["asset_path"]
                    ),
                )
            elif visual_info["type"] == "texture":
                if not is_prim_path_valid(prim_path):
                    print(
                        f"[WARN] Skip random_visual texture for invalid prim path: {prim_path}"
                    )
                    continue
                set_texture(
                    prim_path,
                    os.path.join(
                        default_config["ASSETS_DIR"], visual_info["asset_path"]
                    ),
                )
            else:
                raise ValueError(f"Invalid visual type: {visual_info['type']}")
    return layout


def is_valid_object_path(path: str) -> bool:
    return (
        os.path.exists(path)
        and not os.path.isdir(path)
        and str(path).endswith(".usd")
        and str(path) != ""
    )


def recovery_scene_render(
    scene: "Scene",
    task_data: dict,
    default_config: dict,
    remove_table: bool = False,
) -> None:
    def _set_robot_joint_positions(joint_positions) -> None:
        embodiment = scene.robot_list[0]
        joint_positions_arr = np.asarray(joint_positions)
        if joint_positions_arr.shape[0] == len(embodiment.default_dof_indices):
            embodiment.set_joint_positions(joint_positions_arr)
            return
        embodiment.robot.set_joint_positions(joint_positions_arr)

    if "initial_layout" not in task_data:
        return None
    restore_particle_layout(scene, task_data)
    layout = copy.deepcopy(task_data["initial_layout"])
    scene.robot_list[0].robot.set_world_pose(
        layout[scene.robot_list[0].robot.name]["position"],
        layout[scene.robot_list[0].robot.name]["orientation"],
    )
    if scene.robot_list[0].robot.name in layout:
        _set_robot_joint_positions(
            layout[scene.robot_list[0].robot.name]["joint_positions"]
        )
    else:
        _set_robot_joint_positions(
            np.array([0.012, -0.57, 0.0, -2.81, 0.0, 3.37, 0.741, 0.04, 0.04])
        )
    if scene.robot_list[0].robot.name in layout:
        layout.pop(scene.robot_list[0].robot.name)
    object_list_keys = list(scene.object_list.keys())
    for key in object_list_keys:
        if key not in layout:
            object_prim_path = str(scene.object_list[key].prim_path).rstrip("/")
            has_layout_descendant = any(
                str(object_info.get("prim_path", "")).startswith(object_prim_path + "/")
                for object_info in layout.values()
            )
            if has_layout_descendant:
                continue
            if scene.object_list[key].prim.IsActive():
                scene.object_list[key].prim.SetActive(False)
            scene.object_list.pop(key)
        else:
            if not scene.object_list[key].prim.IsActive():
                scene.object_list[key].prim.SetActive(True)
    for object in scene.object_list.values():
        set_semantic_label(
            str(object.prim_path), str(object.prim_path).split("/")[-1][4:]
        )
    for key in layout:
        if key not in scene.object_list and key not in scene.articulation_list:
            path = layout[key]["path"].split("GenManip-Sim/saved/assets/")[-1]
            if is_valid_object_path(os.path.join(default_config["ASSETS_DIR"], path)):
                scene.object_list[key] = preload_object(
                    os.path.join(default_config["ASSETS_DIR"], path),
                    scene.uuid,
                    layout[key]["prim_path"].split("/")[-1][4:],
                    scene.world,
                    add_colliders=False,
                    add_rigid_body=False,
                    remove_collider=True,
                )
                if not scene.object_list[key].prim.IsActive():
                    scene.object_list[key].prim.SetActive(True)
            else:
                print(f"[WARN] Skip object {key} for invalid path: {path}")
    for key, object_info in layout.items():
        if is_prim_path_valid(object_info["prim_path"]):
            if key in scene.object_list:
                scene.object_list[key].set_world_pose(
                    object_info["position"], object_info["orientation"]
                )
                scene.object_list[key].set_local_scale(object_info["scale"])
            # TODO: set articulation world pose and scale
            # elif key in scene.articulation_list:
            #     scene.articulation_list[key].set_world_pose(
            #         object_info["position"], object_info["orientation"]
            #     )
            #     scene.articulation_list[key].set_local_scale(object_info["scale"])
    # scene["cacheDict"]["meshDict"] = objectList2meshList(
    #     scene["object_list"],
    #     os.path.join(
    #         default_config["ASSETS_DIR"],
    #         "mesh_data",
    #         eval_config["task_name"],
    #     ),
    # )
    restore_initial_prim_attributes(scene, task_data)

    if remove_table:
        if scene.object_list["00000000000000000000000000000000"].prim.IsActive():
            scene.object_list["00000000000000000000000000000000"].prim.SetActive(False)

    if "random_visuals" in task_data:
        for prim_path, visual_info in task_data["random_visuals"].items():
            if visual_info["type"] == "mdl":
                if not is_prim_path_valid(prim_path):
                    print(
                        f"[WARN] Skip random_visual mdl for invalid prim path: {prim_path}"
                    )
                    continue
                set_mdl(
                    prim_path,
                    os.path.join(
                        default_config["ASSETS_DIR"], visual_info["asset_path"]
                    ),
                )
            elif visual_info["type"] == "dome_light":
                create_dome_light(
                    prim_path,
                    os.path.join(
                        default_config["ASSETS_DIR"], visual_info["asset_path"]
                    ),
                )
            elif visual_info["type"] == "texture":
                if not is_prim_path_valid(prim_path):
                    print(
                        f"[WARN] Skip random_visual texture for invalid prim path: {prim_path}"
                    )
                    continue
                set_texture(
                    prim_path,
                    os.path.join(
                        default_config["ASSETS_DIR"], visual_info["asset_path"]
                    ),
                )
            else:
                raise ValueError(f"Invalid visual type: {visual_info['type']}")
    return None
