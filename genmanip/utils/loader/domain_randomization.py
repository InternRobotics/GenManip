"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import random
import re
from typing import TYPE_CHECKING

import cv2
from mplib.pymp import Pose
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim  # type: ignore
from omni.isaac.sensor import Camera  # type: ignore
from pxr import Gf, Sdf, Usd  # type: ignore

from genmanip.core.scene.scene_config import LayoutConfig, ObjectConfig, SceneConfig
from genmanip.utils.loader.asset_search import get_asset_search_root
from genmanip.utils.loader.hardcode_rule import verify_cup_and_plate
from genmanip.utils.loader.utils import (
    add_object_to_scene_from_preload_list,
    adjust_object_scale_by_thickness,
    get_object_scale,
    remove_object_from_scene_by_preload,
    reset_object_xyz,
    reset_articulation_positions,
    resize_object_in_scene_by_uid,
)
from genmanip.utils.pointcloud.pointcloud import (
    get_current_pcList_by_meshList,
    get_mesh_info_by_load,
)
from genmanip.demogen.random_place.random_place import (
    setup_random_all_range,
    setup_random_all_range_buffered,
    setup_random_custom_tableset,
    setup_random_obj1_range,
    setup_random_tableset,
    setup_random_tableset_buffered,
    setup_random_tableset_by_centric_range,
    setup_scene_graph_placement,
    verify_placement,
)
from genmanip.utils.usd_utils.camera_utils import get_src
from genmanip.utils.usd_utils import (
    add_usd_to_world,
    change_material_info,
    change_table_mdl,
    clean_prim_velocity,
    create_dome_light,
    get_prim_bbox,
    resize_object_by_lwh,
    set_colliders,
    set_mdl,
    set_texture,
)
from genmanip.core.robot.base import BaseEmbodiment
from genmanip.utils.planner.mplib.utils import get_mplib_planner
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.utils.standalone.meta_utils import any_projection
from genmanip.utils.standalone.robot_utils import (
    joint_positions_to_position_and_orientation,
)

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


import logging


# add a debug function to hold the scene when layout generation fails, to help debug the layout failure issue by observing the scene in Isaac Sim GUI and checking the logs
# export GENMANIP_DEBUG_LAYOUT_FAIL_STEPS=100000
def _debug_hold_on_layout_failure(scene: "Scene", logger: logging.Logger) -> None:
    steps_raw = os.getenv("GENMANIP_DEBUG_LAYOUT_FAIL_STEPS", "0")
    try:
        steps = int(steps_raw)
    except ValueError:
        logger.warning(
            "Invalid GENMANIP_DEBUG_LAYOUT_FAIL_STEPS=%s; skip debug hold.",
            steps_raw,
        )
        return
    if steps <= 0:
        return
    logger.warning(
        "Layout failed, entering debug render loop for %d steps "
        "(GENMANIP_DEBUG_LAYOUT_FAIL_STEPS).",
        steps,
    )
    for _ in range(steps):
        scene.world.step(render=True)


def random_camera_pose(
    camera: Camera,
    camera_cfg: dict,
    max_translation_noise: float = 0.05,
    max_orientation_noise: float = 10.0,
) -> None:
    translation = np.array(camera_cfg["position"])
    orientation = np.array(camera_cfg["orientation"])

    random_direction = np.random.randn(3)
    random_direction /= np.linalg.norm(random_direction)
    random_distance = np.random.uniform(0, max_translation_noise)
    perturbed_translation = translation + random_direction * random_distance

    original_rot = R.from_quat(orientation, scalar_first=True)
    random_axis = np.random.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    random_angle_deg = np.random.uniform(-max_orientation_noise, max_orientation_noise)
    random_angle_rad = np.radians(random_angle_deg)
    perturbation_rot = R.from_rotvec(random_axis * random_angle_rad)
    perturbed_rot = perturbation_rot * original_rot
    perturbed_orientation = perturbed_rot.as_quat(scalar_first=True)

    camera.set_local_pose(
        translation=perturbed_translation,
        orientation=perturbed_orientation,
        camera_axes=camera_cfg["camera_axes"],
    )


def random_camera_list_pose(
    camera_list: dict[str, Camera],
    camera_cfg: dict[str, dict],
    camera_randomization_cfg: dict,
) -> None:
    for camera_name in camera_randomization_cfg.keys():
        random_camera_pose(
            camera_list[camera_name],
            camera_cfg[camera_name],
            max_translation_noise=camera_randomization_cfg[camera_name][
                "max_translation_noise"
            ],
            max_orientation_noise=camera_randomization_cfg[camera_name][
                "max_orientation_noise"
            ],
        )


def random_robot_pose(robot: Robot, random_range: float | dict) -> None:
    position, _ = robot.get_world_pose()
    if isinstance(random_range, (int, float)):
        new_position = (
            random.uniform(position[0] - random_range, position[0] + random_range),
            random.uniform(position[1] - random_range, position[1] + random_range),
            position[2],
        )
    elif isinstance(random_range, dict) and all(
        key in random_range for key in ["x", "y", "z"]
    ):
        new_position = (
            random.uniform(
                position[0] - random_range["x"], position[0] + random_range["x"]
            ),
            random.uniform(
                position[1] - random_range["y"], position[1] + random_range["y"]
            ),
            random.uniform(
                position[2] - random_range["z"], position[2] + random_range["z"]
            ),
        )
    else:
        raise ValueError(
            "random_range must be a number or a dict with 'x', 'y', 'z' keys"
        )

    robot.set_world_pose(position=new_position)


def random_robot_eepose(embodiment: BaseEmbodiment, current_dir: str) -> int:
    assert (
        embodiment.embodiment_name == "franka"
    ), "Only franka robot is supported for random robot eepose"
    robot = embodiment.robot
    planner = get_mplib_planner(
        robot, robot_type=embodiment.embodiment_name, current_dir=current_dir
    )
    position, orientation = joint_positions_to_position_and_orientation(
        robot.get_joint_positions()
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
        Pose(p=position.astype(float), q=orientation.astype(float)),
        robot.get_joint_positions()[:9],
        return_closest=True,
    )
    if joint_positions[0] != "Success":
        return -1
    elif joint_positions is None:
        return -1
    else:
        joint_positions = joint_positions[1]
        if joint_positions is None:
            return -1
        robot.set_joint_positions(
            np.concatenate([joint_positions[1][:7], embodiment.gripper_open]),
        )
        return 0


def replace_table(
    object_list: dict,
    table_path: str,
    uuid: str,
    without_collider: bool = False,
) -> None:
    # "00000000000000000000000000000000" is the default table key
    object = object_list["00000000000000000000000000000000"]
    # Disable the default table
    try:
        object_list["00000000000000000000000000000000"].prim.SetActive(False)
    except (AttributeError, RuntimeError, ValueError) as exc:
        print(f"Warning: failed to deactivate default table before replacement: {exc}")
    # Get the new table uid from table path, table path is like "/path/to/folder/table_uid/instance.usd"
    table_uid = table_path.split("/")[-2]
    # If the new table is already in the scene, use the existing prim
    if get_prim_at_path(f"/World/{uuid}/obj_{table_uid}").IsValid():
        object_list["00000000000000000000000000000000"] = XFormPrim(
            prim_path=f"/World/{uuid}/obj_{table_uid}",
            name=f"obj_{table_uid}",
        )
        object_list["00000000000000000000000000000000"].prim.SetActive(True)
    # If the new table is not in the scene, add it to the scene
    else:
        object_list["00000000000000000000000000000000"] = add_usd_to_world(
            asset_path=table_path,
            prim_path=f"/World/{uuid}/obj_{table_uid}",
            name=f"obj_{table_uid}",
            orientation=R.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()[
                [3, 0, 1, 2]
            ],
        )
    # Resize the table and set the position
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
        if not without_collider:
            set_colliders(object_list["00000000000000000000000000000000"].prim_path)
    except (AttributeError, RuntimeError, ValueError, OSError):
        delete_prim(object_list["00000000000000000000000000000000"].prim_path)
        object_list["00000000000000000000000000000000"] = object
        object_list["00000000000000000000000000000000"].prim.SetActive(True)


def replace_table_for_eval(
    object_list: dict,
    table_path: str,
    uuid: str,
    real_table_uid: str,
    without_collider: bool = False,
) -> None:
    # "00000000000000000000000000000000" is the default table key
    object = object_list["00000000000000000000000000000000"]
    # Disable the default table
    try:
        # If the new table is not the real table, disable the default table
        if (
            str(object_list["00000000000000000000000000000000"].prim_path).split(
                "obj_"
            )[-1]
            != real_table_uid
        ):
            object_list["00000000000000000000000000000000"].prim.SetActive(False)
        else:
            object_list["00000000000000000000000000000000"].prim.GetAttribute(
                "visibility"
            ).Set("invisible")
    except (AttributeError, RuntimeError, ValueError) as exc:
        print(
            f"Warning: failed to update default table visibility during eval setup: {exc}"
        )
    # Get the new table uid from table path, table path is like "/path/to/folder/table_uid/instance.usd"
    table_uid = table_path.split("/")[-2]
    # If the new table is already in the scene, use the existing prim
    if get_prim_at_path(f"/World/{uuid}/obj_{table_uid}").IsValid():
        object_list["00000000000000000000000000000000"] = XFormPrim(
            prim_path=f"/World/{uuid}/obj_{table_uid}",
            name=f"obj_{table_uid}",
        )
        object_list["00000000000000000000000000000000"].prim.SetActive(True)
    else:
        object_list["00000000000000000000000000000000"] = add_usd_to_world(
            asset_path=table_path,
            prim_path=f"/World/{uuid}/obj_{table_uid}",
            name=f"obj_{table_uid}",
            orientation=R.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()[
                [3, 0, 1, 2]
            ],
        )
    # Resize the table and set the position
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
        if not without_collider:
            set_colliders(object_list["00000000000000000000000000000000"].prim_path)
    except (AttributeError, RuntimeError, ValueError, OSError):
        delete_prim(object_list["00000000000000000000000000000000"].prim_path)
        object_list["00000000000000000000000000000000"] = object
        object_list["00000000000000000000000000000000"].prim.SetActive(True)
        object_list["00000000000000000000000000000000"].prim.GetAttribute(
            "visibility"
        ).Set("invisible")


def load_scene_as_background(
    scene_info: dict,
    assets_dir: str,
    uuid: str,
    table_uid: str,
) -> tuple[XFormPrim, XFormPrim]:
    def deactivate_selected_prims(
        prim: Usd.Prim, selected_names: list[str], random_names: list[str]
    ):

        for child_prim in prim.GetAllChildren():
            prim_name = child_prim.GetName().lower()
            for name in selected_names:
                if name.lower() in prim_name:
                    child_prim.SetActive(False)

            for name in random_names:
                if name.lower() in prim_name:
                    flag = True if random.random() > 0.5 else False
                    child_prim.SetActive(flag)

            deactivate_selected_prims(child_prim, selected_names, random_names)

    FIXED_HEIGHT = 0.99931

    original_table_prim = get_prim_at_path(f"/World/{uuid}/obj_{table_uid}")
    if original_table_prim.IsActive():
        original_table_prim.SetActive(False)

    ground_plane_prim = get_prim_at_path(f"/World/{uuid}/obj_defaultGroundPlane")
    deactivate_selected_prims(ground_plane_prim, ["collision"], [])

    room = add_usd_to_world(
        asset_path=os.path.join(
            assets_dir,
            "miscs",
            scene_info["scene"]["path"],
        ),
        prim_path=f"/World/{uuid}/room",
        name="room",
        translation=[
            scene_info["scene"]["translation"][1],
            -scene_info["scene"]["translation"][0],
            FIXED_HEIGHT
            - scene_info["scene"]["translation"][2]
            - 2 * scene_info["table"]["translation"][2],
        ],
        orientation=R.from_euler(
            "xyz", [0, 0, scene_info["scene"]["euler"][2] - 90], degrees=True
        ).as_quat()[[3, 0, 1, 2]],
        scale=scene_info["scene"]["scale"],
    )
    deactivate_selected_prims(
        room.prim,
        ["pan", "hearth", "ceiling", "__default_setting", "other", "microwave"],
        ["light"],
    )
    table = add_usd_to_world(
        asset_path=os.path.join(
            assets_dir,
            "miscs",
            scene_info["table"]["path"],
        ),
        prim_path=f"/World/{uuid}/table",
        name="table",
        translation=[
            0,
            0,
            FIXED_HEIGHT
            - scene_info["scene"]["translation"][2]
            - scene_info["table"]["translation"][2],
        ],
        orientation=R.from_euler("xyz", [0, 0, -90], degrees=True).as_quat()[
            [3, 0, 1, 2]
        ],
        scale=scene_info["table"]["scale"],
    )
    return room, table


def random_objaverse_table_texture(scene: "Scene", default_config: dict) -> None:
    light_intensity = 0
    texture_path = None
    while light_intensity < 80:
        texture_path = random.choice(scene.assets_library.wall_textures_paths)
        image = cv2.imread(
            os.path.abspath(
                f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
            )
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        light_intensity = np.mean(np.array(image))
    if texture_path is None:
        raise ValueError("No texture path found")
    change_material_info(
        f"{scene.object_list['00000000000000000000000000000000'].prim_path}",
        texture_path=os.path.abspath(
            f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
        ),
        translation=(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)),
        rotation=0,
        scale=(0.4, 0.4),
    )


def random_wall_texture(scene: "Scene", default_config: dict) -> None:
    # randomize 5 walls' texture, [left, right, front, back, top]
    for i in range(5):
        light_intensity = 0
        texture_path = None
        while light_intensity < 80:
            texture_path = random.choice(scene.assets_library.wall_textures_paths)
            image = cv2.imread(
                os.path.abspath(
                    f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
                )
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            light_intensity = np.mean(np.array(image))
        if texture_path is None:
            raise ValueError("No texture path found")
        scene.background["wall_textures"][i].set_texture(
            os.path.abspath(
                f"{default_config['ASSETS_DIR']}/miscs/textures/{texture_path}"
            )
        )


def random_texture_once(
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
    table_without_collider: bool = False,
) -> float:
    # randomize the dome light
    if scene_config.domain_randomization.random_environment.hdr:
        light_path = random.choice(scene.assets_library.domelights_paths)
        create_dome_light(
            f"/World/{scene.uuid}/obj_defaultGroundPlane/GroundPlane/DomeLight",
            f"{default_config['ASSETS_DIR']}/miscs/hdrs/{light_path}",
        )
    # randomize table texture for the default table, table prim tree is under objaverse format
    if (
        scene_config.domain_randomization.random_environment.table_texture
        and not scene_config.domain_randomization.random_environment.table_type
    ):
        random_objaverse_table_texture(scene, default_config)
    # randomize wall texture
    if scene_config.domain_randomization.random_environment.wall_texture:
        random_wall_texture(scene, default_config)
    # randomize table and table texture for grutopia format table, table prim tree is under grutopia format
    if scene_config.domain_randomization.random_environment.table_type:
        table_path = random.choice(scene.assets_library.table_paths)
        replace_table(
            scene.object_list,
            table_path,
            scene.uuid,
            without_collider=table_without_collider,
        )
        if scene_config.domain_randomization.random_environment.table_texture:
            change_table_mdl(
                scene.object_list["00000000000000000000000000000000"].prim_path,
                texture_path_list=[
                    os.path.abspath(
                        f"{default_config['ASSETS_DIR']}/object_usds/grutopia_usd/Table/Materials/{table_mdl_path}"
                    )
                    for table_mdl_path in scene.assets_library.table_mdl_paths
                ],
            )
    # randomize camera pose
    if scene_config.domain_randomization.camera_randomization is not None:
        current_dir = default_config["current_dir"]
        camera_data = load_yaml(
            os.path.join(
                current_dir,
                scene_config.domain_randomization.cameras.config_path,
            )
        )
        random_camera_list_pose(
            scene.camera_list,
            camera_data,
            scene_config.domain_randomization.camera_randomization,
        )
    # render the scene and get the light intensity
    for _ in range(10):
        scene.world.render()
    if "obs_camera" in scene.camera_list:
        image = get_src(scene.camera_list["obs_camera"], "rgb")
    elif "top_camera" in scene.camera_list:
        image = get_src(scene.camera_list["top_camera"], "rgb")
    else:
        raise ValueError("No main camera found")
    if image is None:
        raise ValueError("No image found")
    # get the light intensity
    light_intensity = np.mean(np.array(image))
    return float(light_intensity)


def random_texture_once_for_eval(
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
    table_without_collider: bool = True,
) -> float:
    # randomize the dome light
    if scene_config.domain_randomization.random_environment.hdr:
        light_path = random.choice(scene.assets_library.domelights_paths)
        create_dome_light(
            f"/World/{scene.uuid}/obj_defaultGroundPlane/GroundPlane/DomeLight",
            f"{default_config['ASSETS_DIR']}/miscs/hdrs/{light_path}",
        )
    # randomize table texture for the default table, table prim tree is under objaverse format
    if (
        scene_config.domain_randomization.random_environment.table_texture
        and not scene_config.domain_randomization.random_environment.table_type
    ):
        random_objaverse_table_texture(scene, default_config)
    # randomize wall texture
    if scene_config.domain_randomization.random_environment.wall_texture:
        random_wall_texture(scene, default_config)
    # randomize table and table texture for grutopia format table, table prim tree is under grutopia format
    if scene_config.domain_randomization.random_environment.table_type:
        table_path = random.choice(scene.assets_library.table_paths)
        replace_table_for_eval(
            scene.object_list,
            table_path,
            scene.uuid,
            real_table_uid=scene_config.table_uid,
            without_collider=table_without_collider,
        )
        if scene_config.domain_randomization.random_environment.table_texture:
            change_table_mdl(
                scene.object_list["00000000000000000000000000000000"].prim_path,
                texture_path_list=[
                    os.path.abspath(
                        f"{default_config['ASSETS_DIR']}/object_usds/grutopia_usd/Table/Materials/{table_mdl_path}"
                    )
                    for table_mdl_path in scene.assets_library.table_mdl_paths
                ],
            )
    # render the scene and get the light intensity
    for _ in range(10):
        scene.world.render()
    if "obs_camera" in scene.camera_list:
        image = get_src(scene.camera_list["obs_camera"], "rgb")
    elif "top_camera" in scene.camera_list:
        image = get_src(scene.camera_list["top_camera"], "rgb")
    else:
        raise ValueError("No main camera found")
    if image is None:
        raise ValueError("No image found")
    light_intensity = np.mean(np.array(image))
    return float(light_intensity)


def random_texture(
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
    table_without_collider: bool = False,
) -> int:
    cnt = 0
    while cnt < 10:
        # randomize the texture of the scene
        light_intensity = random_texture_once(
            scene,
            default_config,
            scene_config,
            table_without_collider=table_without_collider,
        )
        # if the light intensity is >= 80, break the loop
        if light_intensity >= 80:
            break
        cnt += 1
    # if the light intensity is < 80 after 10 times, print error message and return -1
    if cnt == 10:
        print("random texture failed")
        return -1
    return 0


def random_texture_for_eval(
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
) -> int:
    cnt = 0
    while cnt < 10:
        # randomize the texture of the scene
        light_intensity = random_texture_once_for_eval(
            scene,
            default_config,
            scene_config,
        )
        if light_intensity >= 80:
            break
        cnt += 1
    if cnt == 10:
        print("random texture failed")
        return -1
    return 0


def domain_randomization(
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
    task_data: dict,
    logger: logging.Logger,
) -> dict | int:
    # randomize robot base position
    if scene_config.domain_randomization.random_environment.robot_base_position:
        for robot in scene.robot_list:
            if isinstance(
                scene_config.domain_randomization.random_environment.robot_base_position,
                dict,
            ):
                random_robot_pose(
                    robot.robot,
                    scene_config.domain_randomization.random_environment.robot_base_position.random_range,
                )
            else:
                random_robot_pose(robot.robot, 0.1)
    # randomize robot eepose
    if scene_config.domain_randomization.random_environment.robot_eepose:
        for robot in scene.robot_list:
            random_robot_eepose(robot, default_config["current_dir"])
    # setup random table layout
    logger.info("Setup random table layout")
    if scene_config.layout_config.type == "random_all":
        IS_OK = setup_random_tableset(
            scene.object_list,
            scene.cache_library.mesh_dict,
            scene_config.layout_config.ignored_objects,
        )
    elif scene_config.layout_config.type == "random_all_buffered":
        IS_OK = setup_random_tableset_buffered(
            scene.object_list,
            scene.cache_library.mesh_dict,
            scene_config.layout_config.ignored_objects,
            task_data["goal"][0][0]["obj1_uid"],
            task_data["goal"][0][0]["obj2_uid"],
        )
    elif scene_config.layout_config.type == "centric_random_range":
        IS_OK = setup_random_tableset_by_centric_range(
            scene.object_list,
            scene.cache_library.mesh_dict,
            scene_config.layout_config,
            scene_config.layout_config.ignored_objects,
            scene_config.layout_config.partial_ignore,
        )
    elif scene_config.layout_config.type == "random_obj1_range":
        IS_OK = setup_random_obj1_range(
            scene.object_list,
            scene.cache_library.mesh_dict,
            task_data,
            scene_config.layout_config,
            scene.meta_infos["world_pose_list"],
        )
    elif scene_config.layout_config.type == "random_custom_tableset":
        if scene_config.layout_config.custom_tableset is None:
            raise ValueError("Custom tableset is not defined")
        IS_OK = setup_random_custom_tableset(
            scene.object_list,
            scene.articulation_list,
            scene.cache_library.mesh_dict,
            scene_config.layout_config.custom_tableset,
            in_order=scene_config.layout_config.in_order,
            ignored_objects=scene_config.layout_config.ignored_objects,
        )
    elif scene_config.layout_config.type == "random_all_range":
        IS_OK = setup_random_all_range(
            scene.object_list,
            scene.cache_library.mesh_dict,
            scene_config.layout_config,
            scene_config.layout_config.ignored_objects,
        )
    elif scene_config.layout_config.type == "random_all_range_buffered":
        IS_OK = setup_random_all_range_buffered(
            scene.object_list,
            scene.cache_library.mesh_dict,
            scene_config.layout_config,
            scene_config.layout_config.ignored_objects,
            task_data,
        )
    elif scene_config.layout_config.type == "scene_graph_placement":
        IS_OK = setup_scene_graph_placement(
            scene.object_list,
            scene.cache_library.mesh_dict,
            scene_config,
        )
    else:
        IS_OK = 0
    if IS_OK == -1:
        _debug_hold_on_layout_failure(scene, logger)
        logger.error("Random table layout failed")
        return IS_OK

    # randomize visuals
    task_data["random_visuals"] = random_visuals(default_config, scene_config)

    # clean prim velocity because the prim velocity may not zero from the previous scene
    for key in scene.object_list:
        clean_prim_velocity(scene.object_list[key].prim_path)
    logger.info("Verify placement")
    # verify if the placement is stable
    if (
        len(task_data["goal"]) > 0
        and len(task_data["goal"][0]) > 0
        and "obj1_uid" in task_data["goal"][0][0]
    ):
        if task_data["goal"][0][0]["obj1_uid"] in scene.object_list.keys():
            IS_OK = verify_placement(
                scene.object_list[task_data["goal"][0][0]["obj1_uid"]],
                scene.world,
            )
            if not IS_OK:
                logger.error("Verify placement failed")
                return -1
    # clean prim velocity because the prim velocity may not zero from the previous scene
    for key in scene.object_list:
        clean_prim_velocity(scene.object_list[key].prim_path)

    # task should not be finished after randomization
    sr = scene.metric_manager.step(scene)
    finished = sr == 1

    if finished:
        logger.error("Check finished failed")
        return -1
    return 0


def find_match_assets(assets_dir: str, path_pattern: str) -> list[str]:
    normalized_pattern, search_root = get_asset_search_root(assets_dir, path_pattern)
    compiled_pattern = re.compile(normalized_pattern)
    assets_list = []
    for root, _, files in os.walk(search_root):
        for file in files:
            asset_path = os.path.join(root, file)
            relative_path = os.path.relpath(asset_path, assets_dir)
            normalized_relative_path = relative_path.replace(os.sep, "/")
            if compiled_pattern.match(file) or compiled_pattern.match(
                normalized_relative_path
            ):
                assets_list.append(asset_path)
    return assets_list


def random_visuals(
    default_config: dict, scene_config: SceneConfig
) -> dict[str, dict[str, str]]:
    result = {}
    for visual in scene_config.domain_randomization.random_environment.random_visuals:
        asset_list = find_match_assets(
            default_config["ASSETS_DIR"], visual.assets_pattern
        )
        if len(asset_list) == 0:
            raise ValueError(f"No match assets found for {visual.assets_pattern}")
        asset_path = random.choice(asset_list)
        if visual.type == "mdl":
            set_mdl(visual.prim_path, asset_path)
        elif visual.type == "dome_light":
            create_dome_light(visual.prim_path, asset_path)
        elif visual.type == "texture":
            set_texture(visual.prim_path, asset_path)
        else:
            raise ValueError(f"Invalid visual type: {visual.type}")
        normalized_asset_path = os.path.relpath(
            asset_path, default_config["ASSETS_DIR"]
        )
        result[visual.prim_path] = {
            "type": visual.type,
            "asset_path": normalized_asset_path,
        }
    return result


def set_shader_color(
    stage: Usd.Stage,
    prim_path: str,
    color: tuple[float, float, float] | list[float],
    attribute: str = "inputs:diffuseColor",
) -> None:
    """Set a shader color input on the current in-memory USD stage."""
    path = Sdf.Path(prim_path)
    if not path.IsAbsolutePath():
        raise ValueError(f"Shader prim path must be absolute: {prim_path}")

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Shader prim does not exist: {prim_path}")

    color_attr = prim.GetAttribute(attribute)
    if not color_attr.IsValid() and attribute == "inputs:diffuseColor":
        color_attr = prim.GetAttribute("inputs:diffuse_color")

    if not color_attr.IsValid():
        available = sorted(
            attr.GetName()
            for attr in prim.GetAttributes()
            if "color" in attr.GetName().lower()
        )
        raise RuntimeError(
            f"Shader color input {attribute!r} is missing on {prim_path}; "
            f"available color attributes: {available}"
        )

    if len(color) != 3:
        raise ValueError(f"Color must contain 3 values, got: {color}")

    value = Gf.Vec3f(*(float(component) for component in color))
    if not color_attr.Set(value):
        raise RuntimeError(f"Failed to set {color_attr.GetPath()} to {value}")


def apply_material_overrides(stage: Usd.Stage, scene_config: SceneConfig) -> None:
    """Apply fixed material colors after scene recovery and randomization."""
    for override in scene_config.domain_randomization.material_overrides:
        set_shader_color(
            stage=stage,
            prim_path=override.prim_path,
            attribute=override.attribute,
            color=override.value,
        )


def reset_scene(scene: "Scene") -> None:
    reset_object_xyz(scene.object_list, scene.meta_infos["world_pose_list"])
    reset_articulation_positions(scene)
    for robot, joint_positions, joint_velocities, robot_pose in zip(
        scene.robot_list,
        scene.meta_infos["joint_positions"],
        scene.meta_infos["joint_velocities"],
        scene.meta_infos["robot_pose_list"],
    ):
        robot.robot.set_joint_positions(joint_positions)
        robot.robot._articulation_view.set_joint_position_targets(joint_positions)
        robot.robot.set_joint_velocities(joint_velocities)
        robot.robot.set_world_pose(*robot_pose)
    for robot in scene.robot_list:
        robot.reset()


def satisfy_replace_existed_object(
    replace_object_config: dict[str, ObjectConfig],
    key: str,
    replaced_uid: str,
    added_uid_list: list[str],
    object_config_key_list: list[str],
) -> bool:
    if "cup_plate_replace" in replace_object_config[key].option:
        if verify_cup_and_plate(
            object_config_key_list,
            key,
            object_config_key_list.index(key),
            replaced_uid,
            added_uid_list,
        ):
            return True
        return False
    else:
        return True


def _layout_config_projection(
    scene: "Scene",
    layout_config: LayoutConfig,
) -> LayoutConfig:
    if layout_config.type == "random_custom_tableset":
        custom_dict = {}
        if isinstance(layout_config.custom_tableset, list):
            if layout_config.custom_tableset is None:
                raise ValueError("Custom tableset is not defined")
            layout_config.custom_tableset = random.choice(layout_config.custom_tableset)
        if not isinstance(layout_config.custom_tableset, dict):
            raise ValueError("custom_tableset must be a dict")
        for key in layout_config.custom_tableset:
            custom_dict[scene.cache_library.meta_to_fine_projection[key]] = (
                layout_config.custom_tableset[key]
            )
            if layout_config.custom_tableset[key]["type"] == "scene_graph":
                custom_dict[scene.cache_library.meta_to_fine_projection[key]][
                    "obj2_uid"
                ] = scene.cache_library.meta_to_fine_projection[
                    layout_config.custom_tableset[key]["obj2_uid"]
                ]
        layout_config.custom_tableset = custom_dict
    return layout_config


def _goal_config_projection(
    scene: "Scene",
    goal_config: dict,
) -> dict:
    if isinstance(goal_config, list):
        for i in range(len(goal_config)):
            goal_config[i] = _goal_config_projection(scene, goal_config[i])
    else:
        goal_config = any_projection(
            goal_config, scene.cache_library.meta_to_fine_projection
        )
    return goal_config


def _action_config_projection(
    scene: "Scene",
    action_config: list[dict],
) -> list[dict]:
    for i in range(len(action_config)):
        action_config[i] = any_projection(
            action_config[i],
            scene.cache_library.meta_to_fine_projection,
        )
    return action_config


def _add_additional_object_from_path_projection(
    scene: "Scene",
    replace_object_config: dict[str, ObjectConfig],
    key: str,
    scene_config: SceneConfig,
    default_config: dict,
) -> None:
    # 5-3-0. get real uid of the object
    if replace_object_config[key].uid != "":
        replaced_uid = replace_object_config[key].uid
    else:
        replaced_uid = replace_object_config[key].path.split("/")[-1].split(".")[0]

    # 5-3-1. add object to the scene
    add_object_to_scene_from_preload_list(
        replaced_uid, scene, default_config, scene_config
    )

    # 5-3-2. get the scale of the object
    scale = get_object_scale(
        replace_object_config, key, replaced_uid, scene.object_pool
    )

    # 5-3-3. resize the object by scale
    # 5-3-3-1. if scale is in meter, and is float or list[float]
    if scale is not None and replace_object_config[key].fixed_scale is None:
        resize_object_in_scene_by_uid(
            replaced_uid, scene, default_config, scale, scene_config
        )

    # 5-3-3-2. if scale is relative to the original object, and is float or list[float]
    elif (
        replace_object_config[key].fixed_scale is not None
        or replace_object_config[key].relative_scale is not None
    ):
        scale = scene.object_list[replaced_uid].get_local_scale()
        if replace_object_config[key].relative_scale is not None:
            if not isinstance(replace_object_config[key].relative_scale, list):
                scale = scale * replace_object_config[key].relative_scale
            elif isinstance(replace_object_config[key].relative_scale, list):
                scale = scale * replace_object_config[key].relative_scale
            else:
                raise ValueError(
                    f"Invalid relative scale type: {type(replace_object_config[key].relative_scale)}"
                )
        elif replace_object_config[key].fixed_scale is not None:
            if not isinstance(replace_object_config[key].fixed_scale, list):
                scale = [replace_object_config[key].fixed_scale] * 3
            elif isinstance(replace_object_config[key].fixed_scale, list):
                scale = replace_object_config[key].fixed_scale
            else:
                raise ValueError(
                    f"Invalid fixed scale type: {type(replace_object_config[key].fixed_scale)}"
                )
        scene.object_list[replaced_uid].set_local_scale(scale)
        mesh_info = get_mesh_info_by_load(
            scene.object_list[replaced_uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                os.path.dirname(
                    scene.cache_library.preloaded_object_path_list[replaced_uid]
                ),
                f"{replaced_uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[replaced_uid] = mesh_info
    # 5-3-3-3. if scale is not defined, raise error
    else:
        raise ValueError(
            f"Object {replaced_uid} has no scale information, previous logic is archived, please add `fixed_scale: 1.0` to object config to keep object to its original scale"
        )

    # 5-3-3-4. adjust object scale by thickness if needed
    if "adjust_thickness" in replace_object_config[key].option:
        adjust_object_scale_by_thickness(
            scene,
            replaced_uid,
            default_config,
            scene_config,
            0.06,
        )

    # 5-3-4. record the meta_to_fine_projection
    scene.cache_library.meta_to_fine_projection[key] = replaced_uid


def _existed_object_projection(
    scene: "Scene",
    replace_object_config: dict[str, ObjectConfig],
    key: str,
    replaced_uid: str,
    default_config: dict,
    scene_config: SceneConfig,
) -> None:
    # 5-1-1. activate object if it is not active
    if not scene.cache_library.preloaded_object_list[replaced_uid].prim.IsActive():
        scene.cache_library.preloaded_object_list[replaced_uid].prim.SetActive(True)

    # 5-1-2. add object to object_list and compute mesh cache for the object
    if replaced_uid not in scene.object_list:
        scene.object_list[replaced_uid] = scene.cache_library.preloaded_object_list[
            replaced_uid
        ]
        mesh_info = get_mesh_info_by_load(
            scene.object_list[replaced_uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                f"{replaced_uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[replaced_uid] = mesh_info

    fixed_size = replace_object_config[key].fixed_size
    if fixed_size is not None:
        resize_object_in_scene_by_uid(
            replaced_uid,
            scene,
            default_config,
            fixed_size,
            scene_config,
        )

    # 5-1-3. record the meta_to_fine_projection
    scene.cache_library.meta_to_fine_projection[key] = replaced_uid


def _load_object_from_path_projection(
    scene: "Scene",
    replace_object_config: dict[str, ObjectConfig],
    key: str,
    replaced_uid: str,
    default_config: dict,
    scene_config: SceneConfig,
) -> None:
    # 5-2-1. add object to the scene
    add_object_to_scene_from_preload_list(
        replaced_uid, scene, default_config, scene_config
    )

    # 5-2-2. get the scale of the object
    scale = get_object_scale(
        replace_object_config, key, replaced_uid, scene.object_pool
    )

    # 5-2-3. resize the object by scale
    # 5-2-3-1. if scale is in meter, and is float or list[float]
    if scale is not None and replace_object_config[key].fixed_scale is None:
        resize_object_in_scene_by_uid(
            replaced_uid, scene, default_config, scale, scene_config
        )

    # 5-2-3-2. if scale is relative to the original object, and is float or list[float]
    elif (
        replace_object_config[key].fixed_scale is not None
        or replace_object_config[key].relative_scale is not None
    ):
        scale = scene.object_list[replaced_uid].get_local_scale()
        if replace_object_config[key].relative_scale is not None:
            if not isinstance(replace_object_config[key].relative_scale, list):
                scale = scale * replace_object_config[key].relative_scale
            elif isinstance(replace_object_config[key].relative_scale, list):
                scale = scale * replace_object_config[key].relative_scale
            else:
                raise ValueError(
                    f"Invalid relative scale type: {type(replace_object_config[key].relative_scale)}"
                )
        elif replace_object_config[key].fixed_scale is not None:
            if not isinstance(replace_object_config[key].fixed_scale, list):
                scale = [replace_object_config[key].fixed_scale] * 3
            elif isinstance(replace_object_config[key].fixed_scale, list):
                scale = replace_object_config[key].fixed_scale
            else:
                raise ValueError(
                    f"Invalid fixed scale type: {type(replace_object_config[key].fixed_scale)}"
                )
        scene.object_list[replaced_uid].set_local_scale(scale)
        mesh_info = get_mesh_info_by_load(
            scene.object_list[replaced_uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                os.path.dirname(
                    scene.cache_library.preloaded_object_path_list[replaced_uid]
                ),
                f"{replaced_uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[replaced_uid] = mesh_info
    # 5-2-3-3. if scale is not defined, raise error
    else:
        raise ValueError(
            f"Object {replaced_uid} has no scale information, previous logic is archived, please add `fixed_scale: 1.0` to object config to keep object to its original scale"
        )

    # 5-2-3-4. adjust object scale by thickness if needed
    if "adjust_thickness" in replace_object_config[key].option:
        adjust_object_scale_by_thickness(
            scene,
            replaced_uid,
            default_config,
            scene_config,
            0.06,
        )

    # 5-2-4. record the meta_to_fine_projection
    scene.cache_library.meta_to_fine_projection[key] = replaced_uid


def build_up_scene(
    scene: "Scene",
    scene_config: SceneConfig,
    default_config: dict,
    task_data: dict,
) -> "Scene":
    """
    initialize meta_to_fine_projection
    for config:
    object1:
        type: load_object_from_path
        ...
    `object1` is called meta key, which may represent a set of objects in the scene, each time for one object. The uid of the object is called fine key.
    meta_to_fine_projection is a dict, the key is the meta key, the value is the fine key.
    by recording it, we can disactivate last added object and add new object to the scene.
    """
    # 1. initialize meta_to_fine_projection
    if scene.cache_library.meta_to_fine_projection == {}:
        scene.cache_library.meta_to_fine_projection = {}
        for key in scene.cache_library.preload_hash_feature:
            scene.cache_library.meta_to_fine_projection[key] = ""

    # 2. initialize configs
    replace_object_config = scene_config.object_config
    added_uid_list = []
    object_config_key_list = list(replace_object_config.keys())
    object_config_key_list.sort()

    # 3. choose new object to replace for each object in the object_config
    for key in object_config_key_list:
        while True:
            replaced_uid = random.choice(
                scene.cache_library.preloaded_object_uid_list[
                    scene.cache_library.preload_hash_feature[key]
                ]
            )
            if replaced_uid not in added_uid_list and satisfy_replace_existed_object(
                replace_object_config,
                key,
                replaced_uid,
                added_uid_list,
                object_config_key_list,
            ):
                break
        added_uid_list.append(replaced_uid)

    # 4. remove the old object
    for key, replaced_uid in scene.cache_library.meta_to_fine_projection.items():
        if (
            replaced_uid in scene.object_list
            and replace_object_config[key].type == "load_object_from_path"
        ):
            remove_object_from_scene_by_preload(replaced_uid, scene)

    # 5. add new object and resize it
    for key, replaced_uid in zip(object_config_key_list, added_uid_list):
        # 5-1. existed object
        if replace_object_config[key].type == "existed_object":
            _existed_object_projection(
                scene,
                replace_object_config,
                key,
                replaced_uid,
                default_config,
                scene_config,
            )
        # 5-2. load_object_from_path
        elif replace_object_config[key].type == "load_object_from_path":
            _load_object_from_path_projection(
                scene,
                replace_object_config,
                key,
                replaced_uid,
                default_config,
                scene_config,
            )
        # 5-3. add_additional_object_from_path
        elif replace_object_config[key].type == "add_additional_object_from_path":
            _add_additional_object_from_path_projection(
                scene, replace_object_config, key, scene_config, default_config
            )

    # 6. replace the uid in the goal
    if scene_config.generation_config.action_path.actions is not None:
        scene_config.generation_config.action_path.actions = _action_config_projection(
            scene, scene_config.generation_config.action_path.actions
        )
    task_data["goal"] = _goal_config_projection(scene, task_data["goal"])

    # 7. replace uid in layout config
    scene_config.layout_config = _layout_config_projection(
        scene, scene_config.layout_config
    )

    # 8. add object_infos to task_data
    task_data["object_infos"] = {}
    for obj_uid in scene.object_list:
        obj_info = scene.object_pool.get_object_info(obj_uid)
        if obj_info is not None and obj_uid not in task_data["object_infos"]:
            task_data["object_infos"][obj_uid] = obj_info
    for obj_uid in scene.articulation_list:
        obj_info = scene.object_pool.get_object_info(obj_uid)
        if obj_info is not None and obj_uid not in task_data["object_infos"]:
            task_data["object_infos"][obj_uid] = obj_info
    for obj_uid in scene.cache_library.preloaded_object_list:
        obj_info = scene.object_pool.get_object_info(obj_uid)
        if obj_info is not None and obj_uid not in task_data["object_infos"]:
            task_data["object_infos"][obj_uid] = obj_info
    return scene
