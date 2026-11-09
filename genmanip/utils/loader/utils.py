"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import random
from typing import TYPE_CHECKING

import numpy as np

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.articulations import Articulation  # type: ignore
from genmanip.utils.pointcloud.pointcloud import (
    get_current_meshList,
    get_mesh_info_by_load,
)
from genmanip.utils.usd_utils.prim_utils import resize_object, resize_object_by_lwh
from genmanip.utils.standalone.pc_utils import compute_aabb_lwh, compute_mesh_bbox
from genmanip.utils.annotation.object_pool import ObjectPool
from genmanip.core.scene.scene_config import ObjectConfig, SceneConfig

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


def reset_object_xyz(
    object_list: dict[str, XFormPrim], xyz: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    for key in object_list:
        if key == "00000000000000000000000000000000" or key == "defaultGroundPlane":
            continue
        if key in xyz:
            object_list[key].set_world_pose(*xyz[key])


def collect_articulation_list(
    scene: "Scene", articulation_list: dict[str, Articulation]
) -> dict[str, np.ndarray]:
    init_positions_list = {}
    for articulation_id, articulation in articulation_list.items():
        if scene.articulation_data[articulation_id]["is_articulated"]:
            init_positions_list[articulation_id] = articulation.get_joint_positions()
    return init_positions_list


def collect_world_pose_list(
    object_list: dict[str, XFormPrim],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    world_pose_list = {}
    for key in object_list:
        if key == "00000000000000000000000000000000" or key == "defaultGroundPlane":
            continue
        world_pose_list[key] = object_list[key].get_world_pose()
    return world_pose_list


def reset_articulation_positions(scene: "Scene") -> None:
    if "articulation_pose_list" not in scene.meta_infos:
        print("No articulation data found")
        return
    articulation_list = scene.articulation_list
    articulation_pose_list = scene.meta_infos["articulation_pose_list"]
    skip_joint_position_reset = {
        uid
        for object_config in scene.scene_config.object_config.values()
        if not object_config.reset_joint_positions
        for uid in object_config.uid_list
    }
    for articulation_id, articulation in articulation_list.items():
        if articulation_id in skip_joint_position_reset:
            continue
        if scene.articulation_data[articulation_id]["is_articulated"]:
            articulation_joint_positions = articulation_pose_list[articulation_id]
            articulation.set_joint_positions(articulation_joint_positions)
            articulation.set_joint_velocities(
                np.zeros_like(articulation_joint_positions)
            )


def remove_object_from_scene_by_preload(uid: str, scene: "Scene") -> None:
    # Disactivate the object in object list
    if scene.object_list[uid].prim.IsActive():
        scene.object_list[uid].prim.SetActive(False)
    # Remove the object from mesh cache
    scene.cache_library.mesh_dict.pop(uid)
    # Remove the object from object list
    scene.object_list.pop(uid)


def add_object_to_scene_from_preload_list(
    uid: str,
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
) -> None:
    # Add the object from preloaded object list to object list
    scene.object_list[uid] = scene.cache_library.preloaded_object_list[uid]
    # Activate the object in object list
    if not scene.object_list[uid].prim.IsActive():
        scene.object_list[uid].prim.SetActive(True)
    # Restore object scale to the pre-scale baseline so per-episode scaling
    # (e.g., relative_scale) will not accumulate across resets.
    preload_meta = scene.cache_library.preload_object_meta_info.setdefault(uid, {})
    if "base_local_scale" not in preload_meta:
        preload_meta["base_local_scale"] = (
            scene.object_list[uid].get_local_scale().tolist()
        )
    else:
        scene.object_list[uid].set_local_scale(
            np.array(preload_meta["base_local_scale"])
        )
    # Compute the object's mesh information
    if uid in scene.cache_library.preloaded_object_path_list:
        # if object is in preloaded object path list, object mesh path should be in the same folder as the object path in `mesh_data`, like `mesh_data/task_name/folder_name/uid.obj` while usd is in `object_usd/folder_name/uid.usd`
        mesh_info = get_mesh_info_by_load(
            scene.object_list[uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                os.path.dirname(scene.cache_library.preloaded_object_path_list[uid]),
                f"{uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[uid] = mesh_info
    else:
        # else, object mesh path should be `mesh_data/task_name/uid.obj`
        mesh_info = get_mesh_info_by_load(
            scene.object_list[uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                f"{uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[uid] = mesh_info


def replace_object_in_scene_by_uid(
    previous_uid: str,
    replaced_uid: str,
    scene: "Scene",
    default_config: dict,
    scene_config: SceneConfig,
) -> None:
    # Remove the previous object from scene dict
    if previous_uid in scene.object_list:
        remove_object_from_scene_by_preload(previous_uid, scene)
    # Add the replaced object to scene dict
    add_object_to_scene_from_preload_list(
        replaced_uid, scene, default_config, scene_config
    )


def resize_object_in_scene_by_uid(
    uid: str,
    scene: "Scene",
    default_config: dict,
    scale: float | list[float],
    scene_config: SceneConfig,
) -> None:
    # 1. Get the current mesh list from scene dict
    meshlist = get_current_meshList(scene.object_list, scene.cache_library.mesh_dict)

    # 2. Resize the object in scene dict by scale factor and mesh information
    if isinstance(scale, float):
        resize_object(
            scene.object_list[uid],
            scale,
            meshlist[uid],
        )
    elif isinstance(scale, list):
        print(f"resize object {uid} by {scale}")
        resize_object_by_lwh(
            scene.object_list[uid],
            l=scale[0],
            w=scale[1],
            h=scale[2],
            mesh=meshlist[uid],
        )
    else:
        raise ValueError(f"Invalid scale type: {type(scale)}")

    # 3. Update the object's mesh information
    if uid in scene.cache_library.preloaded_object_path_list:
        mesh_info = get_mesh_info_by_load(
            scene.object_list[uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                os.path.dirname(scene.cache_library.preloaded_object_path_list[uid]),
                f"{uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[uid] = mesh_info
    else:
        mesh_info = get_mesh_info_by_load(
            scene.object_list[uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                f"{uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[uid] = mesh_info


def adjust_object_scale_by_thickness(
    scene: "Scene",
    uid: str,
    default_config: dict,
    scene_config: SceneConfig,
    min_thickness: float,
) -> None:
    meshlist = get_current_meshList(scene.object_list, scene.cache_library.mesh_dict)
    mesh = meshlist[uid]
    aabb = compute_mesh_bbox(mesh)
    if np.min(compute_aabb_lwh(aabb)) > min_thickness:
        l, w, h = compute_aabb_lwh(aabb)
        min_thickness_ratio = min_thickness / np.min([l, w])
        min_thickness_ratio = max(min_thickness_ratio, min_thickness / np.min([l, w]))
        l *= min_thickness_ratio
        w *= min_thickness_ratio
        h *= min_thickness_ratio
        resize_object_by_lwh(
            scene.object_list[uid],
            l=l,
            w=w,
            h=h,
            mesh=mesh,
        )
        mesh_info = get_mesh_info_by_load(
            scene.object_list[uid],
            os.path.join(
                default_config["ASSETS_DIR"],
                "mesh_data",
                scene_config.task_name,
                os.path.dirname(scene.cache_library.preloaded_object_path_list[uid]),
                f"{uid}.obj",
            ),
        )
        if mesh_info is not None:
            scene.cache_library.mesh_dict[uid] = mesh_info


def get_object_scale(
    replace_object_config: dict[str, ObjectConfig],
    key: str,
    replaced_uid: str,
    object_pool: ObjectPool,
) -> float | list[float] | None:
    if "plain_replace" in replace_object_config[key].option:
        scale = None
    else:
        object_info = object_pool.get_object_info(replaced_uid)
        if object_info is None:
            scale = random.uniform(0.08, 0.12)
        elif "grep_min_scale" in replace_object_config[key].option:
            scale = object_info["scale"][0]
        else:
            scale = random.uniform(
                object_info["scale"][0],
                object_info["scale"][1],
            )
    if replace_object_config[key].clip_range is not None and scale is not None:
        clip_range = replace_object_config[key].clip_range
        if clip_range is None:
            raise ValueError(f"Clip range is not defined for object {key}")
        clip_range_min = clip_range["min"]
        clip_range_max = clip_range["max"]
        if not isinstance(clip_range_min, float):
            raise ValueError(f"Clip range min {clip_range_min} is not a float")
        if not isinstance(clip_range_max, float):
            raise ValueError(f"Clip range max {clip_range_max} is not a float")
        scale = np.clip(float(scale), clip_range_min, clip_range_max)
    if replace_object_config[key].fixed_size is not None:
        scale = replace_object_config[key].fixed_size
    # if scale is None:
    #     raise ValueError(f"Scale is not defined for object {key}")
    return scale
