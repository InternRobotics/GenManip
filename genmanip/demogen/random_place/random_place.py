"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

"""
We defined a place relation with the following parameters:
    - object1: the object to be placed
    - object2: the object on which object1 is to be placed
    - platform: the platform on which object1 is to be placed, sometimes when relation is "on/below", the platform is the object2
"""

import copy
import os
import random

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.articulations import Articulation  # type: ignore
from omni.isaac.core import World  # type: ignore

from genmanip.core.scene.scene_config import LayoutConfig, SceneConfig
from genmanip.demogen.random_place.scene_graph_placement import process_scene_graph
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    check_subgoal_finished_rigid,
    get_related_position,
)
from genmanip.utils.pointcloud.pointcloud import (
    get_current_meshList,
    get_current_pcList_by_meshList,
    meshlist_to_pclist,
)
from genmanip.utils.pointcloud.utils import MeshInfo
from genmanip.utils.standalone.pc_utils import (
    bbox_to_polygon,
    check_mesh_collision,
    compute_lrfb_area,
    compute_mesh_bbox,
    compute_mesh_center,
    compute_near_area,
    compute_pcd_bbox,
    find_fixed_polygon_placement,
    find_polygon_placement,
    find_polygon_placement_with_rotation,
    get_max_distance_to_polygon,
    get_platform_available_area,
    get_xy_contour,
    sample_point_in_2d_line,
    sample_points_in_convex_hull,
    visualize_polygons,
)


def _dump_polygon_debug(
    polygons: list[BaseGeometry], reason: str = "layout_fail"
) -> None:
    if os.getenv("GENMANIP_DEBUG_POLYGONS", "0") != "1":
        return
    output_dir = os.getenv("GENMANIP_DEBUG_POLYGONS_DIR", "tmp/polygon_debug")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{reason}_{random.randint(0, 10**9 - 1):09d}.png"
    output_path = os.path.join(output_dir, filename)
    try:
        visualize_polygons(polygons, output_path=output_path)
        print(f"[DEBUG] polygon dump saved to: {output_path}")
    except Exception as exc:
        print(f"[DEBUG] failed to dump polygon debug image: {exc}")


def rotate_object_around_z(object: XFormPrim, angle_range: tuple[float, float]) -> None:
    position, rotation = object.get_world_pose()
    current_rotation = R.from_quat(rotation[[1, 2, 3, 0]])
    angle = np.random.uniform(*angle_range)
    z_rotation = R.from_euler("z", angle, degrees=True)
    new_rotation = z_rotation * current_rotation
    new_quat = new_rotation.as_quat()[[3, 0, 1, 2]]
    object.set_world_pose(position, new_quat)


def place_object_between_object1_and_object2(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    object_uid: str,
    object1_uid: str,
    object2_uid: str,
    platform_uid: str,
    attemps: int = 100,
) -> int:
    meshlist = get_current_meshList(object_list, meshDict)
    pointcloud_list = meshlist_to_pclist(meshlist)
    line_points = sample_point_in_2d_line(
        compute_mesh_center(meshlist[object1_uid]),
        compute_mesh_center(meshlist[object2_uid]),
        1000,
    )
    object_bottom_point = pointcloud_list[object_uid][
        np.argmin(pointcloud_list[object_uid][:, 2])
    ]
    vec_axis2bottom = object_list[object_uid].get_world_pose()[0] - object_bottom_point
    available_area = get_platform_available_area(
        pointcloud_list[platform_uid],
        pointcloud_list,
        [platform_uid, object_uid],
    )
    available_area = available_area.buffer(
        -get_max_distance_to_polygon(
            get_xy_contour(pointcloud_list[object_uid]),
            Point(object_bottom_point[0], object_bottom_point[1]),
        )
    )
    for _ in range(attemps):
        random_point = Point(random.choice(line_points))
        if available_area.contains(random_point):
            platform_pc = pointcloud_list[platform_uid]
            position = vec_axis2bottom + np.array(
                [random_point.x, random_point.y, np.max(platform_pc[:, 2])]
            )
            object_list[object_uid].set_world_pose(position=position)
            return 0
    return -1


def place_object_in_object(
    object_list: dict[str, XFormPrim],
    meshDict: dict,
    object_uid: str,
    container_uid: str,
) -> int:
    meshlist = get_current_meshList(object_list, meshDict)
    container_mesh = meshlist[container_uid]
    points = sample_points_in_convex_hull(container_mesh, 1000)
    object_trans, _ = object_list[object_uid].get_world_pose()
    object_center = compute_mesh_center(meshlist[object_uid])
    trans_vector = object_trans - object_center
    for point in points:
        target_trans = point + trans_vector
        object_list[object_uid].set_world_pose(position=target_trans)
        meshlist = get_current_meshList(object_list, meshDict)
        if check_mesh_collision(meshlist[object_uid], meshlist[container_uid]):
            continue
        pclist = meshlist_to_pclist(meshlist)
        relation = get_related_position(pclist[object_uid], pclist[container_uid])
        if relation == "in":
            return 0
    return -1


def place_object_to_object_by_relation(
    object1_uid: str,
    object2_uid: str,
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    relation: str,
    platform_uid: str | None = None,
    extra_erosion: float = 0.00,
    another_object2_uid: str | None = None,  # for "between" relation
    ignored_uid: list[str] = [],
    debug: bool = False,
    fixed_position: bool = False,
    mesh_top_only: bool = False,
) -> int:
    object1 = object_list[object1_uid]
    mesh_list = get_current_meshList(object_list, meshDict)
    pointcloud_list = meshlist_to_pclist(mesh_list)
    combined_cloud = []
    for key in pointcloud_list:
        combined_cloud.append(pointcloud_list[key])
    combined_cloud = np.vstack(combined_cloud)
    ignored_uid_ = copy.deepcopy(ignored_uid)
    if platform_uid is not None:
        ignored_uid_.extend([object1_uid, object2_uid, platform_uid])
        available_area = get_platform_available_area(
            pointcloud_list[platform_uid],
            pointcloud_list,
            ignored_uid_,
        ).buffer(-extra_erosion)
    else:
        available_area = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    if relation == "on" or relation == "top":
        IS_OK = randomly_place_object_on_object(
            pointcloud_list[object1_uid],
            combined_cloud if not mesh_top_only else pointcloud_list[object2_uid],
            object1,
            available_polygon=get_xy_contour(
                pointcloud_list[object2_uid], contour_type="concave_hull"
            ),
            collider_polygon=available_area,
            fixed_position=fixed_position,
            mesh_top_only=mesh_top_only,
        )
    elif relation == "near":
        if platform_uid is None:
            raise ValueError("platform_uid is required for near relation")
        near_area = compute_near_area(mesh_list[object1_uid], mesh_list[object2_uid])
        if debug:
            visualize_polygons(
                [
                    near_area,
                ]
                + [
                    get_xy_contour(pcd, contour_type="concave_hull")
                    for pcd in pointcloud_list.values()
                ]
            )
        IS_OK = randomly_place_object_on_object(
            pointcloud_list[object1_uid],
            combined_cloud,
            object1,
            available_polygon=near_area.intersection(
                get_xy_contour(
                    pointcloud_list[platform_uid], contour_type="convex_hull"
                )
            ),
            collider_polygon=available_area,
            fixed_position=fixed_position,
        )
    elif (
        relation == "left"
        or relation == "right"
        or relation == "front"
        or relation == "back"
    ):
        if platform_uid is None:
            raise ValueError("platform_uid is required for near relation")
        place_area = compute_lrfb_area(
            relation, mesh_list[object1_uid], mesh_list[object2_uid]
        )
        near_area = compute_near_area(mesh_list[object1_uid], mesh_list[object2_uid])
        place_area = place_area.intersection(near_area)
        if debug:
            visualize_polygons(
                [
                    place_area,
                    near_area,
                ]
                + [
                    get_xy_contour(pcd, contour_type="concave_hull")
                    for pcd in pointcloud_list.values()
                ]
            )
        IS_OK = randomly_place_object_on_object(
            pointcloud_list[object1_uid],
            combined_cloud,
            object1,
            available_polygon=place_area.intersection(
                get_xy_contour(
                    pointcloud_list[platform_uid], contour_type="convex_hull"
                )
            ),
            collider_polygon=available_area,
            fixed_position=fixed_position,
        )
    elif relation == "in":
        IS_OK = place_object_in_object(object_list, meshDict, object1_uid, object2_uid)
    elif relation == "between":
        if platform_uid is None:
            raise ValueError("platform_uid is required for between relation")
        if another_object2_uid is None:
            raise ValueError("another_object2_uid is required for between relation")
        IS_OK = place_object_between_object1_and_object2(
            object_list,
            meshDict,
            object1_uid,
            object2_uid,
            another_object2_uid,
            platform_uid,
        )
    else:
        IS_OK = -1
    if IS_OK == -1:
        return -1
    pclist = get_current_pcList_by_meshList(object_list, meshDict)
    if relation != "between":
        subgoal = {
            "obj1_uid": object1_uid,
            "obj2_uid": object2_uid,
            "position": relation,
        }
        finished = check_subgoal_finished_rigid(
            relation, pclist[object1_uid], pclist[object2_uid]
        )
    else:
        if another_object2_uid is None:
            raise ValueError("another_object2_uid is required for between relation")
        subgoal = {
            "obj1_uid": object1_uid,
            "obj2_uid": object2_uid,
            "position": relation,
            "another_obj2_uid": another_object2_uid,
        }
        finished = check_subgoal_finished_rigid(
            relation,
            pclist[object1_uid],
            pclist[object2_uid],
            pclist[another_object2_uid],
        )
    if finished or fixed_position:
        return 0
    else:
        return -1


def rotate_quaternion_z(quat: np.ndarray, angle_rad: float) -> np.ndarray:
    r = R.from_quat(quat[[1, 2, 3, 0]])
    r_z = R.from_euler("z", angle_rad)
    return (r_z * r).as_quat()[[3, 0, 1, 2]]


def randomly_place_object_on_object(
    object1_pc: np.ndarray,
    object2_pc: np.ndarray,
    object1: XFormPrim,
    available_polygon: BaseGeometry = Polygon(
        [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    ),
    collider_polygon: BaseGeometry = Polygon(
        [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    ),
    fixed_position: bool = False,
    mesh_top_only: bool = False,
    allow_rotation_fallback: bool = True,
) -> int:
    object1_polygon = get_xy_contour(object1_pc, contour_type="concave_hull")
    object1_pc_bottom = object1_pc[np.argmin(object1_pc[:, 2])][2]
    object2_polygon = get_xy_contour(object2_pc, contour_type="concave_hull")
    object1_pcd = o3d.geometry.PointCloud()
    object1_pcd.points = o3d.utility.Vector3dVector(object1_pc)
    object1_bbox = compute_pcd_bbox(object1_pcd)
    obj1_2d_bbox = [
        object1_bbox.get_min_bound()[0],
        object1_bbox.get_min_bound()[1],
        object1_bbox.get_max_bound()[0] - object1_bbox.get_min_bound()[0],
        object1_bbox.get_max_bound()[1] - object1_bbox.get_min_bound()[1],
    ]
    object2_polygon = object2_polygon.intersection(available_polygon).intersection(
        collider_polygon
    )
    object1_center = object1.get_world_pose()[0][:2]
    if fixed_position:
        valid_placements = find_fixed_polygon_placement(
            object2_polygon, object1_polygon
        )
    else:
        valid_placements = find_polygon_placement(
            object2_polygon, object1_polygon, 10000
        )
        if len(valid_placements) == 0 and allow_rotation_fallback:
            valid_placements = find_polygon_placement_with_rotation(
                object2_polygon, object1_polygon, object1_center, 10000
            )
    if len(valid_placements) == 0:
        _dump_polygon_debug(
            [available_polygon, collider_polygon, object2_polygon, object1_polygon],
            reason="no_valid_placement",
        )
        return -1
    translation, angle = valid_placements[-1]
    position, orientation = object1.get_world_pose()
    position[:2] += translation
    if mesh_top_only:
        cropped_object2_pc = object2_pc
    else:
        updated_obj1_2d_bbox = obj1_2d_bbox
        bbox_buffer_x = 0.05 * updated_obj1_2d_bbox[2]
        bbox_buffer_y = 0.05 * updated_obj1_2d_bbox[3]
        updated_obj1_2d_bbox[0] += translation[0] - bbox_buffer_x
        updated_obj1_2d_bbox[1] += translation[1] - bbox_buffer_y
        updated_obj1_2d_bbox[2] += bbox_buffer_x * 2
        updated_obj1_2d_bbox[3] += bbox_buffer_y * 2
        cropped_object2_pc = object2_pc[
            np.where(
                (object2_pc[:, 0] >= updated_obj1_2d_bbox[0])
                & (
                    object2_pc[:, 0]
                    <= updated_obj1_2d_bbox[0] + updated_obj1_2d_bbox[2]
                )
                & (object2_pc[:, 1] >= updated_obj1_2d_bbox[1])
                & (
                    object2_pc[:, 1]
                    <= updated_obj1_2d_bbox[1] + updated_obj1_2d_bbox[3]
                )
            )
        ]
        if len(cropped_object2_pc) == 0:
            _dump_polygon_debug(
                [
                    available_polygon,
                    collider_polygon,
                    object2_polygon,
                    object1_polygon,
                ],
                reason="empty_top_surface",
            )
            return -1
    object2_pc_top = cropped_object2_pc[np.argmax(cropped_object2_pc[:, 2])][2]
    object1_to_object2_axis2 = object2_pc_top - object1_pc_bottom
    position[2] += object1_to_object2_axis2
    orientation = rotate_quaternion_z(orientation, angle)
    object1.set_world_pose(position=position, orientation=orientation)
    return 0


def setup_random_tableset_by_centric_range(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    centric_random_range: LayoutConfig,
    background_objects: list[str],
    partial_ignore: dict[str, list[str]] = {},
) -> int:
    for key in object_list:
        if (
            key == "00000000000000000000000000000000"
            or key == "defaultGroundPlane"
            or key in background_objects
        ):
            continue
        if centric_random_range.angle_bilateral:
            rotate_object_around_z(
                object_list[key],
                (
                    -centric_random_range.angle,
                    centric_random_range.angle,
                ),
            )
        else:
            rotate_object_around_z(object_list[key], (0, centric_random_range.angle))
        meshlist = get_current_meshList(object_list, meshDict)
        aabb = compute_mesh_bbox(meshlist[key])
        available_polygon = bbox_to_polygon(
            aabb.get_min_bound()[0] - centric_random_range.w / 2,
            aabb.get_min_bound()[1] - centric_random_range.h / 2,
            aabb.get_max_bound()[0] - aabb.get_min_bound()[0] + centric_random_range.w,
            aabb.get_max_bound()[1] - aabb.get_min_bound()[1] + centric_random_range.h,
        )
        pclist = meshlist_to_pclist(meshlist)
        ignored_uid_list = [] if key not in partial_ignore else partial_ignore[key]
        available_area = get_platform_available_area(
            pclist["00000000000000000000000000000000"],
            pclist,
            [key, "00000000000000000000000000000000"] + ignored_uid_list,
        )
        IS_OK = randomly_place_object_on_object(
            pclist[key],
            pclist["00000000000000000000000000000000"],
            object_list[key],
            available_polygon=available_polygon,
            collider_polygon=available_area,
        )
        if IS_OK == -1:
            return -1
    return 0


def setup_random_custom_tableset(
    object_list: dict[str, XFormPrim],
    articulation_list: dict[str, Articulation],
    meshDict: dict[str, MeshInfo],
    custom_tableset_config: dict[str, dict] | list[dict[str, dict]],
    in_order: bool = False,
    ignored_objects: list[str] = [],
) -> int:
    if isinstance(custom_tableset_config, list):
        custom_tableset_config = random.choice(custom_tableset_config)
    custom_tableset_config_keys = list(custom_tableset_config.keys())
    if not in_order:
        random.shuffle(custom_tableset_config_keys)
    for key in custom_tableset_config:
        if custom_tableset_config[key]["type"] in ("centric_range", "local_jitter"):
            if "position" in custom_tableset_config[key]:
                object_list[key].set_world_pose(
                    position=custom_tableset_config[key]["position"]
                )
            continue
        if custom_tableset_config[key].get("keep_original_orientation", False):
            object_list[key].set_world_pose(position=[10.0, 0.0, 0.0])
        else:
            object_list[key].set_world_pose(
                position=[10.0, 0.0, 0.0],
                orientation=[0.5, 0.5, 0.5, 0.5],
            )
    for key in custom_tableset_config_keys:
        if key not in object_list and key not in articulation_list:
            continue
        if key in object_list:
            if custom_tableset_config[key]["type"] == "local_jitter":
                # Perturb a prearranged pose without reprojecting its height onto a table.
                # This mode does not perform footprint packing; use only for local noise.
                w = float(custom_tableset_config[key]["w"])
                h = float(custom_tableset_config[key]["h"])
                if not np.isfinite([w, h]).all() or min(w, h) < 0:
                    raise ValueError("local_jitter w/h must be finite and nonnegative")
                position, orientation = object_list[key].get_world_pose()
                position = np.array(position, dtype=float, copy=True)
                position[:2] += [
                    random.uniform(-w / 2, w / 2),
                    random.uniform(-h / 2, h / 2),
                ]
                object_list[key].set_world_pose(position, orientation)
                continue
            additional_height = custom_tableset_config[key].get("additional_height", 0)
            buffer_size = custom_tableset_config[key].get("buffer_size", 0.0)
            if custom_tableset_config[key].get("reset_orientation", False):
                object_list[key].set_world_pose(
                    orientation=custom_tableset_config[key].get(
                        "orientation", [0.5, 0.5, 0.5, 0.5]
                    ),
                )
            if custom_tableset_config[key]["type"] == "centric_range":
                centric_random_range = custom_tableset_config[key]
                skip_on_table_check = centric_random_range.get(
                    "skip_on_table_check", False
                )
                if centric_random_range["angle_bilateral"]:
                    rotate_object_around_z(
                        object_list[key],
                        (
                            -centric_random_range["angle"],
                            centric_random_range["angle"],
                        ),
                    )
                else:
                    rotate_object_around_z(
                        object_list[key], (0, centric_random_range["angle"])
                    )
                meshlist = get_current_meshList(object_list, meshDict)
                aabb = compute_mesh_bbox(meshlist[key])
                if skip_on_table_check:
                    available_polygon = bbox_to_polygon(
                        aabb.get_min_bound()[0] - centric_random_range["w"] / 2,
                        aabb.get_min_bound()[1] - centric_random_range["h"] / 2,
                        aabb.get_max_bound()[0]
                        - aabb.get_min_bound()[0]
                        + centric_random_range["w"],
                        aabb.get_max_bound()[1]
                        - aabb.get_min_bound()[1]
                        + centric_random_range["h"],
                    )
                    random_point = None
                    min_x, min_y, max_x, max_y = available_polygon.bounds
                    for _ in range(1000):
                        point = Point(
                            random.uniform(min_x, max_x),
                            random.uniform(min_y, max_y),
                        )
                        if available_polygon.contains(point):
                            random_point = point
                            break
                    if random_point is None:
                        return -1
                    current_pose = object_list[key].get_world_pose()
                    current_pose[0][0] = random_point.x
                    current_pose[0][1] = random_point.y
                    current_pose[0][2] += additional_height
                    object_list[key].set_world_pose(current_pose[0], current_pose[1])
                    continue
                else:
                    available_polygon = bbox_to_polygon(
                        aabb.get_min_bound()[0] - centric_random_range["w"] / 2,
                        aabb.get_min_bound()[1] - centric_random_range["h"] / 2,
                        aabb.get_max_bound()[0]
                        - aabb.get_min_bound()[0]
                        + centric_random_range["w"],
                        aabb.get_max_bound()[1]
                        - aabb.get_min_bound()[1]
                        + centric_random_range["h"],
                    )
                pclist = meshlist_to_pclist(meshlist)
                available_area = get_platform_available_area(
                    pclist["00000000000000000000000000000000"],
                    pclist,
                    [key, "00000000000000000000000000000000"] + ignored_objects,
                    buffer_size=buffer_size,
                )
                IS_OK = randomly_place_object_on_object(
                    pclist[key],
                    pclist["00000000000000000000000000000000"],
                    object_list[key],
                    available_polygon=available_polygon,
                    collider_polygon=available_area,
                )
                current_pose = object_list[key].get_world_pose()
                current_pose[0][2] += additional_height
                object_list[key].set_world_pose(current_pose[0])
                if IS_OK == -1:
                    return -1
            elif custom_tableset_config[key]["type"] == "global_range":
                global_range = custom_tableset_config[key]
                available_polygon = bbox_to_polygon(
                    global_range["random_range_x"],
                    global_range["random_range_y"],
                    global_range["random_range_w"],
                    global_range["random_range_h"],
                )
                rotate_object_around_z(
                    object_list[key],
                    (0, global_range["random_range_angle"]),
                )
                pclist = get_current_pcList_by_meshList(object_list, meshDict)
                available_area = get_platform_available_area(
                    pclist["00000000000000000000000000000000"],
                    pclist,
                    [key, "00000000000000000000000000000000"] + ignored_objects,
                    buffer_size=buffer_size,
                )
                IS_OK = randomly_place_object_on_object(
                    pclist[key],
                    pclist["00000000000000000000000000000000"],
                    object_list[key],
                    available_polygon=available_polygon,
                    collider_polygon=available_area,
                    allow_rotation_fallback=global_range.get(
                        "allow_rotation_fallback", True
                    ),
                )
                current_pose = object_list[key].get_world_pose()
                current_pose[0][2] += additional_height
                object_list[key].set_world_pose(current_pose[0])
                if IS_OK == -1:
                    return -1
            elif custom_tableset_config[key]["type"] == "fixed_global_range":
                global_range = custom_tableset_config[key]
                skip_on_table_check = global_range.get("skip_on_table_check", False)
                available_polygon = bbox_to_polygon(-10, -10, 20, 20)
                rotate_object_around_z(
                    object_list[key],
                    (0, global_range["random_range_angle"]),
                )
                if not skip_on_table_check:
                    pclist = get_current_pcList_by_meshList(object_list, meshDict)
                    available_area = get_platform_available_area(
                        pclist["00000000000000000000000000000000"],
                        pclist,
                        [key, "00000000000000000000000000000000"] + ignored_objects,
                        buffer_size=buffer_size,
                    )
                    IS_OK = randomly_place_object_on_object(
                        pclist[key],
                        pclist["00000000000000000000000000000000"],
                        object_list[key],
                        available_polygon=available_polygon,
                        collider_polygon=available_area,
                    )
                else:
                    IS_OK = 0
                # TODO: didnt check collision
                current_pose = object_list[key].get_world_pose()
                current_pose[0][0] = global_range["pos_x"]
                current_pose[0][1] = global_range["pos_y"]
                current_pose[0][2] += additional_height
                current_pose[0][2] = global_range.get("pos_z", current_pose[0][2])
                object_list[key].set_world_pose(current_pose[0])
                if IS_OK == -1:
                    return -1
            elif custom_tableset_config[key]["type"] == "scene_graph":
                if custom_tableset_config[key]["relation"] == "near":
                    graph_config = custom_tableset_config[key]
                    mesh_list = get_current_meshList(object_list, meshDict)
                    pointcloud_list = meshlist_to_pclist(mesh_list)
                    object1_uid = key
                    object2_uid = custom_tableset_config[key]["obj2_uid"]
                    platform_uid = "00000000000000000000000000000000"
                    object1 = object_list[object1_uid]
                    combined_cloud = []
                    for ky in pointcloud_list:
                        combined_cloud.append(pointcloud_list[ky])
                    combined_cloud = np.vstack(combined_cloud)
                    near_area = compute_near_area(
                        mesh_list[object1_uid],
                        mesh_list[object2_uid],
                        near_distance=graph_config["near_distance"],
                    )
                    limited_area = bbox_to_polygon(
                        graph_config["random_range_x"],
                        graph_config["random_range_y"],
                        graph_config["random_range_w"],
                        graph_config["random_range_h"],
                    )
                    near_area = near_area.intersection(limited_area)
                    available_area = get_platform_available_area(
                        pointcloud_list["00000000000000000000000000000000"],
                        pointcloud_list,
                        [key, "00000000000000000000000000000000"] + ignored_objects,
                        buffer_size=buffer_size,
                    )
                    IS_OK = randomly_place_object_on_object(
                        pointcloud_list[object1_uid],
                        combined_cloud,
                        object1,
                        available_polygon=near_area.intersection(
                            get_xy_contour(
                                pointcloud_list[platform_uid],
                                contour_type="convex_hull",
                            )
                        ),
                        collider_polygon=available_area,
                    )
                    if IS_OK == -1:
                        return -1
            elif custom_tableset_config[key]["type"] == "fixed_random_global_range":
                global_range = custom_tableset_config[key]
                available_polygon = bbox_to_polygon(-10, -10, 20, 20)
                rotate_object_around_z(
                    object_list[key],
                    (0, global_range["random_range_angle"]),
                )
                pclist = get_current_pcList_by_meshList(object_list, meshDict)
                available_area = get_platform_available_area(
                    pclist["00000000000000000000000000000000"],
                    pclist,
                    [key, "00000000000000000000000000000000"] + ignored_objects,
                    buffer_size=buffer_size,
                )
                IS_OK = randomly_place_object_on_object(
                    pclist[key],
                    pclist["00000000000000000000000000000000"],
                    object_list[key],
                    available_polygon=available_polygon,
                    collider_polygon=available_area,
                )
                # TODO: didnt check collision
                current_pose = object_list[key].get_world_pose()
                current_pose[0][0] = global_range["pos_x"]
                current_pose[0][1] = global_range["pos_y"]
                current_pose[0][0] += random.uniform(0, global_range["random_range_w"])
                current_pose[0][1] += random.uniform(0, global_range["random_range_h"])
                current_pose[0][2] += additional_height
                object_list[key].set_world_pose(current_pose[0])
                if IS_OK == -1:
                    return -1
        elif key in articulation_list:
            # todo: refine the logic and need more secure
            if custom_tableset_config[key]["type"] == "global_range":
                global_range = custom_tableset_config[key]

                rotate_object_around_z(
                    articulation_list[key],
                    (
                        -global_range["random_range_angle"],
                        global_range["random_range_angle"],
                    ),
                )

                articulation_prim_path = articulation_list[key].prim_path
                articulation_prim = XFormPrim(articulation_prim_path)
                current_pose = articulation_list[key].get_world_pose()
                current_pose[0][0] = global_range["random_range_x"] + random.uniform(
                    -global_range["random_range_w"], global_range["random_range_w"]
                )
                current_pose[0][1] = global_range["random_range_y"] + random.uniform(
                    -global_range["random_range_h"], global_range["random_range_h"]
                )
                articulation_prim.set_world_pose(current_pose[0])
    return 0


def setup_random_all_range(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    random_all_range_config: LayoutConfig,
    background_objects: list[str],
) -> int:
    for key in object_list:
        if (
            key != "defaultGroundPlane"
            and key != "00000000000000000000000000000000"
            and key not in background_objects
        ):
            object_list[key].set_world_pose(
                position=[10.0, 0.0, 0.0],
                orientation=[0.5, 0.5, 0.5, 0.5],
            )
    custom_tableset_config_keys = list(object_list.keys())
    random.shuffle(custom_tableset_config_keys)
    for key in custom_tableset_config_keys:
        if (
            key == "00000000000000000000000000000000"
            or key == "defaultGroundPlane"
            or key in background_objects
        ):
            continue
        global_range = random_all_range_config
        available_polygon = bbox_to_polygon(
            global_range.random_range_x,
            global_range.random_range_y,
            global_range.random_range_w,
            global_range.random_range_h,
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range.random_range_angle),
        )
        pclist = get_current_pcList_by_meshList(object_list, meshDict)
        available_area = get_platform_available_area(
            pclist["00000000000000000000000000000000"],
            pclist,
            [key, "00000000000000000000000000000000"],
        )
        IS_OK = randomly_place_object_on_object(
            pclist[key],
            pclist["00000000000000000000000000000000"],
            object_list[key],
            available_polygon=available_polygon,
            collider_polygon=available_area,
        )
        if IS_OK == -1:
            return -1
    return 0


def setup_scene_graph_placement(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    scene_config: SceneConfig,
) -> int:
    object_list_key = list(object_list.keys())
    object_list_key.remove("00000000000000000000000000000000")
    object_list_key.remove("defaultGroundPlane")
    scene_graph_list = process_scene_graph(scene_config, object_list_key)
    for object_uid in object_list_key:
        object_list[object_uid].set_world_pose(position=[10.0, 0.0, 0.0])
    for edge_list in scene_graph_list:
        if len(edge_list) == 0:
            continue
        meshlist = get_current_meshList(object_list, meshDict)
        pclist = meshlist_to_pclist(meshlist)
        platform_uid = "00000000000000000000000000000000"
        key_uid = None
        for edge in edge_list:
            if edge["position"] == "on" or edge["position"] == "top":
                platform_uid = edge["obj2_uid"]
                key_uid = edge["obj1_uid"]
                break
        available_polygon = get_xy_contour(
            pclist[platform_uid], contour_type="concave_hull"
        )
        for edge in edge_list:
            if edge["position"] != "on" and edge["position"] != "top":
                scene_graph_available_area = compute_lrfb_area(
                    edge["position"],
                    meshlist[edge["obj1_uid"]],
                    meshlist[edge["obj2_uid"]],
                )
                available_polygon = available_polygon.intersection(
                    scene_graph_available_area
                )
        if key_uid is None:
            raise ValueError("key_uid is required for scene graph placement")
        collison_area = get_platform_available_area(
            pclist[platform_uid],
            pclist,
            [key_uid, platform_uid, "00000000000000000000000000000000"],
        )
        IS_OK = randomly_place_object_on_object(
            pclist[key_uid],
            pclist[platform_uid],
            object_list[key_uid],
            available_polygon=available_polygon,
            collider_polygon=collison_area,
        )
        if IS_OK == -1:
            return -1
    return 0


def setup_random_all_range_buffered(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    random_all_range_config: LayoutConfig,
    background_objects: list[str],
    task_data: dict,
    buffer_size: float = 0.05,
) -> int:
    for key in object_list:
        if (
            key != "defaultGroundPlane"
            and key != "00000000000000000000000000000000"
            and key not in background_objects
        ):
            object_list[key].set_world_pose(
                position=[10.0, 0.0, 0.0],
                orientation=[0.5, 0.5, 0.5, 0.5],
            )
    obj1_uid_list = [
        task_data["goal"][0][i]["obj1_uid"] for i in range(len(task_data["goal"][0]))
    ]
    obj2_uid_list = [
        task_data["goal"][0][i]["obj2_uid"] for i in range(len(task_data["goal"][0]))
    ]
    background_uid_list = [
        uid
        for uid in object_list
        if uid not in list(set(obj1_uid_list + obj2_uid_list))
    ]
    random.shuffle(obj1_uid_list)
    for key in obj1_uid_list:
        global_range = random_all_range_config
        available_polygon = bbox_to_polygon(
            global_range.random_range_x,
            global_range.random_range_y,
            global_range.random_range_w,
            global_range.random_range_h,
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range.random_range_angle),
        )
        pclist = get_current_pcList_by_meshList(object_list, meshDict)
        available_area = get_platform_available_area(
            pclist["00000000000000000000000000000000"],
            pclist,
            [key, "00000000000000000000000000000000"],
        )
        for obj1_uid in obj1_uid_list:
            available_area = available_area.difference(
                get_xy_contour(pclist[obj1_uid], contour_type="concave_hull").buffer(
                    buffer_size
                )
            )
        for obj2_uid in obj2_uid_list:
            available_area = available_area.difference(
                get_xy_contour(pclist[obj2_uid], contour_type="concave_hull").buffer(
                    buffer_size
                )
            )
        IS_OK = randomly_place_object_on_object(
            pclist[key],
            pclist["00000000000000000000000000000000"],
            object_list[key],
            available_polygon=available_polygon,
            collider_polygon=available_area,
        )
        if IS_OK == -1:
            return -1

    random.shuffle(obj2_uid_list)
    for key in obj2_uid_list:
        global_range = random_all_range_config
        available_polygon = bbox_to_polygon(
            global_range.random_range_x,
            global_range.random_range_y,
            global_range.random_range_w,
            global_range.random_range_h,
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range.random_range_angle),
        )
        pclist = get_current_pcList_by_meshList(object_list, meshDict)
        available_area = get_platform_available_area(
            pclist["00000000000000000000000000000000"],
            pclist,
            [key, "00000000000000000000000000000000"],
        )
        for obj1_uid in obj1_uid_list:
            available_area = available_area.difference(
                get_xy_contour(pclist[obj1_uid], contour_type="concave_hull").buffer(
                    buffer_size
                )
            )
        for obj2_uid in obj2_uid_list:
            available_area = available_area.difference(
                get_xy_contour(pclist[obj2_uid], contour_type="concave_hull").buffer(
                    buffer_size
                )
            )
        IS_OK = randomly_place_object_on_object(
            pclist[key],
            pclist["00000000000000000000000000000000"],
            object_list[key],
            available_polygon=available_polygon,
            collider_polygon=available_area,
        )
        if IS_OK == -1:
            return -1
    random.shuffle(background_uid_list)
    for key in background_uid_list:
        if (
            key == "00000000000000000000000000000000"
            or key == "defaultGroundPlane"
            or key in background_objects
        ):
            continue
        global_range = random_all_range_config
        available_polygon = bbox_to_polygon(
            global_range.random_range_x,
            global_range.random_range_y,
            global_range.random_range_w,
            global_range.random_range_h,
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range.random_range_angle),
        )
        pclist = get_current_pcList_by_meshList(object_list, meshDict)
        available_area = get_platform_available_area(
            pclist["00000000000000000000000000000000"],
            pclist,
            [key, "00000000000000000000000000000000"],
        )
        for obj1_uid in obj1_uid_list:
            available_area = available_area.difference(
                get_xy_contour(pclist[obj1_uid], contour_type="concave_hull").buffer(
                    buffer_size
                )
            )
        for obj2_uid in obj2_uid_list:
            available_area = available_area.difference(
                get_xy_contour(pclist[obj2_uid], contour_type="concave_hull").buffer(
                    buffer_size
                )
            )
        IS_OK = randomly_place_object_on_object(
            pclist[key],
            pclist["00000000000000000000000000000000"],
            object_list[key],
            available_polygon=available_polygon,
            collider_polygon=available_area,
        )
        if IS_OK == -1:
            return -1
    return 0


def setup_random_tableset(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    background_objects: list[str],
) -> int:
    for key in object_list:
        if (
            key != "defaultGroundPlane"
            and key != "00000000000000000000000000000000"
            and key not in background_objects
        ):
            object_list[key].set_world_pose(
                position=[10.0, 0.0, 0.0],
                orientation=[0.5, 0.5, 0.5, 0.5],
            )
    object_list_key = list(object_list.keys())
    random.shuffle(object_list_key)
    for key in object_list_key:
        if (
            key != "defaultGroundPlane"
            and key != "00000000000000000000000000000000"
            and key not in background_objects
        ):
            rotate_object_around_z(object_list[key], (0, 360))
            pclist = get_current_pcList_by_meshList(object_list, meshDict)
            available_area = get_platform_available_area(
                pclist["00000000000000000000000000000000"],
                pclist,
                [key, "00000000000000000000000000000000"],
            )
            IS_OK = randomly_place_object_on_object(
                pclist[key],
                pclist["00000000000000000000000000000000"],
                object_list[key],
                available_polygon=bbox_to_polygon(-2.0, -2.0, 4.0, 4.0),
                collider_polygon=available_area,
            )
            if IS_OK == -1:
                return -1
    return 0


def setup_random_tableset_buffered(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    background_objects: list[str],
    object_uid: str,
    container_uid: str,
) -> int:
    for key in object_list:
        if (
            key != "defaultGroundPlane"
            and key != "00000000000000000000000000000000"
            and key not in background_objects
        ):
            object_list[key].set_world_pose(position=(1000.0, 0.0, 0.0))
    object_list_key = list(object_list.keys())
    random.shuffle(object_list_key)
    if container_uid in object_list_key:
        object_list_key.remove(container_uid)
        object_list_key.insert(0, container_uid)
    for key in object_list_key:
        if (
            key != "defaultGroundPlane"
            and key != "00000000000000000000000000000000"
            and key not in background_objects
        ):
            rotate_object_around_z(object_list[key], (0, 360))
            pclist = get_current_pcList_by_meshList(object_list, meshDict)
            available_area = get_platform_available_area(
                pclist["00000000000000000000000000000000"],
                pclist,
                [key, "00000000000000000000000000000000"],
            )
            available_area = available_area.difference(
                get_xy_contour(
                    pclist[container_uid], contour_type="concave_hull"
                ).buffer(0.2)
            )
            if key == container_uid or key == object_uid:
                available_area = available_area.intersection(
                    get_xy_contour(
                        pclist["00000000000000000000000000000000"],
                        contour_type="concave_hull",
                    ).buffer(-0.2)
                )
            IS_OK = randomly_place_object_on_object(
                pclist[key],
                pclist["00000000000000000000000000000000"],
                object_list[key],
                available_polygon=bbox_to_polygon(-2.0, -2.0, 4.0, 4.0),
                collider_polygon=available_area,
            )
            if IS_OK == -1:
                return -1
    return 0


def setup_random_obj1_range(
    object_list: dict[str, XFormPrim],
    meshDict: dict[str, MeshInfo],
    task_data: dict,
    obj1_random_range: LayoutConfig,
    world_pose_list: dict[str, tuple[np.ndarray, np.ndarray]],
) -> int:
    task_info = copy.deepcopy(task_data)
    if isinstance(task_info["goal"][0][0]["obj1_uid"], list):
        task_info["goal"][0][0]["obj1_uid"] = task_info["goal"][0][0]["obj1_uid"][0]
    else:
        task_info["goal"][0][0]["obj1_uid"] = task_info["goal"][0][0]["obj1_uid"]
    if task_info["goal"][0][0]["obj1_uid"] in world_pose_list:
        object_list[task_info["goal"][0][0]["obj1_uid"]].set_world_pose(
            *world_pose_list[task_info["goal"][0][0]["obj1_uid"]]
        )
    else:
        object_list[task_info["goal"][0][0]["obj1_uid"]].set_world_pose(
            position=[10.0, 0.0, 0.0],
            orientation=[0.5, 0.5, 0.5, 0.5],
        )
    available = bbox_to_polygon(
        obj1_random_range.random_range_x,
        obj1_random_range.random_range_y,
        obj1_random_range.random_range_w,
        obj1_random_range.random_range_h,
    )
    rotate_object_around_z(
        object_list[task_info["goal"][0][0]["obj1_uid"]],
        (0, obj1_random_range.random_range_angle),
    )
    if obj1_random_range.force_no_check:
        return 0
    pclist = get_current_pcList_by_meshList(object_list, meshDict)
    IS_OK = setup_target_scene_by_polygon(object_list, pclist, task_info, available)
    return IS_OK


def setup_target_scene_by_polygon(
    object_list: dict[str, XFormPrim],
    pointcloud_list: dict[str, np.ndarray],
    data: dict,
    polygon: Polygon,
) -> int:
    collider_area = get_platform_available_area(
        pointcloud_list["00000000000000000000000000000000"],
        pointcloud_list,
        [
            data["goal"][0][0]["obj1_uid"],
            # data["goal"][0][0]["obj2_uid"],
            "00000000000000000000000000000000",
        ],
    )
    IS_OK = randomly_place_object_on_object(
        pointcloud_list[data["goal"][0][0]["obj1_uid"]],
        pointcloud_list["00000000000000000000000000000000"],
        object_list[data["goal"][0][0]["obj1_uid"]],
        polygon,
        collider_area,
        # strict=False,
    )
    return IS_OK


def verify_placement(object1: XFormPrim, world: World) -> bool:
    translation, orientation = object1.get_world_pose()
    for _ in range(50):
        world.step(render=False)
    obj1_translation, obj1_orientation = object1.get_world_pose()
    if np.linalg.norm(obj1_translation - translation) > 0.1:
        print(f"translation not correct, {obj1_translation} vs {translation}")
        return False
    # q and -q encode the same rotation; PhysX may switch hemisphere at rest.
    orientation_delta = min(
        np.linalg.norm(obj1_orientation - orientation),
        np.linalg.norm(obj1_orientation + orientation),
    )
    if orientation_delta > 1:
        print(f"orientation not correct, {obj1_orientation} vs {orientation}")
        return False
    return True
