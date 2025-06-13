"""
We defined a place relation with the following parameters:
    - object1: the object to be placed
    - object2: the object on which object1 is to be placed
    - platform: the platform on which object1 is to be placed, sometimes when relation is "on/below", the platform is the object2
"""

import copy
import numpy as np
import random
from shapely.geometry import Polygon, Point
from typing import List
from genmanip.utils.pc_utils import compute_near_area
from genmanip.core.pointcloud.pointcloud import get_current_meshList, meshlist_to_pclist
from genmanip.demogen.evaluate.evaluate import get_related_position
from genmanip.utils.pc_utils import (
    check_mesh_collision,
    compute_mesh_center,
    compute_mesh_bbox,
    compute_lrfb_area,
    get_max_distance_to_polygon,
    get_platform_available_area,
    get_xy_contour,
    sample_point_in_2d_line,
    sample_points_in_convex_hull,
    find_polygon_placement,
    find_polygon_placement_with_rotation,
    bbox_to_polygon,
    visualize_polygons,
)
from genmanip.core.random_place.scene_graph_placement import process_scene_graph
from genmanip.utils.pc_utils import compute_pcd_bbox
import open3d as o3d
from genmanip.demogen.evaluate.evaluate import check_subgoal_finished_rigid

from omni.isaac.core.prims import XFormPrim  # type: ignore

from scipy.spatial.transform import Rotation as R


def rotate_object_around_z(object, angle_range):
    position, rotation = object.get_world_pose()
    current_rotation = R.from_quat(rotation[[1, 2, 3, 0]])
    angle = np.random.uniform(*angle_range)
    z_rotation = R.from_euler("z", angle, degrees=True)
    new_rotation = z_rotation * current_rotation
    new_quat = new_rotation.as_quat()[[3, 0, 1, 2]]
    object.set_world_pose(position, new_quat)


def place_object_between_object1_and_object2(
    object_list,
    meshDict,
    object_uid,
    object1_uid,
    object2_uid,
    platform_uid,
    attemps=100,
):
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


def place_object_in_object(object_list, meshDict, object_uid, container_uid):
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
    object_list,
    meshDict,
    relation: str,
    platform_uid: str = None,
    extra_erosion: float = 0.00,
    another_object2_uid: str = None,  # for "between" relation
    ignored_uid: List[str] = [],
    debug: bool = False,
):
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
    object1_pc = pointcloud_list[object1_uid]
    object1_bottom_point = object1_pc[np.argmin(object1_pc[:, 2])]
    object1_xyr = get_max_distance_to_polygon(
        get_xy_contour(pointcloud_list[object1_uid]),
        Point(object1_bottom_point[0], object1_bottom_point[1]),
    )
    if relation == "on" or relation == "top":
        IS_OK = randomly_place_object_on_object(
            pointcloud_list[object1_uid],
            combined_cloud,
            object1,
            available_polygon=get_xy_contour(
                pointcloud_list[object2_uid], contour_type="concave_hull"
            ),
            collider_polygon=available_area,
        )
    elif relation == "near":
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
        )
    elif (
        relation == "left"
        or relation == "right"
        or relation == "front"
        or relation == "back"
    ):
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
        )
    elif relation == "in":
        IS_OK = place_object_in_object(object_list, meshDict, object1_uid, object2_uid)
    elif relation == "between":
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
    meshlist = get_current_meshList(object_list, meshDict)
    pclist = meshlist_to_pclist(meshlist)
    if relation != "between":
        subgoal = {
            "obj1_uid": object1_uid,
            "obj2_uid": object2_uid,
            "position": relation,
        }
        finished = check_subgoal_finished_rigid(
            subgoal, pclist[object1_uid], pclist[object2_uid]
        )
    else:
        subgoal = {
            "obj1_uid": object1_uid,
            "obj2_uid": object2_uid,
            "position": relation,
            "another_obj2_uid": another_object2_uid,
        }
        finished = check_subgoal_finished_rigid(
            subgoal,
            pclist[object1_uid],
            pclist[object2_uid],
            pclist[another_object2_uid],
        )
    if finished:
        return 0
    else:
        return -1


from scipy.spatial.transform import Rotation as R


def rotate_quaternion_z(quat, angle_rad):
    r = R.from_quat(quat[[1, 2, 3, 0]])
    r_z = R.from_euler("z", angle_rad)
    return (r_z * r).as_quat()[[3, 0, 1, 2]]


def randomly_place_object_on_object(
    object1_pc: np.ndarray,
    object2_pc: np.ndarray,
    object1: XFormPrim,
    available_polygon: Polygon = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
    collider_polygon: Polygon = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
):
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
    valid_placements = find_polygon_placement(object2_polygon, object1_polygon, 10000)
    if len(valid_placements) == 0:
        valid_placements = find_polygon_placement_with_rotation(
            object2_polygon, object1_polygon, object1_center, 10000
        )
        if len(valid_placements) == 0:
            return -1
    translation, angle = valid_placements[-1]
    position, orientation = object1.get_world_pose()
    position[:2] += translation
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
            & (object2_pc[:, 0] <= updated_obj1_2d_bbox[0] + updated_obj1_2d_bbox[2])
            & (object2_pc[:, 1] >= updated_obj1_2d_bbox[1])
            & (object2_pc[:, 1] <= updated_obj1_2d_bbox[1] + updated_obj1_2d_bbox[3])
        )
    ]
    if len(cropped_object2_pc) == 0:
        return -1
    object2_pc_top = cropped_object2_pc[np.argmax(cropped_object2_pc[:, 2])][2]
    object1_to_object2_axis2 = object2_pc_top - object1_pc_bottom
    position[2] += object1_to_object2_axis2
    orientation = rotate_quaternion_z(orientation, angle)
    object1.set_world_pose(position=position, orientation=orientation)
    return 0


def setup_random_tableset_by_centric_range(
    object_list, meshDict, centric_random_range, background_objects, partial_ignore={}
):
    for key in object_list:
        if (
            key == "00000000000000000000000000000000"
            or key == "defaultGroundPlane"
            or key in background_objects
        ):
            continue
        if centric_random_range["angle_bilateral"]:
            rotate_object_around_z(
                object_list[key],
                (
                    -centric_random_range["angle"],
                    centric_random_range["angle"],
                ),
            )
        else:
            rotate_object_around_z(object_list[key], (0, centric_random_range["angle"]))
        meshlist = get_current_meshList(object_list, meshDict)
        aabb = compute_mesh_bbox(meshlist[key])
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
    object_list, articulation_list, meshDict, custom_tableset_config
):
    custom_tableset_config_keys = list(custom_tableset_config.keys())
    random.shuffle(custom_tableset_config_keys)
    for key in custom_tableset_config_keys:
        if key not in object_list and key not in articulation_list:
            continue
        if key in object_list:
            if custom_tableset_config[key]["type"] == "centric_range":
                centric_random_range = custom_tableset_config[key]
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
                meshlist = get_current_meshList(object_list, meshDict)
                pclist = meshlist_to_pclist(meshlist)
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


def setup_random_all_range(
    object_list, meshDict, random_all_range_config, background_objects
):
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
            global_range["random_range_x"],
            global_range["random_range_y"],
            global_range["random_range_w"],
            global_range["random_range_h"],
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range["random_range_angle"]),
        )
        meshlist = get_current_meshList(object_list, meshDict)
        pclist = meshlist_to_pclist(meshlist)
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


def setup_scene_graph_placement(
    object_list,
    meshDict,
    demogen_config,
):
    object_list_key = list(object_list.keys())
    object_list_key.remove("00000000000000000000000000000000")
    object_list_key.remove("defaultGroundPlane")
    scene_graph_list = process_scene_graph(demogen_config, object_list_key)
    for object_uid in object_list_key:
        object_list[object_uid].set_world_pose(position=[10.0, 0.0, 0.0])
    for edge_list in scene_graph_list:
        if len(edge_list) == 0:
            continue
        meshlist = get_current_meshList(object_list, meshDict)
        pclist = meshlist_to_pclist(meshlist)
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
    object_list,
    meshDict,
    random_all_range_config,
    background_objects,
    task_data,
    buffer_size=0.05,
):
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
            global_range["random_range_x"],
            global_range["random_range_y"],
            global_range["random_range_w"],
            global_range["random_range_h"],
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range["random_range_angle"]),
        )
        meshlist = get_current_meshList(object_list, meshDict)
        pclist = meshlist_to_pclist(meshlist)
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
            global_range["random_range_x"],
            global_range["random_range_y"],
            global_range["random_range_w"],
            global_range["random_range_h"],
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range["random_range_angle"]),
        )
        meshlist = get_current_meshList(object_list, meshDict)
        pclist = meshlist_to_pclist(meshlist)
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
            global_range["random_range_x"],
            global_range["random_range_y"],
            global_range["random_range_w"],
            global_range["random_range_h"],
        )
        rotate_object_around_z(
            object_list[key],
            (0, global_range["random_range_angle"]),
        )
        meshlist = get_current_meshList(object_list, meshDict)
        pclist = meshlist_to_pclist(meshlist)
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


def setup_random_tableset(object_list, meshDict, background_objects):
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
            meshlist = get_current_meshList(object_list, meshDict)
            pclist = meshlist_to_pclist(meshlist)
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
    object_list, meshDict, background_objects, object_uid, container_uid
):
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
            meshlist = get_current_meshList(object_list, meshDict)
            pclist = meshlist_to_pclist(meshlist)
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
    object_list,
    meshDict,
    task_data,
    obj1_random_range,
    world_pose_list,
):
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
        obj1_random_range["random_range_x"],
        obj1_random_range["random_range_y"],
        obj1_random_range["random_range_w"],
        obj1_random_range["random_range_h"],
    )
    rotate_object_around_z(
        object_list[task_info["goal"][0][0]["obj1_uid"]],
        (0, obj1_random_range["random_range_angle"]),
    )
    meshlist = get_current_meshList(object_list, meshDict)
    pclist = meshlist_to_pclist(meshlist)
    IS_OK = setup_target_scene_by_polygon(object_list, pclist, task_info, available)
    return IS_OK


def setup_target_scene_by_polygon(object_list, pointcloud_list, data, polygon):
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


def verify_placement(object1, world):
    translation, orientation = object1.get_world_pose()
    for _ in range(50):
        world.step(render=False)
    obj1_translation, obj1_orientation = object1.get_world_pose()
    if np.linalg.norm(obj1_translation - translation) > 0.1:
        print(f"translation not correct, {obj1_translation} vs {translation}")
        return False
    if np.linalg.norm(obj1_orientation - orientation) > 1:
        print(f"orientation not correct, {obj1_orientation} vs {orientation}")
        return False
    return True
