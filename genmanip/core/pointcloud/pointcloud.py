from filelock import SoftFileLock
from copy import deepcopy
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
from pathlib import Path
from genmanip.core.pointcloud.transform import (
    forward_transform_mesh,
    inverse_transform_mesh,
    transform_between_meshes,
    transform_between_point_clouds,
)
from genmanip.core.usd_utils.prim_utils import get_mesh_from_prim
from genmanip.utils.pc_utils import get_pcd_from_mesh

from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_parent, get_prim_path  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore


def get_current_mesh(object: XFormPrim, mesh_dict):
    scale = np.array([1, 1, 1])
    trans, quat = object.get_world_pose()
    quat = quat[[1, 2, 3, 0]]
    return transform_between_meshes(
        mesh_dict["mesh"],
        mesh_dict["scale"],
        mesh_dict["quat"],
        mesh_dict["trans"],
        scale,
        quat,
        trans,
    )


def get_current_meshList(object_list, mesh_dict):
    updatedMeshList = {}
    for key in mesh_dict:
        mesh = get_current_mesh(object_list[key], mesh_dict[key])
        if mesh is not None:
            updatedMeshList[key] = mesh
    return updatedMeshList


def get_current_pointCloud(object: XFormPrim, point_dict):
    scale = np.array([1, 1, 1])
    trans, quat = object.get_world_pose()
    quat = quat[[1, 2, 3, 0]]
    return transform_between_point_clouds(
        point_dict["points"],
        point_dict["scale"],
        point_dict["quat"],
        point_dict["trans"],
        scale,
        quat,
        trans,
    )


def get_current_pointCloutList(object_list, point_dict):
    updatedPointCloudList = {}
    for key in point_dict:
        updatedPointCloudList[key] = get_current_pointCloud(
            object_list[key], point_dict[key]
        )
    return updatedPointCloudList


def get_mesh_info_by_load(object: XFormPrim, mesh_path: str):
    lock = SoftFileLock(mesh_path + "_soft.lock", timeout=600.0)
    try:
        with lock:
            if not os.path.exists(mesh_path):
                if not os.path.exists(mesh_path):
                    Path(mesh_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        mesh = get_mesh_from_prim(object.prim)
                    except:
                        return None
                    scale = object.get_local_scale()
                    trans, quat = object.get_local_pose()
                    quat = quat[[1, 2, 3, 0]]
                    mesh = inverse_transform_mesh(mesh, scale, quat, trans)
                    print(f"save mesh to {mesh_path}")
                    o3d.io.write_triangle_mesh(mesh_path, mesh)
            mesh = o3d.io.read_triangle_mesh(mesh_path)
    except:
        raise Exception(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )
    mesh_info = {}
    mesh_info["mesh"] = get_world_mesh(mesh, object.prim_path)
    mesh_info["trans"], mesh_info["quat"] = object.get_world_pose()
    mesh_info["quat"] = mesh_info["quat"][[1, 2, 3, 0]]
    mesh_info["scale"] = np.array([1, 1, 1])
    return mesh_info


def get_mesh_info(object):
    try:
        mesh = get_mesh_from_prim(object.prim)
    except:
        return None
    scale = object.get_local_scale()
    trans, quat = object.get_local_pose()
    quat = quat[[1, 2, 3, 0]]
    mesh = inverse_transform_mesh(mesh, scale, quat, trans)
    mesh_info = {}
    mesh_info["mesh"] = get_world_mesh(mesh, object.prim_path)
    mesh_info["trans"], mesh_info["quat"] = object.get_world_pose()
    mesh_info["quat"] = mesh_info["quat"][[1, 2, 3, 0]]
    mesh_info["scale"] = np.array([1, 1, 1])
    return mesh_info


def get_world_mesh(mesh, prim_path):
    prim = get_prim_at_path(prim_path)
    mesh = deepcopy(mesh)
    while get_prim_path(prim) != "/":
        scale = np.array(prim.GetAttribute("xformOp:scale").Get())
        trans = np.array(prim.GetAttribute("xformOp:translate").Get())
        quat = prim.GetAttribute("xformOp:orient").Get()
        if quat is not None:
            r = quat.GetReal()
            i, j, k = quat.GetImaginary()
            quat = np.array([i, j, k, r])
        else:
            quat = np.array([0, 0, 0, 1])
        mesh = forward_transform_mesh(mesh, scale, quat, trans)
        prim = get_prim_parent(prim)
    return mesh


def meshDict2pointCloudDict(mesh_dict):
    pointcloud_dict = {}
    for key in mesh_dict:
        pointCloud_info = mesh_info2pointCloud_info(mesh_dict[key])
        if pointCloud_info is not None:
            pointcloud_dict[key] = pointCloud_info
    return pointcloud_dict


def meshlist_to_pclist(meshlist):
    pointcloudlist = {}
    for key in meshlist:
        try:
            if key == "00000000000000000000000000000000":
                pointcloudlist[key] = np.asarray(
                    get_pcd_from_mesh(meshlist[key], num_points=100000).points
                )
            else:
                pointcloudlist[key] = np.asarray(
                    get_pcd_from_mesh(meshlist[key], num_points=10000).points
                )
        except:
            continue
    return pointcloudlist


def mesh_info2pointCloud_info(mesh_info):
    try:
        points = np.asarray(get_pcd_from_mesh(mesh_info["mesh"]).points)
        scale = mesh_info["scale"]
        trans = mesh_info["trans"]
        quat = mesh_info["quat"]
        return {
            "points": points,
            "scale": scale,
            "trans": trans,
            "quat": quat,
        }
    except:
        return None


def objectList2meshList(object_list, mesh_folder_path=None):
    mesh_dict = {}
    for key in tqdm(object_list):
        if key == "defaultGroundPlane":
            continue
        if mesh_folder_path is not None:
            mesh_info = get_mesh_info_by_load(
                object_list[key], os.path.join(mesh_folder_path, f"{key}.obj")
            )
        else:
            mesh_info = get_mesh_info(object_list[key])
        if mesh_info is not None:
            mesh_dict[key] = mesh_info
    return mesh_dict


def objectList2pointCloudList(object_list, visualize=False):
    mesh_dict = objectList2meshList(object_list)
    point_dict = meshDict2pointCloudDict(mesh_dict)
    if visualize:
        all_points = []
        for key in point_dict:
            all_points.append(point_dict[key]["points"])
        all_points = np.vstack(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.visualization.draw_geometries([pcd])
    return point_dict
