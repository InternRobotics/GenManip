"""
We recommend using a independent conda environment to run this script.

conda create -n usd python=3.10
conda activate usd
pip install usd-core==24.11
pip install open3d numpy shapely
"""

import json
import numpy as np
import open3d as o3d
import os
from pxr import Usd, UsdGeom, Sdf  # type: ignore
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from tqdm import tqdm


def move_prim(stage: Usd.Stage, old_path: str, new_path: str):
    old_path = Sdf.Path(old_path)
    new_path = Sdf.Path(new_path)
    old_prim = stage.GetPrimAtPath(old_path)
    if not old_prim or not old_prim.IsValid():
        print(f"[Error] Prim at {old_path} does not exist.")
        return
    if stage.GetPrimAtPath(new_path).IsValid():
        print(f"[Error] Prim at {new_path} already exists.")
        return
    layer = stage.GetEditTarget().GetLayer()
    if not Sdf.CopySpec(layer, old_path, layer, new_path):
        print(f"[Error] Failed to copy spec from {old_path} to {new_path}")
        return
    stage.RemovePrim(old_path)


def add_prim_from_asset(stage: Usd.Stage, prim_path: str, asset_path: str):
    prim_path = Sdf.Path(prim_path)
    if stage.GetPrimAtPath(prim_path).IsValid():
        print(f"[Error] Prim at {prim_path} already exists.")
        return
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_path)


def save_stage_as(stage: Usd.Stage, new_path: str):
    new_layer = Sdf.Layer.CreateNew(new_path)
    if not new_layer:
        return
    stage.Export(new_path)


def delete_prim(stage: Usd.Stage, prim_path: str):
    prim_path = Sdf.Path(prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def print_prim_tree(prim, prefix="", is_last=True):
    RESET = "\033[0m"
    COLORS = {
        "Xform": "\033[94m",
        "Mesh": "\033[95m",
        "Camera": "\033[93m",
        "Light": "\033[96m",
        "Default": "\033[92m",
        "Type": "\033[91m",
    }
    connector = "└── " if is_last else "├── "
    prim_name = prim.GetName()
    prim_type = prim.GetTypeName()
    color = COLORS.get(prim_type, COLORS["Default"])
    color_type = COLORS["Type"]
    colored_name = f"{color}{prim_name}{RESET}"
    colored_type = f"{color_type} ({prim_type}){RESET}" if prim_type else ""
    print(prefix + connector + colored_name + colored_type)
    children = prim.GetChildren()
    count = len(children)
    for i, child in enumerate(children):
        is_last_child = i == count - 1
        extension = "    " if is_last else "│   "
        print_prim_tree(child, prefix + extension, is_last_child)


def _decompose_affine_transform(
    transform_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(transform_matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {matrix.shape}")

    translation = matrix[:3, 3].copy()
    linear = matrix[:3, :3].copy()
    scale = np.linalg.norm(linear, axis=0)

    rotation_matrix = np.eye(3, dtype=float)
    non_zero = np.abs(scale) > 1e-12
    if np.any(non_zero):
        rotation_matrix[:, non_zero] = linear[:, non_zero] / scale[non_zero]

    if np.linalg.det(rotation_matrix) < 0 and np.any(non_zero):
        axis_idx = int(np.argmax(np.abs(scale)))
        rotation_matrix[:, axis_idx] *= -1.0
        scale[axis_idx] *= -1.0

    return translation, scale, rotation_matrix


def _resolve_local_transform(
    prim: Usd.Prim,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform_attr = prim.GetAttribute("xformOp:transform").Get()
    if transform_attr is not None:
        matrix = np.array(transform_attr, dtype=float).T
        return _decompose_affine_transform(matrix)

    translation_attr = prim.GetAttribute("xformOp:translate").Get()
    translation = (
        np.zeros(3, dtype=float)
        if translation_attr is None
        else np.asarray(translation_attr, dtype=float)
    )

    scale_attr = prim.GetAttribute("xformOp:scale").Get()
    scale = (
        np.ones(3, dtype=float)
        if scale_attr is None
        else np.asarray(scale_attr, dtype=float)
    )

    scale_units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
    if scale_units_resolve is not None:
        scale = scale * np.asarray(scale_units_resolve, dtype=float)

    orient_attr = prim.GetAttribute("xformOp:orient").Get()
    rotate_xyz = prim.GetAttribute("xformOp:rotateXYZ").Get()
    rotate_zyx = prim.GetAttribute("xformOp:rotateZYX").Get()
    if orient_attr is not None:
        quat_wxyz = np.array(
            [orient_attr.GetReal(), *orient_attr.GetImaginary()], dtype=float
        )
    elif rotate_xyz is not None or rotate_zyx is not None:
        if rotate_xyz is not None:
            seq = "xyz"
            angles = np.asarray(rotate_xyz, dtype=float)
        else:
            seq = "zyx"
            angles = np.asarray(rotate_zyx, dtype=float)
        quat_xyzw = R.from_euler(seq, angles, degrees=True).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    else:
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    rotate_x_units_resolve = prim.GetAttribute("xformOp:rotateX:unitsResolve").Get()
    if rotate_x_units_resolve is not None:
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        euler[0] += float(rotate_x_units_resolve)
        quat_xyzw = R.from_euler("xyz", euler, degrees=True).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    rotation_matrix = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return translation, scale, rotation_matrix


def _apply_local_transform(
    points: list[np.ndarray],
    translation: np.ndarray,
    scale: np.ndarray,
    rotation_matrix: np.ndarray,
) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    points_array = np.asarray(points, dtype=float)
    transformed = (rotation_matrix @ (points_array * scale).T).T + translation
    return [p for p in transformed]


def recursive_parse(prim):
    translation, scale, rotation_matrix = _resolve_local_transform(prim)
    points_total = []
    faceuv_total = []
    normals_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []
    mesh_total = []
    children = prim.GetChildren()
    for child in children:
        points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_list = (
            recursive_parse(child)
        )
        base_num = len(points_total)
        for idx in faceVertexIndices:
            faceVertexIndices_total.append(base_num + idx)
        faceVertexCounts_total += faceVertexCounts
        faceuv_total += faceuv
        normals_total += normals
        points_total += points
        mesh_total += mesh_list
    if prim.IsA(UsdGeom.Mesh) and len(children) == 0:
        mesh_path = str(prim.GetPath()).split("/")[-1]
        if not mesh_path == "SM_Dummy":
            mesh_total.append(mesh_path)
            points = prim.GetAttribute("points").Get()
            normals = prim.GetAttribute("normals").Get()
            faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
            faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
            faceuv = prim.GetAttribute("primvars:st").Get()
            if points is None:
                points = []
            if normals is None:
                normals = []
            if faceVertexCounts is None:
                faceVertexCounts = []
            if faceVertexIndices is None:
                faceVertexIndices = []
            if faceuv is None:
                faceuv = []
            normals = [_ for _ in normals]
            faceVertexCounts = [_ for _ in faceVertexCounts]
            faceVertexIndices = [_ for _ in faceVertexIndices]
            faceuv = [_ for _ in faceuv]
            ps = []
            for p in points:
                x, y, z = p
                p = np.array((x, y, z))
                ps.append(p)
            points = ps
            base_num = len(points_total)
            for idx in faceVertexIndices:
                faceVertexIndices_total.append(base_num + idx)
            faceVertexCounts_total += faceVertexCounts
            faceuv_total += faceuv
            normals_total += normals
            points_total += points
    if prim.IsA(UsdGeom.Cube):
        size_attr = prim.GetAttribute("size").Get()
        half_size = float(size_attr) / 2 if size_attr is not None else 0.5
        points = [
            np.array([-half_size, -half_size, -half_size]),
            np.array([half_size, -half_size, -half_size]),
            np.array([half_size, half_size, -half_size]),
            np.array([-half_size, half_size, -half_size]),
            np.array([-half_size, -half_size, half_size]),
            np.array([half_size, -half_size, half_size]),
            np.array([half_size, half_size, half_size]),
            np.array([-half_size, half_size, half_size]),
        ]
        face_vertex_counts = [4, 4, 4, 4, 4, 4]
        face_vertex_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            4,
            7,
            3,
            1,
            5,
            6,
            2,
            3,
            2,
            6,
            7,
            0,
            1,
            5,
            4,
        ]
        points_total += points
        faceVertexCounts_total += face_vertex_counts
        faceVertexIndices_total += face_vertex_indices
        mesh_total.append(str(prim.GetPath()).split("/")[-1])

    new_points = _apply_local_transform(
        points_total, translation, scale, rotation_matrix
    )
    return (
        new_points,
        faceuv_total,
        normals_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
        mesh_total,
    )


def get_world_mesh_from_prim(prim):
    (
        points,
        faceuv,
        normals,
        faceVertexCounts,
        faceVertexIndices,
        mesh_list,
    ) = recursive_parse(prim)

    return (
        points,
        faceuv,
        normals,
        faceVertexCounts,
        faceVertexIndices,
        mesh_list,
    )


def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def get_mesh_from_prim(prim):
    points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_total = (
        recursive_parse(prim)
    )
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    return mesh


def get_pcd_from_mesh(mesh, num_points=1000):
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def compute_pcd_bbox(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb


def compute_mesh_bbox(mesh):
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_bbox(pcd)


def get_prim_bbox(prim):
    mesh = get_mesh_from_prim(prim)
    return compute_mesh_bbox(mesh)


def bbox_to_polygon(x, y, w, h):
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Polygon(points)


if __name__ == "__main__":
    stage = Usd.Stage.Open(
        os.environ.get("GENMANIP_SCENE_USD", "saved/assets/scene_usds/test_mona_lisa/scene.usd")
    )

    print_prim_tree(stage.GetPseudoRoot())
