"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.

---

Dependencies:

pip install open3d scipy numpy usd-core==24.11
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import numpy as np
import open3d as o3d
import os
from pathlib import Path
from pxr import Usd, UsdGeom, Gf, Sdf  # type: ignore
import random
from scipy.spatial.transform import Rotation as R
import shutil
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = os.environ.get("MESA_SOURCE_ROOT", str(REPO_ROOT / "saved/MesaTask-10K/Layout_info_USD"))
OUTPUT_ROOT = os.environ.get("MESA_OUTPUT_ROOT", str(REPO_ROOT / "tmp/mesa_scenes"))
ASSETS_DIR = os.environ.get("GENMANIP_ASSETS_DIR", str(REPO_ROOT / "saved/assets"))
MAX_WORKERS = 32
NUM_SCENES = 200


def _decompose_affine_transform(
    transform_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return translation, scale, quat_wxyz, rotation_matrix


def _resolve_local_transform(
    prim: Usd.Prim,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform_attr = prim.GetAttribute("xformOp:transform").Get()
    if transform_attr is not None:
        matrix = np.array(transform_attr, dtype=float).T
        translation, scale, quat_wxyz, rotation_matrix = _decompose_affine_transform(
            matrix
        )
        return translation, scale, rotation_matrix

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


def recursive_parse(prim: Usd.Prim) -> tuple[list, list, list, list, list, list]:
    translation, scale, rotation_matrix = _resolve_local_transform(prim)
    points_total = []
    faceuv_total = []
    normals_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []
    mesh_total = []
    if prim.IsA(UsdGeom.Mesh):
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
    elif prim.IsA(UsdGeom.Cube):
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
    else:
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


def get_mesh_from_points_and_faces(
    points, faceVertexCounts, faceVertexIndices
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        elif count == 4:
            face_indices = faceVertexIndices[idx : idx + 4]
            triangles.append([face_indices[0], face_indices[1], face_indices[2]])
            triangles.append([face_indices[0], face_indices[2], face_indices[3]])
        elif count > 4:
            face_indices = faceVertexIndices[idx : idx + count]
            for i in range(1, count - 1):
                triangles.append(
                    [face_indices[0], face_indices[i], face_indices[i + 1]]
                )
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def get_mesh_from_prim(prim: Usd.Prim) -> o3d.geometry.TriangleMesh:
    points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_total = (
        recursive_parse(prim)
    )
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    return mesh


def hash_name(name: str) -> str:
    """
    Generate a hash from the given name.

    Args:
        name: The original name to hash.

    Returns:
        A hashed string.
    """
    return hashlib.md5(name.encode()).hexdigest()[:8]


def translation_transform(xformable: UsdGeom.Xformable, translation: list) -> None:
    """
    Translate the transform of a prim by the given translation.
    """
    xform_api = UsdGeom.XformCommonAPI(xformable)
    current_translation, rotation, scale, pivot, rotation_order = (
        xform_api.GetXformVectors(Usd.TimeCode.Default())
    )
    new_translation = Gf.Vec3d(
        current_translation[0] + translation[0],
        current_translation[1] + translation[1],
        current_translation[2] + translation[2],
    )
    xform_api.SetTranslate(new_translation)


def scale_transform(xformable: UsdGeom.Xformable, scale_factor: float) -> None:
    """
    Scale the transform of a prim by the given factor.

    Args:
        xformable: The xformable prim to scale.
        scale_factor: The scale factor to apply.
    """
    # Get the current transform operations
    xform_api = UsdGeom.XformCommonAPI(xformable)

    # Get current translation, rotation, scale
    translation, rotation, scale, pivot, rotation_order = xform_api.GetXformVectors(
        Usd.TimeCode.Default()
    )

    # Scale the translation
    new_translation = Gf.Vec3d(
        translation[0] * scale_factor,
        translation[1] * scale_factor,
        translation[2] * scale_factor,
    )

    # Scale the scale
    new_scale = Gf.Vec3f(
        scale[0] * scale_factor, scale[1] * scale_factor, scale[2] * scale_factor
    )

    # Set the new transform
    xform_api.SetXformVectors(
        new_translation,
        rotation,
        new_scale,
        pivot,
        rotation_order,
        Usd.TimeCode.Default(),
    )


def rotation_transform_z90(xformable: UsdGeom.Xformable) -> None:
    """
    Rotate the transform of a prim by 90 degrees around Z axis.

    Args:
        xformable: The xformable prim to rotate.
    """
    xform_api = UsdGeom.XformCommonAPI(xformable)
    translation, rotation, scale, pivot, rotation_order = xform_api.GetXformVectors(
        Usd.TimeCode.Default()
    )
    new_translation = Gf.Vec3d(translation[1], -translation[0], translation[2])
    new_rotation = Gf.Vec3f(rotation[0], rotation[1], rotation[2] - 90)
    xform_api.SetXformVectors(
        new_translation,
        new_rotation,
        scale,
        pivot,
        rotation_order,
        Usd.TimeCode.Default(),
    )


def set_prim_transform(
    prim: Usd.Prim, translation: list, orientation: list, scale: list
) -> None:
    """
    Set the transform of a prim.

    Args:
        prim: The prim to set transform for.
        translation: Translation as [x, y, z].
        orientation: Orientation as quaternion [w, x, y, z].
        scale: Scale as [x, y, z].
    """
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return

    xform_api = UsdGeom.XformCommonAPI(xformable)
    quat_scipy = [orientation[1], orientation[2], orientation[3], orientation[0]]
    rotation = R.from_quat(quat_scipy).as_euler("xyz", degrees=True)
    xform_api.SetTranslate(Gf.Vec3d(translation[0], translation[1], translation[2]))
    xform_api.SetRotate(
        Gf.Vec3f(rotation[0], rotation[1], rotation[2]),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    xform_api.SetScale(Gf.Vec3f(scale[0], scale[1], scale[2]))


def process_single_scene(input_usd_path: str) -> None:
    try:
        output_base_path = os.path.join(OUTPUT_ROOT, "mesa_scenes")
        output_usd_path = os.path.join(
            output_base_path,
            str(input_usd_path).replace(SOURCE_ROOT, ""),
        )

        scene_list_path = os.path.join(ASSETS_DIR, "miscs", "scene_list.json")
        Path(os.path.dirname(output_usd_path)).mkdir(parents=True, exist_ok=True)
        original_material_path = os.path.join(
            os.path.dirname(input_usd_path), "textures"
        )
        new_material_path = os.path.join(os.path.dirname(output_usd_path), "textures")
        if os.path.exists(original_material_path) and not os.path.exists(
            new_material_path
        ):
            print("copying materials")
            shutil.copytree(original_material_path, new_material_path)
        with open(scene_list_path, "r") as f:
            scene_list = json.load(f)
        scene_info = random.choice(scene_list)
        FIXED_HEIGHT = 0.0
        os.makedirs(os.path.dirname(output_usd_path), exist_ok=True)

        print("opening input stage")
        stage = Usd.Stage.Open(input_usd_path)
        if not stage:
            print(f"Failed to open stage at {input_usd_path}")
            return
        print("creating output stage")
        output_stage = Usd.Stage.CreateNew(output_usd_path)
        UsdGeom.SetStageUpAxis(output_stage, UsdGeom.GetStageUpAxis(stage))
        UsdGeom.SetStageMetersPerUnit(
            output_stage, UsdGeom.GetStageMetersPerUnit(stage)
        )
        root_prims = stage.GetPseudoRoot().GetChildren()
        root_prim = None
        for prim in root_prims:
            if prim.GetName() == "root":
                root_prim = prim
                break
        if not root_prim:
            print("No 'root' prim found in input stage")
            return
        output_root = output_stage.DefinePrim("/root", "Xform")
        output_stage.SetDefaultPrim(output_root)
        exclude_names = ["Camera", "Light", "env_light"]
        scale_factor = 0.01
        child_prim_list = []
        max_x, max_y, min_x, min_y = (
            float("-inf"),
            float("-inf"),
            float("inf"),
            float("inf"),
        )
        for child in root_prim.GetChildren():
            child_type = child.GetTypeName()
            child_name = child.GetName()
            if child_name in exclude_names:
                print(f"excluding {child_name}")
                continue
            name_hash = hash_name(child_name)
            if child_type == "Xform":
                new_name = f"obj_{name_hash}"
            else:
                new_name = child_name
            new_prim_path = f"/root/{new_name}"
            print(f"copying {child_name} to {new_prim_path}")
            Sdf.CopySpec(
                stage.GetRootLayer(),
                child.GetPath(),
                output_stage.GetRootLayer(),
                Sdf.Path(new_prim_path),
            )
            new_prim = output_stage.GetPrimAtPath(new_prim_path)
            xformable = UsdGeom.Xformable(new_prim)
            if xformable:
                scale_transform(xformable, scale_factor)
                rotation_transform_z90(xformable)
            mesh = get_mesh_from_prim(new_prim)
            mesh_bbox = mesh.get_axis_aligned_bounding_box()
            max_x = max(max_x, mesh_bbox.max_bound[0])
            max_y = max(max_y, mesh_bbox.max_bound[1])
            min_x = min(min_x, mesh_bbox.min_bound[0])
            min_y = min(min_y, mesh_bbox.min_bound[1])
            child_prim_list.append(new_prim)

        print("transforming children")
        for prim in child_prim_list:
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                translation_transform(
                    xformable, [-(max_x + min_x) / 2, -(max_y + min_y) / 2, 0]
                )
        object_half_width_x = (max_x - min_x) / 2
        object_half_width_y = (max_y - min_y) / 2

        print("adding GRScene")
        while True:
            scene_info = random.choice(scene_list)
            room_usd_path = os.path.join(
                ASSETS_DIR, "miscs", scene_info["scene"]["path"]
            )
            room_prim = output_stage.DefinePrim("/root/room", "Xform")
            room_prim.GetReferences().AddReference(room_usd_path)
            room_translation = [
                scene_info["scene"]["translation"][1],
                -scene_info["scene"]["translation"][0],
                FIXED_HEIGHT
                - scene_info["scene"]["translation"][2]
                - 2 * scene_info["table"]["translation"][2],
            ]
            room_orientation = R.from_euler(
                "xyz", [0, 0, scene_info["scene"]["euler"][2] - 90], degrees=True
            ).as_quat()[[3, 0, 1, 2]]
            room_scale = scene_info["scene"]["scale"]
            set_prim_transform(
                room_prim, room_translation, room_orientation, room_scale
            )
            table_usd_path = os.path.join(
                ASSETS_DIR, "miscs", scene_info["table"]["path"]
            )
            table_prim = output_stage.DefinePrim("/root/obj_table", "Xform")
            table_prim.GetReferences().AddReference(table_usd_path)
            table_translation = [
                0,
                0,
                FIXED_HEIGHT
                - scene_info["scene"]["translation"][2]
                - scene_info["table"]["translation"][2],
            ]
            table_orientation = R.from_euler(
                "xyz", [0, 0, -90], degrees=True
            ).as_quat()[[3, 0, 1, 2]]
            table_scale = scene_info["table"]["scale"]
            set_prim_transform(
                table_prim, table_translation, table_orientation, table_scale
            )
            table_mesh = get_mesh_from_prim(table_prim)
            table_bbox = table_mesh.get_axis_aligned_bounding_box()
            if (
                table_bbox.max_bound[0] >= object_half_width_x
                and table_bbox.min_bound[0] <= -object_half_width_x
                and table_bbox.max_bound[1] >= object_half_width_y
                and table_bbox.min_bound[1] <= -object_half_width_y
            ):
                break
            else:
                output_stage.RemovePrim("/root/room")
                output_stage.RemovePrim("/root/obj_table")
        output_stage.GetRootLayer().Save()
        print(f"\nOutput saved to: {output_usd_path}")
    except (IndexError, KeyError, OSError, RuntimeError, ValueError, TypeError) as e:
        print(f"Error: {e}")
        return None


def main() -> None:
    scene_files = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        if "scene.usd" in files:
            scene_files.append(os.path.join(root, "scene.usd"))
    scene_files = random.sample(scene_files, NUM_SCENES)
    print(
        f"Starting batch processing with {MAX_WORKERS} workers and {len(scene_files)} scenes..."
    )
    start_time = time.time()

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(process_single_scene, scene_files):
            completed += 1
            if completed % 10 == 0:
                print(f"[{completed}/{len(scene_files)}] {result}")
    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
