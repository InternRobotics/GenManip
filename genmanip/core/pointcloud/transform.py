import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def forward_transform_mesh(mesh, scale_factors, quaternion, translation_vector):
    vertices = mesh.vertices
    vertices = vertices * np.array(scale_factors)
    rotation = R.from_quat(quaternion)
    vertices = rotation.apply(vertices)
    vertices = vertices + translation_vector
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return transformed_mesh


def forward_transform_point_cloud(
    points, scale_factors, quaternion, translation_vector
):
    points = points * np.array(scale_factors)
    rotation = R.from_quat(quaternion)
    points = rotation.apply(points)
    points = points + translation_vector
    return points


def inverse_transform_mesh(mesh, scale_factors, quaternion, translation_vector):
    vertices = mesh.vertices
    vertices = vertices - translation_vector
    rotation = R.from_quat(quaternion)
    vertices = rotation.inv().apply(vertices)
    vertices = vertices / np.array(scale_factors)
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return transformed_mesh


def inverse_transform_point_cloud(
    points, scale_factors, quaternion, translation_vector
):
    points = points - translation_vector
    rotation = R.from_quat(quaternion)
    points = rotation.inv().apply(points)
    points = points / np.array(scale_factors)
    return points


def transform_between_meshes(
    mesh_A, scale_A, quat_A, trans_A, scale_B, quat_B, trans_B
):
    mesh_in_world_frame = inverse_transform_mesh(mesh_A, scale_A, quat_A, trans_A)
    transformed_mesh_B = forward_transform_mesh(
        mesh_in_world_frame, scale_B, quat_B, trans_B
    )
    return transformed_mesh_B


def transform_between_point_clouds(
    points_A, scale_A, quat_A, trans_A, scale_B, quat_B, trans_B
):
    points_in_world_frame = inverse_transform_point_cloud(
        points_A, scale_A, quat_A, trans_A
    )
    transformed_points_B = forward_transform_point_cloud(
        points_in_world_frame, scale_B, quat_B, trans_B
    )

    return transformed_points_B
