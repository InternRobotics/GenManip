"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from pxr import Usd  # type: ignore


@dataclass(frozen=True)
class UsdLocalTransform:
    translation: np.ndarray
    scale: np.ndarray
    quat_wxyz: np.ndarray
    rotation_matrix: np.ndarray


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return quat_wxyz[[1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=float)
    return quat_xyzw[[3, 0, 1, 2]]


def quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_matrix()


def decompose_affine_transform(
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
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    return translation, scale, quat_wxyz, rotation_matrix


def resolve_prim_local_transform(prim: Usd.Prim) -> UsdLocalTransform:
    # Baked meshes can retain inactive xformOp attributes from their source layer.
    # An explicitly empty order means identity, regardless of those attributes.
    order_attr = prim.GetAttribute("xformOpOrder")
    order = order_attr.Get() if order_attr else None
    if order is not None and len(order) == 0:
        return UsdLocalTransform(
            translation=np.zeros(3, dtype=float),
            scale=np.ones(3, dtype=float),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3, dtype=float),
        )

    translation_attr = prim.GetAttribute("xformOp:translate").Get()
    scale_attr = prim.GetAttribute("xformOp:scale").Get()
    scale_units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
    orient_attr = prim.GetAttribute("xformOp:orient").Get()
    rotate_xyz = prim.GetAttribute("xformOp:rotateXYZ").Get()
    rotate_zyx = prim.GetAttribute("xformOp:rotateZYX").Get()
    rotate_x_units_resolve = prim.GetAttribute("xformOp:rotateX:unitsResolve").Get()

    has_explicit_trs = any(
        attr is not None
        for attr in (
            translation_attr,
            scale_attr,
            scale_units_resolve,
            orient_attr,
            rotate_xyz,
            rotate_zyx,
            rotate_x_units_resolve,
        )
    )

    if not has_explicit_trs:
        transform_attr = prim.GetAttribute("xformOp:transform").Get()
        if transform_attr is not None:
            matrix = np.array(transform_attr, dtype=float).T
            translation, scale, quat_wxyz, rotation_matrix = decompose_affine_transform(
                matrix
            )
            return UsdLocalTransform(
                translation=translation,
                scale=scale,
                quat_wxyz=quat_wxyz,
                rotation_matrix=rotation_matrix,
            )

    if translation_attr is None:
        translation = np.zeros(3, dtype=float)
    else:
        translation = np.asarray(translation_attr, dtype=float)

    if scale_attr is None:
        scale = np.ones(3, dtype=float)
    else:
        scale = np.asarray(scale_attr, dtype=float)

    # if scale_units_resolve is not None:
    #     scale = scale * np.asarray(scale_units_resolve, dtype=float)

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
        quat_wxyz = quat_xyzw_to_wxyz(R.from_euler(seq, angles, degrees=True).as_quat())
    else:
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # if rotate_x_units_resolve is not None:
    #     euler = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_euler("xyz", degrees=True)
    #     euler[0] += float(rotate_x_units_resolve)
    #     quat_wxyz = quat_xyzw_to_wxyz(
    #         R.from_euler("xyz", euler, degrees=True).as_quat()
    #     )

    rotation_matrix = quat_wxyz_to_rotation_matrix(quat_wxyz)
    return UsdLocalTransform(
        translation=translation,
        scale=scale,
        quat_wxyz=quat_wxyz,
        rotation_matrix=rotation_matrix,
    )


def apply_local_transform_to_points(
    points: list[np.ndarray], transform: UsdLocalTransform
) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    points_array = np.asarray(points, dtype=float)
    transformed = (
        transform.rotation_matrix @ (points_array * transform.scale).T
    ).T + transform.translation
    return [p for p in transformed]
