import numpy as np
from scipy.spatial.transform import Rotation as R


def adjust_orientation(ori):
    ori = R.from_quat(ori[[1, 2, 3, 0]])
    if ori.apply(np.array([1, 0, 0]))[0] < 0:
        ori = R.from_euler("z", 180, degrees=True) * ori
    return ori.as_quat()[[3, 0, 1, 2]]


def rot_orientation_by_z_axis(ori, angle):
    ori = R.from_quat(ori[[1, 2, 3, 0]])
    ori = R.from_euler("z", angle, degrees=True) * ori
    return ori.as_quat()[[3, 0, 1, 2]]


def adjust_translation_along_quaternion(
    translation, quaternion, distance, aug_distance=0.0
):
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    arbitrary_vector = (
        np.array([1, 0, 0]) if direction_vector[0] == 0 else np.array([0, 1, 0])
    )
    perp_vector1 = np.cross(direction_vector, arbitrary_vector)
    perp_vector2 = np.cross(direction_vector, perp_vector1)
    perp_vector1 /= np.linalg.norm(perp_vector1)
    perp_vector2 /= np.linalg.norm(perp_vector2)
    random_shift = np.random.uniform(-aug_distance, aug_distance, size=2)
    new_translation += random_shift[0] * perp_vector1 + random_shift[1] * perp_vector2
    return new_translation


def compute_final_pose(P_A0, Q_A0, P_B0, Q_B0, P_A1, Q_A1):
    rot_A0 = R.from_quat(Q_A0[[1, 2, 3, 0]])
    rot_A1 = R.from_quat(Q_A1[[1, 2, 3, 0]])
    rot_B0 = R.from_quat(Q_B0[[1, 2, 3, 0]])
    rot_BA = rot_A0.inv() * rot_B0
    t_BA = rot_A0.inv().apply(P_B0 - P_A0)
    P_B1 = rot_A1.apply(t_BA) + P_A1
    rot_B1 = rot_A1 * rot_BA
    Q_B1 = rot_B1.as_quat()[[3, 0, 1, 2]]
    return P_B1, Q_B1


def compute_delta_eepose(pose1, pose2):
    """
    return the delta eepose between two poses: pose1 - pose2
    """
    pose1_transform = pose_to_transform(pose1)
    pose2_transform = pose_to_transform(pose2)
    delta_transform = pose1_transform @ np.linalg.inv(pose2_transform)
    return transform_to_pose(delta_transform)


def transform_to_pose(transform):
    trans = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return trans, quat


def pose_to_transform(pose):
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return transform


def compute_pose2(current_pose1, previous_pose1, previous_pose2):
    current_pose1_transform = pose_to_transform(current_pose1)
    previous_pose1_transform = pose_to_transform(previous_pose1)
    previous_pose2_transform = pose_to_transform(previous_pose2)
    current_pose2_transform = (
        previous_pose2_transform
        @ np.linalg.inv(previous_pose1_transform)
        @ current_pose1_transform
    )
    return transform_to_pose(current_pose2_transform)
