from scipy.spatial.transform import Rotation as R
import numpy as np

def adjust_translation_along_quaternion(translation, quaternion, distance):
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    return new_translation


def pose_to_transform(pose):
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return transform
