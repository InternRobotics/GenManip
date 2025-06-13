import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R


def joint_positions_to_ee_pose_translation_euler(joint_positions):
    ee_pose = rtb.models.Panda().fkine(q=joint_positions, end="panda_hand").A
    translation = ee_pose[:3, 3]
    euler_angles = R.from_matrix(ee_pose[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate([translation, euler_angles])


def joint_positions_to_position_and_orientation(joint_positions):
    ee_pose = rtb.models.Panda().fkine(q=joint_positions, end="panda_hand").A
    translation = ee_pose[:3, 3]
    orientation = R.from_matrix(ee_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return translation, orientation
