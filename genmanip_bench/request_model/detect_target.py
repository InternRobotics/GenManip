import numpy as np
import open3d as o3d
from genmanip.core.usd_utils.prim_utils import get_world_pose_by_prim_path


def triangle_normal(A, B, C):
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    normal = normal / np.linalg.norm(normal)  # 单位化
    return normal


def extrude_triangle(A, B, C, distance):
    normal = triangle_normal(A, B, C)
    A1, B1, C1 = A + distance * normal, B + distance * normal, C + distance * normal
    A2, B2, C2 = A - distance * normal, B - distance * normal, C - distance * normal
    return [A1, B1, C1], [A2, B2, C2]


def is_point_in_triangle(p, A, B, C):
    # 使用重心坐标法
    v0, v1, v2 = C - A, B - A, p - A
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def is_point_in_prism(p, A, B, C, distance=0.2):
    normal = triangle_normal(A, B, C)
    d = np.dot(normal, A)  # 平面常数
    proj = np.dot(normal, p)

    # 检查是否在两平面之间
    if not (d - distance <= proj <= d + distance):
        return False

    # 投影点到三角形平面
    projected_p = p - (proj - d) * normal

    return is_point_in_triangle(projected_p, A, B, C)


def sample_points_in_mesh(
    mesh: o3d.geometry.TriangleMesh,
    voxel_size: float = 0.05,
    samples_per_voxel: int = 5,
    jitter_ratio: float = 0.5,
) -> o3d.geometry.PointCloud:
    """
    从 mesh 的体积（非仅表面）中采样点。

    参数：
        mesh: 输入的封闭 TriangleMesh。
        voxel_size: 用于 voxelization 的体素大小。
        samples_per_voxel: 每个体素中采样的点数。
        jitter_ratio: 每个点 jitter 的范围比例 (0~1)，决定采样点的随机性。

    返回：
        o3d.geometry.PointCloud：采样得到的点云。
    """

    # 创建 voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh, voxel_size=voxel_size
    )

    # 获取所有体素中心
    centers = [
        voxel_grid.get_voxel_center_coordinate(v.grid_index)
        for v in voxel_grid.get_voxels()
    ]
    centers = np.array(centers)

    # jitter 掉每个中心，产生多个点
    def jitter_points(points, voxel_size, jitter_ratio, samples_per_voxel):
        jittered = []
        for p in points:
            for _ in range(samples_per_voxel):
                offset = (np.random.rand(3) - 0.5) * jitter_ratio * voxel_size
                jittered.append(p + offset)
        return np.array(jittered)

    jittered_points = jitter_points(
        centers, voxel_size, jitter_ratio, samples_per_voxel
    )

    # 转成 PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(jittered_points)
    return pcd


def detect_target_is_grasped(franka, mesh):
    # as numpy
    point_cloud = np.asarray(sample_points_in_mesh(mesh).points)
    joint_positions = franka.get_joint_positions()
    if franka.name == "franka_robot":
        if np.abs(np.sum(joint_positions[7:]) - 0.08) > 0.001:
            left_finger_position, _ = get_world_pose_by_prim_path(
                franka.prim_path + "/panda_leftfinger"
            )
            right_finger_position, _ = get_world_pose_by_prim_path(
                franka.prim_path + "/panda_rightfinger"
            )
            hand_position, _ = get_world_pose_by_prim_path(
                franka.prim_path + "/panda_hand"
            )
            for i in range(point_cloud.shape[0]):
                if is_point_in_prism(
                    point_cloud[i],
                    left_finger_position,
                    right_finger_position,
                    hand_position,
                ):
                    return True
            return False
        else:
            return False
    elif franka.name == "franka_robotiq":
        if np.abs(np.sum(joint_positions[7:9])) > 0.033:
            left_finger_position, _ = get_world_pose_by_prim_path(
                franka.prim_path + "/Robotiq_2F_85/left_inner_finger_pad"
            )
            right_finger_position, _ = get_world_pose_by_prim_path(
                franka.prim_path + "/Robotiq_2F_85/right_inner_finger_pad"
            )
            hand_position, _ = get_world_pose_by_prim_path(
                franka.prim_path + "/Robotiq_2F_85"
            )
            for i in range(point_cloud.shape[0]):
                if is_point_in_prism(
                    point_cloud[i],
                    left_finger_position,
                    right_finger_position,
                    hand_position,
                ):
                    return True
            return False
        else:
            return False
    return False
