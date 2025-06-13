import numpy as np
import open3d as o3d
from pxr import Usd, UsdPhysics, UsdGeom, Gf, Sdf, PhysxSchema  # type: ignore
from typing import Optional, Sequence  # type: ignore

from genmanip.utils.pc_utils import (
    compute_aabb_lwh,
    compute_mesh_bbox,
    compute_mesh_center,
    get_mesh_from_points_and_faces,
    get_pcd_from_mesh,
)

from omni.isaac.core.prims import RigidPrim, GeometryPrim  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore
from omni.isaac.core.utils.semantics import add_update_semantics  # type: ignore
from pxr import Usd, UsdPhysics, PhysxSchema  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_parent, get_prim_path, delete_prim  # type: ignore
import omni.usd  # type: ignore

"""
collider type:
none
convexDecomposition
convexHull
boundingSphere
boundingCube
"""


def set_semantic_label(prim_path: str, label):
    prim = get_prim_at_path(prim_path)
    add_update_semantics(prim, semantic_label=label, type_label="class")
    # prim = get_prim_at_path(prim_path)
    # if prim.GetTypeName() == "Mesh":
    #     add_update_semantics(prim, semantic_label=label, type_label="class")
    # all_children = prim.GetAllChildren()
    # for child in all_children:
    #     set_semantic_label(str(child.GetPath()), label)


def add_usd_to_world(
    asset_path: str,
    prim_path: str,
    name: str,
    translation: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
    add_rigid_body: bool = False,
    add_colliders: bool = False,
    collision_approximation="convexDecomposition",
):
    reference = add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim_path = str(reference.GetPrimPath())
    prim = XFormPrim(
        prim_path,
        name=name,
        translation=translation,
        orientation=orientation,
        scale=scale,
    )
    usd_prim = prim.prim
    if not usd_prim.IsValid():
        print(f"Prim at path {prim_path} is not valid.")
        return prim
    if add_rigid_body:
        set_rigid_body(prim_path)
        print(f"RigidBodyAPI applied to {prim_path}")
    if add_colliders:
        set_colliders(prim_path, collision_approximation)
        print(f"CollisionAPI applied to {prim_path}")
    set_semantic_label(
        str(usd_prim.GetPath()), str(usd_prim.GetPath()).split("/")[-1][4:]
    )
    return prim


def get_mesh_from_prim(prim):
    points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_total = (
        recursive_parse(prim)
    )
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    return mesh


def get_prim_bbox(prim):
    mesh = get_mesh_from_prim(prim)
    return compute_mesh_bbox(mesh)


def get_prim_center(prim):
    mesh = get_mesh_from_prim(prim)
    return compute_mesh_center(mesh)


def recursive_parse(prim):
    translation = prim.GetAttribute("xformOp:translate").Get()
    if translation is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation)
    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is None:
        scale = np.ones(3)
    else:
        scale = np.array(scale)
    orient = prim.GetAttribute("xformOp:orient").Get()
    if orient is None:
        orient = np.zeros([4, 1])
        orient[0] = 1.0
    else:
        r = orient.GetReal()
        i, j, k = orient.GetImaginary()
        orient = np.array([r, i, j, k]).reshape(4, 1)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(orient)
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
    new_points = []
    for i, p in enumerate(points_total):
        pn = np.array(p)
        pn *= scale
        pn = np.matmul(rotation_matrix, pn)
        pn += translation
        new_points.append(pn)
    return (
        new_points,
        faceuv_total,
        normals_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
        mesh_total,
    )


def resize_object_by_lwh(pre_object, l, w, h, mesh=None):
    if mesh is None:
        aabb = get_prim_bbox(pre_object.prim)
    else:
        aabb = compute_mesh_bbox(mesh)
    x, y, z = compute_aabb_lwh(aabb)
    length_rate = l / x
    width_rate = w / y
    height_rate = h / z
    local_scale = pre_object.get_local_scale()
    local_scale[0] *= width_rate
    local_scale[1] *= length_rate
    local_scale[2] *= height_rate
    pre_object.set_local_scale(local_scale)


def resize_object(pre_object, size, mesh=None):
    if mesh is None:
        aabb = get_prim_bbox(pre_object.prim)
    else:
        aabb = compute_mesh_bbox(mesh)
    x, y, z = compute_aabb_lwh(aabb)
    length_rate = size / x if size else 1.0
    width_rate = size / y if size else 1.0
    height_rate = size / z if size else 1.0
    local_scale = pre_object.get_local_scale()
    if x >= y and x >= z:
        pre_object.set_local_scale(local_scale * length_rate)
        return
    elif y >= x and y >= z:
        pre_object.set_local_scale(local_scale * width_rate)
        return
    elif z >= x and z >= y:
        pre_object.set_local_scale(local_scale * height_rate)
        return


def sample_points_from_prim(prim, num_points=1000):
    mesh = get_mesh_from_prim(prim)
    pcd = get_pcd_from_mesh(mesh, num_points)
    return np.asarray(pcd.points)


def setup_physics_scene():
    stage = omni.usd.get_context().get_stage()
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physxSceneAPI.GetEnableGPUDynamicsAttr().Set(False)
    physxSceneAPI.GetEnableStabilizationAttr().Set(True)
    physxSceneAPI.GetEnableCCDAttr().Set(True)
    physxSceneAPI.GetBroadphaseTypeAttr().Set("GPU")
    physxSceneAPI.GetSolverTypeAttr().Set("TGS")
    physxSceneAPI.GetGpuTotalAggregatePairsCapacityAttr().Set(10 * 1024 * 1024)
    physxSceneAPI.GetGpuFoundLostAggregatePairsCapacityAttr().Set(10 * 1024 * 1024)


def remove_colliders(prim_path):
    prim = get_prim_at_path(prim_path)
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsCollisionAPI" in schema_list:
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    if "PhysicsMeshCollisionAPI" in schema_list:
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    if "PhysxConvexHullCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
    if "PhysxConvexDecompositionCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
    if "PhysxSDFMeshCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
    if "PhysxTriangleMeshCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
    for child in prim.GetAllChildren():
        remove_colliders(str(child.GetPath()))


def set_contact_offset(prim_path, contact_offset):
    prim = get_prim_at_path(prim_path)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr().Set(contact_offset)
    return prim


def set_gravity(prim_path, gravity_enabled):
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateDisableGravityAttr().Set(not gravity_enabled)
    return prim


def remove_contact_offset(prim_path):
    prim = get_prim_at_path(prim_path)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr().Set(float('-inf'))
    return prim


def set_colliders(prim_path, collision_approximation="convexDecomposition", convex_hulls = None):
    remove_colliders(prim_path)
    prim = get_prim_at_path(prim_path)
    collider = UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collider.CreateApproximationAttr().Set(collision_approximation)
    collider.GetCollisionEnabledAttr().Set(True)
    if collision_approximation == "convexDecomposition":
        collision_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        collision_api.CreateHullVertexLimitAttr().Set(64)
        if convex_hulls is not None:
            collision_api.CreateMaxConvexHullsAttr().Set(convex_hulls)
        else:
            collision_api.CreateMaxConvexHullsAttr().Set(64)
        collision_api.CreateMinThicknessAttr().Set(0.1)
        collision_api.CreateShrinkWrapAttr().Set(True)
        collision_api.CreateErrorPercentageAttr().Set(0.1)
        # collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # collision_api.CreateContactOffsetAttr().Set(0.1)
    elif collision_approximation == "convexHull":
        collision_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
        collision_api.CreateHullVertexLimitAttr().Set(64)
        collision_api.CreateMinThicknessAttr().Set(0.00001)
    elif collision_approximation == "sdf":
        collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        collision_api.CreateSdfResolutionAttr().Set(1024)
    return prim


def add_contact_offset(prim_path, contact_offset=0.1):
    prim = get_prim_at_path(prim_path)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr().Set(contact_offset)
    return prim


def set_max_convex_hulls(prim_path, max_convex_hulls):
    prim = get_prim_at_path(prim_path)
    prim.CreateAttribute(
        "physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int
    ).Set(max_convex_hulls)
    return prim


def set_rigid_body_CCD(prim_path, ccd_enabled):
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateEnableCCDAttr().Set(ccd_enabled)
    return prim


def set_mass(prim_path, mass):
    prim = get_prim_at_path(prim_path)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)
    return prim


def set_rigid_body(prim_path):
    prim = get_prim_at_path(prim_path)
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    set_rigid_body_CCD(prim_path, True)
    return prim


def get_all_joints(prim):
    joint_list = []

    def recurse_prim(current_prim):
        for child in current_prim.GetChildren():
            if child.IsA(UsdPhysics.Joint):
                joint_type = child.GetTypeName()
                if joint_type == "PhysicsPrismaticJoint":
                    joint = UsdPhysics.PrismaticJoint(child)
                elif joint_type == "PhysicsRevoluteJoint":
                    joint = UsdPhysics.RevoluteJoint(child)
                else:
                    # joint = UsdPhysics.Joint(child)
                    continue
                joint_list.append(joint)
            recurse_prim(child)

    recurse_prim(prim)

    return joint_list


def get_world_pose_by_prim_path(prim_path):
    xform_prim = XFormPrim(prim_path)
    return xform_prim.get_world_pose()


def set_world_pose_by_prim_path(prim_path, world_pose):
    xform_prim = XFormPrim(prim_path)
    xform_prim.set_world_pose(*world_pose)


def get_joint_info(joint):
    path = str(joint.GetPath())
    axis = joint.GetAxisAttr().Get()
    lower_limit = joint.GetLowerLimitAttr().Get()
    upper_limit = joint.GetUpperLimitAttr().Get()
    joint_info = {}
    joint_info[path] = {
        "axis": axis,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
    }
    return joint_info


def set_joint_info(joint, joint_info=None):
    if not joint_info:
        print(f"joint info is None")
        return joint
    joint.GetAxisAttr().Set(joint_info["axis"])
    joint.GetLowerLimitAttr().Set(joint_info["lower_limit"])
    joint.GetUpperLimitAttr().Set(joint_info["upper_limit"])
    return joint


def get_leaf_prims(prim):
    # prim here include Xform and Mesh
    leaf_prims = set()

    def recurse_prim(current_prim):
        prim_type_name = current_prim.GetTypeName()
        if prim_type_name in ["Xform", "Mesh"]:
            leaf_prims.add(current_prim)
        if current_prim.GetChildren():
            for child in current_prim.GetChildren():
                recurse_prim(child)

    recurse_prim(prim)

    return list(leaf_prims)


def get_prim_info(prim):
    prim_info = {}
    prim_type = prim.GetTypeName()
    prim_path = str(prim.GetPath())

    translation_gf = prim.GetAttribute("xformOp:translate").Get()
    if translation_gf is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation_gf)

    orientation_gf = prim.GetAttribute("xformOp:orient").Get()
    if orientation_gf is None:
        orientation = np.zeros(4)
        orientation[0] = 1.0
    else:
        r = orientation_gf.GetReal()
        i, j, k = orientation_gf.GetImaginary()
        orientation = np.array([r, i, j, k])

    scale_gf = prim.GetAttribute("xformOp:scale").Get()
    if scale_gf is None:
        scale = np.ones(3)
    else:
        scale = np.array(scale_gf)

    mass_center = None
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI):
        mass_center_gf = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()
        mass_center = np.array(mass_center_gf).tolist()

    prim_info[prim_path] = {
        "translation": translation.tolist(),
        "orientation": orientation.tolist(),
        "scale": scale.tolist(),
        "mass_center": mass_center,
    }
    return prim_info


def set_prim_info(prim, prim_info=None):
    if prim_info is None:
        print(f"prim info is None")
        return prim
    translation = np.array(prim_info["translation"])
    translation_gf = Gf.Vec3f(translation[0], translation[1], translation[2])
    orientation = np.array(prim_info["orientation"])
    # orientation_gf = Gf.Quatf(
    #     orientation[0], orientation[1], orientation[2], orientation[3]
    # )
    scale = np.array(prim_info["scale"])
    scale_gf = Gf.Vec3f(scale[0], scale[1], scale[2])

    prim_xform = UsdGeom.Xform(prim)

    # translation xform
    trans_attr = prim.GetAttribute("xformOp:translate")
    if not trans_attr:
        trans_attr = prim_xform.AddTranslateOp()
    trans_attr.Set(translation_gf)

    # orient xform
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not orient_attr:
        orient_attr = prim_xform.AddOrientOp()
    orient_type = orient_attr.GetTypeName()
    if orient_type == "quatd":
        orientation_gf = Gf.Quatd(
            orientation[0], orientation[1], orientation[2], orientation[3]
        )
    else:  # Quatf
        orientation_gf = Gf.Quatf(
            orientation[0], orientation[1], orientation[2], orientation[3]
        )
    orient_attr.Set(orientation_gf)

    # scale xform
    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        scale_attr = prim_xform.AddScaleOp()
    scale_attr.Set(scale_gf)

    # prim.GetAttribute("xformOp:translate").Set(translation_gf)
    # prim.GetAttribute("xformOp:orient").Set(orientation_gf)
    # prim.GetAttribute("xformOp:scale").Set(scale_gf)

    if prim_info["mass_center"] is not None:
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
        else:
            mass = UsdPhysics.MassAPI(prim)
        mass_center = np.array(prim_info["mass_center"])
        mass_center_gf = Gf.Vec3f(
            float(mass_center[0]), float(mass_center[1]), float(mass_center[2])
        )
        mass.GetCenterOfMassAttr().Set(mass_center_gf)
        # print(mass_center_gf)
    print(translation_gf, orientation_gf, scale_gf)
    return prim
