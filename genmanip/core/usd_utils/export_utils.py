from functools import partial
from pathlib import Path
from pxr import Sdf, Gf, Usd, UsdGeom, UsdUI  # type: ignore
from typing import List

import omni.usd  # type: ignore


def __set_xform_prim_transform(prim: UsdGeom.Xformable, transform: Gf.Matrix4d):
    prim = UsdGeom.Xformable(prim)
    _, _, scale, rot_mat, translation, _ = transform.Factor()
    angles = rot_mat.ExtractRotation().Decompose(
        Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis()
    )
    rotation = Gf.Vec3f(angles[2], angles[1], angles[0])

    for xform_op in prim.GetOrderedXformOps():
        attr = xform_op.GetAttr()
        prim.GetPrim().RemoveProperty(attr.GetName())
    prim.ClearXformOpOrder()

    UsdGeom.XformCommonAPI(prim).SetTranslate(translation)
    UsdGeom.XformCommonAPI(prim).SetRotate(rotation)
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(scale[0], scale[1], scale[2]))


def export(path: str, prims: List[Usd.Prim]):
    """Export prim to external USD file"""
    filename = Path(path).stem

    # TODO: stage.Flatten() is extreamly slow
    stage = omni.usd.get_context().get_stage()
    source_layer = stage.Flatten()
    target_layer = Sdf.Layer.CreateNew(path)
    target_stage = Usd.Stage.Open(target_layer)
    axis = UsdGeom.GetStageUpAxis(stage)
    UsdGeom.SetStageUpAxis(target_stage, axis)

    # All prims will be put under /Root
    if len(prims) > 1:
        root_path = Sdf.Path.absoluteRootPath.AppendChild("Root")
        UsdGeom.Xform.Define(target_stage, root_path)
    else:
        root_path = Sdf.Path.absoluteRootPath

    keep_transforms = len(prims) > 1

    center_point = Gf.Vec3d(0.0)
    transforms = []
    if keep_transforms:
        bound_box = Gf.BBox3d()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
        )
        for prim in prims:
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                local_bound_box = bbox_cache.ComputeWorldBound(prim)
                transforms.append(
                    xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                )
                bound_box = Gf.BBox3d.Combine(bound_box, local_bound_box)
            else:
                transforms.append(None)

        center_point = bound_box.ComputeCentroid()
    else:
        transforms.append(Gf.Matrix4d(1.0))

    for i in range(len(transforms)):
        if transforms[i]:
            transforms[i] = transforms[i].SetTranslateOnly(
                transforms[i].ExtractTranslation() - center_point
            )

    # Set default prim name
    if len(prims) > 1:
        target_layer.defaultPrim = root_path.name
        # target_layer.SetDefaultPrim(prims[0].GetPath().name)

        for i in range(len(prims)):
            source_path = prims[i].GetPath()
            if len(prims) > 1 and transforms[i]:
                target_xform_path = root_path.AppendChild(source_path.name)
                target_xform_path = Sdf.Path(
                    omni.usd.get_stage_next_free_path(
                        target_stage, target_xform_path, False
                    )
                )
                target_xform = UsdGeom.Xform.Define(target_stage, target_xform_path)
                __set_xform_prim_transform(target_xform, transforms[i])
                target_path = target_xform_path.AppendChild(source_path.name)
            else:
                target_path = root_path.AppendChild(source_path.name)
                target_path = Sdf.Path(
                    omni.usd.get_stage_next_free_path(target_stage, target_path, False)
                )

            all_external_references = set([])

            def on_prim_spec_path(root_path, prim_spec_path):
                if prim_spec_path.IsPropertyPath():
                    return

                if prim_spec_path == Sdf.Path.absoluteRootPath:
                    return

                prim_spec = source_layer.GetPrimAtPath(prim_spec_path)
                if not prim_spec or not prim_spec.HasInfo(Sdf.PrimSpec.ReferencesKey):
                    return

                op = prim_spec.GetInfo(Sdf.PrimSpec.ReferencesKey)
                items = []
                items = op.ApplyOperations(items)

                for item in items:
                    if not item.primPath.HasPrefix(root_path):
                        all_external_references.add(item.primPath)

            # Traverse the source prim tree to find all references that are outside of the source tree.
            source_layer.Traverse(source_path, partial(on_prim_spec_path, source_path))

            # Copy dependencies
            for path in all_external_references:
                Sdf.CreatePrimInLayer(target_layer, path)
                Sdf.CopySpec(source_layer, path, target_layer, path)

            Sdf.CreatePrimInLayer(target_layer, target_path)
            Sdf.CopySpec(source_layer, source_path, target_layer, target_path)

            prim = target_stage.GetPrimAtPath(target_path)
            if transforms[i]:
                __set_xform_prim_transform(prim, Gf.Matrix4d(1.0))

            # Edit UI info of compound
            spec = target_layer.GetPrimAtPath(target_path)
            attributes = spec.attributes

            if UsdUI.Tokens.uiDisplayGroup not in attributes:
                attr = Sdf.AttributeSpec(
                    spec, UsdUI.Tokens.uiDisplayGroup, Sdf.ValueTypeNames.Token
                )
                attr.default = "Material Graphs"

            if UsdUI.Tokens.uiDisplayName not in attributes:
                attr = Sdf.AttributeSpec(
                    spec, UsdUI.Tokens.uiDisplayName, Sdf.ValueTypeNames.Token
                )
                attr.default = target_path.name

            if "ui:order" not in attributes:
                attr = Sdf.AttributeSpec(spec, "ui:order", Sdf.ValueTypeNames.Int)
                attr.default = 1024
    else:
        source_path = prims[0].GetPath()

        target_path = root_path.AppendChild(source_path.name)

        target_path = Sdf.Path(
            omni.usd.get_stage_next_free_path(target_stage, target_path, False)
        )
        target_layer.defaultPrim = source_path.name

        # all_external_references = set([])

        # def on_prim_spec_path(root_path, prim_spec_path):
        #     if prim_spec_path.IsPropertyPath():
        #         return

        #     if prim_spec_path == Sdf.Path.absoluteRootPath:
        #         return

        #     prim_spec = source_layer.GetPrimAtPath(prim_spec_path)
        #     if not prim_spec or not prim_spec.HasInfo(Sdf.PrimSpec.ReferencesKey):
        #         return

        #     op = prim_spec.GetInfo(Sdf.PrimSpec.ReferencesKey)
        #     items = []
        #     items = op.ApplyOperations(items)

        #     for item in items:
        #         if not item.primPath.HasPrefix(root_path):
        #             all_external_references.add(item.primPath)

        # # Traverse the source prim tree to find all references that are outside of the source tree.
        # source_layer.Traverse(source_path, partial(on_prim_spec_path, source_path))

        # Copy dependencies
        # for path in all_external_references:
        #     Sdf.CreatePrimInLayer(target_layer, path)
        #     Sdf.CopySpec(source_layer, path, target_layer, path)

        Sdf.CreatePrimInLayer(target_layer, target_path)
        Sdf.CopySpec(source_layer, source_path, target_layer, target_path)

        prim = target_stage.GetPrimAtPath(target_path)

        # Edit UI info of compound
        spec = target_layer.GetPrimAtPath(target_path)
        attributes = spec.attributes

        if UsdUI.Tokens.uiDisplayGroup not in attributes:
            attr = Sdf.AttributeSpec(
                spec, UsdUI.Tokens.uiDisplayGroup, Sdf.ValueTypeNames.Token
            )
            attr.default = "Material Graphs"

        if UsdUI.Tokens.uiDisplayName not in attributes:
            attr = Sdf.AttributeSpec(
                spec, UsdUI.Tokens.uiDisplayName, Sdf.ValueTypeNames.Token
            )
            attr.default = target_path.name

        if "ui:order" not in attributes:
            attr = Sdf.AttributeSpec(spec, "ui:order", Sdf.ValueTypeNames.Int)
            attr.default = 1024

    # Save
    target_layer.Save()
