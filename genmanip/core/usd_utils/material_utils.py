import random
from pxr import Sdf, UsdShade, Gf  # type: ignore
import omni.usd  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
from omni.isaac.core.materials.omni_pbr import OmniPBR  # type: ignore


def create_omni_pbr(prim_path, scale=(0.1125, 0.2)):
    stage = omni.usd.get_context().get_stage()
    mtl_path = Sdf.Path(prim_path)
    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    omni_pbr = OmniPBR(prim_path=prim_path)
    shader = get_prim_at_path(omni_pbr.prim_path + "/Shader")
    shader.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateAttribute("inputs:texture_scale", Sdf.ValueTypeNames.Float2).Set(scale)
    shader.CreateAttribute("inputs:texture_translate", Sdf.ValueTypeNames.Float2).Set(
        (0.0, 0.0)
    )
    shader.CreateAttribute("inputs:texture_rotate", Sdf.ValueTypeNames.Float).Set(0.0)
    return omni_pbr


def change_material_info(
    prim_path, texture_path=None, translation=None, rotation=None, scale=None
):
    prim = get_prim_at_path(prim_path)
    for child in prim.GetAllChildren():
        if str(child.GetPath()).endswith("Looks"):
            for grandchild in child.GetAllChildren():
                if str(grandchild.GetPath()).endswith("Table_1"):
                    for grandgrandchild in grandchild.GetAllChildren():
                        if str(grandgrandchild.GetPath()).endswith("baseColorTex"):
                            tex = grandgrandchild
                            try:
                                if texture_path is not None:
                                    tex.GetAttribute("inputs:texture").Set(texture_path)
                                if translation is not None:
                                    tex.GetAttribute("inputs:offset").Set(translation)
                                if rotation is not None:
                                    tex.GetAttribute("inputs:rotation").Set(rotation)
                                if scale is not None:
                                    tex.GetAttribute("inputs:scale").Set(
                                        Gf.Vec2f(scale[0], scale[1])
                                    )
                            except Exception as e:
                                print(f"Error changing material info: {e}")


def change_table_mdl(prim_path, texture_path_list):
    prim = get_prim_at_path(prim_path)
    for child in prim.GetAllChildren():
        if str(child.GetPath()).endswith("Looks"):
            for grandchild in child.GetAllChildren():
                for grandgrandchild in grandchild.GetAllChildren():
                    try:
                        texture_path = random.choice(texture_path_list)
                        grandgrandchild.GetAttribute("info:mdl:sourceAsset").Set(
                            texture_path
                        )
                        grandgrandchild.GetAttribute(
                            "info:mdl:sourceAsset:subIdentifier"
                        ).Set(texture_path.split("/")[-1].split(".")[0])
                    except Exception as e:
                        print(f"Error changing table mdl: {e}")
