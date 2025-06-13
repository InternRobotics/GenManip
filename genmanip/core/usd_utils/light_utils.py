from pxr import UsdLux  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore

def create_dome_light(prim_path, hdr):
    light_prim = get_prim_at_path(prim_path)
    light = UsdLux.DomeLight(light_prim)
    light.CreateTextureFileAttr(hdr)
    light.CreateIntensityAttr(1000.0)
    return light
