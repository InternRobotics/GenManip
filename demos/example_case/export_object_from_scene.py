import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(current_dir)


from isaacsim import SimulationApp  # type: ignore

simulation_app = SimulationApp({"headless": True})

from genmanip.utils.loader.scene import load_world_xform_prim
from genmanip.utils.usd_utils.export_utils import export

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore

source_dir = os.environ.get("GENMANIP_SCENE_DIR", os.path.join(current_dir, "saved/assets/scene_usds"))
usd_path_list = [
    os.path.join(source_dir, usd_path)
    for usd_path in os.listdir(source_dir)
    if usd_path.endswith(".usda")
]

export_dir = os.environ.get("GENMANIP_EXPORT_DIR", os.path.join(current_dir, "tmp/exported_objects"))
Path(export_dir).mkdir(parents=True, exist_ok=True)
for usd_path in tqdm(usd_path_list):
    scene_xform, uuid = load_world_xform_prim(usd_path)
    prim = get_prim_at_path(f"/World/{uuid}")
    child_info = {}
    for child in prim.GetAllChildren():
        if (
            child.GetName().split("/")[-1] == "franka"
            or child.GetName().split("/")[-1] == "obj_defaultGroundPlane"
            or child.GetName().split("/")[-1] == "obj_aa49db8a801d402dac6cf1579536502c"
        ):
            continue
        child_info[child.GetName()[4:]] = {}
        child_xform = XFormPrim(str(child.GetPath()))
        position, orientation = child_xform.get_world_pose()
        child_info[child.GetName()[4:]]["position"] = position.tolist()
        child_info[child.GetName()[4:]]["orientation"] = orientation.tolist()
        child_info[child.GetName()[4:]][
            "scale"
        ] = child_xform.get_local_scale().tolist()
        from genmanip.utils.usd_utils import get_prim_bbox
        from genmanip.utils.standalone.pc_utils import compute_aabb_lwh

        try:
            l, w, h = compute_aabb_lwh(get_prim_bbox(child_xform.prim))
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            print(f"Error computing bounding box for {child.GetName()}: {e}")
            l, w, h = 0, 0, 0
        child_info[child.GetName()[4:]]["bounding_box"] = [l, w, h]
        # child_xform.set_world_pose(position=[0, 0, 0], orientation=[1, 0, 0, 0])
        # child_xform.set_local_scale([1, 1, 1])
        # export(
        #     os.path.join(export_dir, f"{uuid}-" + child.GetName()[4:] + ".usd"), [child]
        # )

    with open(os.path.join(export_dir, f"{uuid}-child_info.json"), "w") as f:
        json.dump(child_info, f)
simulation_app.close()
