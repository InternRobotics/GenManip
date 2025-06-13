import os
from pathlib import Path
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from genmanip.core.loading.loading import (
    create_camera_list,
    load_world_xform_prim,
    get_object_list,
)
from genmanip.core.sensor.camera import get_src
from genmanip.utils.file_utils import load_default_config
from genmanip.utils.utils import setup_logger
from genmanip.core.loading.loading import relate_franka_from_data
from omni.isaac.core import World  # type: ignore
from genmanip.utils.file_utils import load_yaml
from genmanip.core.pointcloud.pointcloud import objectList2meshList
from genmanip.core.usd_utils.prim_utils import set_colliders

logger = setup_logger()
default_config = load_default_config(
    current_dir=current_dir, config_name="__None__.json", anygrasp_mode="local"
)
ASSETS_DIR = default_config["ASSETS_DIR"]
TEST_USD_NAME = default_config["TEST_USD_NAME"]
TABLE_UID = "aa49db8a801d402dac6cf1579536502c"
camera_data = load_yaml("configs/cameras/fixed_camera.yml")
world = World()
scene_xform, uuid = load_world_xform_prim(
    os.path.join(ASSETS_DIR, "scene_usds/base_scenes/base.usda")
)
print(uuid)
camera_list = create_camera_list(camera_data, uuid)
franka_list = [relate_franka_from_data(uuid)]
object_list = get_object_list(uuid, scene_xform, TABLE_UID)
meshDict = objectList2meshList(object_list)
for obj in object_list.values():
    set_colliders(obj.prim_path, "convexDecomposition")
world.reset()
while (
    get_src(camera_list["obs_camera"], "depth") is None
    or get_src(camera_list["realsense"], "depth") is None
    or get_src(camera_list["camera1"], "depth") is None
):
    world.step()
for _ in range(10):
    world.step()
Path("tmp").mkdir(parents=True, exist_ok=True)
image = Image.fromarray(get_src(camera_list["obs_camera"], "rgb"))
image.save("tmp/test.png")
simulation_app.close()
