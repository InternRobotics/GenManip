import csv
import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(current_dir)

from isaacsim import SimulationApp  # type: ignore

simulation_app = SimulationApp({"headless": False})

from genmanip.utils.loader.scene import (
    create_camera_list,
    load_world_xform_prim,
    get_object_list,
)
from genmanip.utils.usd_utils.camera_utils import get_src
from genmanip.utils.standalone.file_utils import load_default_config
from genmanip.utils.standalone.utils import setup_logger
from genmanip.utils.loader.robot import relate_franka_from_data
from omni.isaac.core import World  # type: ignore
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.utils.pointcloud.pointcloud import objectList2meshList
from genmanip.utils.usd_utils import set_colliders

logger = setup_logger()
default_config = load_default_config(
    current_dir=current_dir, config_name="__None__.json", anygrasp_mode="local"
)
ASSETS_DIR = default_config["ASSETS_DIR"]
TEST_USD_NAME = default_config["TEST_USD_NAME"]
TABLE_UID = "aa49db8a801d402dac6cf1579536502c"
camera_data = load_yaml("configs/cameras/fixed_camera.yml")
world = World()
# scene_xform, uuid = load_world_xform_prim(
#     os.path.join(ASSETS_DIR, "scene_usds/debug_scenes/kitchen_scenes/base.usda")
# )
scene_xform, uuid = load_world_xform_prim(os.environ.get("GENMANIP_SCENE_USD", os.path.join(ASSETS_DIR, "object.usda")))
print(uuid)
camera_list = create_camera_list(camera_data, uuid)
robot_list = [relate_franka_from_data(uuid)]
for franka in robot_list:
    world.scene.add(franka)
object_list = get_object_list(uuid, scene_xform, TABLE_UID)
# meshDict = objectList2meshList(object_list)
for obj in object_list.values():
    set_colliders(obj.prim_path, "convexDecomposition")
world.reset()
while (
    get_src(camera_list["obs_camera"], "depth") is None
    or get_src(camera_list["realsense"], "depth") is None
    or get_src(camera_list["camera1"], "depth") is None
):
    world.step()
import numpy as np
from genmanip.utils.usd_utils.camera_utils import set_camera_look_at

for i in range(100):
    world.step()

print("start")
from genmanip.utils.standalone.robot_utils import joint_position_to_end_effector_pose

target_pos, target_ori = joint_position_to_end_effector_pose(
    robot_list[0].get_joint_positions()
)
target_pos += robot_list[0].get_world_pose()[0]
set_camera_look_at(
    camera_list["obs_camera"], target_pos, distance=1.0, elevation=90.0, azimuth=0.0
)
original_pos, original_ori = camera_list["obs_camera"].get_world_pose()
original_focal = camera_list["obs_camera"].get_focal_length()
initial_distance = np.linalg.norm(np.array(target_pos) - np.array(original_pos))
# save render as video
focal_length = original_focal
distance = initial_distance
k = focal_length / distance
image_list = []
import cv2

for i in range(10):
    world.step()
for i in range(10):
    world.step()
    image = get_src(camera_list["obs_camera"], "rgb")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_list.append(image)
zoom_ratios = np.logspace(0, 5.0, 100, base=1.5)
for i in range(100):
    zoom_ratio = zoom_ratios[i]
    new_distance = initial_distance * zoom_ratio  # 或用别的缓慢函数
    new_focal = k * new_distance  # 保持目标大小不变
    forward_vec = np.array(original_pos) - np.array(target_pos)
    forward_dir = forward_vec / np.linalg.norm(forward_vec)
    new_pos = np.array(target_pos) + forward_dir * new_distance
    camera_list["obs_camera"].set_world_pose(new_pos.tolist(), original_ori)
    camera_list["obs_camera"].set_focal_length(new_focal)
    world.step()
    image = get_src(camera_list["obs_camera"], "rgb")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_list.append(image)

video_writer = cv2.VideoWriter(
    "tmp/test.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (image_list[0].shape[1], image_list[0].shape[0]),
)
for image in image_list:
    video_writer.write(image)
video_writer.release()

simulation_app.close()
