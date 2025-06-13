import numpy as np
import os
from tqdm import tqdm
import sys
from filelock import SoftFileLock
from isaacsim import SimulationApp

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from argparse import ArgumentParser
from genmanip.utils.file_utils import load_json, load_yaml

parser = ArgumentParser()
parser.add_argument(
    "-cfg",
    "--config",
    type=str,
    default="",
    required=True,
    help="Path to the YAML config file",
)
parser.add_argument("-l", "--local", action="store_true")
parser.add_argument("-r", "--render_first_frame", action="store_true")
parser.add_argument("-wod", "--without_depth", action="store_true")
parser.add_argument("-a", "--add_random_position_camera", action="store_true")
# TODO: random choice frame and render
args = parser.parse_args()
config = load_yaml(args.config)

simulation_app = SimulationApp({"headless": not args.local})

from genmanip.utils.file_utils import load_default_config
from genmanip.utils.utils import setup_logger
from genmanip.utils.file_utils import load_dict_from_pkl, make_dir
from genmanip.utils.utils import parse_demogen_config
from genmanip.core.sensor.camera import set_camera_look_at
from genmanip.core.usd_utils.prim_utils import remove_colliders
from genmanip.core.loading.domain_randomization import random_texture
from genmanip.core.loading.loading import (
    build_scene_from_config,
    clear_scene,
    warmup_world,
    preprocess_scene,
    collect_meta_infos,
    load_object_pool,
)
from genmanip.core.loading.loading import recovery_scene_render
from genmanip.demogen.recoder.render_recorder import Logger
from genmanip.demogen.recoder.utils import parse_planning_result

simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
logger = setup_logger()
if args.local:
    default_config = load_default_config(current_dir, "__None__.json", "local")
else:
    default_config = load_default_config(current_dir, "default.json")
demogen_config_list = parse_demogen_config(config)
for demogen_config in demogen_config_list:
    make_dir(
        os.path.join(
            default_config["DEMONSTRATION_DIR"], demogen_config["task_name"], "render"
        )
    )
    scene = build_scene_from_config(
        demogen_config,
        default_config,
        current_dir,
        physics_dt=1 / 600000.0,
        rendering_dt=1 / 600000.0,
        is_eval=True,
        is_render=True,
    )
    load_object_pool(scene, demogen_config, current_dir)
    preprocess_scene(scene, demogen_config)
    warmup_world(scene)
    collect_meta_infos(scene)
    remove_colliders(scene["object_list"]["defaultGroundPlane"].prim_path)
    dir_list = os.listdir(
        os.path.join(
            default_config["DEMONSTRATION_DIR"],
            demogen_config["task_name"],
            "trajectory",
        )
    )
    logger.info(f"rendering {len(dir_list)} trajectories")
    for dir in dir_list:
        if not os.path.isdir(
            os.path.join(
                default_config["DEMONSTRATION_DIR"],
                demogen_config["task_name"],
                "trajectory",
                dir,
            )
        ):
            continue
        if os.path.isdir(
            os.path.join(
                default_config["DEMONSTRATION_DIR"],
                demogen_config["task_name"],
                "trajectory",
                dir,
            )
        ) and os.path.exists(
            os.path.join(
                default_config["DEMONSTRATION_DIR"],
                demogen_config["task_name"],
                "render",
                dir,
            )
        ):
            logger.info(f"skip {dir} because it is already rendered")
            continue
        lock_file = os.path.join(
            default_config["DEMONSTRATION_DIR"],
            demogen_config["task_name"],
            "render",
            f"render_{dir}_soft.lock",
        )
        lock = SoftFileLock(lock_file, timeout=0)
        try:
            executed = False
            with lock:
                if os.path.isdir(
                    os.path.join(
                        default_config["DEMONSTRATION_DIR"],
                        demogen_config["task_name"],
                        "trajectory",
                        dir,
                    )
                ) and not os.path.exists(
                    os.path.join(
                        default_config["DEMONSTRATION_DIR"],
                        demogen_config["task_name"],
                        "render",
                        dir,
                    )
                ):
                    make_dir(
                        os.path.join(
                            default_config["DEMONSTRATION_DIR"],
                            demogen_config["task_name"],
                            "render",
                            dir,
                        )
                    )
                    meta_info = load_dict_from_pkl(
                        os.path.join(
                            default_config["DEMONSTRATION_DIR"],
                            demogen_config["task_name"],
                            "trajectory",
                            dir,
                            "meta_info.pkl",
                        )
                    )
                    input_camera_dict = scene["camera_list"].copy()
                    if args.add_random_position_camera:
                        # avoid the camera is in the back of the robot
                        random_azimuth = np.random.uniform(-150, 150)
                        random_elevation = np.random.uniform(30, 50)
                        distance = np.random.uniform(0.7, 1.2)
                        set_camera_look_at(
                            input_camera_dict["camera1"],
                            np.array([0, 0, 1.1]),
                            distance=distance,
                            azimuth=random_azimuth,
                            elevation=random_elevation,
                        )
                    else:
                        input_camera_dict.pop("camera1")
                    recorder = Logger(
                        input_camera_dict,
                        scene["robot_info"]["robot_list"][0].robot,
                        meta_info["task_data"]["instruction"],
                        log_dir=os.path.join(
                            default_config["DEMONSTRATION_DIR"],
                            demogen_config["task_name"],
                            "render",
                            dir,
                        ),
                        task_data=meta_info["task_data"],
                        tcp_config=scene["tcp_configs"]["franka"],
                    )
                    recovery_scene_render(
                        scene, meta_info["task_data"], demogen_config, default_config
                    )
                    random_texture(scene, default_config, demogen_config)
                    data_list = parse_planning_result(
                        dir, default_config, demogen_config, scene
                    )
                    for _ in range(10):
                        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
                            data_list[0]["qpos"]
                        )
                        for key in scene["object_list"]:
                            if key == "00000000000000000000000000000000":
                                continue
                            scene["object_list"][key].set_world_pose(
                                data_list[0]["obj_info"][key]["position"],
                                data_list[0]["obj_info"][key]["orientation"],
                            )
                            scene["object_list"][key].set_local_scale(
                                data_list[0]["obj_info"][key]["scale"]
                            )
                        scene["world"].step()
                    print("rendering data with length: ", len(data_list))
                    for data in tqdm(data_list):
                        scene["robot_info"]["robot_list"][0].robot.set_joint_positions(
                            data["qpos"]
                        )
                        for key in scene["object_list"]:
                            if key == "00000000000000000000000000000000":
                                continue
                            scene["object_list"][key].set_world_pose(
                                data["obj_info"][key]["position"],
                                data["obj_info"][key]["orientation"],
                            )
                            scene["object_list"][key].set_local_scale(
                                data["obj_info"][key]["scale"]
                            )
                        scene["world"].step()
                        recorder.load_dynamic_info(
                            data["obj_info"],
                            data["action"],
                            data["qpos"],
                            data["qvel"],
                            data["gripper_close"],
                            data["name"],
                        )
                        if args.render_first_frame:
                            break
                    recorder.save(without_depth=args.without_depth)
                    executed = True
            if executed and os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            logger.info(f"error in rendering {dir}: {e}")
            pass
    clear_scene(scene, demogen_config, current_dir)
simulation_app.close()
