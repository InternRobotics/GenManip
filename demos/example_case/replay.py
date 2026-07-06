"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from argparse import ArgumentParser
from isaacsim import SimulationApp  # type: ignore
import os
import sys

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(current_dir)

from genmanip.utils.standalone.file_utils import load_yaml

parser = ArgumentParser()
parser.add_argument(
    "-cfg",
    "--config",
    type=str,
    default="configs/minimal.yaml",
    required=True,
    help="Path to the YAML config file",
)
parser.add_argument("-l", "--local", action="store_true")
args = parser.parse_args()
config = load_yaml(args.config)

# Preload typing_extensions before SimulationApp adjusts sys.path.
# Isaac Sim can prefer its older prebundle; preloading ensures pydantic sees
# the newer typing_extensions with Sentinel.
import typing_extensions  # noqa: F401

simulation_app = SimulationApp({"headless": not args.local})

from genmanip.utils.standalone.file_utils import load_default_config
from genmanip.utils.standalone.utils import setup_logger
from genmanip.core.evaluator.utils import parse_lmdb_data
from genmanip.utils.standalone.file_utils import load_dict_from_pkl, make_dir
from genmanip.utils.standalone.utils import parse_eval_config
from genmanip.demogen.workflow.utils import (
    check_eval_finished,
    adjust_arm_gripper_action_by_embodiment,
)
from genmanip.utils.loader.domain_randomization import reset_scene
from genmanip.utils.loader.scene import clear_scene
from genmanip.utils.loader.scene import recovery_scene
from genmanip.utils.usd_utils import remove_colliders
from genmanip.core.scene.scene import Scene
from genmanip.core.robot.dualarm_manip import DualArmEmbodiment
from genmanip.core.scene.scene_config import SceneConfig
from genmanip.utils.standalone.version_utils import process_archived_config

simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
logger = setup_logger()
default_config = load_default_config(current_dir, "default.json")
eval_config_list = parse_eval_config(config)
for eval_config in eval_config_list:
    make_dir(os.path.join(default_config["EVAL_RESULT_DIR"], eval_config["task_name"]))
    scene_config = SceneConfig(**process_archived_config(eval_config))
    seed = check_eval_finished(scene_config, default_config)
    if seed == -1:
        continue
    seed = str(seed).zfill(3)
    make_dir(
        os.path.join(default_config["EVAL_RESULT_DIR"], eval_config["task_name"], seed)
    )
    scene = Scene(scene_config=scene_config)
    scene.initialize(
        default_config,
        physics_dt=1 / 30,
        rendering_dt=1 / 30,
        only_depth_rep_for_camera=True,
    )
    scene.post_initialize()
    while simulation_app.is_running():
        meta_info = load_dict_from_pkl(
            os.path.join(
                default_config["TASKS_DIR"],
                eval_config["task_name"],
                f"{seed}/meta_info.pkl",
            )
        )
        planning_data = parse_lmdb_data(
            os.path.join(
                default_config["TASKS_DIR"],
                eval_config["task_name"],
                f"{seed}",
            )
        )
        reset_scene(scene)
        layout = recovery_scene(
            scene, meta_info["task_data"], scene_config.task_name, default_config
        )
        scene_config.generation_config.goal = meta_info["task_data"]["goal"]
        if "defaultGroundPlane" in scene.object_list:
            remove_colliders(scene.object_list["defaultGroundPlane"].prim_path)
        for _ in range(50):
            scene.world.step()
        for i in range(len(planning_data["action"])):
            arm_action = planning_data["action"][i]
            gripper_action = planning_data["gripper_action"][i]
            action = adjust_arm_gripper_action_by_embodiment(
                arm_action,
                gripper_action,
                scene.robot_list[0].embodiment_name,
            )
            if isinstance(scene.robot_list[0], DualArmEmbodiment):
                base_motion = planning_data["base_motion"][i]
                scene.robot_list[0].delta_move_to(
                    base_motion[0],
                    base_motion[1],
                    base_motion[2],
                )
            scene.robot_list[0].robot_view.set_joint_position_targets(
                action,
                joint_indices=scene.robot_list[0].default_dof_indices,
            )
            scene.world.step(render=True)
        seed = check_eval_finished(scene_config, default_config)
        if seed == -1:
            break
        seed = str(seed).zfill(3)
        make_dir(
            os.path.join(
                default_config["EVAL_RESULT_DIR"], eval_config["task_name"], seed
            )
        )
    clear_scene(scene, scene_config, current_dir)
simulation_app.close()
