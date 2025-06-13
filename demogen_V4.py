import argparse
import copy
import numpy as np
import os
import sys
from tqdm import tqdm
from genmanip.utils.file_utils import (
    load_default_config,
    load_yaml,
    make_dir,
    record_log,
)

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="configs/1.yaml",
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--record",
        type=str,
        required=False,
        default="just for record",
    )
    parser.add_argument(
        "-l",
        "--local",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


args = parse_args()
config = load_yaml(args.config)
is_local = args.local
is_evalgen = args.eval
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": not is_local})

from genmanip.core.loading.loading import (
    build_scene_from_config,
    clear_scene,
    collect_meta_infos,
    update_meta_infos,
    preload_objects,
    preprocess_scene,
    warmup_world,
    create_planner,
    load_object_pool,
)
from genmanip.core.loading.domain_randomization import (
    build_up_scene,
    domain_randomization,
    reset_scene,
)
from genmanip.core.pointcloud.pointcloud import (
    get_current_meshList,
    meshlist_to_pclist,
)
from genmanip.demogen.evaluate.evaluate import check_finished
from genmanip.demogen.planning.planning import apply_action_by_config
from genmanip.demogen.planning.utils import (
    check_evalgen_finished,
    check_planning_finished,
    corse_process_task_data,
    refine_task_data,
)
from genmanip.demogen.recoder.planning_recorder import Logger, collect_task_data
from genmanip.utils.utils import (
    parse_demogen_config,
    setup_logger,
    parse_evalgen_config,
    check_usda_exist,
    check_proxy_exist,
)
from genmanip.core.usd_utils.prim_utils import remove_colliders

simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
logger = setup_logger()
if is_local:
    default_config = load_default_config(current_dir, "__None__.json", "local")
else:
    default_config = load_default_config(current_dir, "default.json")
default_config["current_dir"] = current_dir
if not is_evalgen:
    demogen_config_list = parse_demogen_config(config)
else:
    demogen_config_list = parse_evalgen_config(config)
for demogen_config in demogen_config_list:
    assert (
        "layout_config" in demogen_config
    ), "Your config is out of date, please update it."
    if not is_evalgen:
        make_dir(
            os.path.join(
                default_config["DEMONSTRATION_DIR"],
                demogen_config["task_name"],
                "trajectory",
            )
        )
        if check_planning_finished(demogen_config, default_config):
            continue
    else:
        make_dir(os.path.join(default_config["TASKS_DIR"], demogen_config["task_name"]))
        if check_evalgen_finished(demogen_config, default_config):
            continue
    if check_proxy_exist():
        logger.warning("Proxy exists, may cost disconnect anygrasp server...")
    if not check_usda_exist(default_config, demogen_config):
        logger.error(
            f"USD file does not exist, do you run usda_gen.py in the right way? Use python standalone_tools/usda_gen.py -f saved/assets/scene_usds -r to generate the USDA file."
        )
        continue
    else:
        logger.info(f"USD file exists")
    scene = build_scene_from_config(
        demogen_config,
        default_config,
        current_dir,
        physics_dt=1 / 30,
        rendering_dt=1 / 30,
    )
    load_object_pool(scene, demogen_config, current_dir)
    preprocess_scene(scene, demogen_config)
    preload_objects(scene, default_config, demogen_config)
    remove_colliders(scene["object_list"]["defaultGroundPlane"].prim_path)
    warmup_world(scene)
    collect_meta_infos(scene)
    total_success = 0
    while simulation_app.is_running():
        if not is_evalgen:
            if check_planning_finished(demogen_config, default_config):
                break
        else:
            if check_evalgen_finished(demogen_config, default_config):
                break
        reset_scene(scene)
        remove_colliders(scene["object_list"]["defaultGroundPlane"].prim_path)
        task_data = corse_process_task_data(demogen_config)
        build_up_scene(scene, demogen_config, default_config, task_data)
        task_data = refine_task_data(task_data, demogen_config)
        if (
            domain_randomization(
                scene, default_config, demogen_config, task_data, mode="demogen"
            )
            == -1
        ):
            continue
        update_meta_infos(scene)
        scene["planner_list"] = create_planner(scene, demogen_config, current_dir)
        task_data = collect_task_data(
            scene["object_list"],
            scene["robot_info"]["robot_list"],
            load_yaml(
                os.path.join(
                    current_dir,
                    demogen_config["domain_randomization"]["cameras"]["config_path"],
                )
            ),
            task_data,
            scene["cacheDict"]["preloaded_object_path_list"],
        )
        input_camera_dict = scene["camera_list"].copy()
        input_camera_dict.pop("camera1")
        recorder = Logger(
            input_camera_dict,
            scene["robot_info"]["robot_list"][0].robot,
            scene["object_list"],
            task_data["instruction"],
            log_dir=(
                os.path.join(
                    default_config["DEMONSTRATION_DIR"],
                    demogen_config["task_name"],
                    "trajectory",
                )
                if not is_evalgen
                else os.path.join(
                    default_config["TASKS_DIR"],
                    demogen_config["task_name"],
                )
            ),
            task_data=task_data,
            tcp_config=scene["tcp_configs"]["franka"],
        )
        for _ in range(100):
            scene["world"].step(render=False)
        for idx, action_info in enumerate(task_data["task_path"]):
            try:
                is_success = apply_action_by_config(
                    scene,
                    action_info,
                    default_config,
                    demogen_config,
                    recorder,
                    idx,
                )
                if not is_success:
                    raise Exception("Subgoal not completed")
            except Exception as e:
                del recorder
                logger.error(str(e))
                if not is_evalgen:
                    record_log(
                        os.path.join(
                            default_config["DEMONSTRATION_DIR"],
                            demogen_config["task_name"],
                            "trajectory",
                        ),
                        str(e),
                    )
                else:
                    record_log(
                        os.path.join(
                            default_config["TASKS_DIR"],
                            demogen_config["task_name"],
                        ),
                        str(e),
                    )
                break
        if "recorder" not in locals() or "recorder" not in globals():
            logger.error("Task not completed, retry......")
            continue
        for _ in tqdm(range(30)):
            scene["world"].step(render=False)
        meshlist = get_current_meshList(
            scene["object_list"], scene["cacheDict"]["meshDict"]
        )
        pclist = meshlist_to_pclist(meshlist)
        if len(task_data["goal"]) == 0 or len(task_data["goal"][0]) == 0:
            finished = True
        else:
            finished = (
                check_finished(task_data["goal"], pclist, scene["articulation_list"])
                == 1
            )
        if finished:
            recorder.save(demogen_config["task_name"], args.config)
            if not is_evalgen:
                record_log(
                    os.path.join(
                        default_config["DEMONSTRATION_DIR"],
                        demogen_config["task_name"],
                        "trajectory",
                    ),
                    "success",
                )
                total_success += 1
            else:
                record_log(
                    os.path.join(
                        default_config["TASKS_DIR"],
                        demogen_config["task_name"],
                    ),
                    "success",
                )
    clear_scene(scene, demogen_config, current_dir)
simulation_app.close()
