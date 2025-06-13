import os
from tqdm import tqdm
import sys

from isaacsim import SimulationApp

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from argparse import ArgumentParser
from genmanip.utils.file_utils import load_json, load_yaml

parser = ArgumentParser()
parser.add_argument("-r", "--receive_port", type=int, default=10000)
parser.add_argument("-s", "--send_port", type=int, default=10001)
parser.add_argument(
    "-cfg",
    "--config",
    type=str,
    default="",
    required=True,
    help="Path to the YAML config file",
)
parser.add_argument("-l", "--local", action="store_true")
parser.add_argument("-n", "--num_steps", type=int, default=600)
args = parser.parse_args()
config = load_yaml(args.config)

simulation_app = SimulationApp({"headless": not args.local})

from genmanip.demogen.evaluate.evaluate import check_finished
from genmanip.utils.file_utils import load_default_config
from genmanip.utils.utils import setup_logger
from genmanip_bench.evaluate.evaluator import Evaluator, parse_lmdb_data
from genmanip_bench.request_model.socket_utils import (
    create_send_port_and_wait,
    create_receive_port_and_attach,
)
from genmanip.core.pointcloud.pointcloud import (
    get_current_meshList,
    meshlist_to_pclist,
)
from genmanip.utils.file_utils import load_dict_from_pkl, make_dir
from genmanip.utils.utils import parse_eval_config
from genmanip.demogen.planning.utils import check_eval_finished
from genmanip.core.loading.loading import (
    build_scene_from_config,
    clear_scene,
    warmup_world,
    preprocess_scene,
    collect_meta_infos,
    load_object_pool,
)
from genmanip.core.loading.loading import recovery_scene
from genmanip.core.usd_utils.prim_utils import remove_colliders

simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
logger = setup_logger()
receive_port = create_receive_port_and_attach(args.receive_port)
send_port = create_send_port_and_wait(args.send_port)
if args.local:
    default_config = load_default_config(current_dir, "__None__.json", "local")
else:
    default_config = load_default_config(current_dir, "default.json")
eval_config_list = parse_eval_config(config)
for eval_config in eval_config_list:
    make_dir(os.path.join(default_config["EVAL_RESULT_DIR"], eval_config["task_name"]))
    seed = check_eval_finished(eval_config, default_config)
    if seed == -1:
        continue
    seed = str(seed).zfill(3)
    make_dir(
        os.path.join(default_config["EVAL_RESULT_DIR"], eval_config["task_name"], seed)
    )
    scene = build_scene_from_config(
        eval_config,
        default_config,
        current_dir,
        is_eval=True,
        physics_dt=1 / 60,
        rendering_dt=1 / 60,
    )
    load_object_pool(scene, eval_config, current_dir)
    preprocess_scene(scene, eval_config)
    warmup_world(scene)
    collect_meta_infos(scene)
    evaluator = Evaluator(
        scene,
        eval_config["instruction"],
        os.path.join(default_config["EVAL_RESULT_DIR"], eval_config["task_name"]),
        current_dir,
        send_port=send_port,
        receive_port=receive_port,
        is_relative_action=True,
    )
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
        recovery_scene(
            scene, evaluator, meta_info["task_data"], eval_config, default_config
        )
        eval_config["generation_config"]["goal"] = meta_info["task_data"]["goal"]
        evaluator.update_task_data(meta_info["task_data"], planning_data)
        remove_colliders(scene["object_list"]["defaultGroundPlane"].prim_path)
        for _ in range(50):
            scene["world"].step()
        evaluator.initialize(seed)
        finished = False
        for _ in tqdm(range(args.num_steps)):
            action = evaluator.request_action()
            scene["robot_info"]["robot_view_list"][0].set_joint_position_targets(action)
            scene["world"].step(render=True)
            evaluator.record()
            meshlist = get_current_meshList(
                scene["object_list"], scene["cacheDict"]["meshDict"]
            )
            pclist = meshlist_to_pclist(meshlist)
            finished = (
                finished + 1
                if (
                    check_finished(
                        eval_config["generation_config"]["goal"],
                        pclist,
                        scene["articulation_list"],
                    )
                    == 1
                )
                else 0
            )
            if finished != 0:
                print(f"finished {finished} times")
            if finished > 100:
                break
        meshlist = get_current_meshList(
            scene["object_list"], scene["cacheDict"]["meshDict"]
        )
        pclist = meshlist_to_pclist(meshlist)
        evaluator.finish(
            finished,
            check_finished(
                eval_config["generation_config"]["goal"],
                pclist,
                scene["articulation_list"],
            ),
        )
        seed = check_eval_finished(eval_config, default_config)
        if seed == -1:
            break
        seed = str(seed).zfill(3)
        make_dir(
            os.path.join(
                default_config["EVAL_RESULT_DIR"], eval_config["task_name"], seed
            )
        )
    clear_scene(scene, eval_config, current_dir)
simulation_app.close()
