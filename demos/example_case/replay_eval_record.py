"""
Replay actions saved by the evaluator in an episode meta_record.pkl.

This script is intended for checking whether a recorded evaluation action
sequence still reproduces a collision or penetration issue. It restores the
same saved task episode, then replays the per-step joint targets and base
motions stored in meta_record["model_output"].

Accepted arguments:
    -m, --meta-record:
        Path to the evaluator meta_record.pkl. The default points to the
        microwave_test_mini/003 episode that originally motivated this script.

    -cfg, --config:
        Optional YAML config containing evaluation_configs. When omitted, the
        script infers benchmark_id, task_name, and seed from the meta_record
        path and searches configs/tasks/<benchmark_id> for a matching task_name.
        If no source config is found there, it falls back to
        saved/tasks/<task_name>/<seed>/config.yaml.

    --task-name:
        Optional task_name override, for example
        ebench/mobile_manip/microwave_test_mini. Use this if the meta_record
        path layout cannot be inferred or if you want to replay with a specific
        task directory.

    --seed:
        Optional seed override, for example 003. By default it is inferred from
        the meta_record parent directory.

    -l, --local:
        Show the Isaac Sim UI. Without this flag the script runs headless.

    --start:
        First model_output action index to replay. Default: 0.

    --end:
        Stop before this model_output action index. Default: replay to the end.

    --settle-steps:
        Number of physics warmup steps after scene recovery and before replay.
        Default: 50.

    --hold-final:
        Number of rendered steps to hold after replay for inspection.
        Use -1 to keep the UI open until Isaac Sim is closed. Default: 120.

    --score-every:
        Print the metric score every N replayed actions. Use 0 to disable.
        Default: 50.

    --save-video:
        Save replay camera videos to disk. Without this flag no frames are
        collected.

    --video-dir:
        Output directory for replay videos. Default:
        <meta_record_parent>/replay_videos.

    --video-cameras:
        Optional camera names to save, for example left_camera right_camera.
        Default: save all cameras in the recovered scene.

    --video-fps:
        Output video fps. Default: 30.

    --video-stride:
        Save one video frame every N replay actions. Default: 1.

Examples:
    /isaac-sim/python.sh demos/example_case/replay_eval_record.py --local

    /isaac-sim/python.sh demos/example_case/replay_eval_record.py \\
        --local --hold-final -1

    /isaac-sim/python.sh demos/example_case/replay_eval_record.py \\
        -m /path/to/meta_record.pkl --start 300 --end 450 --local

    /isaac-sim/python.sh demos/example_case/replay_eval_record.py \\
        --save-video --video-cameras left_camera overlook_camera

    /isaac-sim/python.sh demos/example_case/replay_eval_record.py \
        -m /path/to/meta_record.pkl \
        --save-video \
        --video-cameras overlook_camera
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import copy
import glob
import os
import pickle
import sys

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(current_dir)


DEFAULT_META_RECORD = None


parser = ArgumentParser(
    description=(
        "Replay per-step arm/base actions saved in an evaluator meta_record.pkl."
    ),
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-m",
    "--meta-record",
    type=str,
    required=True,
    help=(
        "Path to evaluator meta_record.pkl. The path is also used to infer "
        "benchmark_id, task_name, and seed when those are not supplied."
    ),
)
parser.add_argument(
    "-cfg",
    "--config",
    type=str,
    default=None,
    help=(
        "Optional YAML config containing evaluation_configs. If omitted, search "
        "configs/tasks/<benchmark_id> for the inferred task_name, then fall back "
        "to saved/tasks/<task_name>/<seed>/config.yaml."
    ),
)
parser.add_argument(
    "--task-name",
    type=str,
    default=None,
    help=(
        "Optional task_name override, e.g. "
        "ebench/mobile_manip/microwave_test_mini."
    ),
)
parser.add_argument(
    "--seed",
    type=str,
    default=None,
    help="Optional seed override. It is zero-padded to three digits.",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="Show Isaac Sim UI. Without this flag the script runs headless.",
)
parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="First model_output action index to replay.",
)
parser.add_argument(
    "--end",
    type=int,
    default=None,
    help="Stop before this model_output action index. None means replay all actions.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=50,
    help="Physics warmup steps after scene recovery and before replay.",
)
parser.add_argument(
    "--hold-final",
    type=int,
    default=120,
    help="Extra rendered steps after replay. Use -1 to keep the UI open.",
)
parser.add_argument(
    "--score-every",
    type=int,
    default=50,
    help="Print metric score every N replayed actions. Use 0 to disable printing.",
)
parser.add_argument(
    "--save-video",
    action="store_true",
    help="Save replay camera videos to disk.",
)
parser.add_argument(
    "--video-dir",
    type=str,
    default=None,
    help=(
        "Directory for output mp4 files. None means "
        "<meta_record_parent>/replay_videos."
    ),
)
parser.add_argument(
    "--video-cameras",
    nargs="*",
    default=None,
    help=(
        "Camera names to save. Omit this argument to save all scene cameras. "
        "Use names from scene.camera_list, e.g. left_camera overlook_camera."
    ),
)
parser.add_argument(
    "--video-fps",
    type=float,
    default=30.0,
    help="Output video fps.",
)
parser.add_argument(
    "--video-stride",
    type=int,
    default=1,
    help="Save one video frame every N replay actions.",
)
args = parser.parse_args()

# SimulationApp forwards sys.argv to Kit, whose parser does not understand this
# script's replay arguments such as -m/--meta-record/--save-video.
sys.argv = [sys.argv[0]]

# Preload typing_extensions before SimulationApp adjusts sys.path.
import typing_extensions  # noqa: F401

from isaacsim import SimulationApp  # type: ignore

simulation_app = SimulationApp({"headless": not args.local})

import numpy as np

from genmanip.core.robot.dualarm_manip import DualArmEmbodiment
from genmanip.core.scene.scene import Scene
from genmanip.core.scene.scene_config import SceneConfig
from genmanip.utils.loader.domain_randomization import reset_scene
from genmanip.utils.loader.scene import clear_scene, recovery_scene
from genmanip.utils.standalone.file_utils import (
    load_default_config,
    load_dict_from_pkl,
    load_yaml,
)
from genmanip.utils.standalone.frame_utils import create_video_from_image_list
from genmanip.utils.standalone.utils import parse_eval_config, setup_logger
from genmanip.utils.standalone.version_utils import process_archived_config
from genmanip.utils.usd_utils import get_eval_camera_data, remove_colliders


simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
logger = setup_logger()


def _infer_episode_from_meta_record(meta_record_path: str) -> tuple[str, str, str]:
    path = Path(meta_record_path).resolve()
    parts = path.parts
    try:
        idx = parts.index("eval_results")
    except ValueError as exc:
        raise ValueError(
            f"Cannot infer episode from non-eval result path: {meta_record_path}"
        ) from exc

    rel = parts[idx + 1 : -1]
    if len(rel) < 4:
        raise ValueError(
            "Expected saved/eval_results/<benchmark>/<run_id>/<task_name...>/<seed>/meta_record.pkl, "
            f"got: {meta_record_path}"
        )
    benchmark_id = rel[0]
    seed = rel[-1]
    task_name = "/".join(rel[2:-1])
    return benchmark_id, task_name, seed


def _find_eval_config_by_task_name(task_name: str, benchmark_id: str) -> dict:
    config_root = Path(current_dir) / "configs" / "tasks" / benchmark_id
    patterns = [
        str(config_root / "**" / "*.yml"),
        str(config_root / "**" / "*.yaml"),
    ]
    for pattern in patterns:
        for config_path in glob.iglob(pattern, recursive=True):
            config = load_yaml(config_path)
            for eval_config in parse_eval_config(config):
                if eval_config.get("task_name") == task_name:
                    logger.info("Using eval config: %s", config_path)
                    return eval_config
    raise FileNotFoundError(
        f"Could not find an evaluation config with task_name={task_name!r} under {config_root}"
    )


def _load_eval_config(
    config_path: str | None, task_name: str, benchmark_id: str, seed: str
) -> dict:
    if config_path is not None:
        config = load_yaml(config_path)
        for eval_config in parse_eval_config(config):
            if eval_config.get("task_name") == task_name:
                return eval_config
        if len(parse_eval_config(config)) == 1:
            eval_config = parse_eval_config(config)[0]
            eval_config["task_name"] = task_name
            return eval_config
        raise ValueError(
            f"{config_path} has no evaluation config matching task_name={task_name!r}"
        )

    try:
        return _find_eval_config_by_task_name(task_name, benchmark_id)
    except FileNotFoundError:
        saved_config_path = (
            Path(current_dir) / "saved" / "tasks" / task_name / seed / "config.yaml"
        )
        logger.warning(
            "Falling back to saved task config and overriding task_name: %s",
            saved_config_path,
        )
        config = load_yaml(str(saved_config_path))
        eval_config = parse_eval_config(config)[0]
        eval_config["task_name"] = task_name
        return eval_config


def _load_meta_record(meta_record_path: str) -> dict:
    with open(meta_record_path, "rb") as f:
        meta_record = pickle.load(f)
    if "model_output" not in meta_record:
        raise KeyError(f"{meta_record_path} does not contain model_output")
    return meta_record


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _resolve_video_dir(meta_record_path: str, video_dir: str | None) -> Path:
    if video_dir is not None:
        return Path(video_dir).expanduser().resolve()
    return Path(meta_record_path).resolve().parent / "replay_videos"


def _init_video_frames(scene: Scene, camera_names: list[str] | None) -> dict:
    available = list(scene.camera_list.keys())
    selected = available if camera_names is None else camera_names
    missing = [name for name in selected if name not in scene.camera_list]
    if missing:
        raise ValueError(
            f"Unknown video camera(s): {missing}. Available cameras: {available}"
        )
    return {name: [] for name in selected}


def _record_video_frame(scene: Scene, video_frames: dict[str, list]) -> None:
    camera_data = get_eval_camera_data(scene.camera_list)
    for camera_name in video_frames:
        frame = camera_data.get(camera_name, {}).get("rgb")
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0:
            video_frames[camera_name].append(frame.copy())


def _save_videos(video_frames: dict[str, list], video_dir: Path, fps: float) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, frames in video_frames.items():
        if not frames:
            logger.warning("No frames captured for camera %s; skip video", camera_name)
            continue
        output_path = video_dir / f"{camera_name}.mp4"
        create_video_from_image_list(frames, str(output_path), fps=fps)
        logger.info("Saved replay video: %s", output_path)


def _replay_action(scene: Scene, action_item: dict, render: bool) -> float | None:
    robot = scene.robot_list[0]
    arm_action = _as_float_array(action_item["arm_action"])
    base_motion = _as_float_array(action_item.get("base_motion", [0.0, 0.0, 0.0]))

    robot.robot_view.set_joint_position_targets(
        arm_action,
        joint_indices=robot.default_dof_indices,
    )

    if isinstance(robot, DualArmEmbodiment):
        robot.delta_move_to(*base_motion)

    scene.world.step(render=render)
    if scene.metric_manager is None:
        return None
    return scene.metric_manager.step(scene)


def main() -> None:
    benchmark_id, inferred_task_name, inferred_seed = _infer_episode_from_meta_record(
        args.meta_record
    )
    task_name = args.task_name or inferred_task_name
    seed = str(args.seed or inferred_seed).zfill(3)

    default_config = load_default_config(current_dir, "default.json")
    eval_config = _load_eval_config(args.config, task_name, benchmark_id, seed)
    scene_config = SceneConfig(**process_archived_config(copy.deepcopy(eval_config)))

    meta_info_path = os.path.join(
        default_config["TASKS_DIR"], task_name, seed, "meta_info.pkl"
    )
    meta_info = load_dict_from_pkl(meta_info_path)
    meta_record = _load_meta_record(args.meta_record)
    model_output = meta_record["model_output"]
    end = len(model_output) if args.end is None else min(args.end, len(model_output))
    start = max(0, args.start)
    video_stride = max(1, args.video_stride)
    video_dir = _resolve_video_dir(args.meta_record, args.video_dir)

    logger.info("Task: %s seed=%s", task_name, seed)
    logger.info("Replay actions: [%s, %s) / %s", start, end, len(model_output))

    scene = Scene(scene_config=scene_config)
    scene.initialize(
        default_config,
        physics_dt=scene_config.physics_dt,
        rendering_dt=scene_config.rendering_dt,
        only_depth_rep_for_camera=True,
        only_color_rep_for_camera=True,
    )
    scene.post_initialize()

    try:
        reset_scene(scene)
        recovery_scene(scene, meta_info["task_data"], task_name, default_config)
        scene.build_metrics_manager(
            meta_info["task_data"]["goal"], skip_steps=10, succ_cnts=50
        )

        if "defaultGroundPlane" in scene.object_list:
            remove_colliders(scene.object_list["defaultGroundPlane"].prim_path)

        for _ in range(max(0, args.settle_steps)):
            scene.world.step(render=args.local)

        video_frames = None
        if args.save_video:
            video_frames = _init_video_frames(scene, args.video_cameras)
            logger.info(
                "Saving replay video frames for cameras=%s stride=%s output=%s",
                list(video_frames.keys()),
                video_stride,
                video_dir,
            )

        last_score = None
        for i in range(start, end):
            if not simulation_app.is_running():
                break
            last_score = _replay_action(scene, model_output[i], render=True)
            if video_frames is not None and (i - start) % video_stride == 0:
                _record_video_frame(scene, video_frames)
            if args.score_every > 0 and (i == start or (i + 1) % args.score_every == 0):
                logger.info("step=%s score=%s", i + 1, last_score)

        logger.info("Replay finished at action=%s score=%s", end, last_score)
        if video_frames is not None:
            _save_videos(video_frames, video_dir, args.video_fps)

        if args.hold_final < 0:
            while simulation_app.is_running():
                scene.world.step(render=True)
        else:
            for _ in range(args.hold_final):
                if not simulation_app.is_running():
                    break
                scene.world.step(render=True)
    finally:
        clear_scene(scene, scene_config, current_dir)
        simulation_app.close()


if __name__ == "__main__":
    main()
