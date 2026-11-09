"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import copy
import json
import os
import pathlib
import pickle
import warnings
import shutil
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Sequence

import numpy as np
import cv2

from huggingface_hub import snapshot_download
import lmdb

from genmanip.utils.rerun.rerun_utils import (
    RerunSaveStatus,
    log_episode_to_rerun,
)
from genmanip.utils.standalone.file_utils import load_yaml, check_benchmark_version
from genmanip.utils.standalone.frame_utils import (
    create_video_from_image_folder,
    save_image,
)
from genmanip.utils.standalone.file_utils import make_dir
from genmanip.utils.standalone.robot_utils import joint_position_to_end_effector_pose
from genmanip.utils.standalone.transform_utils import (
    pose_to_transform,
    transform_to_pose,
)

try:
    # if mediapy is installed, use it to create video
    import mediapy as mp
    from genmanip.utils.standalone.frame_utils import (
        create_video_from_image_list_with_mediapy as create_video_from_image_list,
    )
except (ImportError, ModuleNotFoundError):
    # if mediapy is not installed, use the original function
    from genmanip.utils.standalone.frame_utils import (
        create_video_from_image_list as create_video_from_image_list,
    )


def _is_relative_to(path: pathlib.Path, base: pathlib.Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _resolve_under(base_dir: pathlib.Path, raw_path: str) -> pathlib.Path:
    candidate = pathlib.Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not _is_relative_to(resolved, base_dir):
        raise ValueError(f"Path escapes allowed directory: {raw_path}")
    return resolved


def parse_configs_and_benchmark_id(
    config_or_benchmark_id: str, current_dir: str
) -> tuple[list[dict[str, Any]], str, bool]:
    config_or_benchmark_id = str(config_or_benchmark_id).strip()
    if config_or_benchmark_id == "":
        raise ValueError("config_or_benchmark_id must be a non-empty string")

    workspace_root = pathlib.Path(current_dir).resolve()
    tasks_root = (workspace_root / "configs" / "tasks").resolve()
    package_root = (
        workspace_root / "saved" / "assets" / "collected_packages"
    ).resolve()
    if not _is_relative_to(tasks_root, workspace_root):
        raise ValueError("tasks root escapes workspace")
    # if not _is_relative_to(package_root, workspace_root):
    #     raise ValueError("package root escapes workspace")

    is_genmanip_package = False

    # if config_or_benchmark_id indicates a yml file
    default_yml = (
        config_or_benchmark_id
        if config_or_benchmark_id.endswith(".yml")
        else f"{config_or_benchmark_id}.yml"
    )
    yml_path = _resolve_under(tasks_root, default_yml)
    if config_or_benchmark_id.endswith(".yml") or yml_path.exists():
        if not yml_path.exists():
            raise ValueError(f"Config file {yml_path} not found")
        config = load_yaml(str(yml_path))
        configs = [config]
        benchmark_id = yml_path.relative_to(tasks_root).parts[0]
        return configs, benchmark_id, is_genmanip_package

    # if config_or_benchmark_id is a directory
    config_dir = _resolve_under(tasks_root, config_or_benchmark_id)
    if config_dir.is_dir():
        # json files contain config relative paths or nested json path files,
        # yml files are config files
        json_path = config_dir / f"{config_dir.name}.json"
        if not json_path.exists():
            raise ValueError(
                f"the indicated path {json_path} does not contain a json file of path list"
            )

        config_paths = Queue()
        # read paths from json file
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            for item in data:
                config_paths.put(item)

        configs = []
        # deal with nested json files
        while not config_paths.empty():
            config_path = config_paths.get()
            if not isinstance(config_path, str) or config_path.strip() == "":
                raise ValueError("Invalid config path in json list")
            resolved_config_path = _resolve_under(tasks_root, config_path)
            if str(config_path).endswith(".yml"):
                config = load_yaml(str(resolved_config_path))
                assert (
                    config is not None
                ), f"the indicated path {resolved_config_path} does not contain a valid config"
                configs.append(config)
            elif str(config_path).endswith(".json"):
                # read paths from nested json file
                with open(resolved_config_path, "r", encoding="utf-8") as fp:
                    tmp_paths = json.load(fp)
                    for tmp_path in tmp_paths:
                        config_paths.put(tmp_path)
            else:
                raise ValueError(f"Unsupported config list entry: {config_path}")

        benchmark_id = config_dir.relative_to(tasks_root).parts[0]
        return configs, benchmark_id, is_genmanip_package

    local_package_config_path = _resolve_under(
        package_root, f"{config_or_benchmark_id.split('/')[-1]}/tasks/config.yaml"
    )
    if local_package_config_path.exists():
        print("Loading benchmark from local directory...")
        config = load_yaml(str(local_package_config_path))
        configs = [config]
        benchmark_id = config_or_benchmark_id
        is_genmanip_package = True
        return configs, benchmark_id, is_genmanip_package

    print("Downloading benchmark from HuggingFace...")
    if check_benchmark_version(config_or_benchmark_id):
        package_root.mkdir(parents=True, exist_ok=True)
        (workspace_root / "saved" / "tasks").mkdir(parents=True, exist_ok=True)
        package_dir = _resolve_under(
            package_root, config_or_benchmark_id.split("/")[-1]
        )
        snapshot_download(
            repo_id=config_or_benchmark_id,
            repo_type="dataset",
            local_dir=str(package_dir),
        )
        downloaded_config_path = _resolve_under(
            package_root, f"{config_or_benchmark_id.split('/')[-1]}/tasks/config.yaml"
        )
        if downloaded_config_path.exists():
            config = load_yaml(str(downloaded_config_path))
            benchmark_id = config_or_benchmark_id
            configs = [config]
        else:
            raise ValueError(f"Config file {config_or_benchmark_id} not found")
        is_genmanip_package = True
        return configs, benchmark_id, is_genmanip_package
    raise ValueError(f"Config file {config_or_benchmark_id} not found")


def get_scalar_data_from_lmdb(data_path: str, key: str | bytes) -> dict:
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"]["scalar_data"].index(key)
    key_key = meta_info["keys"]["scalar_data"][key_index]
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(key_key))
    return data


def ensure_empty_dir(path: str, max_retries: int = 3) -> None:
    """
    Ensure a directory exists and is empty.

    Robust to NFS race conditions (Errno 39: Directory not empty).
    Uses retry logic with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            make_dir(path)
            return
        except OSError as e:
            # Errno 39: Directory not empty (race condition during rmtree)
            # Errno 116: Stale file handle (NFS cache invalidation)
            if e.errno in (39, 116) and attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))  # Exponential backoff
                continue
            raise


def remove_dir_best_effort(path: str | None, max_retries: int = 3) -> bool:
    """Best-effort directory removal robust to NFS races and stale handles."""
    if not path:
        return True
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            if e.errno in (39, 116) and attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))
                continue
            return False
    return False


def apply_delta_pose(
    delta_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    delta_position = np.array(delta_pose[0])
    delta_orientation = np.array(delta_pose[1])
    current_position = np.array(current_pose[0])
    current_orientation = np.array(current_pose[1])
    delta_transform = pose_to_transform((delta_position, delta_orientation))
    current_transform = pose_to_transform((current_position, current_orientation))
    new_transform = delta_transform @ current_transform
    return transform_to_pose(new_transform)


def _normalize_ee_pose_actions(action: Any) -> list[tuple[Any, Any, Any]]:
    if len(action) == 3 and len(action[0]) == 3 and len(action[1]) == 4:
        return [(action[0], action[1], action[2])]
    return list(action)


def parse_embodiment_action(
    action: Any,
    *,
    control_type: str,
    embodiment: Any,
    is_relative_action: bool,
    last_joint_position: np.ndarray,
    last_ee_pose: list[tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]] | None]:
    embodiment_joint_num = embodiment.arm_dof_num + embodiment.gripper_dof_num

    if control_type == "joint_position":
        action_arr = np.array(action, dtype=np.float64)
        for i in range(len(action_arr) // embodiment_joint_num):
            if is_relative_action:
                start_idx = i * embodiment_joint_num
                end_idx = start_idx + embodiment.arm_dof_num
                action_arr[start_idx:end_idx] += last_joint_position[start_idx:end_idx]
        return action_arr, action_arr, last_ee_pose

    if control_type != "ee_pose":
        raise ValueError(f"Unsupported control_type: {control_type}")

    if last_ee_pose is None:
        raise ValueError("last_ee_pose is required for ee_pose control")

    actions = _normalize_ee_pose_actions(action)
    action_arr = np.array([])
    for i, act in enumerate(actions):
        position, orientation, gripper_width = act

        if is_relative_action:
            abs_pose = apply_delta_pose((position, orientation), last_ee_pose[i])
            position = abs_pose[0].tolist()
            orientation = abs_pose[1].tolist()
            last_ee_pose[i] = abs_pose
        else:
            last_ee_pose[i] = (np.asarray(position), np.asarray(orientation))

        if len(actions) == 1:
            planner = embodiment.planner
        elif i == 0:
            planner = embodiment.left_planner  # type: ignore[attr-defined]
        elif i == 1:
            planner = embodiment.right_planner  # type: ignore[attr-defined]
        else:
            raise ValueError("Invalid ee_pose action")

        if planner is None:
            raise ValueError("Planner is not initialized")

        start_idx = i * embodiment_joint_num
        end_idx = start_idx + embodiment.arm_dof_num
        cur_joint = embodiment.robot.get_joint_positions()[
            embodiment.default_dof_indices
        ][start_idx:end_idx]
        ik_result = planner.ik_single(position + orientation, cur_joint)
        if ik_result is None:
            ik_result = cur_joint

        action_arr = np.concatenate(
            [action_arr, ik_result[: embodiment.arm_dof_num], gripper_width]
        )

    return action_arr, action_arr, last_ee_pose


def parse_lmdb_data(lmdb_path: str) -> dict:
    data = {}
    meta_info = pickle.load(open(f"{lmdb_path}/meta_info.pkl", "rb"))
    arm_action = get_scalar_data_from_lmdb(lmdb_path, b"arm_action")
    gripper_action = get_scalar_data_from_lmdb(lmdb_path, b"gripper_action")
    try:
        base_motion = get_scalar_data_from_lmdb(lmdb_path, b"base_motion")
    except (KeyError, ValueError, TypeError, IndexError, lmdb.Error):
        warnings.warn(
            "base_motion is missing in lmdb data. Using zero-motion fallback.",
            stacklevel=2,
        )
        base_motion = [np.array([0.0, 0.0, 0.0])] * len(arm_action)
    key_action = []
    idx_dict = {}
    for frame_status in meta_info["task_data"]["frame_status"]:
        idx = int(frame_status.split("/")[0])
        if idx not in idx_dict:
            idx_dict[idx] = []
        idx_dict[idx].append(frame_status)
    for idx in idx_dict.keys():
        if (
            f"{idx}/pre_grasp" in meta_info["task_data"]["frame_status"]
            and f"{idx}/post_grasp" in meta_info["task_data"]["frame_status"]
            and f"{idx}/post_place" in meta_info["task_data"]["frame_status"]
        ):
            key_action.append(
                [
                    joint_position_to_end_effector_pose(
                        arm_action[
                            meta_info["task_data"]["frame_status"][f"{idx}/pre_grasp"]
                        ]
                    ),
                    joint_position_to_end_effector_pose(
                        arm_action[
                            meta_info["task_data"]["frame_status"][f"{idx}/post_grasp"]
                        ]
                    ),
                    joint_position_to_end_effector_pose(
                        arm_action[
                            meta_info["task_data"]["frame_status"][f"{idx}/post_place"]
                        ]
                    ),
                ]
            )
        else:
            for action_name in idx_dict[idx]:
                key_action.append(
                    [
                        joint_position_to_end_effector_pose(
                            arm_action[
                                meta_info["task_data"]["frame_status"][action_name]
                            ]
                        )
                    ]
                )
    data["key_action"] = key_action
    data["action"] = arm_action
    data["gripper_action"] = gripper_action
    data["base_motion"] = base_motion
    return data


def save_dict_to_json_atomic(data: dict, file_path: str) -> None:
    """Atomically write JSON data to file."""
    path = pathlib.Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                # Temp file may already have been removed.
                pass
            except OSError as exc:
                print(f"Warning: failed to remove temporary file {tmp_path}: {exc}")
        raise


@dataclass
class EpisodeRecorder:
    traj_log_dir: str
    result_info_dir: str
    camera_names: list[str]
    instruction: str
    save_process: bool = True
    save_every: int = 10
    image_list: dict[str, list[Any]] = field(default_factory=dict)
    meta_record: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    policy_inference_time_total_ms: float = 0.0
    episode_steps: int = 0

    @classmethod
    def create(
        cls,
        *,
        traj_log_dir: str,
        result_info_dir: str,
        camera_names: Sequence[str],
        instruction: str,
        save_process: bool = True,
        save_every: int = 10,
        with_render: bool = True,
    ) -> "EpisodeRecorder":
        ensure_empty_dir(traj_log_dir)
        traj_real = os.path.realpath(os.path.abspath(traj_log_dir))
        result_real = os.path.realpath(os.path.abspath(result_info_dir))
        if traj_real == result_real:
            raise ValueError(
                "result_info_dir must be different from traj_log_dir for evaluator outputs"
            )
        if save_process and with_render and save_every > 0:
            for camera_name in camera_names:
                make_dir(os.path.join(traj_log_dir, camera_name))
        image_list = {name: [] for name in camera_names}
        meta_record = {
            "joint_positions": [],
            "gripper_positions": [],
            "base_positions": [],
            "ee_pose": [],
            "instruction": instruction,
            "model_output": [],
            "robot_id": None,
        }
        return cls(
            traj_log_dir=traj_log_dir,
            result_info_dir=result_info_dir,
            camera_names=list(camera_names),
            instruction=instruction,
            save_process=save_process,
            save_every=save_every,
            image_list=image_list,
            meta_record=meta_record,
        )

    def record_model_output(self, *, arm_action: Any, base_motion: Any) -> None:
        if not self.save_process:
            return
        self.meta_record["model_output"].append(
            {"arm_action": arm_action, "base_motion": base_motion}
        )

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0

    @staticmethod
    def _encode_frame(frame: np.ndarray, jpeg_quality: int = 85) -> bytes | None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            return None
        return buf.tobytes()

    def record_obs(
        self,
        _obs: dict,
        precomputed_jpeg: dict | None = None,
        raw_frames: dict | None = None,
    ) -> int:
        if not self.save_process:
            self.steps += 1
            return self.steps
        # Avoid copy.deepcopy(_obs) — the raw RGB frames (~11 MB/step for 4
        # cameras) are only read here, not mutated, and the caller serialises
        # obs synchronously after this call returns. Deep-copying every step
        # was allocating ~30 GB per episode.
        obs = _obs
        precomputed_jpeg = precomputed_jpeg or {}
        for camera_name in self.camera_names:
            key = f"video.{camera_name}_view"
            if key not in obs:
                continue
            # Prefer pre-encoded JPEG bytes (avoids a second cv2 encode).
            pre = precomputed_jpeg.get(camera_name)
            if pre is not None:
                encoded = pre.get("data")
                if encoded:
                    self.image_list[camera_name].append(encoded)
                    # Still honour PNG save_every using the raw frame if available.
                    if (
                        self.save_every > 0
                        and self.steps % self.save_every == 0
                        and raw_frames is not None
                        and camera_name in raw_frames
                    ):
                        camera_dir = os.path.join(self.traj_log_dir, camera_name)
                        if os.path.exists(camera_dir):
                            save_image(
                                raw_frames[camera_name],
                                os.path.join(
                                    camera_dir,
                                    f"{str(self.steps).zfill(5)}.png",
                                ),
                            )
                    continue

            frame = obs[key]
            if not self._is_valid_frame(frame):
                warnings.warn(
                    f"Skip invalid frame for camera '{camera_name}' at step {self.steps} "
                    f"under {self.result_info_dir}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            encoded = self._encode_frame(frame)
            if encoded is None:
                warnings.warn(
                    f"JPEG encode failed for camera '{camera_name}' at step {self.steps} "
                    f"under {self.result_info_dir}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            self.image_list[camera_name].append(encoded)
            should_save_frame = (
                self.save_every > 0 and self.steps % self.save_every == 0
            )
            if should_save_frame:
                camera_dir = os.path.join(self.traj_log_dir, camera_name)
                if os.path.exists(camera_dir):
                    save_image(
                        frame,
                        os.path.join(camera_dir, f"{str(self.steps).zfill(5)}.png"),
                    )

        if "state.joints" in obs:
            self.meta_record["joint_positions"].append(obs["state.joints"])
        if "state.gripper" in obs:
            self.meta_record["gripper_positions"].append(obs["state.gripper"])
        if "state.base" in obs:
            self.meta_record["base_positions"].append(obs["state.base"])
        if "state.ee_pose" in obs:
            self.meta_record["ee_pose"].append(obs["state.ee_pose"])
        if self.meta_record.get("robot_id") is None and "robot_id" in obs:
            self.meta_record["robot_id"] = obs["robot_id"]
        self.steps += 1
        return self.steps

    def save_meta_record(self) -> None:
        with open(os.path.join(self.result_info_dir, "meta_record.pkl"), "wb") as f:
            pickle.dump(self.meta_record, f)

    def build_finalize_payload(
        self,
        score: float,
        episode_start_time: str,
        episode_end_time: str,
        metric_score: list,
        episode_steps: int,
    ) -> dict[str, Any]:
        return {
            "traj_log_dir": self.traj_log_dir,
            "result_info_dir": self.result_info_dir,
            "camera_names": list(self.camera_names),
            "save_process": self.save_process,
            "image_list": self.image_list,
            "meta_record": self.meta_record,
            "policy_inference_time_total_ms": self.policy_inference_time_total_ms,
            "episode_steps": episode_steps,
            "score": score,
            "episode_start_time": episode_start_time,
            "episode_end_time": episode_end_time,
            "metric_score": metric_score,
        }

    def finalize(
        self,
        score: float,
        episode_start_time: str,
        episode_end_time: str,
        metric_score: list,
    ) -> None:
        if self.save_process:
            action_list = []
            for item in self.meta_record["model_output"]:
                action_list.append(item["arm_action"])
            action_list = np.array(action_list) if action_list else None
            joint_list = (
                np.array(self.meta_record["joint_positions"])
                if self.meta_record["joint_positions"]
                else None
            )
            gripper_list = (
                np.array(self.meta_record["gripper_positions"])
                if self.meta_record["gripper_positions"]
                else None
            )
            base_list = (
                np.array(self.meta_record["base_positions"])
                if self.meta_record["base_positions"]
                else None
            )
            rrd_path = os.path.join(self.result_info_dir, "episode.rrd")
            rerun_status = log_episode_to_rerun(
                self.image_list,
                action_list,
                joint_list,
                gripper_list,
                base_list,
                rrd_path=rrd_path,
                robot_id=self.meta_record.get("robot_id"),
            )
            if rerun_status not in (
                RerunSaveStatus.SAVED,
                RerunSaveStatus.SKIPPED_NO_SDK,
            ):
                raise RuntimeError(
                    "Refuse to write result_info.json because episode.rrd was not "
                    f"saved: status={rerun_status.value}"
                )
            if rerun_status != RerunSaveStatus.SKIPPED_NO_SDK:
                if not os.path.exists(rrd_path):
                    raise RuntimeError(
                        "Refuse to write result_info.json because episode.rrd is "
                        f"missing: {rrd_path}"
                    )
                if os.path.getsize(rrd_path) <= 0:
                    raise RuntimeError(
                        "Refuse to write result_info.json because episode.rrd is "
                        f"empty: {rrd_path}"
                    )

            for camera_name in self.camera_names:
                if len(self.image_list[camera_name]) > 0:
                    create_video_from_image_list(
                        self.image_list[camera_name],
                        os.path.join(self.result_info_dir, camera_name + ".mp4"),
                    )
                self.image_list[camera_name] = []

            self.save_meta_record()

        remove_dir_best_effort(self.traj_log_dir)

        save_dict_to_json_atomic(
            {
                "score": score,
                "policy_avg_inference_time_ms": (
                    self.policy_inference_time_total_ms / self.episode_steps
                    if self.episode_steps > 0
                    else -1
                ),
                "steps": self.episode_steps,
                "success_rate": 1 if abs(score - 1) < 1e-6 else 0,
                "log_info": {
                    "episode_start_time": episode_start_time,
                    "episode_end_time": episode_end_time,
                    "metric_score": metric_score,
                },
            },
            os.path.join(self.result_info_dir, "result_info.json"),
        )


def finalize_episode_recorder_payload(payload: dict[str, Any]) -> None:
    recorder = EpisodeRecorder(
        traj_log_dir=payload["traj_log_dir"],
        result_info_dir=payload["result_info_dir"],
        camera_names=list(payload["camera_names"]),
        instruction=str(payload["meta_record"].get("instruction", "")),
        save_process=bool(payload["save_process"]),
        image_list=payload["image_list"],
        meta_record=payload["meta_record"],
        policy_inference_time_total_ms=float(payload["policy_inference_time_total_ms"]),
        episode_steps=int(payload["episode_steps"]),
    )
    recorder.finalize(
        float(payload["score"]),
        str(payload["episode_start_time"]),
        str(payload["episode_end_time"]),
        list(payload["metric_score"]),
    )
