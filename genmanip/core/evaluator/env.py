from datetime import datetime
import time
import numpy as np
import os
from typing import Any
import uuid

from genmanip.core.robot.dualarm_manip import DualArmEmbodiment
from genmanip.core.scene.scene import Scene
from genmanip.core.scene.scene_config import SceneConfig
from genmanip.core.evaluator.utils import (
    EpisodeRecorder,
    parse_embodiment_action,
    remove_dir_best_effort,
)
from genmanip.utils.loader.domain_randomization import (
    apply_material_overrides,
    random_texture_for_eval,
    reset_scene,
)
from genmanip.utils.loader.scene import (
    clear_scene,
    recovery_scene,
)
from genmanip.utils.standalone.file_utils import (
    load_dict_from_pkl,
    make_dir,
)
from genmanip.utils.standalone.io_utils import serialize_data, _jpeg
from genmanip.utils.standalone.utils import setup_logger, tuple_to_list
from genmanip.utils.standalone.version_utils import process_archived_config
from genmanip.utils.usd_utils import get_eval_camera_data, remove_colliders


class IsaacEvalEnvRay:
    def __init__(
        self,
        args,
        simulation_app,
        default_config,
        current_dir,
        benchmark_id: str = "new-benchmark",
    ):
        self.args = args
        self.simulation_app = simulation_app
        self.default_config = default_config
        self.current_dir = current_dir
        self.benchmark_id = benchmark_id

        self.logger = setup_logger()

        self.current_eval_config = None
        self.scene_config = None
        self.current_seed = None

        self.scene = None

        self._step_cnt = 0
        self.num_steps = args.num_steps
        self.done = False

        self.current_joint_position = None
        self.last_joint_position: np.ndarray | None = None
        self.last_ee_pose: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.instruction = None
        self._recorder: EpisodeRecorder | None = None
        self.traj_log_dir = None
        self.traj_uid = uuid.uuid4().hex

        self.run_id = args.run_id
        self.save_process = bool(getattr(args, "save_process", True))
        self.episode_recorder_save_every = max(
            0, int(getattr(args, "episode_recorder_save_every", 10))
        )

        self.episode_start_time = None
        self.episode_end_time = None
        self.invalid_state_tail_steps = 30
        self.forced_episode_score: float | None = None
        self._policy_inference_started_at: float | None = None

    def _detect_invalid_state(
        self,
        *,
        embodiment,
        prev_arm_position: np.ndarray,
        prev_gripper_position: np.ndarray,
        prev_base_position: np.ndarray,
        cached_joint_positions=None,
    ) -> str | None:
        # OPT: reuse cached full joint positions if provided.
        if cached_joint_positions is None:
            full_jp = embodiment.robot.get_joint_positions()
        else:
            full_jp = cached_joint_positions
        current_joint_position = full_jp[embodiment.default_dof_indices]
        current_arm_position = current_joint_position[
            embodiment.default_arm_dof_indices
        ]
        current_gripper_position = current_joint_position[
            embodiment.default_gripper_dof_indices
        ]

        if isinstance(embodiment, DualArmEmbodiment):
            current_base_position = full_jp[embodiment.base_dof_indices]
        else:
            current_base_position = np.zeros(3, dtype=np.float64)

        finite_checks = {
            "arm": np.asarray(current_arm_position, dtype=np.float64),
            "gripper": np.asarray(current_gripper_position, dtype=np.float64),
            "base": np.asarray(current_base_position, dtype=np.float64),
        }
        for name, value in finite_checks.items():
            if not np.all(np.isfinite(value)):
                return f"non_finite_{name}_state"

        # Arm joint abs limit (rad).
        arm_abs_limit = 10.0
        # Arm joint per-step delta limit (rad).
        arm_step_limit = 1.0
        # Base x/y abs limit (m).
        base_xy_abs_limit = 5.0
        # Base yaw abs limit (rad).
        base_yaw_abs_limit = 4.0 * np.pi
        # Base x/y per-step delta limit (m).
        base_xy_step_limit = 0.2
        # Base yaw per-step delta limit (rad).
        base_yaw_step_limit = np.deg2rad(20.0)

        if np.any(np.abs(current_arm_position) > arm_abs_limit):
            return "arm_state_out_of_range"
        if np.any(np.abs(current_arm_position - prev_arm_position) > arm_step_limit):
            return "arm_state_jump_too_large"

        # Gripper state range (m or joint units).
        gripper_open_max = (
            max(embodiment.gripper_open) if embodiment.gripper_open else 0.0
        )
        gripper_lower_limit = -0.01
        gripper_upper_limit = gripper_open_max + 0.01
        if np.any(current_gripper_position < gripper_lower_limit) or np.any(
            current_gripper_position > gripper_upper_limit
        ):
            return "gripper_state_out_of_range"
        # Gripper per-step delta limit (m or joint units).
        if np.any(np.abs(current_gripper_position - prev_gripper_position) > 0.05):
            return "gripper_state_jump_too_large"

        if np.any(np.abs(current_base_position[:2]) > base_xy_abs_limit):
            return "base_xy_out_of_range"
        if abs(current_base_position[2]) > base_yaw_abs_limit:
            return "base_yaw_out_of_range"
        if np.any(
            np.abs(current_base_position[:2] - prev_base_position[:2])
            > base_xy_step_limit
        ):
            return "base_xy_jump_too_large"
        if abs(current_base_position[2] - prev_base_position[2]) > base_yaw_step_limit:
            return "base_yaw_jump_too_large"

        return None

    def _require_scene(self) -> Scene:
        if self.scene is None:
            raise ValueError("Scene not initialized")
        return self.scene

    def _require_scene_config(self) -> SceneConfig:
        if self.scene_config is None:
            raise ValueError("Current eval config not initialized")
        return self.scene_config

    def _require_seed(self) -> str:
        if self.current_seed is None:
            raise ValueError("Current seed not initialized")
        return self.current_seed

    def _require_recorder(self) -> EpisodeRecorder:
        if self._recorder is None:
            raise ValueError("Episode recorder not initialized")
        return self._recorder

    def _get_embodiment(self):
        scene = self._require_scene()
        if len(scene.robot_list) == 0:
            raise ValueError("No robot in scene")
        return scene.robot_list[0]

    def _maybe_initialize_task(self, current_eval_config: dict) -> None:
        processed = process_archived_config(current_eval_config)
        if (
            self.current_eval_config is not None
            and processed["task_name"] == self.current_eval_config["task_name"]
        ):
            self.logger.info(
                "Task unchanged; reusing existing scene for task=%s",
                processed["task_name"],
            )
            return

        if self.scene_config is not None and self.scene is not None:
            self.logger.info(
                "Clearing previous scene for task=%s",
                self.current_eval_config["task_name"],
            )
            clear_scene(self.scene, self.scene_config, self.current_dir)
            self.scene = None
            self.logger.info("Previous scene cleared")

        self.current_eval_config = processed
        self.logger.info(f"Starting new task: {self.current_eval_config['task_name']}")

        self.logger.info("Building SceneConfig")
        self.scene_config = SceneConfig(**processed)
        self.logger.info(
            "SceneConfig ready: task=%s num_steps=%s physics_dt=%s rendering_dt=%s",
            self.scene_config.task_name,
            self.scene_config.num_steps,
            self.scene_config.physics_dt,
            self.scene_config.rendering_dt,
        )
        self.num_steps = (
            self.scene_config.num_steps
            if self.args.num_steps is None
            else self.args.num_steps
        )

        self.logger.info("Constructing Scene object")
        self.scene = Scene(scene_config=self.scene_config)
        self.logger.info("Scene object constructed")
        self.logger.info("Initializing Scene")
        self.scene.initialize(
            self.default_config,
            physics_dt=self.scene_config.physics_dt,
            rendering_dt=self.scene_config.rendering_dt,
            only_depth_rep_for_camera=True,
            only_color_rep_for_camera=True,
        )
        self.logger.info("Scene.initialize completed")
        self.logger.info("Running Scene.post_initialize")
        self.scene.post_initialize()
        self.logger.info("Scene.post_initialize completed")

        if "camera1" in self.scene.camera_list:
            self.scene.camera_list.pop("camera1")
            self.logger.info("Removed camera1 from scene camera list")

    def reset(self, seed, current_eval_config, default_config: dict | None = None):
        if default_config is not None:
            self.default_config = default_config
            self.logger.info("Updated default_config for reset")
        self.logger.info(
            "Reset start: seed=%s requested_task=%s",
            seed,
            process_archived_config(current_eval_config).get("task_name", "<unknown>"),
        )
        self._maybe_initialize_task(current_eval_config)

        self.current_seed = seed
        self.traj_uid = uuid.uuid4().hex
        self.episode_start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        self.logger.info(f"Resetting episode: {self.current_seed}")

        self.logger.info("Setting up episode layout")
        self._setup_episode_layout()
        self.logger.info("Episode layout ready")

        self._step_cnt = 0
        self.forced_episode_score = None
        scene = self._require_scene()
        _ = self._require_scene_config()

        self.done = False
        embodiment = scene.robot_list[0]

        self.logger.info("Reading initial robot joint positions")
        self.current_joint_position = embodiment.robot.get_joint_positions()[
            embodiment.default_dof_indices
        ]
        self.last_joint_position = self.current_joint_position.copy()
        self.logger.info("Initializing last end-effector pose")
        self.last_ee_pose = self._init_last_ee_pose()

        self.logger.info("Collecting initial observation")
        obs = self.get_obs()
        self.logger.info("Initial observation collected")
        info = {
            "task": self._require_scene_config().task_name,
            "seed": self.current_seed,
        }
        self.logger.info("Reset complete: task=%s seed=%s", info["task"], info["seed"])
        self._policy_inference_started_at = time.perf_counter()
        return obs, info

    def _init_last_ee_pose(self) -> list[tuple[np.ndarray, np.ndarray]]:
        embodiment = self._get_embodiment()
        joint_positions = embodiment.robot.get_joint_positions()
        if isinstance(embodiment, DualArmEmbodiment):
            left_pose, right_pose = embodiment.fk_single(joint_positions)
            return [left_pose, right_pose]  # type: ignore
        return [embodiment.fk_single(joint_positions)]

    def get_obs(self, cached_joint_positions=None):
        scene = self._require_scene()
        embodiment = scene.robot_list[0]
        recorder = self._require_recorder()
        timestep = recorder.steps
        scene_config = self._require_scene_config()
        seed = self._require_seed()
        task_name = scene_config.task_name

        # OPT: cache full get_joint_positions() once; avoids 3x GPU->CPU syncs.
        if cached_joint_positions is None:
            full_jp = embodiment.robot.get_joint_positions()
        else:
            full_jp = cached_joint_positions

        self.current_joint_position = full_jp[embodiment.default_dof_indices]
        if isinstance(embodiment, DualArmEmbodiment):
            current_base_position = full_jp[embodiment.base_dof_indices]
        else:
            current_base_position = [0.0, 0.0, 0.0]
        current_arm_position = self.current_joint_position[
            embodiment.default_arm_dof_indices
        ]
        current_gripper_position = self.current_joint_position[
            embodiment.default_gripper_dof_indices
        ]

        _t0 = time.perf_counter()
        camera_data = {}
        if not self.args.without_render:
            camera_data = get_eval_camera_data(scene.camera_list)
        self._phase_add("get_obs.camera", time.perf_counter() - _t0)

        _t0 = time.perf_counter()
        ee_pose = embodiment.fk_single(full_jp)
        self._phase_add("get_obs.fk", time.perf_counter() - _t0)

        if self.instruction is None:
            raise ValueError("Instruction not initialized")
        obs = {
            "instruction": self.instruction,
            "state.joints": current_arm_position,
            "state.gripper": current_gripper_position,
            "state.base": current_base_position,
            "state.ee_pose": tuple_to_list(ee_pose),
            "timestep": timestep,
            "reset": timestep == 0,
            "episode_id": str(
                os.path.join(self.benchmark_id, self.run_id, task_name, seed)
            ),
            "robot_id": scene_config.robots[0].type,
        }

        # OPT: encode each camera RGB once with TurboJPEG, then share the
        # bytes between recorder storage and serialize_data wire format.
        _t0 = time.perf_counter()
        from turbojpeg import TJPF_RGB

        encoded_cams: dict[str, dict] = {}
        raw_cams: dict[str, np.ndarray] = {}
        for camera, img in camera_data.items():
            rgb = img["rgb"]
            key = f"video.{camera}_view"
            if (
                isinstance(rgb, np.ndarray)
                and rgb.dtype == np.uint8
                and rgb.ndim == 3
                and rgb.shape[2] in (3, 4)
            ):
                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]
                raw_cams[camera] = rgb
                jb = _jpeg.encode(rgb, quality=85, pixel_format=TJPF_RGB)
                encoded_cams[camera] = {
                    "type": "jpeg_bytes",
                    "dtype": str(rgb.dtype),
                    "shape": rgb.shape,
                    "data": jb,
                }
                obs[key] = encoded_cams[camera]
            else:
                obs[key] = rgb
        self._phase_add("get_obs.jpeg", time.perf_counter() - _t0)

        _t0 = time.perf_counter()
        recorder.record_obs(obs, precomputed_jpeg=encoded_cams, raw_frames=raw_cams)
        self._phase_add("get_obs.record", time.perf_counter() - _t0)

        _t0 = time.perf_counter()
        out = serialize_data(obs)
        self._phase_add("get_obs.serialize", time.perf_counter() - _t0)
        return out

    def _record_invalid_state_tail(self) -> None:
        scene = self._require_scene()
        steps = max(0, int(self.invalid_state_tail_steps))
        if steps <= 0:
            return

        self.logger.info(
            "Recording %s additional steps after invalid state before termination",
            steps,
        )
        for _ in range(steps):
            scene.world.step(render=not self.args.without_render)
            self.get_obs()

    # ── Phase timer (for optimization profiling) ──────────────────
    def _phase_add(self, label: str, dt: float) -> None:
        pt = getattr(self, "_phase_totals", None)
        if pt is None:
            self._phase_totals = {}
            self._phase_counts = {}
            pt = self._phase_totals
        self._phase_totals[label] = pt.get(label, 0.0) + dt
        self._phase_counts[label] = self._phase_counts.get(label, 0) + 1

    def _phase_report(self) -> None:
        pt = getattr(self, "_phase_totals", None)
        if not pt:
            return
        total = sum(pt.values())
        parts = sorted(pt.items(), key=lambda kv: -kv[1])
        msg = (
            "[phase] total=%.3fs" % total
            + " | "
            + " ".join(
                "%s=%.1fms(%d,%.1f%%)"
                % (
                    k,
                    1000 * v / max(1, self._phase_counts[k]),
                    self._phase_counts[k],
                    100 * v / total if total else 0.0,
                )
                for k, v in parts
            )
        )
        self.logger.debug(msg)
        self._phase_totals.clear()
        self._phase_counts.clear()

    def step(
        self,
        action: dict[str, Any],
        skip_obs: bool = False,
        skip_render: bool = False,
        subframes: int = 0,
    ):
        if not self.simulation_app.is_running():
            return None, 0, True, {"error": "Sim closed"}
        if self.scene is None:
            return None, 0, True, {"error": "Scene not initialized"}
        if self.scene_config is None:
            return None, 0, True, {"error": "Current eval config not initialized"}
        if self._recorder is None:
            return None, 0, True, {"error": "Episode recorder not initialized"}

        # Done
        if self.done:
            return None, 0, True, {"info": "Done"}

        if self._policy_inference_started_at is not None:
            inference_time_ms = (
                time.perf_counter() - self._policy_inference_started_at
            ) * 1000.0
            self._recorder.policy_inference_time_total_ms += inference_time_ms
            self._policy_inference_started_at = None

        # 3-1. Set joint targets
        embodiment = self.scene.robot_list[0]

        _t = time.perf_counter()
        # OPT: one get_joint_positions for prev_* (was 1-2 calls).
        prev_full_jp = embodiment.robot.get_joint_positions()
        prev_joint_position = prev_full_jp[embodiment.default_dof_indices]
        prev_arm_position = prev_joint_position[embodiment.default_arm_dof_indices]
        prev_gripper_position = prev_joint_position[
            embodiment.default_gripper_dof_indices
        ]
        if isinstance(embodiment, DualArmEmbodiment):
            prev_base_position = prev_full_jp[embodiment.base_dof_indices]
        else:
            prev_base_position = np.zeros(3, dtype=np.float64)
        self._phase_add("prev_state", time.perf_counter() - _t)

        _t = time.perf_counter()
        base_motion = action.get("base_motion", [0.0, 0.0, 0.0])
        base_is_rel = action.get("base_is_rel", True)
        arm_action = self.parse_action(
            action["action"],
            control_type=action["control_type"],
            is_relative_action=action.get("is_rel", self.args.is_relative_action),
        )
        self._recorder.record_model_output(
            arm_action=arm_action, base_motion=base_motion
        )
        self.scene.robot_list[0].robot_view.set_joint_position_targets(
            arm_action,
            joint_indices=self.scene.robot_list[0].default_dof_indices,
        )

        if isinstance(self.scene.robot_list[0], DualArmEmbodiment):
            if base_is_rel:
                self.scene.robot_list[0].delta_move_to(*base_motion)
            else:
                target_base_pose = np.asarray(base_motion, dtype=float).copy()
                if target_base_pose.shape[0] != 3:
                    raise ValueError("base_motion must be a 3-element sequence")
                target_base_pose[2] = np.deg2rad(target_base_pose[2])
                self.scene.robot_list[0].target_base_pose = target_base_pose
                self.scene.robot_list[0].robot_view.set_joint_position_targets(
                    target_base_pose,
                    joint_indices=self.scene.robot_list[0].base_dof_indices,
                )
        self._phase_add("set_action", time.perf_counter() - _t)

        # 3-2. Step Physics. skip_render is a lite-mode hint: on intermediate
        # chunk steps we do not need fresh camera frames, so we can tell Isaac
        # Sim to skip the rendering pass entirely. Only honoured when the
        # server-side recorder isn't capturing videos (save_process=False),
        # otherwise the recorder would miss frames.
        _lite_render = skip_render and not self.save_process
        _render_on = (not self.args.without_render) and not _lite_render
        _t = time.perf_counter()
        self.scene.world.step(render=_render_on)
        self._phase_add("world_step", time.perf_counter() - _t)

        # Optional subframe catch-up: after the main render pass, call
        # scene.world.render() N times so the RTX renderer accumulates
        # TAA / denoiser / path-traced GI history before the camera
        # annotator is read. Only useful on the final-of-chunk step, right
        # before get_obs runs. Has no effect when rendering is already off.
        if _render_on and subframes > 0:
            _ts = time.perf_counter()
            for _ in range(subframes):
                self.scene.world.render()
            self._phase_add("subframes", time.perf_counter() - _ts)

        # OPT: cache get_joint_positions() post-step; reused by invalid check + get_obs.
        _t = time.perf_counter()
        cur_full_jp = embodiment.robot.get_joint_positions()
        self._phase_add("post_jp", time.perf_counter() - _t)

        _t = time.perf_counter()
        invalid_reason = self._detect_invalid_state(
            embodiment=embodiment,
            prev_arm_position=prev_arm_position,
            prev_gripper_position=prev_gripper_position,
            prev_base_position=prev_base_position,
            cached_joint_positions=cur_full_jp,
        )
        self._phase_add("invalid_check", time.perf_counter() - _t)

        if invalid_reason is not None:
            self.logger.warning(
                "Invalid robot state detected at step %s: %s",
                self._step_cnt,
                invalid_reason,
            )
            self._record_invalid_state_tail()
            # Metrics are not updated on the invalid frame or during the tail.
            self.forced_episode_score = (
                self.scene.metric_manager.calc_overall_score()
                if self.scene_config.preserve_score_on_invalid_state
                else 0.0
            )
            self.done = True
            return (
                self.get_obs(cached_joint_positions=cur_full_jp),
                self.forced_episode_score,
                True,
                {
                    "info": self.forced_episode_score,
                    "termination_reason": invalid_reason,
                },
            )

        # 3-4 & 3-5. Check Success Logic
        _t = time.perf_counter()
        score = self.scene.metric_manager.step(self.scene)
        self._phase_add("metric_step", time.perf_counter() - _t)
        self._step_cnt += 1

        # Check Termination
        done = False
        if score and abs(score - 1) < 1e-6:
            self.logger.info("Goal reached consistently. Done.")
            done = True
        elif self._step_cnt >= self.num_steps:
            done = True

        # Lite path: client said "I do not care about this obs", and the
        # server recorder is off, so we can skip the expensive camera USD
        # readout + JPEG encode. Final step in a chunk always takes the full
        # path so the client still gets a valid observation.
        _lite_obs = skip_obs and not self.save_process
        if _lite_obs:
            # Keep the recorder step counter in sync even though we are
            # intentionally skipping get_obs on intermediate chunk steps.
            # Otherwise the final-of-chunk obs would report timestep=N-chunks
            # instead of the real step index.
            if self._recorder is not None:
                self._recorder.steps += 1
            self._phase_add("get_obs_skipped", 0.0)
            obs = None
        else:
            _t = time.perf_counter()
            obs = self.get_obs(cached_joint_positions=cur_full_jp)
            self._phase_add("get_obs", time.perf_counter() - _t)

        if self._step_cnt % 100 == 0:
            self._phase_report()

        if not skip_obs:
            self._policy_inference_started_at = time.perf_counter()

        return (
            obs,
            score,
            done,
            {"info": score},
        )

    def post_episode_process(self, done_info):
        if self.done and self._recorder is None:
            return None
        if self.scene is None:
            return None
        if self.scene_config is None:
            return None

        episode_score = (
            self.forced_episode_score
            if self.forced_episode_score is not None
            else self.scene.metric_manager.calc_overall_score()
        )

        self.episode_end_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        self.done = True
        recorder = self._require_recorder()
        if self.episode_start_time is None or self.episode_end_time is None:
            raise ValueError("Episode start or end time not initialized")
        finalize_payload = recorder.build_finalize_payload(
            episode_score,
            self.episode_start_time,
            self.episode_end_time,
            self.scene.metric_manager.metric_score,
            self._step_cnt,
        )
        self._recorder = None
        self.traj_log_dir = None
        return {
            "score": episode_score,
            "finalize_payload": finalize_payload,
        }

    def close(self):
        self.simulation_app.close()

    def abort_episode(self) -> None:
        """Best-effort cleanup for an unfinished episode after lock loss."""
        removed = remove_dir_best_effort(self.traj_log_dir)
        if not removed and self.traj_log_dir is not None:
            self.logger.warning(
                "Failed to remove unfinished episode image dir: %s",
                self.traj_log_dir,
            )
        self._recorder = None
        self.traj_log_dir = None
        self.done = True

    def _setup_episode_layout(self):
        scene = self._require_scene()
        scene_config = self._require_scene_config()
        seed = self._require_seed()
        task_dir = self.default_config["TASKS_DIR"]
        task_name = scene_config.task_name
        # Load data
        meta_info = load_dict_from_pkl(
            os.path.join(task_dir, task_name, f"{seed}/meta_info.pkl")
        )
        reset_scene(scene)

        # Recovery Scene
        recovery_scene(
            scene,
            meta_info["task_data"],
            task_name,
            self.default_config,
        )

        # Randomization
        if self.args.random_randomization:
            random_texture_for_eval(scene, self.default_config, scene_config)
        apply_material_overrides(scene.world.stage, scene_config)

        # Update Config & Evaluator
        self.instruction = meta_info["task_data"]["instruction"]

        # Remove Ground Plane Colliders
        if "defaultGroundPlane" in scene.object_list:
            remove_colliders(scene.object_list["defaultGroundPlane"].prim_path)

        # Warmup
        for _ in range(scene_config.warmup_steps):
            scene.world.step()

        base_traj_dir = os.path.join(
            self.default_config["EVAL_RESULT_DIR"],
            self.benchmark_id,
            self.run_id,
            task_name,
            seed,
        )
        make_dir(base_traj_dir)
        self.traj_log_dir = os.path.join(base_traj_dir, self.traj_uid)
        self._recorder = EpisodeRecorder.create(
            traj_log_dir=self.traj_log_dir,
            result_info_dir=base_traj_dir,
            camera_names=list(scene.camera_list.keys()),
            instruction=self.instruction,
            save_process=self.save_process,
            save_every=self.episode_recorder_save_every,
            with_render=not self.args.without_render,
        )

        scene.build_metrics_manager(
            meta_info["task_data"]["goal"], skip_steps=10, succ_cnts=50
        )

    def parse_action(
        self,
        action,
        control_type: str = "joint_position",
        is_relative_action: bool = False,
    ) -> np.ndarray:
        embodiment = self._get_embodiment()
        if self.last_joint_position is None:
            raise ValueError("Last joint position not initialized")
        action_arr, last_joint_position, last_ee_pose = parse_embodiment_action(
            action,
            control_type=control_type,
            embodiment=embodiment,
            is_relative_action=is_relative_action,
            last_joint_position=self.last_joint_position,
            last_ee_pose=self.last_ee_pose,
        )
        self.last_joint_position = last_joint_position
        self.last_ee_pose = last_ee_pose
        return action_arr

    def finish(self, score: float) -> None:
        if self.episode_start_time is None or self.episode_end_time is None:
            raise ValueError("Episode start or end time not initialized")
        recorder = self._require_recorder()
        recorder.finalize(score, self.episode_start_time, self.episode_end_time)
