from genmanip.core.evaluator.request_logging import traced_worker
import os
import logging
import sys
import time

import ray
import torch

from genmanip.core.evaluator.logging_utils import (
    build_file_logger,
    get_run_log_dir,
    make_log_file_timestamp,
    redirect_std_streams,
)

cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
first_visible = cuda_visible.split(",")[0].strip() if cuda_visible else "0"
active_gpu = int(first_visible) if first_visible.isdigit() else 0

GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
JOB_MEMORY_GB = (
    20  # Approximate memory per job in GB, adjust this to maximize GPU utilization
)
NUM_GPUS_REQUIRED_PER_JOB = 1.0 / max(
    1, int(GPU_MEMORY / 1024 / 1024 / 1024 / JOB_MEMORY_GB)
)


def import_modules():
    """
    Import modules that are used in the worker and avoid DLL Hell problem.
    """
    import pydantic
    import torch
    import numpy
    from pydantic import BaseModel, Field


def _read_proc_status_memory_kb() -> dict[str, int]:
    memory_fields = ("VmRSS", "RssAnon", "RssFile", "VmSize", "VmSwap")
    result: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                for field in memory_fields:
                    if line.startswith(f"{field}:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            result[field] = int(parts[1])
    except OSError:
        return {}
    return result


@ray.remote(num_gpus=NUM_GPUS_REQUIRED_PER_JOB, max_restarts=3, max_task_retries=3)
class IsaacWorker:
    def __init__(
        self,
        worker_id: str,
        pool_log_id: str,
        args,
        default_config,
        current_dir,
        benchmark_id,
    ):
        self.worker_id = worker_id
        self.pool_log_id = pool_log_id
        self.args = args
        self.benchmark_id = benchmark_id

        if current_dir not in sys.path:
            sys.path.append(current_dir)

        # ---------- setup per-worker logger ----------
        run_log_dir = get_run_log_dir(current_dir, getattr(args, "run_id", None))
        worker_log_dir = os.path.join(run_log_dir, "worker")
        os.makedirs(worker_log_dir, exist_ok=True)
        self.worker_log_id = make_log_file_timestamp()
        log_path = os.path.join(
            worker_log_dir,
            f"pool_{self.pool_log_id}_worker_{self.worker_id}_{self.worker_log_id}.log",
        )
        self.log_path = log_path

        logger_name = (
            f"IsaacWorker-{getattr(args, 'run_id', 'default')}-{self.worker_id}"
        )
        self.logger = build_file_logger(logger_name, log_path)
        redirect_std_streams(self.logger, self.logger)

        self.logger.info(f"===== IsaacWorker {self.worker_id} started =====")

        # ---------- GPU Info ----------
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        self.logger.info(f"CUDA_VISIBLE_DEVICES = {cuda_visible}")

        self.logger.info(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            self.logger.info(f"torch device {i}: {torch.cuda.get_device_name(i)}")

        # ---------- Isaac modules ----------
        import_modules()

        from isaacsim import SimulationApp  # type: ignore

        self.simulation_app = SimulationApp(
            {
                "headless": not args.local,
                "multi_gpu": False,
                "active_gpu": active_gpu,  # Omniverse renderer: physical GPU id
                "physics_gpu": 0,
            }
        )

        self.simulation_app._carb_settings.set(
            "/physics/cooking/ujitsoCollisionCooking", False
        )
        self.simulation_app._carb_settings.set(
            "/app/omni.graph.scriptnode/opt_in", True
        )

        from genmanip.core.evaluator.env import IsaacEvalEnvRay

        self.env = IsaacEvalEnvRay(
            args,
            self.simulation_app,
            default_config,
            current_dir,
            benchmark_id=self.benchmark_id,
        )
        self.current_task_info = None
        self._episode_step_time_total = 0.0
        self._episode_step_count = 0
        self._post_reset_warmup = False

        self.logger.info("IsaacWorker init finished.")

    def _log_memory_snapshot(self, tag: str) -> None:
        mem = _read_proc_status_memory_kb()
        if not mem:
            self.logger.info("Memory snapshot unavailable tag=%s", tag)
            return
        self.logger.info(
            "Memory snapshot tag=%s VmRSS=%.2fGiB RssAnon=%.2fGiB "
            "RssFile=%.2fGiB VmSize=%.2fGiB VmSwap=%.2fGiB",
            tag,
            mem.get("VmRSS", 0) / 1024 / 1024,
            mem.get("RssAnon", 0) / 1024 / 1024,
            mem.get("RssFile", 0) / 1024 / 1024,
            mem.get("VmSize", 0) / 1024 / 1024,
            mem.get("VmSwap", 0) / 1024 / 1024,
        )

    @traced_worker
    def get_memory_snapshot(self) -> dict[str, float]:
        mem = _read_proc_status_memory_kb()
        if not mem:
            return {}
        return {key: value / 1024 / 1024 for key, value in mem.items()}

    @traced_worker
    def reset(
        self, seed: str, current_eval_config: dict, default_config: dict | None = None
    ):
        reset_start = time.perf_counter()
        try:
            self.logger.info(f"Reset called with seed={seed}")
            self._episode_step_time_total = 0.0
            self._episode_step_count = 0
            self._post_reset_warmup = True
            self._log_memory_snapshot("before_reset")
            obs, self.current_task_info = self.env.reset(
                seed, current_eval_config, default_config
            )
            self._log_memory_snapshot("after_reset")
            reset_elapsed = time.perf_counter() - reset_start

            if self.current_task_info is None:
                self.logger.info(
                    "No task info returned from env.reset(). reset_time=%.4fs",
                    reset_elapsed,
                )
                return None

            self.logger.info(
                "Running Task: %s, Seed: %s, reset_time=%.4fs",
                self.current_task_info["task"],
                self.current_task_info["seed"],
                reset_elapsed,
            )
            return obs
        except Exception:
            self.logger.exception("Worker %s reset failed", self.worker_id)
            raise

    @traced_worker
    def step(self, action):
        try:
            start_time = time.time()
            obs, _, done, info = self.env.step(action)
            step_time = time.time() - start_time
            self._episode_step_time_total += step_time
            self._episode_step_count += 1
            self.logger.debug("Env %s step time: %.4f", self.worker_id, step_time)
            if done:
                avg_step_time = (
                    self._episode_step_time_total / self._episode_step_count
                    if self._episode_step_count > 0
                    else 0.0
                )
                self.logger.info(
                    "Episode done. avg_step_time=%.4f step_count=%s info=%s",
                    avg_step_time,
                    self._episode_step_count,
                    info,
                )
            return obs, done, info
        except Exception:
            self.logger.exception("Worker %s step failed", self.worker_id)
            raise

    @traced_worker
    def step_chunk(
        self,
        action_list,
        render_mode: str = "lite",
        subframes: int = 2,
        render_suffix: int = 2,
    ):
        """Execute a chunk of actions in one Ray RPC.

        N-1 intermediate actions are stepped with skip_obs=True and
        skip_render=True (the server drops the expensive camera fetch +
        JPEG encode and tells Isaac Sim to skip the rendering pass). The
        final action runs the full env.step so the returned obs is
        populated as usual.

        If any intermediate step terminates the episode (invalid state /
        goal reached / num_steps exceeded), we fetch a full obs at that
        point, stop the chunk, and return how many steps actually ran so
        the client can account for them.

        When env.save_process is True the lite flags are ignored inside
        env.step so video recording stays complete — the chunk still
        collapses into a single Ray RPC though, which is itself a win.
        """
        try:
            if not isinstance(action_list, list) or not action_list:
                raise ValueError("action_list must be a non-empty list")

            last_idx = len(action_list) - 1
            executed = 0
            final_obs = None
            final_done = False
            final_info: dict = {}

            start_time = time.time()
            # render_mode:
            #   "always" — every step runs world.step(render=True); intermediate
            #              obs is still skipped (no camera read-back / JPEG)
            #              to save ~60 ms/step, but the RTX pipeline keeps
            #              accumulating TAA / denoiser state every tick.
            #              Per-action cost ≈ 95 ms, PSNR ≈ 47 dB.
            #   "lite"   — intermediate steps skip both obs AND the render
            #              pass; the final step catches up with `subframes`
            #              extra world.render() passes. Per-action cost
            #              ≈ 48 ms (for subframes=2), PSNR ≈ 36-37 dB mean.
            # Post-reset warmup: the first chunk after env.reset() forces
            # render every step so Hydra can absorb the new scene before
            # lite-mode skipping begins. Without this, frames in the first
            # chunk after a reset are visibly lagged. One-shot — cleared
            # after this chunk runs.
            if self._post_reset_warmup:
                lite = False
                self._post_reset_warmup = False
            else:
                lite = render_mode == "lite"
            # In lite mode, the last (render_suffix + 1) steps of each chunk
            # render fully so Hydra can drain its async pipeline before the
            # camera readout at the chunk boundary.
            render_threshold = last_idx - max(0, int(render_suffix))
            for i, action in enumerate(action_list):
                is_last = i == last_idx
                should_render = is_last or i >= render_threshold
                obs, _, done, info = self.env.step(
                    action,
                    skip_obs=not is_last,
                    skip_render=lite and (not should_render),
                    subframes=(subframes if (lite and is_last) else 0),
                )
                executed += 1
                final_done = bool(done)
                final_info = info
                if final_done:
                    # An intermediate step terminated early — obs may be None
                    # because we requested skip_obs. Fetch a full obs so the
                    # client still sees the final state.
                    if obs is None:
                        obs = self.env.get_obs()
                    final_obs = obs
                    break
                if is_last:
                    final_obs = obs

            total = time.time() - start_time
            self._episode_step_time_total += total
            self._episode_step_count += executed
            self.logger.debug(
                "Env %s step_chunk executed=%d time=%.4f",
                self.worker_id,
                executed,
                total,
            )
            if final_done:
                avg_step_time = (
                    self._episode_step_time_total / self._episode_step_count
                    if self._episode_step_count > 0
                    else 0.0
                )
                self.logger.info(
                    "Episode done in chunk. executed=%d chunk_time=%.4f "
                    "avg_step_time=%.4f total_step_time=%.4f step_count=%s info=%s",
                    executed,
                    total,
                    avg_step_time,
                    self._episode_step_time_total,
                    self._episode_step_count,
                    final_info,
                )
            return final_obs, final_done, final_info, executed
        except Exception:
            self.logger.exception("Worker %s step failed", self.worker_id)
            raise

    @traced_worker
    def post_episode_process(self):
        process_start = time.perf_counter()
        try:
            result = self.env.post_episode_process(None)
            score = None
            has_finalize_payload = False
            if isinstance(result, dict):
                score = result.get("score")
                has_finalize_payload = result.get("finalize_payload") is not None
            self.logger.info(
                "post_episode_process finished result=%s has_finalize_payload=%s elapsed=%.4fs",
                score if score is not None else result,
                has_finalize_payload,
                time.perf_counter() - process_start,
            )
            return result
        except Exception:
            self._log_memory_snapshot("post_episode_process_exception")
            self.logger.exception(
                "Worker %s post_episode_process failed", self.worker_id
            )
            raise

    @traced_worker
    def abort_episode(self):
        try:
            self.logger.info("abort_episode called.")
            self.env.abort_episode()
        except Exception:
            self.logger.exception("Worker %s abort_episode failed", self.worker_id)
            raise

    @traced_worker
    def close(self):
        try:
            self.logger.info("Closing IsaacWorker...")
            self.env.close()
            self.simulation_app.close()
            self.logger.info("IsaacWorker closed.")
        except Exception:
            self.logger.exception("Worker %s close failed", self.worker_id)
            raise

    @traced_worker
    def is_done(self):
        return self.env.done
