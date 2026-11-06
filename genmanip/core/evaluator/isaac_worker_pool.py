from genmanip.core.evaluator.request_logging import (
    remote_call, ray_get, submit_traced, traced,
)
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import os
import time
from pathlib import Path
import threading
from typing import Dict, Any, List

import ray
import torch
from ray.exceptions import GetTimeoutError, RayActorError

from genmanip.core.evaluator.exceptions import InsufficientResourcesError
from genmanip.core.evaluator.logging_utils import (
    build_file_logger,
    get_run_log_dir,
    make_log_file_timestamp,
)
from genmanip.core.evaluator.progress_manager import (
    ProgressManager,
    HEARTBEAT_INTERVAL_SECONDS,
)
from genmanip.core.evaluator.isaac_worker import IsaacWorker
from genmanip.core.evaluator.utils import (
    finalize_episode_recorder_payload,
    parse_configs_and_benchmark_id,
)
from genmanip.utils.standalone.file_utils import load_default_config, make_dir
from genmanip.utils.standalone.utils import parse_eval_config
from genmanip.utils.standalone.version_utils import process_archived_config

GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
JOB_MEMORY_GB = (
    20  # Approximate memory per job in GB, adjust this to maximize GPU utilization
)
NUM_GPUS_REQUIRED_PER_JOB = 1.0 / max(
    1, int(GPU_MEMORY / 1024 / 1024 / 1024 / JOB_MEMORY_GB)
)
FINALIZE_LOCK_HEARTBEAT_TIMEOUT_SECONDS = 3000.0
LOCK_LOST_FAILURE_THRESHOLD = 3
NEXT_TASK_WAIT_TIMEOUT_SECONDS = 600.0
NEXT_TASK_WAIT_POLL_SECONDS = 1.0
NEXT_TASK_RECONCILE_INTERVAL_SECONDS = 30.0


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


class IsaacWorkerPool:
    """
    A pool of Isaac workers for parallel evaluation.

    This class is purely functional - it only handles:
    - Worker creation and destruction
    - Step execution

    All task lifecycle management (configs, seeds, results, progress)
    is delegated to ProgressManager.

    Args:
        args: Command line arguments.
        current_dir: Current working directory.
        world_size: Number of GPUs available.
    """

    def __init__(self, args, current_dir, world_size=None):
        self.lock = threading.Lock()
        self.workers: Dict[str, Any] = {}
        self._reset_executor = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="episode-reset"
        )
        self._finalize_executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="episode-finalize"
        )
        self._pending_reset_futures: Dict[str, Future] = {}
        self._pending_finalize_futures: Dict[str, Future] = {}
        self._pending_finalize_episodes: Dict[str, tuple[str, str, float]] = {}
        self._worker_cancel_events: Dict[str, threading.Event] = {}
        self._worker_lock_failure_counts: Dict[str, int] = {}
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_pending_finalize_locks,
            name="episode-finalize-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        self.args = args
        self.current_dir = current_dir
        self.workspace_root = Path(self.current_dir).resolve()
        self.pool_log_id = make_log_file_timestamp()
        self.logger: logging.Logger | None = None
        self.world_size = (
            world_size if world_size is not None else torch.cuda.device_count()
        )
        self.max_workers = int(self.world_size / NUM_GPUS_REQUIRED_PER_JOB)

        self.default_config = load_default_config(
            self.current_dir, "__None__.json", "local" if self.args.local else "default"
        )
        # worker env timeout
        self.step_call_timeout = 1200
        self.reset_call_timeout = 1200
        self.worker_restart_memory_gib = float(
            getattr(self.args, "worker_restart_memory_gib", 20.0)
        )
        self.benchmark_id: str | None = None
        self.progress_manager: ProgressManager | None = None
        self.worker_env_vars: dict[str, str] = {}
        self.configure_logging(getattr(self.args, "run_id", None))

    def configure_logging(self, run_id: str | None) -> None:
        log_dir = get_run_log_dir(self.current_dir, run_id)
        self.logger = build_file_logger(
            logger_name=f"IsaacWorkerPool[{run_id or 'default'}][{self.pool_log_id}]",
            log_path=f"{log_dir}/worker_pool_{self.pool_log_id}.log",
        )
        self.logger.info(
            "Configured worker pool logging. run_id=%s", run_id or "default"
        )

    def _create_progress_manager(
        self, benchmark_id: str, run_id: str
    ) -> ProgressManager:
        """Create a new ProgressManager instance."""
        return ProgressManager(
            result_base_dir=os.path.join(self.current_dir, "saved/eval_results"),
            benchmark_id=benchmark_id,
            run_id=run_id,
        )

    def _validate_run_id(self, run_id: str) -> str:
        if not isinstance(run_id, str) or run_id.strip() == "":
            raise ValueError("run_id must be a non-empty string")
        run_id_path = Path(run_id)
        if run_id_path.is_absolute():
            raise ValueError("run_id must not be an absolute path")
        if any(part in ("", ".", "..") for part in run_id_path.parts):
            raise ValueError("run_id contains invalid path segments")
        return run_id

    def _validate_run_output_dir(self, benchmark_id: str, run_id: str) -> None:
        result_base_dir = (self.workspace_root / "saved" / "eval_results").resolve()
        run_dir = (result_base_dir / benchmark_id / run_id).resolve()
        if not _is_relative_to(run_dir, result_base_dir):
            raise ValueError(
                f"run output path escapes workspace: benchmark_id={benchmark_id}, run_id={run_id}"
            )

    def _parse_task_config(
        self, config: dict
    ) -> tuple[list[dict], dict[str, list[str]], dict[str, int]]:
        """Parse evaluation config into task lists."""
        task_config_list = parse_eval_config(config)
        task_config_list = [process_archived_config(cfg) for cfg in task_config_list]

        task_seed_list = {}
        task_num_test_list = {}
        for cfg in task_config_list:
            task_seed_list[cfg["task_name"]] = [
                str(i).zfill(3) for i in range(cfg["num_test"])
            ]
            task_num_test_list[cfg["task_name"]] = cfg["num_test"]

        return task_config_list, task_seed_list, task_num_test_list

    def _resolve_worker_env_vars(
        self, all_configs: list[tuple[list[dict], dict[str, list[str]], dict[str, int]]]
    ) -> dict[str, str]:
        env_vars_reference: dict[str, str] | None = None
        for task_config_list, _, _ in all_configs:
            for cfg in task_config_list:
                raw_env_vars = cfg.get("env_vars", None)
                if raw_env_vars is None:
                    continue
                if not isinstance(raw_env_vars, dict):
                    if self.logger is not None:
                        self.logger.warning(
                            "Ignore non-dict env_vars in task %s: %r",
                            cfg.get("task_name", "<unknown>"),
                            type(raw_env_vars),
                        )
                    continue

                env_vars: dict[str, str] = {}
                for key, value in raw_env_vars.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        if self.logger is not None:
                            self.logger.warning(
                                "Ignore invalid env var in task %s: %r=%r",
                                cfg.get("task_name", "<unknown>"),
                                key,
                                value,
                            )
                        continue
                    env_vars[key] = value.replace(
                        "{ASSETS_DIR}", self.default_config.get("ASSETS_DIR", "")
                    )

                if env_vars_reference is None:
                    env_vars_reference = env_vars
                elif env_vars != env_vars_reference:
                    if self.logger is not None:
                        self.logger.warning(
                            "Inconsistent env_vars across tasks. "
                            "Use the first discovered env_vars for worker init."
                        )

        return env_vars_reference if env_vars_reference is not None else {}

    def start_new_job(self, run_id: str, evaluation_config_path_list: list[str]):
        """
        Start a new evaluation job.

        Kills existing workers, initializes ProgressManager, and loads configs.

        Args:
            run_id: Run identifier (auto-generated if empty/None)
            evaluation_config_path_list: List of config file paths
        """
        if (
            not isinstance(evaluation_config_path_list, list)
            or not evaluation_config_path_list
        ):
            raise ValueError(
                "evaluation_config_path_list must be a non-empty list of strings"
            )
        if any(
            (not isinstance(config_path, str)) or (config_path.strip() == "")
            for config_path in evaluation_config_path_list
        ):
            raise ValueError(
                "evaluation_config_path_list must contain non-empty strings only"
            )

        self._wait_for_pending_finalizers()
        self.kill_workers(list(self.workers.keys()))

        # Determine run_id
        effective_run_id = (
            run_id
            if (run_id != "") and (run_id is not None)
            else (
                self.args.run_id
                if getattr(self.args, "run_id", None) is not None
                else datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
            )
        )
        effective_run_id = self._validate_run_id(effective_run_id)
        self.args.run_id = effective_run_id
        self.configure_logging(effective_run_id)

        # Parse configs and determine benchmark_id
        self.benchmark_id = None
        all_configs: list[tuple[list[dict], dict[str, list[str]], dict[str, int]]] = []

        for evaluation_config_path_or_group in evaluation_config_path_list:
            evaluation_config_list, benchmark_id, is_genmanip_package = (
                parse_configs_and_benchmark_id(
                    evaluation_config_path_or_group, self.current_dir
                )
            )
            if is_genmanip_package:
                self.default_config["TASKS_DIR"] = os.path.join(
                    self.current_dir,
                    "saved/assets/collected_packages",
                    benchmark_id.split("/")[-1],
                    "tasks",
                )
            if self.benchmark_id is None:
                self.benchmark_id = benchmark_id
            else:
                assert (
                    self.benchmark_id == benchmark_id
                ), f"Only one benchmark is allowed at a time: {self.benchmark_id} vs {benchmark_id}"

            for evaluation_config in evaluation_config_list:
                parsed = self._parse_task_config(evaluation_config)
                all_configs.append(parsed)

        if self.benchmark_id is None:
            raise ValueError("Benchmark ID is not set")

        self.worker_env_vars = self._resolve_worker_env_vars(all_configs)
        self._validate_run_output_dir(self.benchmark_id, effective_run_id)
        # Create progress manager
        self.progress_manager = self._create_progress_manager(
            self.benchmark_id, effective_run_id
        )

        # Add all configs to progress manager
        for task_config_list, task_seed_list, task_num_test_list in all_configs:
            self.progress_manager.add_evaluation_config(
                task_config_list, task_seed_list, task_num_test_list
            )

        # Load existing progress from filesystem
        self.progress_manager.load_progress()
        self.progress_manager.print_progress()

    # ==================== Worker Management ====================

    def create_workers(self, worker_ids: List[str]):
        """Create new workers by ID."""
        with self.lock:
            new_worker_cnt = sum(1 for w_id in worker_ids if w_id not in self.workers)
            total_requested = len(self.workers) + new_worker_cnt
            if total_requested > self.max_workers:
                raise InsufficientResourcesError(
                    requested=total_requested, available=self.max_workers
                )

            for w_id in worker_ids:
                cancel_event = self._worker_cancel_events.get(str(w_id))
                if cancel_event is None:
                    cancel_event = threading.Event()
                    self._worker_cancel_events[str(w_id)] = cancel_event
                else:
                    cancel_event.clear()
                self._worker_lock_failure_counts[str(w_id)] = 0
                if w_id not in self.workers:
                    self.workers[w_id] = self._spawn_worker(w_id)

    def kill_workers(self, worker_ids: List[str]):
        """Kill workers and clean up their tasks."""
        with self.lock:
            for w_id in worker_ids:
                future = self._pending_reset_futures.pop(str(w_id), None)
                cancel_event = self._worker_cancel_events.get(str(w_id))
                if cancel_event is None:
                    cancel_event = threading.Event()
                    self._worker_cancel_events[str(w_id)] = cancel_event
                cancel_event.set()
                if future is not None:
                    future.cancel()
                if w_id in self.workers:
                    ray.kill(self.workers[w_id], no_restart=True)
                    del self.workers[w_id]
                self._worker_lock_failure_counts.pop(str(w_id), None)

                # Clean up worker's tasks via progress manager
                if self.progress_manager is not None:
                    self.progress_manager.cleanup_worker(str(w_id))

        if self.progress_manager is not None:
            self.progress_manager.print_progress()

    def _heartbeat_pending_finalize_locks(self) -> None:
        interval_seconds = max(1, HEARTBEAT_INTERVAL_SECONDS // 2)
        while not self._heartbeat_stop_event.wait(interval_seconds):
            progress_manager = self.progress_manager
            if progress_manager is None:
                continue
            with self.lock:
                pending_episodes = list(self._pending_finalize_episodes.items())
            for episode_id, (task_name, seed, started_at) in pending_episodes:
                elapsed = time.monotonic() - started_at
                if elapsed > FINALIZE_LOCK_HEARTBEAT_TIMEOUT_SECONDS:
                    with self.lock:
                        current = self._pending_finalize_episodes.get(episode_id)
                        if current is not None and current[2] == started_at:
                            self._pending_finalize_episodes.pop(episode_id, None)
                    if self.logger is not None:
                        self.logger.warning(
                            "Stop heartbeating finalize lock for episode=%s after %.2fs; "
                            "stale cleanup may reclaim the lock",
                            episode_id,
                            elapsed,
                        )
                    continue
                progress_manager.refresh_episode_lock(task_name, seed)

    def get_active_workers(self) -> List[str]:
        """Get list of active worker IDs."""
        with self.lock:
            return list(self.workers.keys())

    # ==================== Evaluation Execution ====================

    @traced("pool.refresh_locks")
    def _refresh_locks(self, worker_ids: List[str]) -> set[str]:
        """Refresh locks for a list of workers (best-effort)."""
        if self.progress_manager is None:
            return set()
        lost: set[str] = set()
        for w_id in worker_ids:
            ok = self.progress_manager.refresh_worker_lock(str(w_id))
            with self.lock:
                if ok:
                    self._worker_lock_failure_counts[str(w_id)] = 0
                    continue
                failures = self._worker_lock_failure_counts.get(str(w_id), 0) + 1
                self._worker_lock_failure_counts[str(w_id)] = failures
            if failures < LOCK_LOST_FAILURE_THRESHOLD:
                if self.logger is not None:
                    self.logger.warning(
                        "Lock refresh failed for worker=%s failures=%s/%s",
                        w_id,
                        failures,
                        LOCK_LOST_FAILURE_THRESHOLD,
                    )
                continue
            if self.logger is not None:
                self.logger.warning(
                    "Lost lock for worker=%s after %s consecutive refresh failures",
                    w_id,
                    failures,
                )
            else:
                print(f"[lock] lost lock for worker {w_id}")
            lost.add(str(w_id))
        return lost

    @traced("pool.wait_for_next_task")
    def _wait_for_next_task(
        self, w_id: str, timeout_seconds: float = NEXT_TASK_WAIT_TIMEOUT_SECONDS
    ) -> tuple[dict | None, str | None]:
        if self.progress_manager is None:
            raise ValueError("No job started")
        deadline = time.monotonic() + timeout_seconds
        next_reconcile_at = time.monotonic() + NEXT_TASK_RECONCILE_INTERVAL_SECONDS
        while True:
            if self._worker_cancelled(w_id):
                return None, None
            config, task_seed = self.progress_manager.get_next_task(str(w_id))
            if config is not None:
                return config, task_seed
            if self.progress_manager.check_finished():
                return None, None
            now = time.monotonic()
            if now >= next_reconcile_at:
                all_done_on_disk = (
                    self.progress_manager.reconcile_task_state_from_filesystem()
                )
                config, task_seed = self.progress_manager.get_next_task(str(w_id))
                if config is not None:
                    return config, task_seed
                if all_done_on_disk or self.progress_manager.check_finished():
                    return None, None
                next_reconcile_at = now + NEXT_TASK_RECONCILE_INTERVAL_SECONDS
            if time.monotonic() >= deadline:
                all_done_on_disk = (
                    self.progress_manager.reconcile_task_state_from_filesystem()
                )
                config, task_seed = self.progress_manager.get_next_task(str(w_id))
                if config is not None:
                    return config, task_seed
                if all_done_on_disk or self.progress_manager.check_finished():
                    return None, None
                if self.logger is not None:
                    self.logger.warning(
                        "No task became available for worker=%s within %.1fs after "
                        "filesystem reconcile; leaving worker idle until client reset",
                        w_id,
                        timeout_seconds,
                    )
                return None, None
            time.sleep(NEXT_TASK_WAIT_POLL_SECONDS)

    def _worker_cancelled(self, w_id: str) -> bool:
        with self.lock:
            cancel_event = self._worker_cancel_events.get(str(w_id))
        return bool(cancel_event is not None and cancel_event.is_set())

    @traced("pool.reset_worker")
    def _reset_worker(
        self,
        w_id: str,
        task_seed: str,
        config: dict[str, Any],
        include_default_config: bool = False,
        memory_restart_budget: int = 1,
    ) -> Any:
        """Reset one worker and raise a context-rich error on failure."""
        task_name = (
            config.get("task_name", "<unknown>")
            if isinstance(config, dict)
            else "<unknown>"
        )
        if self._worker_cancelled(w_id):
            raise RuntimeError(f"Worker {w_id} reset cancelled before start")
        if memory_restart_budget > 0:
            restarted = self._restart_worker_if_memory_exceeded(
                w_id,
                tag="before_reset",
                task_name=task_name,
                task_seed=task_seed,
            )
            if restarted:
                memory_restart_budget -= 1

        def _invoke_reset() -> Any:
            if self._worker_cancelled(w_id):
                raise RuntimeError(f"Worker {w_id} reset cancelled before RPC dispatch")
            worker = self.workers[w_id]
            if include_default_config:
                return remote_call(worker.reset, task_seed, config, self.default_config, _trace_worker_id=w_id)
            return remote_call(worker.reset, task_seed, config, _trace_worker_id=w_id)

        def _should_retry_reset(error: Exception) -> bool:
            if self._worker_cancelled(w_id):
                return False
            return isinstance(error, TimeoutError) or self._is_worker_unavailable_error(
                error
            )

        def _on_retry_reset(error: Exception, attempt_idx: int) -> bool:
            if self._worker_cancelled(w_id):
                if self.logger is not None:
                    self.logger.info(
                        "Reset retry suppressed for cancelled worker=%s task=%s seed=%s",
                        w_id,
                        task_name,
                        task_seed,
                    )
                return False
            if self.logger is not None:
                self.logger.warning(
                    "Reset attempt %s failed for worker=%s task=%s seed=%s: %s. "
                    "Replacing worker and retrying.",
                    attempt_idx + 1,
                    w_id,
                    task_name,
                    task_seed,
                    error,
                )
            self._replace_worker(w_id, reason=error)
            return True

        try:
            return self._ray_get_with_timeout(
                call_remote=_invoke_reset,
                timeout=self.reset_call_timeout,
                context=f"Worker {w_id} reset task={task_name} seed={task_seed}",
                max_attempts=5,
                should_retry=_should_retry_reset,
                on_retry=_on_retry_reset,
                should_abort=lambda: self._worker_cancelled(w_id),
            )
        except Exception as e:
            raise RuntimeError(
                f"Worker {w_id} reset failed for task={task_name} seed={task_seed}: "
                f"{type(e).__name__}: {e}"
            ) from e

    def reset(self, worker_ids: List[str] | None) -> Dict[str, Any]:
        """
        Reset workers and assign initial tasks.

        Args:
            worker_ids: List of worker IDs to reset

        Returns:
            Dict mapping worker_id to response dict
        """
        if self.progress_manager is None:
            raise ValueError("No job started. Call start_new_job first.")

        if worker_ids is None:
            worker_ids = self.get_active_workers()
            if not worker_ids:
                raise ValueError(
                    "worker_ids not provided and no active workers exist; create workers first"
                )
        if not isinstance(worker_ids, list) or not worker_ids:
            raise ValueError("worker_ids must be a non-empty list")
        worker_ids = [str(w_id) for w_id in worker_ids]
        if any(w_id.strip() == "" for w_id in worker_ids):
            raise ValueError("worker_ids must contain non-empty values")

        self.create_workers(worker_ids)
        if self.logger is not None:
            self.logger.info("Reset requested for workers=%s", worker_ids)

        response = {}
        for w_id in worker_ids:
            pending_resp = self._consume_pending_reset_if_ready(w_id)
            if pending_resp is not None:
                response[w_id] = pending_resp
                continue
            if self._has_pending_reset(w_id):
                response[w_id] = self._make_reset_pending_response()
                continue

            config, task_seed = self.progress_manager.get_next_task(str(w_id))
            if config is None and not self.progress_manager.check_finished():
                if self.logger is not None:
                    self.logger.warning(
                        "No task available during reset for worker=%s; "
                        "reconciling task state from filesystem",
                        w_id,
                    )
                all_done_on_disk = (
                    self.progress_manager.reconcile_task_state_from_filesystem()
                )
                if not all_done_on_disk:
                    config, task_seed = self.progress_manager.get_next_task(str(w_id))

            if config is None:
                resp = {
                    "obs": None,
                    "metric": self.progress_manager.calculate_result(),
                    "episode_result": None,
                }
            else:
                resp = self._start_background_reset(
                    w_id,
                    task_seed,
                    config,
                    include_default_config=True,
                )

            response[w_id] = resp
        return response

    def _make_reset_pending_response(self) -> dict[str, Any]:
        metric = None
        if self.progress_manager is not None:
            metric = self.progress_manager.calculate_result()
        return {
            "obs": {"reset": True, "reset_pending": True},
            "metric": metric,
            "episode_result": None,
            "reset_pending": True,
        }

    def _schedule_async_finalize(
        self,
        task_name: str,
        task_seed: str,
        finalize_payload: dict[str, Any],
    ) -> None:
        episode_id = f"{task_name}/{task_seed}"
        scheduled_at = time.monotonic()

        def _run_finalize() -> None:
            try:
                finalize_episode_recorder_payload(finalize_payload)
            finally:
                if self.progress_manager is not None:
                    self.progress_manager.release_episode_lock(task_name, task_seed)

        with self.lock:
            future = self._pending_finalize_futures.get(episode_id)
            if future is not None and not future.done():
                return
            future = submit_traced(self._finalize_executor, _run_finalize)
            self._pending_finalize_futures[episode_id] = future
            self._pending_finalize_episodes[episode_id] = (
                task_name,
                task_seed,
                scheduled_at,
            )

        def _cleanup(done_future: Future) -> None:
            cleanup_start = time.monotonic()
            with self.lock:
                self._pending_finalize_episodes.pop(episode_id, None)
            try:
                exc = done_future.exception()
            except Exception as callback_error:
                exc = callback_error
                if self.logger is not None:
                    self.logger.exception(
                        "Exception in finalize future callback for episode=%s",
                        episode_id,
                    )
            elapsed = time.monotonic() - scheduled_at
            cleanup_elapsed = time.monotonic() - cleanup_start
            if self.logger is not None:
                if exc is None:
                    self.logger.info(
                        "Async finalize done for episode=%s elapsed=%.2fs cleanup=%.4fs",
                        episode_id,
                        elapsed,
                        cleanup_elapsed,
                    )
                else:
                    self.logger.warning(
                        "Async finalize finished with error for episode=%s "
                        "elapsed=%.2fs cleanup=%.4fs error=%s: %s",
                        episode_id,
                        elapsed,
                        cleanup_elapsed,
                        type(exc).__name__,
                        exc,
                    )

        future.add_done_callback(_cleanup)

    @traced("pool.wait_for_pending_finalizers")
    def _wait_for_pending_finalizers(self) -> None:
        while True:
            with self.lock:
                pending_items = list(self._pending_finalize_futures.items())
            if not pending_items:
                return
            for episode_id, future in pending_items:
                wait_start = time.monotonic()
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Async finalize failed for episode {episode_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if self.logger is not None:
                    self.logger.info(
                        "Waited %.2fs for pending finalize episode=%s",
                        time.monotonic() - wait_start,
                        episode_id,
                    )
                task_name, task_seed = episode_id.rsplit("/", 1)
                if (
                    self.progress_manager is None
                    or not self.progress_manager.is_episode_completed(
                        task_name, task_seed
                    )
                ):
                    raise RuntimeError(
                        "Async finalize completed without result_info.json for "
                        f"episode {episode_id}"
                    )
                with self.lock:
                    current_future = self._pending_finalize_futures.get(episode_id)
                    if current_future is future:
                        self._pending_finalize_futures.pop(episode_id, None)

    def _has_pending_reset(self, w_id: str) -> bool:
        with self.lock:
            future = self._pending_reset_futures.get(str(w_id))
        return future is not None and not future.done()

    def _consume_pending_reset_if_ready(self, w_id: str) -> dict[str, Any] | None:
        with self.lock:
            future = self._pending_reset_futures.get(str(w_id))
            if future is None or not future.done():
                return None
            del self._pending_reset_futures[str(w_id)]
        try:
            return future.result()
        except Exception:
            if self.logger is not None:
                self.logger.exception(
                    "Pending reset future failed for worker=%s", str(w_id)
                )
            if self.progress_manager is not None:
                released = self.progress_manager.release_worker_task(str(w_id))
                if released is not None and self.logger is not None:
                    task_name, task_seed, _ = released
                    self.logger.warning(
                        "Released failed pending reset task back to queue: "
                        "worker=%s task=%s seed=%s",
                        str(w_id),
                        task_name,
                        task_seed,
                    )
            raise

    @traced("pool.spawn_worker")
    def _spawn_worker(self, w_id: str):
        return IsaacWorker.options(
            runtime_env={"env_vars": self.worker_env_vars}
        ).remote(
            w_id,
            self.pool_log_id,
            self.args,
            self.default_config,
            self.current_dir,
            self.benchmark_id,
        )

    @traced("pool.replace_worker")
    def _replace_worker(self, w_id: str, reason: Exception | None = None) -> None:
        reason_text = f"{type(reason).__name__}: {reason}" if reason is not None else ""
        if self._worker_cancelled(w_id):
            if self.logger is not None:
                self.logger.info(
                    "Skip replacing cancelled worker=%s reason=%s", w_id, reason_text
                )
            return
        if self.logger is not None:
            self.logger.warning("Replacing dead worker=%s reason=%s", w_id, reason_text)
        with self.lock:
            old_worker = self.workers.get(w_id)
            if old_worker is not None:
                try:
                    ray.kill(old_worker, no_restart=True)
                except Exception:
                    if self.logger is not None:
                        self.logger.exception(
                            "Failed to kill stale worker handle for worker=%s", w_id
                        )
            self.workers[w_id] = self._spawn_worker(w_id)

    def _get_worker_memory_snapshot(self, w_id: str) -> dict[str, float]:
        worker = self.workers.get(w_id)
        if worker is None:
            return {}
        try:
            return self._ray_get_with_timeout(
                call_remote=lambda: remote_call(worker.get_memory_snapshot, _trace_worker_id=w_id),
                timeout=min(10.0, self.reset_call_timeout),
                context=f"Worker {w_id} get_memory_snapshot",
            )
        except Exception as error:
            if self.logger is not None:
                self.logger.warning(
                    "Failed to get memory snapshot for worker=%s: %s",
                    w_id,
                    error,
                )
            return {}

    def _restart_worker_if_memory_exceeded(
        self,
        w_id: str,
        *,
        tag: str,
        task_name: str,
        task_seed: str,
    ) -> bool:
        threshold_gib = self.worker_restart_memory_gib
        if threshold_gib <= 0:
            return False
        mem = self._get_worker_memory_snapshot(w_id)
        vmrss_gib = float(mem.get("VmRSS", 0.0))
        if vmrss_gib <= threshold_gib:
            return False
        if self.logger is not None:
            self.logger.warning(
                "Worker %s memory exceeds threshold at %s: VmRSS=%.2fGiB threshold=%.2fGiB "
                "task=%s seed=%s. Restarting actor before continuing.",
                w_id,
                tag,
                vmrss_gib,
                threshold_gib,
                task_name,
                task_seed,
            )
        self._replace_worker(
            w_id,
            reason=RuntimeError(
                f"worker memory threshold exceeded at {tag}: {vmrss_gib:.2f}GiB > {threshold_gib:.2f}GiB"
            ),
        )
        return True

    def _ray_get_with_timeout(
        self,
        call_remote,
        timeout: float,
        context: str,
        max_attempts: int = 1,
        should_retry=None,
        on_retry=None,
        should_abort=None,
    ) -> Any:
        attempts = max(1, int(max_attempts))
        last_error: Exception | None = None

        for attempt_idx in range(attempts):
            if should_abort is not None and should_abort():
                raise RuntimeError(f"{context} cancelled")
            attempt_start = time.monotonic()
            try:
                ref = call_remote()
                result = ray_get(ref, timeout=timeout)
                return result
            except GetTimeoutError:
                last_error = TimeoutError(f"{context} timed out after {timeout}s")
            except Exception as e:
                last_error = e

            if attempt_idx >= attempts - 1:
                break

            retry_allowed = (
                should_retry(last_error) if should_retry is not None else False
            )
            if should_abort is not None and should_abort():
                break
            if not retry_allowed:
                break
            if on_retry is not None and not on_retry(last_error, attempt_idx):
                break

        if last_error is None:
            raise RuntimeError(f"{context} failed without a captured exception")
        raise last_error

    def _is_worker_unavailable_error(self, error: Exception) -> bool:
        if isinstance(error, RayActorError):
            return True
        error_text = str(error)
        return any(
            token in error_text
            for token in (
                "ActorDiedError",
                "The actor died",
                "ray.kill",
                "actor is dead",
            )
        )

    def _start_pending_reset(self, w_id: str, info: dict[str, Any]) -> dict[str, Any]:
        if self.logger is not None:
            self.logger.info(
                "Starting background reset for worker=%s info=%s", w_id, info
            )
        with self.lock:
            future = self._pending_reset_futures.get(str(w_id))
            if future is None or future.done():
                self._pending_reset_futures[str(w_id)] = submit_traced(self._reset_executor,
                    self._handle_done, w_id, info
                )
        return self._make_reset_pending_response()

    def _run_reset_task(
        self,
        w_id: str,
        task_seed: str,
        config: dict[str, Any],
        *,
        include_default_config: bool = False,
    ) -> dict[str, Any]:
        if self._worker_cancelled(w_id):
            raise RuntimeError(f"Worker {w_id} background reset cancelled")
        self._refresh_locks([w_id])
        obs = self._reset_worker(
            w_id,
            task_seed,
            config,
            include_default_config=include_default_config,
        )
        if self._worker_cancelled(w_id):
            return {
                "obs": None,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": None,
            }
        self._refresh_locks([w_id])
        return {
            "obs": obs,
            "metric": self.progress_manager.calculate_result(),
            "episode_result": None,
        }

    def _start_background_reset(
        self,
        w_id: str,
        task_seed: str,
        config: dict[str, Any],
        *,
        include_default_config: bool = False,
    ) -> dict[str, Any]:
        if self.logger is not None:
            self.logger.info(
                "Starting background initial reset for worker=%s task=%s seed=%s",
                w_id,
                config.get("task_name", "<unknown>"),
                task_seed,
            )
        with self.lock:
            future = self._pending_reset_futures.get(str(w_id))
            if future is None or future.done():
                self._pending_reset_futures[str(w_id)] = submit_traced(self._reset_executor,
                    self._run_reset_task,
                    w_id,
                    task_seed,
                    config,
                    include_default_config=include_default_config,
                )
        return self._make_reset_pending_response()

    def get_pending_reset_results(self, worker_ids: List[str] | None) -> Dict[str, Any]:
        if worker_ids is None:
            with self.lock:
                worker_ids = list(self._pending_reset_futures.keys())
        if not isinstance(worker_ids, list) or not worker_ids:
            raise ValueError("worker_ids must be a non-empty list")

        response = {}
        for w_id in [str(w_id) for w_id in worker_ids]:
            pending_resp = self._consume_pending_reset_if_ready(w_id)
            if pending_resp is not None:
                response[w_id] = pending_resp
            elif self._has_pending_reset(w_id):
                response[w_id] = self._make_reset_pending_response()
            else:
                raise ValueError(f"worker {w_id} has no pending reset result")
        return response

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step for each worker.

        Args:
            action_dict: Dict mapping worker_id to action

        Returns:
            Dict mapping worker_id to response dict
        """
        if self.progress_manager is None:
            raise ValueError("No job started. Call start_new_job first.")

        response = {}
        actionable_dict: Dict[str, Any] = {}

        # Safety redundancy and backward compatibility for older clients that may step before reset polling completes.
        for w_id, action in action_dict.items():
            pending_resp = self._consume_pending_reset_if_ready(w_id)
            if pending_resp is not None:
                response[str(w_id)] = pending_resp
                continue
            if self._has_pending_reset(w_id):
                response[str(w_id)] = self._make_reset_pending_response()
                continue
            actionable_dict[str(w_id)] = action

        if not actionable_dict:
            return response

        # start to step actionable workers, but first refresh locks and record any
        lost = self._refresh_locks(list(actionable_dict.keys()))
        futures = []
        for w_id, action in actionable_dict.items():
            futures.append(remote_call(self.workers[w_id].step, action, _trace_worker_id=w_id))

        results = ray_get(futures)
        lost.update(self._refresh_locks(list(actionable_dict.keys())))

        for i, w_id in enumerate(actionable_dict.keys()):
            obs, done, info = results[i]
            resp = {
                "obs": obs,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": None,
            }

            if str(w_id) in lost:
                ray_get(remote_call(self.workers[w_id].abort_episode, _trace_worker_id=w_id))
                # Requeue the task so it can be retried by any worker
                self.progress_manager.release_worker_task(str(w_id))
                resp = self._handle_done(w_id, {"error": "lock_lost", "info": "Done"})
                resp["lock_lost"] = True
                response[w_id] = resp
                continue

            if done:
                resp = self._start_pending_reset(w_id, info)

            response[w_id] = resp

        return response

    def step_chunk(
        self,
        action_chunk: List[Dict[str, Any]],
        render_mode: str = "lite",
        subframes: int = 2,
        render_suffix: int = 2,
    ) -> dict[str, Any]:
        """
        Execute multiple steps in one server call.

        Args:
            action_chunk: List of action_dict. Each element has the same format as `step`.

        Returns:
            {
                "obs": final_response_dict,
                "executed_steps": int,
                "stopped_early": bool,
            }

        Fast path: if every entry in the chunk targets exactly the same
        single worker, we dispatch the whole chunk to that worker actor in
        ONE Ray RPC via IsaacWorker.step_chunk. That lets the worker skip
        obs + render on intermediate steps (see env.step skip_obs flag) and
        collapses what used to be N Ray RPCs into 1.
        """
        if self.progress_manager is None:
            raise ValueError("No job started. Call start_new_job first.")
        if not isinstance(action_chunk, list) or len(action_chunk) == 0:
            raise ValueError("action_chunk must be a non-empty list")

        # Detect single-worker chunk
        single_wid = None
        single_worker = True
        for step_action_dict in action_chunk:
            if not isinstance(step_action_dict, dict) or len(step_action_dict) != 1:
                single_worker = False
                break
            (wid,) = step_action_dict.keys()
            if single_wid is None:
                single_wid = wid
            elif single_wid != wid:
                single_worker = False
                break

        if (
            single_worker
            and single_wid is not None
            and single_wid in self.workers
            and not self._has_pending_reset(single_wid)
            and self._consume_pending_reset_if_ready(single_wid) is None
        ):
            # Extract per-worker action list
            per_worker = [
                step_action_dict[single_wid] for step_action_dict in action_chunk
            ]
            lost = self._refresh_locks([single_wid])
            if single_wid in lost:
                # Lock lost before we even dispatched — fall through to slow path
                pass
            else:
                future = remote_call(self.workers[single_wid].step_chunk,
                    per_worker, render_mode, subframes, render_suffix, _trace_worker_id=single_wid
                )
                result = ray_get(future)
                final_obs, final_done, final_info, executed_steps = result
                lost.update(self._refresh_locks([single_wid]))

                final_response: Dict[str, Any] = {}
                stopped_early = bool(final_done) or executed_steps < len(action_chunk)

                if single_wid in lost:
                    ray_get(remote_call(self.workers[single_wid].abort_episode, _trace_worker_id=single_wid))
                    self.progress_manager.release_worker_task(str(single_wid))
                    resp = self._handle_done(
                        single_wid, {"error": "lock_lost", "info": "Done"}
                    )
                    resp["lock_lost"] = True
                    final_response[single_wid] = resp
                else:
                    if final_done:
                        resp = self._start_pending_reset(single_wid, final_info)
                    else:
                        resp = {
                            "obs": final_obs,
                            "metric": self.progress_manager.calculate_result(),
                            "episode_result": None,
                        }
                    final_response[single_wid] = resp

                return {
                    "obs": final_response,
                    "executed_steps": executed_steps,
                    "stopped_early": stopped_early,
                }

        # Slow path (multi-worker or fast-path preconditions not met):
        # per-step dispatch across all workers as before.
        final_response = {}
        executed_steps = 0
        stopped_early = False

        for step_action_dict in action_chunk:
            if not isinstance(step_action_dict, dict) or len(step_action_dict) == 0:
                raise ValueError("Each action in action_chunk must be a non-empty dict")

            final_response = self.step(step_action_dict)
            executed_steps += 1

            for worker_resp in final_response.values():
                if worker_resp.get("reset_pending"):
                    stopped_early = True
                    break
                obs = worker_resp.get("obs")
                if obs is None:
                    stopped_early = True
                    break
                if isinstance(obs, dict) and obs.get("reset"):
                    stopped_early = True
                    break
            if stopped_early:
                break

        return {
            "obs": final_response,
            "executed_steps": executed_steps,
            "stopped_early": stopped_early,
        }

    @traced("pool.handle_done")
    def _handle_done(self, w_id: str, info: dict) -> dict:
        """Handle episode completion for a worker."""
        if self.progress_manager is None:
            raise ValueError("No job started")
        if self._worker_cancelled(w_id):
            return {
                "obs": None,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": None,
            }

        # Get result from worker
        result = None
        finalize_payload = None
        if "error" not in info and "info" in info and info["info"] != "Done":
            try:
                result = self._ray_get_with_timeout(
                    call_remote=lambda: remote_call(self.workers[
                        w_id
                    ].post_episode_process, _trace_worker_id=w_id),
                    timeout=self.step_call_timeout,
                    context=f"Worker {w_id} post_episode_process",
                )
            except Exception as e:
                if self._is_worker_unavailable_error(e):
                    self._replace_worker(w_id, reason=e)
                if self.logger is not None:
                    self.logger.exception(
                        "Post-episode processing failed for worker=%s", w_id
                    )
        if isinstance(result, dict):
            finalize_payload = result.get("finalize_payload")
            result = result.get("score")

        if self._worker_cancelled(w_id):
            return {
                "obs": None,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": None,
            }

        # Record result via progress manager
        worker_task = self.progress_manager.get_worker_task(str(w_id))
        keep_lock_for_finalize = (
            worker_task is not None
            and finalize_payload is not None
            and result is not None
        )
        episode_result = self.progress_manager.record_result(
            str(w_id), result, release_lock=not keep_lock_for_finalize
        )
        if keep_lock_for_finalize:
            task_name, task_seed, _ = worker_task
            self._schedule_async_finalize(task_name, task_seed, finalize_payload)

        if self._worker_cancelled(w_id):
            return {
                "obs": None,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": episode_result,
            }

        # Get next task
        config, task_seed = self.progress_manager.get_next_task(str(w_id))
        if config is None and not self.progress_manager.check_finished():
            if self.logger is not None:
                self.logger.info(
                    "No task immediately available for worker=%s; waiting for locks/finalize to clear",
                    w_id,
                )
            config, task_seed = self._wait_for_next_task(w_id)
        if config is None:
            all_done_on_disk = (
                self.progress_manager.reconcile_task_state_from_filesystem()
            )
            if not all_done_on_disk:
                if self.logger is not None:
                    self.logger.warning(
                        "Rebuilt task state from filesystem before final save; "
                        "unfinished episodes remain for worker=%s, retrying task allocation",
                        w_id,
                    )
                config, task_seed = self._wait_for_next_task(w_id)
            if config is not None:
                self._refresh_locks([w_id])
                obs = self._reset_worker(w_id, task_seed, config)
                if self._worker_cancelled(w_id):
                    return {
                        "obs": None,
                        "metric": self.progress_manager.calculate_result(),
                        "episode_result": episode_result,
                    }
                self._refresh_locks([w_id])
                return {
                    "obs": obs,
                    "metric": self.progress_manager.calculate_result(),
                    "episode_result": episode_result,
                }
            # No more tasks
            self._wait_for_pending_finalizers()
            metric = self.progress_manager.calculate_result()
            self._save_final_result()
            return {"obs": None, "metric": metric, "episode_result": episode_result}

        # Reset worker for next task
        if self._worker_cancelled(w_id):
            return {
                "obs": None,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": episode_result,
            }
        self._refresh_locks([w_id])
        obs = self._reset_worker(w_id, task_seed, config)
        if self._worker_cancelled(w_id):
            return {
                "obs": None,
                "metric": self.progress_manager.calculate_result(),
                "episode_result": episode_result,
            }
        self._refresh_locks([w_id])

        return {
            "obs": obs,
            "metric": self.progress_manager.calculate_result(),
            "episode_result": episode_result,
        }

    @traced("pool.save_final_result")
    def _save_final_result(self):
        """Save final evaluation result."""
        if self.progress_manager is None or self.benchmark_id is None:
            return
        self._wait_for_pending_finalizers()
        if self.logger is not None:
            self.logger.info(
                "Saving final result for benchmark_id=%s", self.benchmark_id
            )
        result_dir = os.path.join(
            self.current_dir,
            "saved/eval_results",
            self.benchmark_id,
            self.args.run_id,
        )
        make_dir(result_dir)
        self.progress_manager.save_final_result(result_dir)

    # ==================== Status API ====================

    def get_status(self, refresh: bool = False) -> dict:
        """
        Get comprehensive job status.

        Returns:
            Status dict for the /status endpoint
        """
        active_workers = self.get_active_workers()

        if self.progress_manager is not None:
            return self.progress_manager.get_status(
                active_workers, refresh_filesystem=refresh
            )

        # No job started
        return {
            "status": "idle",
            "benchmark_id": None,
            "run_id": None,
            "total_episodes": 0,
            "completed_episodes": 0,
            "in_progress_episodes": 0,
            "active_workers": active_workers,
            "results": {},
        }

    # ==================== Lifecycle ====================

    def close(self):
        """Close all workers."""
        futures = [remote_call(worker.close, _trace_worker_id=w_id) for w_id, worker in self.workers.items()]
        ray_get(futures)
        self._heartbeat_stop_event.set()
        self._heartbeat_thread.join(timeout=1.0)
        self._reset_executor.shutdown(wait=False, cancel_futures=True)
        self._finalize_executor.shutdown(wait=False, cancel_futures=True)
