import asyncio
from genmanip.core.evaluator.request_logging import (
    RequestLoggingMiddleware, event, submit_traced,
)
from concurrent.futures import ThreadPoolExecutor
import logging
from fastapi import FastAPI, APIRouter, Request, HTTPException, Response
from fastapi.responses import JSONResponse
import traceback
import uvicorn
import pickle
from urllib.parse import urlencode

from genmanip.core.evaluator.reliable_requests import RequestLedger, ProtocolError

from genmanip.core.evaluator.logging_utils import (
    build_file_logger,
    get_run_log_dir,
    make_log_file_timestamp,
)

# Timeout constants (in seconds)
DEFAULT_STEP_TIMEOUT = 600  # 10 minutes
DEFAULT_RESET_TIMEOUT = 600  # 10 minutes
DEFAULT_KILL_TIMEOUT = 120  # 2 minutes
DEFAULT_CREATE_TIMEOUT = 600  # 10 minutes
DEFAULT_LOAD_CONFIG_TIMEOUT = 60  # 1 minute
DEFAULT_DATA_EXECUTOR_WORKERS = 32
DEFAULT_CONTROL_EXECUTOR_WORKERS = 8


class EvalServer:
    """
    Server class for Eval service.
    Uses FastAPI to create a web server that handles evaluation requests.

    Args:
        host (str): Host address for the server.
        port (int): Port number for the server.
        workers (int): Number of worker processes for the server.

    Routes:
        - /step: POST request to perform a step in the evaluation.
        - /reset: POST request to reset the evaluation environment.
        - /kill: POST request to kill specified workers.
        - /create_workers: create workers by id
        - /start_new_job: load a new task config

    Functions:
        - register_worker_pool(pool): Register a worker pool to handle evaluation tasks.
        - run(): Start the FastAPI server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        workers: int,
        step_timeout: float = DEFAULT_STEP_TIMEOUT,
        reset_timeout: float = DEFAULT_RESET_TIMEOUT,
        kill_timeout: float = DEFAULT_KILL_TIMEOUT,
        create_timeout: float = DEFAULT_CREATE_TIMEOUT,
        load_config_timeout: float = DEFAULT_LOAD_CONFIG_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.workers = workers
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.kill_timeout = kill_timeout
        self.create_timeout = create_timeout
        self.load_config_timeout = load_config_timeout
        self.logger: logging.Logger | None = None
        self.server_log_id = make_log_file_timestamp()
        self._data_executor = ThreadPoolExecutor(
            max_workers=DEFAULT_DATA_EXECUTOR_WORKERS,
            thread_name_prefix="eval-data",
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=DEFAULT_CONTROL_EXECUTOR_WORKERS,
            thread_name_prefix="eval-control",
        )
        self._request_ledger = RequestLedger()
        self.replay_wait = 5.0
        self.app = FastAPI(title="Eval Service")
        self.app.add_middleware(RequestLoggingMiddleware, server=self)
        self._router = APIRouter()
        self._register_routes()
        self.app.include_router(self._router)

    def configure_logging(self, current_dir: str, run_id: str | None) -> None:
        log_dir = get_run_log_dir(current_dir, run_id)
        self.logger = build_file_logger(
            logger_name=f"EvalServer[{run_id or 'default'}][{self.server_log_id}]",
            log_path=f"{log_dir}/server_{self.server_log_id}.log",
        )
        self.logger.info(
            "Configured server logging. run_id=%s server_id=%s",
            run_id or "default",
            self.server_log_id,
        )

    def _log_exception(self, message: str) -> None:
        if self.logger is not None:
            self.logger.exception(message)
        else:
            traceback.print_exc()

    def _register_routes(self):
        route_config = [
            ("/request_protocol", self.request_protocol, ["GET"]),
            ("/reliable_request", self.reliable_request, ["POST"]),
            ("/step", self.step, ["POST"]),
            ("/step_chunk", self.step_chunk, ["POST"]),
            ("/reset_result", self.reset_result, ["POST"]),
            ("/kill", self.kill, ["POST"]),
            ("/reset", self.reset, ["POST"]),
            ("/create_workers", self.create_workers, ["POST"]),
            ("/start_new_job", self.start_new_job, ["POST"]),
            ("/status", self.status, ["GET"]),
        ]

        for path, handler, methods in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
            )

    @staticmethod
    def _internal_error_detail(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    async def _run_in_data_executor(self, func, *args):
        self._log_dispatch(func, args)
        return await asyncio.wrap_future(submit_traced(self._data_executor, func, *args))

    async def _run_in_control_executor(self, func, *args):
        self._log_dispatch(func, args)
        return await asyncio.wrap_future(submit_traced(self._control_executor, func, *args))

    @staticmethod
    def _log_dispatch(func, args):
        operation = func.__name__
        workers = None
        chunk_length = None
        if args and operation == "step" and isinstance(args[0], dict):
            workers = list(args[0])
        elif args and operation == "step_chunk" and isinstance(args[0], list):
            chunk_length = len(args[0])
            workers = sorted({str(w) for action in args[0] if isinstance(action, dict) for w in action})
        elif args and operation in ("reset", "get_pending_reset_results", "create_workers", "kill_workers"):
            workers = args[0]
        event("request.parsed", operation=operation, worker_ids=workers, chunk_length=chunk_length)

    async def request_protocol(self):
        return JSONResponse(
            {"protocol": "genmanip-replay-v1", "epoch": self._request_ledger.epoch,
             "worker_generations": dict(self._request_ledger.generations)},
            headers={"Cache-Control": "no-store"})

    async def reliable_request(self, request: Request) -> Response:
        """Submit or retrieve a request. Disconnects do not cancel its execution."""
        def reply(data, status=200):
            return Response(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
                            status_code=status, media_type="application/octet-stream",
                            headers={"Cache-Control": "no-store"})

        try:
            envelope = pickle.loads(await request.body())
            if not isinstance(envelope, dict):
                raise ProtocolError("Expected a request envelope")
            operation = envelope.get("operation")
            if operation not in ("step", "step_chunk", "reset", "reset_result"):
                raise ProtocolError("Unsupported reliable operation")
            payload = envelope.get("payload")
            if not isinstance(payload, bytes):
                raise ProtocolError("payload must be bytes")
            data = pickle.loads(payload)
            workers = envelope.get("worker_ids")
            if not isinstance(workers, list) or not workers or any(not isinstance(w, str) for w in workers):
                raise ProtocolError("worker_ids must be a non-empty list of strings")
            if operation in ("reset", "reset_result"):
                used = data.get("worker_ids") if isinstance(data, dict) else None
                if not isinstance(used, list) or not used:
                    raise ProtocolError("Explicit worker_ids are required")
            else:
                actions = data if operation == "step_chunk" else [data]
                if not isinstance(actions, list) or not actions or any(not isinstance(a, dict) or not a for a in actions):
                    raise ProtocolError("Expected non-empty action dictionaries")
                used = [w for action in actions for w in action]
            if any(not isinstance(w, str) or w not in workers for w in used):
                raise ProtocolError("Payload refers to workers outside this stream")
            # Keep run routing metadata; only explicit rendering options affect execution.
            options = {k: request.query_params[k] for k in ("render_mode", "subframes", "render_suffix")
                       if k in request.query_params}
            identity = pickle.dumps((operation, payload, sorted(options.items())))
            scope = dict(request.scope)
            scope["genmanip_reliable"] = True
            scope["path"] = "/" + operation
            scope["query_string"] = urlencode(options).encode()

            async def receive():
                return {"type": "http.request", "body": payload, "more_body": False}

            async def execute():
                try:
                    response = await getattr(self, operation)(Request(scope, receive))
                    return {"status_code": response.status_code, "body": response.body}
                except HTTPException as exc:
                    return {"status_code": exc.status_code, "detail": exc.detail}
                except Exception as exc:
                    self._log_exception("Unhandled reliable request exception")
                    return {"status_code": 500, "detail": self._internal_error_detail(exc)}

            event("replay.received", replay_id=f"{envelope.get('stream_id')}:{envelope.get('sequence')}",
                  operation=operation, worker_ids=workers)
            task = self._request_ledger.submit(
                envelope.get("epoch"), envelope.get("stream_id"), envelope.get("sequence"),
                workers, envelope.get("worker_generations"), identity, execute)
            info = {"protocol": "genmanip-replay-v1", "epoch": self._request_ledger.epoch,
                    "stream_id": envelope["stream_id"], "sequence": envelope["sequence"]}
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=self.replay_wait)
            except asyncio.TimeoutError:
                event("replay.pending", replay_id=f"{info['stream_id']}:{info['sequence']}")
                return reply({**info, "state": "pending"}, 202)
            event("replay.completed", replay_id=f"{info['stream_id']}:{info['sequence']}",
                  operation_status=result["status_code"])
            return reply({**info, "state": "completed", **result})
        except (ProtocolError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            # An application error travels in a 200 envelope so proxies cannot replace it.
            return reply({"protocol": "genmanip-replay-v1", "state": "rejected", "detail": str(exc)})

    def _guard_legacy(self, request, workers):
        if not request.scope.get("genmanip_reliable"):
            try:
                self._request_ledger.check_unclaimed(workers)
            except ProtocolError as exc:
                raise HTTPException(status_code=409, detail=str(exc))

    # ================= Registered routes =================
    async def step(self, request: Request) -> Response:
        """
        step vec env by the action, return binary obs/data

        Args:
            request.body(): pickled action_dict

        Return:
            pickled obs_dict / data_dict
                {
                    worker_id:
                    {
                        'obs': ...,
                        'metric': ...,
                        'episode_result': ...,
                    }
                }
        """
        pool = request.app.state.pool
        try:
            raw = await request.body()
            try:
                action_dict = pickle.loads(raw)
            except (
                AttributeError,
                EOFError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ) as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid pickle in request body: {e}"
                )

            self._guard_legacy(request, action_dict.keys())
            try:
                response_data = await asyncio.wait_for(
                    self._run_in_data_executor(pool.step, action_dict),
                    timeout=self.step_timeout,
                )
            except asyncio.TimeoutError:
                event("http.timeout", operation=request.url.path)
                raise HTTPException(
                    status_code=504,
                    detail=f"Step timed out after {self.step_timeout}s",
                )

            # return pickled response_data
            blob = pickle.dumps(response_data, protocol=pickle.HIGHEST_PROTOCOL)
            return Response(content=blob, media_type="application/octet-stream")

        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /step")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def step_chunk(self, request: Request) -> Response:
        """
        Execute an action chunk and return only the final observation/result.

        Args:
            request.body(): pickled list[action_dict]

        Return:
            pickled {
                "obs": final_response_dict,
                "executed_steps": int,
                "stopped_early": bool,
            }
        """
        pool = request.app.state.pool
        try:
            raw = await request.body()
            try:
                action_chunk = pickle.loads(raw)
            except (
                AttributeError,
                EOFError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ) as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid pickle in request body: {e}"
                )
            if not isinstance(action_chunk, list):
                raise HTTPException(
                    status_code=400, detail="action_chunk must be a list[action_dict]"
                )

            self._guard_legacy(request, [w for a in action_chunk for w in a])
            try:
                chunk_timeout = self.step_timeout * max(1, len(action_chunk))
                try:
                    subframes = int(request.query_params.get("subframes", "2"))
                except (TypeError, ValueError):
                    subframes = 2
                try:
                    render_suffix = int(request.query_params.get("render_suffix", "2"))
                except (TypeError, ValueError):
                    render_suffix = 2
                render_mode = request.query_params.get("render_mode", "lite")
                if render_mode not in ("lite", "always"):
                    render_mode = "lite"
                response_data = await asyncio.wait_for(
                    self._run_in_data_executor(
                        pool.step_chunk,
                        action_chunk,
                        render_mode,
                        subframes,
                        render_suffix,
                    ),
                    timeout=chunk_timeout,
                )
            except asyncio.TimeoutError:
                event("http.timeout", operation=request.url.path)
                raise HTTPException(
                    status_code=504,
                    detail=f"Step chunk timed out after {chunk_timeout}s",
                )

            blob = pickle.dumps(response_data, protocol=pickle.HIGHEST_PROTOCOL)
            return Response(content=blob, media_type="application/octet-stream")

        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /step_chunk")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def kill(self, request: Request):
        """
        body: {"data": {"worker_ids": [...]}}
        """
        pool = request.app.state.pool
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(
                    status_code=400, detail="Request body must be a JSON object"
                )
            data = body.get("data")
            if not isinstance(data, dict) or "worker_ids" not in data:
                raise HTTPException(
                    status_code=400, detail="Request body must contain data.worker_ids"
                )
            worker_ids = data["worker_ids"]
            with self._request_ledger.changing(worker_ids):
                try:
                    await asyncio.wait_for(
                        self._run_in_control_executor(pool.kill_workers, worker_ids),
                        timeout=self.kill_timeout,
                    )
                except asyncio.TimeoutError:
                    event("http.timeout", operation=request.url.path)
                    raise HTTPException(
                        status_code=504,
                        detail=f"Kill timed out after {self.kill_timeout}s",
                    )
                return {"data": {"success": True}}
        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /kill")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def reset(self, request: Request) -> Response:
        """
        reset in gym style

        Args:
            request: pickled {"worker_ids": [...]}

        Return:
            pickled obs_dict
        """
        pool = request.app.state.pool
        try:
            raw = await request.body()
            if raw:
                try:
                    req_data = pickle.loads(raw)
                except (
                    AttributeError,
                    EOFError,
                    TypeError,
                    ValueError,
                    pickle.UnpicklingError,
                ) as e:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid pickle in reset body: {e}"
                    )
                if not isinstance(req_data, dict):
                    raise HTTPException(
                        status_code=400, detail="Reset body must decode to a dict"
                    )
                worker_ids = req_data.get("worker_ids", None)
            else:
                worker_ids = None

            self._guard_legacy(request, worker_ids)
            try:
                obs_dict = await asyncio.wait_for(
                    self._run_in_control_executor(pool.reset, worker_ids),
                    timeout=self.reset_timeout,
                )
            except asyncio.TimeoutError:
                event("http.timeout", operation=request.url.path)
                raise HTTPException(
                    status_code=504,
                    detail=f"Reset timed out after {self.reset_timeout}s",
                )

            blob = pickle.dumps(obs_dict, protocol=pickle.HIGHEST_PROTOCOL)
            return Response(content=blob, media_type="application/octet-stream")

        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /reset")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def reset_result(self, request: Request) -> Response:
        """
        Poll pending reset results triggered by /step episode completion.

        Args:
            request: pickled {"worker_ids": [...]} or empty body for all pending workers

        Return:
            pickled response dict keyed by worker_id
        """
        pool = request.app.state.pool
        try:
            raw = await request.body()
            if raw:
                try:
                    req_data = pickle.loads(raw)
                except (
                    AttributeError,
                    EOFError,
                    TypeError,
                    ValueError,
                    pickle.UnpicklingError,
                ) as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid pickle in reset_result body: {e}",
                    )
                if not isinstance(req_data, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="Reset result body must decode to a dict",
                    )
                worker_ids = req_data.get("worker_ids", None)
            else:
                worker_ids = None

            self._guard_legacy(request, worker_ids)
            result_dict = await asyncio.wait_for(
                self._run_in_control_executor(
                    pool.get_pending_reset_results, worker_ids
                ),
                timeout=self.reset_timeout,
            )
            blob = pickle.dumps(result_dict, protocol=pickle.HIGHEST_PROTOCOL)
            return Response(content=blob, media_type="application/octet-stream")
        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /reset_result")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def create_workers(self, request: Request):
        """
        body: {"data": {"worker_ids": [...]}}
        """
        pool = request.app.state.pool
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(
                    status_code=400, detail="Request body must be a JSON object"
                )
            data = body.get("data")
            if not isinstance(data, dict) or "worker_ids" not in data:
                raise HTTPException(
                    status_code=400, detail="Request body must contain data.worker_ids"
                )
            worker_ids = data["worker_ids"]
            with self._request_ledger.changing(worker_ids):
                try:
                    await asyncio.wait_for(
                        self._run_in_control_executor(pool.create_workers, worker_ids),
                        timeout=self.create_timeout,
                    )
                except asyncio.TimeoutError:
                    event("http.timeout", operation=request.url.path)
                    raise HTTPException(
                        status_code=504,
                        detail=f"Create workers timed out after {self.create_timeout}s",
                    )
                return {"data": {"success": True}}
        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /create_workers")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def start_new_job(self, request: Request):
        """
        body: {"data": {"config_path": ...}}
        """
        pool = request.app.state.pool
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(
                    status_code=400, detail="Request body must be a JSON object"
                )
            data = body.get("data")
            if not isinstance(data, dict) or "config_path" not in data:
                raise HTTPException(
                    status_code=400,
                    detail="Request body must contain data.config_path",
                )
            config_path = data["config_path"]
            run_id = data.get("run_id", None)
            if not isinstance(config_path, list) or not config_path:
                raise HTTPException(
                    status_code=400,
                    detail="config_path must be a non-empty list of strings",
                )
            if any(
                (not isinstance(item, str)) or (item.strip() == "")
                for item in config_path
            ):
                raise HTTPException(
                    status_code=400,
                    detail="config_path must contain non-empty strings only",
                )
            if run_id is not None and not isinstance(run_id, str):
                raise HTTPException(
                    status_code=400,
                    detail="run_id must be a string when provided",
                )
            with self._request_ledger.changing():
                try:
                    await asyncio.wait_for(
                        self._run_in_control_executor(
                            pool.start_new_job, run_id, config_path
                        ),
                        timeout=self.load_config_timeout,
                    )
                except asyncio.TimeoutError:
                    event("http.timeout", operation=request.url.path)
                    raise HTTPException(
                        status_code=504,
                        detail=f"Load config timed out after {self.load_config_timeout}s",
                    )
                effective_run_id = getattr(pool.args, "run_id", run_id)
                self.configure_logging(pool.current_dir, effective_run_id)
                if self.logger is not None:
                    self.logger.info(
                        "Started new job. run_id=%s config_count=%s",
                        effective_run_id,
                        len(config_path),
                    )
                return {"data": {"success": True}}
        except HTTPException:
            raise
        except Exception as e:
            self._log_exception("Unhandled exception in /start_new_job")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    async def status(self, request: Request):
        """
        GET /status - Get current job status

        Returns:
            {
                "status": "idle|running|complete|incomplete",
                "benchmark_id": "...",
                "run_id": "...",
                "total_episodes": 100,
                "completed_episodes": 50,
                "in_progress_episodes": 2,
                "active_workers": ["0", "1"],
                "results": {...}
            }
        """
        pool = request.app.state.pool
        try:
            refresh = request.query_params.get("refresh", "0").lower() in (
                "1",
                "true",
                "yes",
            )
            status_data = await self._run_in_control_executor(pool.get_status, refresh)
            return {"data": status_data}
        except Exception as e:
            self._log_exception("Unhandled exception in /status")
            raise HTTPException(status_code=500, detail=self._internal_error_detail(e))

    # ================= Server functions =================
    def register_worker_pool(self, pool):
        self.app.state.pool = pool
        self.configure_logging(pool.current_dir, getattr(pool.args, "run_id", None))
        if self.logger is not None:
            self.logger.info(
                "Registered worker pool. initial_run_id=%s",
                getattr(pool.args, "run_id", None) or "default",
            )

    def run(self):
        if self.logger is not None:
            self.logger.info(
                "Starting uvicorn server on %s:%s with workers=%s",
                self.host,
                self.port,
                self.workers,
            )
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers,
        )
