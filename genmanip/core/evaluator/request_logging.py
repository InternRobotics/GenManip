"""Request tracing across ASGI, executors and Ray (no payload logging)."""

import contextvars
import functools
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager

from genmanip.core.evaluator.logging_utils import build_file_logger

TRACE = contextvars.ContextVar("eval_request_trace", default=None)
WAIT_LOG_SECONDS = 30
_active = {}
_lock = threading.Lock()
_monitor = None
_loggers = {}
_logger_lock = threading.Lock()


def event(name, **fields):
    trace = TRACE.get()
    if trace is None:
        return
    path = trace.get("server_log_path")
    with _logger_lock:
        if path not in _loggers:
            _loggers[path] = (
                build_file_logger("RequestTrace:" + path, path)
                if path
                else logging.getLogger("eval.request")
            )
        logger = _loggers[path]
    logger.info(
        "request_trace %s",
        json.dumps(
            {
                **trace,
                "event": name,
                "pid": os.getpid(),
                "thread": threading.current_thread().name,
                **fields,
            },
            ensure_ascii=True,
            default=str,
        ),
    )


def _report_waits():
    now = time.monotonic()
    with _lock:
        pending = list(_active.values())
    for started, trace, stage, fields, thread_id in pending:
        if now - started < WAIT_LOG_SECONDS:
            continue
        frame = sys._current_frames().get(thread_id)
        token = TRACE.set(trace)
        try:
            event(
                "waiting",
                stage=stage,
                elapsed_s=round(now - started, 3),
                stack="".join(traceback.format_stack(frame, limit=10)) if frame else "",
                **fields
            )
        finally:
            TRACE.reset(token)


def _watchdog():
    while True:
        time.sleep(WAIT_LOG_SECONDS)
        _report_waits()


@contextmanager
def span(stage, **fields):
    global _monitor
    trace = TRACE.get()
    if trace is None:
        yield
        return
    started = time.monotonic()
    key = uuid.uuid4().hex
    fields = {"span_id": key, **fields}
    event("start", stage=stage, **fields)
    with _lock:
        _active[key] = (started, trace, stage, fields, threading.get_ident())
        if _monitor is None:
            _monitor = threading.Thread(
                target=_watchdog, name="request-watchdog", daemon=True
            )
            _monitor.start()
    try:
        yield
    except BaseException as exc:
        event(
            "error",
            stage=stage,
            error_type=type(exc).__name__,
            elapsed_s=round(time.monotonic() - started, 3),
            **fields
        )
        raise
    else:
        event(
            "end", stage=stage, elapsed_s=round(time.monotonic() - started, 3), **fields
        )
    finally:
        with _lock:
            _active.pop(key, None)


def traced(stage):
    """Instrument pool operations, preserving thread-local request context."""

    def decorate(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            fields = {
                key: bound[key]
                for key in ("w_id", "worker_ids", "task_seed")
                if key in bound
            }
            with span(stage, **fields):
                return func(*args, **kwargs)

        return wrapped

    return decorate


def traced_worker(func):
    @functools.wraps(func)
    def wrapped(self, *args, _request_trace=None, **kwargs):
        # Resolve process-local tracing state after Ray deserializes the actor.
        # Capturing TRACE here makes cloudpickle serialize the ContextVar.
        from genmanip.core.evaluator import request_logging as tracing

        token = tracing.TRACE.set(_request_trace)
        try:
            with tracing.span("worker." + func.__name__, worker_id=self.worker_id):
                return func(self, *args, **kwargs)
        finally:
            tracing.TRACE.reset(token)

    # Ray validates arguments using inspect.signature (which follows __wrapped__).
    wrapped.__signature__ = inspect.signature(func).replace(
        parameters=[
            *inspect.signature(func).parameters.values(),
            inspect.Parameter(
                "_request_trace", inspect.Parameter.KEYWORD_ONLY, default=None
            ),
        ]
    )
    # Ray unwraps actor methods before extracting their call signatures.
    # Preserve our explicit signature instead of exposing the original method.
    del wrapped.__wrapped__
    return wrapped


def remote_call(method, *args, _trace_worker_id=None, **kwargs):
    trace = TRACE.get()
    rpc_id = uuid.uuid4().hex
    child_trace = (
        {**trace, "rpc_id": rpc_id, "worker_id": _trace_worker_id} if trace else None
    )
    # Only this private keyword crosses Ray; actions/observations are never logged.
    with span("ray.submit", rpc_id=rpc_id, worker_id=_trace_worker_id):
        ref = method.remote(*args, _request_trace=child_trace, **kwargs)
        event(
            "ray.submitted",
            rpc_id=rpc_id,
            worker_id=_trace_worker_id,
            object_ref=ref.hex(),
        )
        return ref


def ray_get(refs, **kwargs):
    import ray

    items = refs if isinstance(refs, list) else [refs]
    with span(
        "ray.wait",
        object_refs=[ref.hex() for ref in items],
        timeout_s=kwargs.get("timeout"),
    ):
        return ray.get(refs, **kwargs)


def submit_traced(executor, func, *args, **kwargs):
    """Keep queue and execution visible even after the HTTP waiter times out."""
    context = contextvars.copy_context()
    queued = span("executor.queue", operation=func.__name__)
    queued.__enter__()

    def run():
        queued.__exit__(None, None, None)
        with span("executor.run", operation=func.__name__):
            return func(*args, **kwargs)

    try:
        future = executor.submit(context.run, run)
    except BaseException:
        queued.__exit__(*sys.exc_info())
        raise

    def on_done(done):
        if done.cancelled():
            # A queued job can be cancelled before its callable starts.
            context.run(queued.__exit__, None, None, None)

    future.add_done_callback(on_done)
    return future


class RequestLoggingMiddleware:
    """Pure ASGI middleware: trace body reception and response sending as well."""

    def __init__(self, app, server):
        self.app = app
        self.server = server

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id", b"").decode("ascii", errors="replace")
        request_id = (
            incoming
            if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", incoming)
            else uuid.uuid4().hex
        )
        logger = self.server.logger
        paths = (
            [
                h.baseFilename
                for h in logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
            if logger
            else []
        )
        token = TRACE.set(
            {
                "request_id": request_id,
                "method": scope["method"],
                "path": scope["path"],
                "server_log_path": paths[0] if paths else None,
            }
        )
        body_bytes = 0

        async def traced_receive():
            nonlocal body_bytes
            message = await receive()
            if message["type"] == "http.request":
                body_bytes += len(message.get("body", b""))
                if not message.get("more_body", False):
                    event("body.received", bytes=body_bytes)
            elif message["type"] == "http.disconnect":
                event("http.disconnect")
            return message

        async def traced_send(message):
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        (k, v)
                        for k, v in message.get("headers", [])
                        if k.lower() != b"x-request-id"
                    ]
                    + [(b"x-request-id", request_id.encode("ascii"))],
                }
                event("response.start", status=message["status"])
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                event("response.sent")

        try:
            with span("http.request"):
                await self.app(scope, traced_receive, traced_send)
        finally:
            TRACE.reset(token)
