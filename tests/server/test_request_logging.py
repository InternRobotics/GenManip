import asyncio
import inspect
import json
import pickle
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from genmanip.core.evaluator import request_logging as tracing
from genmanip.core.evaluator.eval_server import EvalServer


class RequestLoggingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = EvalServer("127.0.0.1", 8087, 1, step_timeout=0.02)
        self.pool = SimpleNamespace(current_dir=self.temp.name, args=SimpleNamespace(run_id="test"))
        self.server.register_worker_pool(self.pool)
        self.log = next(Path(self.temp.name).rglob("server_*.log"))
        self.trace = {"request_id": "unit-test", "server_log_path": str(self.log)}

    def tearDown(self):
        self.server._data_executor.shutdown(wait=True, cancel_futures=True)
        self.server._control_executor.shutdown(wait=True, cancel_futures=True)
        self.temp.cleanup()

    def events(self):
        return [json.loads(line.split("request_trace ", 1)[1])
                for line in self.log.read_text().splitlines() if "request_trace " in line]

    async def request(self, path, body, request_id="client-123"):
        messages = []
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        async def send(message):
            messages.append(message)
        await self.server.app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [(b"x-request-id", request_id.encode())],
            "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8087),
        }, receive, send)
        return messages

    async def test_http_workers_context_and_payload_privacy(self):
        def step(actions):
            self.assertEqual(tracing.TRACE.get()["request_id"], "client-123")
            return {w: "obs" for w in actions}
        self.pool.step = step
        self.server.step_timeout = 2
        messages = await self.request("/step", pickle.dumps({"worker-7": "private-action"}))
        self.assertEqual(messages[0]["status"], 200)
        self.assertIn((b"x-request-id", b"client-123"), messages[0]["headers"])
        events = self.events()
        self.assertEqual(next(e for e in events if e["event"] == "request.parsed")["worker_ids"], ["worker-7"])
        self.assertTrue(any(e["event"] == "response.sent" for e in events))
        self.assertNotIn("private-action", self.log.read_text())
        self.assertIsNone(tracing.TRACE.get())

    async def test_http_timeout_preserves_running_work_trace(self):
        release = threading.Event()
        def step(actions):
            release.wait(2)
            tracing.event("late.work.finished")
            return {}
        self.pool.step = step
        try:
            messages = await self.request("/step", pickle.dumps({"7": []}))
            self.assertEqual(messages[0]["status"], 504)
            self.assertTrue(any(e["event"] == "http.timeout" for e in self.events()))
        finally:
            release.set()
            await asyncio.to_thread(self.server._data_executor.shutdown, wait=True)
        late = next(e for e in self.events() if e["event"] == "late.work.finished")
        self.assertEqual(late["request_id"], "client-123")
        self.assertFalse(tracing._active)

    async def test_invalid_body_logged_before_parsing(self):
        messages = await self.request("/step", b"not-pickle", "bad\nheader")
        self.assertEqual(messages[0]["status"], 400)
        events = self.events()
        self.assertTrue(any(e["event"] == "body.received" for e in events))
        self.assertEqual(len(events[0]["request_id"]), 32)

    async def test_queue_cancel_and_wait_stack(self):
        release = threading.Event()
        token = tracing.TRACE.set(self.trace)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                running = tracing.submit_traced(executor, release.wait, 2)
                queued = tracing.submit_traced(executor, lambda: None)
                try:
                    with patch.object(tracing, "WAIT_LOG_SECONDS", 0):
                        tracing._report_waits()
                    self.assertTrue(queued.cancel())
                    waiting = [e for e in self.events() if e["event"] == "waiting"]
                    self.assertTrue(any(e["stage"] == "executor.queue" for e in waiting))
                    self.assertTrue(any(e["stack"] for e in waiting))
                finally:
                    release.set()
                running.result()
            self.assertFalse(tracing._active)
        finally:
            tracing.TRACE.reset(token)

    async def test_concurrent_request_context_isolation(self):
        def step(actions):
            tracing.event("pool.observed", worker=list(actions)[0])
            return {}
        self.pool.step = step
        self.server.step_timeout = 2
        await asyncio.gather(*[
            self.request("/step", pickle.dumps({str(i): []}), "request-" + str(i))
            for i in range(8)
        ])
        observed = [e for e in self.events() if e["event"] == "pool.observed"]
        self.assertEqual(len(observed), 8)
        for e in observed:
            self.assertEqual(e["request_id"], "request-" + e["worker"])

    async def test_ray_worker_signature_and_correlation(self):
        from ray._common.signature import extract_signature, flatten_args
        class Worker:
            worker_id = "7"
            @tracing.traced_worker
            def step(self, action):
                tracing.event("inside.worker")
                return action + 1
        signature = extract_signature(Worker.step, ignore_first=True)
        flatten_args(signature, (1,), {"_request_trace": self.trace})
        self.assertIn("_request_trace", inspect.signature(Worker.step).parameters)
        worker = Worker()
        class Method:
            def remote(self, *args, **kwargs):
                self.result = worker.step(*args, **kwargs)
                return SimpleNamespace(hex=lambda: "object-7")
        method = Method()
        token = tracing.TRACE.set(self.trace)
        try:
            ref = tracing.remote_call(method, 1, _trace_worker_id="7")
            with patch("ray.get", return_value=2) as get:
                self.assertEqual(tracing.ray_get(ref, timeout=5), 2)
                get.assert_called_once_with(ref, timeout=5)
        finally:
            tracing.TRACE.reset(token)
        events = self.events()
        worker_event = next(e for e in events if e["event"] == "inside.worker")
        submitted = next(e for e in events if e["event"] == "ray.submitted")
        self.assertEqual(worker_event["rpc_id"], submitted["rpc_id"])
        self.assertEqual(worker_event["worker_id"], "7")
        self.assertEqual(method.result, 2)


if __name__ == "__main__":
    unittest.main()
