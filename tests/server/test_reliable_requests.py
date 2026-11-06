"""Run without Isaac Sim: pytest tests/server/test_reliable_requests.py."""

import asyncio
import pickle
import threading
import unittest
import uuid

import httpx

from genmanip.core.evaluator.eval_server import EvalServer


class Pool:
    def __init__(self):
        self.calls = 0
        self.release = threading.Event()
        self.release.set()
        self.error = None

    def step_chunk(self, actions, *options):
        self.calls += 1
        if not self.release.wait(2):
            raise RuntimeError("test worker timed out")
        if self.error:
            raise self.error
        return {"obs": {"0": "observation"}, "executed_steps": len(actions)}

    def kill_workers(self, workers):
        pass

    def reset(self, workers):
        self.calls += 1
        return {w: "reset-observation" for w in workers}

    def get_pending_reset_results(self, workers):
        self.calls += 1
        return {w: "reset-result" for w in workers}

    def step(self, actions):
        self.calls += 1
        return {w: "step-observation" for w in actions}


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = EvalServer("127.0.0.1", 0, 1)
        self.server.replay_wait = 0.5
        self.pool = Pool()
        self.server.app.state.pool = self.pool
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.server.app), base_url="http://test"
        )
        self.envelope = dict(
            epoch=self.server._request_ledger.epoch,
            stream_id=uuid.uuid4().hex,
            sequence=1,
            worker_ids=["0"],
            worker_generations={"0": 0},
            operation="step_chunk",
            payload=pickle.dumps([{"0": {"action": [1]}}]),
        )

    async def asyncTearDown(self):
        self.pool.release.set()
        await self.client.aclose()
        self.server._data_executor.shutdown(wait=True)
        self.server._control_executor.shutdown(wait=True)

    async def send(self, **changes):
        response = await self.client.post(
            "/reliable_request", content=pickle.dumps({**self.envelope, **changes})
        )
        return response.status_code, pickle.loads(response.content)

    async def test_result_loss_replays_without_execution(self):
        _, first = await self.send()
        _, second = await self.send()
        self.assertEqual(first, second)
        self.assertEqual(self.pool.calls, 1)

    async def test_pending_and_concurrent_duplicates_execute_once(self):
        self.server.replay_wait = 0.01
        self.pool.release.clear()
        responses = await asyncio.gather(self.send(), self.send(), self.send())
        self.assertTrue(
            all(
                status == 202 and result["state"] == "pending"
                for status, result in responses
            )
        )
        self.assertEqual(self.pool.calls, 1)
        self.pool.release.set()
        _, result = await self.send()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.pool.calls, 1)

    async def test_disconnect_does_not_cancel_execution(self):
        self.pool.release.clear()
        self.server.replay_wait = 1
        request = asyncio.create_task(self.send())
        while self.pool.calls == 0:
            await asyncio.sleep(0.001)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.pool.release.set()
        _, result = await self.send()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.pool.calls, 1)

    async def test_stale_out_of_order_and_changed_payload_are_rejected(self):
        await self.send()
        for kwargs in (
            {"sequence": 3},
            {"payload": pickle.dumps([{"0": {"action": [2]}}])},
        ):
            _, result = await self.send(**kwargs)
            self.assertEqual(result["state"], "rejected")
        await self.send(sequence=2)
        _, result = await self.send()
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.pool.calls, 2)

    async def test_new_sequence_while_pending_is_rejected(self):
        self.server.replay_wait = 0.01
        self.pool.release.clear()
        await self.send()
        _, result = await self.send(sequence=2)
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.pool.calls, 1)
        self.pool.release.set()
        await self.send()

    async def test_wrong_epoch_never_executes(self):
        _, result = await self.send(epoch="previous-server")
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.pool.calls, 0)

    async def test_errors_are_cached_and_stop_further_steps(self):
        self.pool.error = AssertionError("worker died")
        _, first = await self.send()
        self.assertEqual(first["status_code"], 500)
        self.assertIn("AssertionError", first["detail"])
        _, second = await self.send()
        self.assertEqual(first, second)
        _, third = await self.send(sequence=2)
        self.assertEqual(third["state"], "rejected")
        self.assertEqual(self.pool.calls, 1)

    async def test_other_stream_and_legacy_step_cannot_use_owned_worker(self):
        await self.send()
        _, result = await self.send(stream_id=uuid.uuid4().hex)
        self.assertEqual(result["state"], "rejected")
        legacy = await self.client.post("/step_chunk", content=self.envelope["payload"])
        self.assertEqual(legacy.status_code, 409)
        self.assertEqual(self.pool.calls, 1)

    async def test_kill_invalidates_old_stream(self):
        await self.send()
        await self.client.post("/kill", json={"data": {"worker_ids": ["0"]}})
        _, result = await self.send()
        self.assertEqual(result["state"], "rejected")
        _, result = await self.send(
            stream_id=uuid.uuid4().hex, worker_generations={"0": 1}
        )
        self.assertEqual(result["state"], "completed")

    async def test_scope_mismatch_rejected_before_execution(self):
        _, result = await self.send(payload=pickle.dumps([{"1": {}}]))
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.pool.calls, 0)

    async def test_step_reset_and_reset_result_replay(self):
        for sequence, operation in enumerate(("reset", "step", "reset_result"), 1):
            payload = pickle.dumps(
                {"0": {}} if operation == "step" else {"worker_ids": ["0"]}
            )
            _, first = await self.send(
                sequence=sequence, operation=operation, payload=payload
            )
            _, second = await self.send(
                sequence=sequence, operation=operation, payload=payload
            )
            self.assertEqual(first["status_code"], 200)
            self.assertEqual(first, second)
        self.assertEqual(self.pool.calls, 3)

    async def test_legacy_client_still_works_on_unclaimed_worker(self):
        result = await self.client.post("/step_chunk", content=self.envelope["payload"])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(pickle.loads(result.content)["executed_steps"], 1)

    async def test_first_request_delayed_across_kill_is_rejected(self):
        await self.client.post("/kill", json={"data": {"worker_ids": ["0"]}})
        _, result = await self.send()
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.pool.calls, 0)

    async def test_render_options_cannot_change_on_retry(self):
        await self.send()
        response = await self.client.post(
            "/reliable_request?render_mode=always", content=pickle.dumps(self.envelope)
        )
        self.assertEqual(pickle.loads(response.content)["state"], "rejected")
        self.assertEqual(self.pool.calls, 1)

    async def test_timeout_result_is_cached_while_executor_finishes(self):
        self.server.step_timeout = 0.01
        self.pool.release.clear()
        _, first = await self.send()
        self.assertEqual(first["status_code"], 504)
        self.pool.release.set()
        _, second = await self.send()
        self.assertEqual(first, second)
        self.assertEqual(self.pool.calls, 1)

    async def test_worker_lifecycle_change_blocks_new_requests(self):
        with self.server._request_ledger.changing(["0"]):
            _, result = await self.send(worker_generations={"0": 1})
            self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.pool.calls, 0)

    async def test_capacity_rejects_new_streams_without_evicting_ids(self):
        self.server._request_ledger.max_streams = 1
        await self.send()
        _, result = await self.send(
            worker_ids=["1"],
            worker_generations={"1": 0},
            stream_id=uuid.uuid4().hex,
            payload=pickle.dumps([{"1": {}}]),
        )
        self.assertEqual(result["state"], "rejected")
        _, result = await self.send()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.pool.calls, 1)
