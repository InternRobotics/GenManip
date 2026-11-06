"""Ray must serialize locally defined traced actor classes."""

import unittest
from ray import cloudpickle
from genmanip.core.evaluator import request_logging as tracing


class WorkerTraceSerializationTest(unittest.TestCase):
    def test_actor_class_roundtrip_and_context_restoration(self):
        class Worker:
            worker_id = "test"

            @tracing.traced_worker
            def step(self, value):
                from genmanip.core.evaluator import request_logging

                return value + 1, request_logging.TRACE.get()

        restored = cloudpickle.loads(cloudpickle.dumps(Worker))
        self.assertEqual(restored().step(4, _request_trace=None), (5, None))
        self.assertIsNone(tracing.TRACE.get())

    def test_ray_actor_metadata_accepts_trace_keyword(self):
        import ray

        try:
            from ray._common.signature import flatten_args
        except ImportError:
            from ray._private.signature import flatten_args

        class Worker:
            @tracing.traced_worker
            def step(self, value):
                return value

        actor = ray.remote(Worker)
        signature = actor.__ray_metadata__.method_meta.signatures["step"]
        flatten_args(signature, (4,), {"_request_trace": None})


if __name__ == "__main__":
    unittest.main()
