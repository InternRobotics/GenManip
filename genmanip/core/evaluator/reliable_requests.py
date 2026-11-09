"""In-memory, ordered request replay for one evaluation server process.

Only the latest result per stream is retained. Sequence high-water marks are
never evicted: an expired result must never turn into a new simulation action.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import uuid


class ProtocolError(ValueError):
    pass


@dataclass
class Stream:
    workers: tuple
    sequence: int = 0
    digest: str = ""
    task: object = None
    retired: bool = False


class RequestLedger:
    def __init__(self, max_streams=4096):
        self.epoch = uuid.uuid4().hex
        self.max_streams = max_streams
        self.streams = {}
        self.owners = {}
        self.generations = {}
        self.changing_workers = set()
        self.changing_all = False

    @contextmanager
    def changing(self, workers=None):
        selected = set(map(str, workers)) if workers is not None else set()
        if (
            self.changing_all
            or (workers is None and self.changing_workers)
            or selected.intersection(self.changing_workers)
        ):
            raise ProtocolError("Worker lifecycle change already in progress")
        if workers is None:
            self.changing_all = True
            self.epoch = uuid.uuid4().hex
        else:
            self.changing_workers.update(selected)
            for worker in selected:
                self.generations[worker] = self.generations.get(worker, 0) + 1
        self.invalidate(workers)
        try:
            yield
        finally:
            if workers is None:
                self.changing_all = False
            self.changing_workers.difference_update(selected)

    def invalidate(self, workers=None):
        """Retire streams before a worker lifecycle change; retain tombstones."""
        selected = set(self.owners) if workers is None else set(map(str, workers))
        for worker in selected:
            stream_id = self.owners.get(worker)
            if stream_id is not None:
                stream = self.streams[stream_id]
                stream.retired = True
                # Keep only a tombstone after the background task finishes.
                if stream.task is not None and not stream.task.done():
                    stream.task.add_done_callback(
                        lambda _, s=stream: setattr(s, "task", None)
                    )
                else:
                    stream.task = None
                for owned in stream.workers:
                    self.owners.pop(owned, None)

    def check_unclaimed(self, workers):
        selected = set(self.owners) if workers is None else set(map(str, workers))
        if selected.intersection(self.owners):
            raise ProtocolError(
                "Workers belong to a reliable request stream; use that stream or kill/reset the client first"
            )

    def submit(
        self, epoch, stream_id, sequence, workers, generations, identity, execute
    ):
        # No await until the task is registered: admission is atomic on the ASGI loop.
        if epoch != self.epoch:
            raise ProtocolError(
                "Server epoch changed; the previous episode cannot be resumed"
            )
        if not isinstance(stream_id, str) or len(stream_id) != 32:
            raise ProtocolError("stream_id must be a UUID hex string")
        try:
            int(stream_id, 16)
        except ValueError:
            raise ProtocolError("stream_id must be a UUID hex string")
        if type(sequence) is not int or sequence < 1:
            raise ProtocolError("sequence must be a positive integer")
        if (
            not isinstance(workers, list)
            or not workers
            or any(not isinstance(w, str) for w in workers)
        ):
            raise ProtocolError("worker_ids must be a non-empty list of strings")
        workers = tuple(sorted(set(workers)))
        if self.changing_all or set(workers).intersection(self.changing_workers):
            raise ProtocolError("Worker lifecycle change in progress")
        if not isinstance(generations, dict) or any(
            generations.get(w) != self.generations.get(w, 0) for w in workers
        ):
            raise ProtocolError(
                "Worker generation changed; the previous episode cannot be resumed"
            )
        digest = hashlib.sha256(identity).hexdigest()
        stream = self.streams.get(stream_id)
        if stream is None:
            if sequence != 1:
                raise ProtocolError("Unknown stream; sequence must start at 1")
            if len(self.streams) >= self.max_streams:
                raise ProtocolError(
                    "Request stream capacity reached; restart the server between jobs"
                )
            if any(w in self.owners for w in workers):
                raise ProtocolError("Workers already belong to another request stream")
            stream = self.streams[stream_id] = Stream(workers)
            self.owners.update({w: stream_id for w in workers})
        if stream.retired or stream.workers != workers:
            raise ProtocolError("Request stream was invalidated or its workers changed")
        if sequence == stream.sequence:
            if digest != stream.digest:
                raise ProtocolError("Same request ID was reused with different content")
            return stream.task
        if sequence != stream.sequence + 1:
            raise ProtocolError(
                "Stale or out-of-order request; old results are not re-executed"
            )
        if stream.task is not None:
            if not stream.task.done():
                raise ProtocolError("Previous request is still pending")
            if stream.task.cancelled() or stream.task.exception() is not None:
                raise ProtocolError(
                    "Previous request failed; this stream cannot continue"
                )
            if stream.task.result().get("status_code", 500) != 200:
                raise ProtocolError(
                    "Previous operation failed; start a new episode explicitly"
                )
        stream.sequence, stream.digest = sequence, digest
        stream.task = asyncio.create_task(execute())
        return stream.task
