"""
Matrix store.

Generated matrices are handed back as IDs so /evaluate and /diagnose can refer
to them without the client round-tripping an n x n array. The default backend is
in-process, which means it does not survive a restart and is not shared between
workers -- run a single worker, or swap in the Redis backend, before scaling out.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StoredMatrix:
    matrix_id: str
    W: np.ndarray
    metadata: dict[str, Any]
    created_at: float


class MatrixStore(ABC):
    @abstractmethod
    def put(self, W: np.ndarray, metadata: dict[str, Any]) -> str: ...

    @abstractmethod
    def get(self, matrix_id: str) -> StoredMatrix | None: ...

    @abstractmethod
    def delete(self, matrix_id: str) -> bool: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...


class InMemoryMatrixStore(MatrixStore):
    """Bounded, TTL-expiring, thread-safe. Evicts least-recently-used on overflow."""

    def __init__(self, max_items: int = 256, ttl_seconds: float = 3600.0):
        self._items: dict[str, StoredMatrix] = {}
        self._last_access: dict[str, float] = {}
        self._lock = threading.RLock()
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds

    def put(self, W: np.ndarray, metadata: dict[str, Any]) -> str:
        matrix_id = f"rxd_{uuid.uuid4().hex[:20]}"
        now = time.time()
        with self._lock:
            self._expire(now)
            while len(self._items) >= self.max_items:
                oldest = min(self._last_access, key=self._last_access.get)
                self._items.pop(oldest, None)
                self._last_access.pop(oldest, None)
            self._items[matrix_id] = StoredMatrix(matrix_id, W, metadata, now)
            self._last_access[matrix_id] = now
        return matrix_id

    def get(self, matrix_id: str) -> StoredMatrix | None:
        now = time.time()
        with self._lock:
            self._expire(now)
            item = self._items.get(matrix_id)
            if item is not None:
                self._last_access[matrix_id] = now
            return item

    def delete(self, matrix_id: str) -> bool:
        with self._lock:
            self._last_access.pop(matrix_id, None)
            return self._items.pop(matrix_id, None) is not None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_bytes = sum(item.W.nbytes for item in self._items.values())
            return {
                "backend": "in_memory",
                "items": len(self._items),
                "max_items": self.max_items,
                "ttl_seconds": self.ttl_seconds,
                "approx_bytes": total_bytes,
            }

    def _expire(self, now: float) -> None:
        stale = [k for k, v in self._items.items() if now - v.created_at > self.ttl_seconds]
        for key in stale:
            self._items.pop(key, None)
            self._last_access.pop(key, None)


class RedisMatrixStore(MatrixStore):
    """
    Redis-backed store for multi-worker deployments.

    Matrices are serialised with numpy's .npy format, which round-trips dtype and
    shape exactly. Requires the `redis` package; construction fails loudly if the
    server is unreachable rather than silently degrading to in-memory.
    """

    def __init__(self, url: str, ttl_seconds: float = 3600.0, prefix: str = "rxd:matrix:"):
        try:
            import redis  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "RedisMatrixStore needs the 'redis' package: pip install redis"
            ) from exc
        self._redis = redis.Redis.from_url(url)
        self._redis.ping()
        self.ttl_seconds = int(ttl_seconds)
        self.prefix = prefix

    def put(self, W: np.ndarray, metadata: dict[str, Any]) -> str:
        import json  # noqa: PLC0415

        matrix_id = f"rxd_{uuid.uuid4().hex[:20]}"
        buf = io.BytesIO()
        np.save(buf, W, allow_pickle=False)
        pipe = self._redis.pipeline()
        pipe.setex(f"{self.prefix}{matrix_id}:W", self.ttl_seconds, buf.getvalue())
        pipe.setex(f"{self.prefix}{matrix_id}:meta", self.ttl_seconds, json.dumps(metadata))
        pipe.execute()
        return matrix_id

    def get(self, matrix_id: str) -> StoredMatrix | None:
        import json  # noqa: PLC0415

        raw = self._redis.get(f"{self.prefix}{matrix_id}:W")
        meta = self._redis.get(f"{self.prefix}{matrix_id}:meta")
        if raw is None or meta is None:
            return None
        W = np.load(io.BytesIO(raw), allow_pickle=False)
        return StoredMatrix(matrix_id, W, json.loads(meta), time.time())

    def delete(self, matrix_id: str) -> bool:
        removed = self._redis.delete(
            f"{self.prefix}{matrix_id}:W", f"{self.prefix}{matrix_id}:meta"
        )
        return bool(removed)

    def stats(self) -> dict[str, Any]:
        return {"backend": "redis", "ttl_seconds": self.ttl_seconds, "prefix": self.prefix}
