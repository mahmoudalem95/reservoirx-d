"""Runtime configuration, all overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

API_VERSION = "1.4.1"
API_PREFIX = "/v1/reservoirx-d"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """
    Limits exist because every endpoint here is CPU-bound and synchronous.

    A /compare over 8 seeds and 4 tasks is 64 reservoir simulations. The caps
    below keep a single request from occupying a worker for minutes; raise them
    deliberately, and put long jobs behind a queue rather than a request.
    """

    store_backend: str = field(default_factory=lambda: os.environ.get("RXD_STORE_BACKEND", "memory"))
    redis_url: str = field(default_factory=lambda: os.environ.get("RXD_REDIS_URL", "redis://localhost:6379/0"))
    store_max_items: int = field(default_factory=lambda: _int("RXD_STORE_MAX_ITEMS", 256))
    store_ttl_seconds: float = field(default_factory=lambda: _float("RXD_STORE_TTL_SECONDS", 3600.0))

    max_node_count: int = field(default_factory=lambda: _int("RXD_MAX_NODE_COUNT", 2000))
    max_sequence_length: int = field(default_factory=lambda: _int("RXD_MAX_SEQUENCE_LENGTH", 20000))
    max_iterations: int = field(default_factory=lambda: _int("RXD_MAX_ITERATIONS", 32))
    max_null_iterations: int = field(default_factory=lambda: _int("RXD_MAX_NULL_ITERATIONS", 500))
    max_tasks_per_request: int = field(default_factory=lambda: _int("RXD_MAX_TASKS", 4))
    max_ratios_per_request: int = field(default_factory=lambda: _int("RXD_MAX_RATIOS", 10))

    public_url: str = field(default_factory=lambda: os.environ.get("RXD_PUBLIC_URL", "http://localhost:8080"))
    cors_origins: str = field(default_factory=lambda: os.environ.get("RXD_CORS_ORIGINS", "*"))
    request_timeout_seconds: float = field(default_factory=lambda: _float("RXD_TIMEOUT_SECONDS", 240.0))

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
