"""ReservoirX-D FastAPI application."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import API_PREFIX, API_VERSION, settings
from .core.graphs import GraphConstructionError
from .models import HealthResponse
from .routers.api import router as v1_router
from .store import InMemoryMatrixStore, RedisMatrixStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("reservoirx_d")

DESCRIPTION = """
Generate directed, degree-regular reservoir weight matrices, then **test** whether
that wiring actually helps on your configuration.

### What this service does differently

Most of the published support for regular-degree connectivity is architecture-specific,
and some of it was never independently reproduced. Rather than shipping those results as
product claims, ReservoirX-D exposes the experiment:

* `POST /compare` runs the regular-vs-skewed comparison on your parameters, paired by
  seed, matched on edge budget, Holm-corrected across tasks.
* `POST /optimize` selects a width on one set of seeds and confirms it on another, and
  reports connection savings it actually counted.
* `GET /claims` returns the provenance and evidential status of every claim behind the
  service, including the two that come from an external pipeline that was not reproduced.

### Construction

NetworkX has no directed analogue of `random_regular_graph`, and the obvious substitute
(`random_regular_graph(...).to_directed()`) yields a symmetric matrix, destroying exactly
the directedness this service depends on. Graphs here are built by configuration-model
stub pairing followed by directed double-edge swaps, which repairs self-loops and parallel
edges while preserving the degree sequence exactly.

### Limits

Every endpoint is synchronous CPU work. `/compare` over 8 seeds and 4 tasks is 64
reservoir simulations. Size your requests accordingly, or put them behind a queue.
"""

TAGS_METADATA = [
    {"name": "generation", "description": "Build reservoir matrices with controlled degree structure."},
    {"name": "analysis", "description": "Structural diagnostics and task performance."},
    {"name": "experiments", "description": "Comparisons and width searches with held-out confirmation."},
    {"name": "meta", "description": "Health, version, evidence registry."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.store_backend == "redis":
        logger.info("using redis matrix store at %s", settings.redis_url)
        app.state.store = RedisMatrixStore(
            settings.redis_url, ttl_seconds=settings.store_ttl_seconds
        )
    else:
        logger.info("using in-memory matrix store (single worker only)")
        app.state.store = InMemoryMatrixStore(
            max_items=settings.store_max_items, ttl_seconds=settings.store_ttl_seconds
        )
    yield
    app.state.store = None


app = FastAPI(
    title="ReservoirX-D",
    version=API_VERSION,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    servers=[
        {"url": settings.public_url, "description": "Configured deployment (RXD_PUBLIC_URL)"},
        {"url": "http://localhost:8080", "description": "Local development"},
    ],
    contact={"name": "ReservoirX-D", "url": "https://github.com/"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s -> %s in %.1fms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@app.exception_handler(GraphConstructionError)
async def graph_error_handler(request: Request, exc: GraphConstructionError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "graph_construction_failed",
            "detail": str(exc),
            "hint": "Lower mean_degree, raise node_count, or reduce skew_cv.",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422, content={"error": "invalid_request", "detail": str(exc), "hint": None}
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness and store state")
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok", version=API_VERSION, store=request.app.state.store.stats()
    )


@app.get("/", tags=["meta"], summary="Service index")
def index() -> dict:
    return {
        "name": "ReservoirX-D",
        "version": API_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": [
            f"{API_PREFIX}/generate",
            f"{API_PREFIX}/diagnose",
            f"{API_PREFIX}/evaluate",
            f"{API_PREFIX}/optimize",
            f"{API_PREFIX}/compare",
            f"{API_PREFIX}/analyze",
            f"{API_PREFIX}/claims",
            f"{API_PREFIX}/tasks",
            f"{API_PREFIX}/version",
        ],
    }


for tag, paths in [
    ("generation", {"/generate"}),
    ("analysis", {"/diagnose", "/evaluate"}),
    ("experiments", {"/optimize", "/compare", "/analyze"}),
    ("meta", {"/claims", "/tasks", "/version"}),
]:
    for route in v1_router.routes:
        if getattr(route, "path", None) in paths:
            route.tags = [tag]

app.include_router(v1_router, prefix=API_PREFIX)
