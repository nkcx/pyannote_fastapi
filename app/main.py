import asyncio
import hashlib
import json
import logging
import os
import secrets
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import torch
import torchaudio
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

import chunked_upload

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("pyannote_service")


REQUESTS_TOTAL = Counter(
    "pyannote_requests_total",
    "HTTP requests processed",
    labelnames=("endpoint", "status"),
)
DIARIZATION_SECONDS = Histogram(
    "pyannote_diarization_duration_seconds",
    "Wall time spent in the pyannote inference call (excluding upload and queue wait)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
)
AUDIO_DURATION_SECONDS = Histogram(
    "pyannote_audio_duration_seconds",
    "Input audio duration in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
)
REALTIME_FACTOR = Histogram(
    "pyannote_realtime_factor",
    "Inference wall time divided by audio duration (lower is faster; <1 = faster than realtime)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
SSE_RESULTS_TOTAL = Counter(
    "pyannote_sse_results_total",
    "Terminal SSE events emitted by /diarize",
    labelnames=("outcome",),
)
QUEUE_DEPTH = Gauge(
    "pyannote_queue_depth",
    "Jobs currently waiting in the diarization queue (excludes jobs being processed)",
)
ACTIVE_REQUESTS = Gauge("pyannote_active_requests", "Jobs currently being processed by a worker")
LEAKED_INFERENCE_THREADS = Gauge(
    "pyannote_leaked_inference_threads",
    "Inference threads still running after their request timed out (cannot be killed; "
    "decrements when the underlying pyannote call finally returns). Sustained non-zero "
    "values indicate the model is getting stuck and the pod should be restarted.",
)
MODEL_LOADED = Gauge("pyannote_model_loaded", "Whether the diarization pipeline is loaded (1=yes)")
PYANNOTE_BUILD = Info(
    "pyannote",
    "Static build/runtime information for the diarization service",
)

SSE_RESULTS_TOTAL.labels(outcome="success")
SSE_RESULTS_TOTAL.labels(outcome="error")

_BEARER_SCHEME = HTTPBearer(bearerFormat="API key", auto_error=False)


def _read_allowed_api_keys() -> frozenset[str]:
    raw = os.environ.get("API_KEYS")
    if raw is None or not raw.strip():
        logger.error(
            "Refusing to start: API_KEYS is unset or empty; set a comma-separated list of keys."
        )
        sys.exit(1)
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


ALLOWED_API_KEYS: frozenset[str] = _read_allowed_api_keys()


def _is_valid_api_key(token: str) -> bool:
    """Constant-time check of a presented token against every allowed key.

    Uses ``secrets.compare_digest`` and deliberately compares against all keys
    without short-circuiting, so neither the match/no-match outcome nor which
    key matched is leaked through response timing. Comparison is done on UTF-8
    bytes: ``compare_digest`` raises on non-ASCII ``str`` inputs, and Bearer
    tokens arrive latin-1 decoded, so a crafted token could otherwise turn a
    clean 401 into a 500.
    """
    token_bytes = token.encode("utf-8")
    valid = False
    for key in ALLOWED_API_KEYS:
        if secrets.compare_digest(token_bytes, key.encode("utf-8")):
            valid = True
    return valid

MODEL_PATH_RAW = os.environ.get("MODEL_PATH", "").strip()
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "pyannote/speaker-diarization-community-1",
).strip()

MODEL_CARD_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
MODEL_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

DIARIZE_WORKERS = max(1, int(os.environ.get("DIARIZE_WORKERS", "1").strip() or "1"))
SSE_HEARTBEAT_SECONDS = max(
    0.5, float(os.environ.get("SSE_HEARTBEAT_SECONDS", "5").strip() or "5")
)
MAX_QUEUE_DEPTH = max(1, int(os.environ.get("MAX_QUEUE_DEPTH", "64").strip() or "64"))

# Hard limits enforced inside the FastAPI process. Configure stricter values at
# the reverse proxy / Cloudflare layer when possible, but never trust them.
MAX_UPLOAD_BYTES = max(
    1, int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)).strip())
)
MAX_AUDIO_SECONDS = max(
    1.0, float(os.environ.get("MAX_AUDIO_SECONDS", "43200").strip() or "43200")
)
# Per-request inference timeout. Note: the underlying thread is not killable,
# but the queue slot and the client request are released, so a stuck job stops
# blocking the API surface. Set to 0 to disable.
INFERENCE_TIMEOUT_SECONDS = max(
    0.0, float(os.environ.get("INFERENCE_TIMEOUT_SECONDS", "7200").strip() or "7200")
)
# Constant-ish delay added to 401 responses to slow credential stuffing.
AUTH_FAIL_DELAY_SECONDS = max(
    0.0, float(os.environ.get("AUTH_FAIL_DELAY_SECONDS", "0.5").strip() or "0.5")
)

# Rate limit configuration. Per-key limits apply when a valid Bearer token shape
# is presented; per-IP limits apply unconditionally. Format follows
# slowapi/limits syntax, e.g. "10/minute", "100/hour", "5/second".
RATE_LIMIT_DIARIZE = os.environ.get("RATE_LIMIT_DIARIZE", "10/minute").strip()
# Per-IP cap for /diarize, evaluated independently of the per-key cap above.
# This is the dominant defence against unauthenticated abuse and against
# attackers rotating Bearer tokens to dodge per-key buckets.
RATE_LIMIT_DIARIZE_IP = os.environ.get("RATE_LIMIT_DIARIZE_IP", "20/minute").strip()
RATE_LIMIT_LIVE = os.environ.get("RATE_LIMIT_LIVE", "120/minute").strip()
RATE_LIMIT_HEALTH = os.environ.get("RATE_LIMIT_HEALTH", "120/minute").strip()
RATE_LIMIT_METRICS = os.environ.get("RATE_LIMIT_METRICS", "60/minute").strip()
RATE_LIMIT_STORAGE_URI = os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://").strip()


def _client_ip(request: Request) -> str:
    """Return the originating client IP, trusting Cloudflare / reverse proxy
    headers in priority order. Falls back to the direct socket peer."""
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        # First hop is the originating client per RFC 7239 conventions.
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    if request.client is not None:
        return request.client.host
    return "unknown"


def _key_prefix(token: str) -> str:
    """Short, non-reversible-ish identifier for an API key (audit logs only)."""
    return token[:4] + "***" if token else "-"


def _rate_limit_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        # Only bucket by token when it is a *valid* key. Keying on arbitrary
        # presented tokens lets an attacker mint unlimited distinct buckets in
        # the limiter store (unbounded memory with memory://; secrets written
        # to redis with a shared backend). Invalid tokens fall through to the
        # per-IP bucket. The token is hashed so raw keys are never stored.
        if token and token in ALLOWED_API_KEYS:
            return f"key:{hashlib.sha256(token.encode()).hexdigest()}"
    return f"ip:{_client_ip(request)}"


def _rate_limit_ip_key(request: Request) -> str:
    return f"ip:{_client_ip(request)}"


limiter = Limiter(
    key_func=_rate_limit_key,
    storage_uri=RATE_LIMIT_STORAGE_URI,
    headers_enabled=True,
)


def _audit(event: str, request: Request, **fields: Any) -> None:
    """Emit a structured audit log line for security-relevant events."""
    auth = request.headers.get("authorization", "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    parts = [
        f"audit_event={event}",
        f"ip={_client_ip(request)}",
        f"path={request.url.path}",
        f"method={request.method}",
        f"key={_key_prefix(token)}",
        f"ua={request.headers.get('user-agent', '-')!r}",
        f"cf_ray={request.headers.get('cf-ray', '-')}",
        f"cf_country={request.headers.get('cf-ipcountry', '-')}",
    ]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    logger.warning(" ".join(parts))

# Opt-in telemetry. pyannote.audio ships an anonymous usage-metrics hook that is
# enabled by default upstream; we disable it unless the operator explicitly
# opts in via PYANNOTE_TELEMETRY=1.
PYANNOTE_TELEMETRY_ENABLED = os.environ.get("PYANNOTE_TELEMETRY", "").strip() in {
    "1",
    "true",
    "yes",
}


def _configure_pyannote_telemetry() -> None:
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics
    except ImportError:
        logger.debug("pyannote.audio.telemetry not available; skipping telemetry configuration.")
        return
    set_telemetry_metrics(PYANNOTE_TELEMETRY_ENABLED)
    logger.info(
        "pyannote_telemetry enabled=%s (set PYANNOTE_TELEMETRY=1 to opt in)",
        PYANNOTE_TELEMETRY_ENABLED,
    )


class SegmentModel(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    num_speakers: int = Field(..., ge=0)
    speakers: list[str]
    segments: list[SegmentModel]
    processing_time_seconds: float
    model: str = "pyannote/speaker-diarization-community-1"
    pyannote_version: str


_pipeline: Any = None
_pyannote_version: str = "unknown"


def _load_pyannote_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _is_testing() -> bool:
    return os.environ.get("PYANNOTE_TESTING", "").strip() in {"1", "true", "yes"}


def _read_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val is not None and val.strip():
            return val.strip()
    return None


def _local_pipeline_dir() -> Path | None:
    if not MODEL_PATH_RAW:
        return None
    path = Path(MODEL_PATH_RAW).expanduser()
    if not path.is_dir():
        logger.error(
            "MODEL_PATH=%s must be an existing directory containing a full offline "
            "pyannote pipeline checkout (see %s, Offline use).",
            MODEL_PATH_RAW,
            MODEL_CARD_URL,
        )
        sys.exit(1)
    if not (path / "config.yaml").exists():
        logger.error(
            "MODEL_PATH=%s must include config.yaml from the upstream pipeline snapshot.",
            MODEL_PATH_RAW,
        )
        sys.exit(1)
    return path


class _DryRunPipeline:
    """Minimal stand-in used only when PYANNOTE_TESTING=1 (see tests)."""

    def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, Annotation]:
        ann = Annotation()
        ann[Segment(0.0, 0.5)] = "SPEAKER_00"
        return {"speaker_diarization": ann}


def _load_pipeline() -> Any:
    if _is_testing():
        logger.warning("PYANNOTE_TESTING is enabled; using dry-run pipeline.")
        return _DryRunPipeline()
    _configure_pyannote_telemetry()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    local_dir = _local_pipeline_dir()
    if local_dir is not None:
        logger.info(
            "loading_pipeline source=local path=%s device=%s",
            local_dir,
            device,
        )
        pipe: Pipeline = Pipeline.from_pretrained(str(local_dir))
        pipe.to(device)
        return pipe

    token = _read_hf_token()
    if token is None:
        logger.error(
            "Refusing to start: model weights are not bundled. Either mount an offline "
            "checkout at MODEL_PATH (after you personally accepted HF access terms and "
            "cloned per %s), or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN for your runtime.",
            MODEL_CARD_URL,
        )
        sys.exit(1)
    logger.info(
        "loading_pipeline source=huggingface_hub model_id=%s device=%s",
        MODEL_ID,
        device,
    )
    pipe = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipe.to(device)
    return pipe


def _extract_diarization_output(raw: Any, exclusive: bool) -> Annotation:
    if isinstance(raw, Annotation):
        return raw
    key = "exclusive_speaker_diarization" if exclusive else "speaker_diarization"
    if isinstance(raw, dict):
        if key not in raw:
            raise KeyError(key)
        value = raw[key]
        if not isinstance(value, Annotation):
            raise TypeError(f"{key} is not an Annotation")
        return value
    if hasattr(raw, key):
        value = getattr(raw, key)
        if not isinstance(value, Annotation):
            raise TypeError(f"{key} is not an Annotation")
        return value
    raise TypeError("Unsupported diarization output type")


def _annotation_to_segments(diarization: Annotation) -> tuple[list[SegmentModel], list[str]]:
    segments: list[SegmentModel] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SegmentModel(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        )
    speakers_sorted = sorted({s.speaker for s in segments})
    return segments, speakers_sorted


class _DiarizationParams(BaseModel):
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    exclusive: bool = False


class _Job:
    """One queued diarization request and its private event stream."""

    def __init__(self, tmp_path: Path, params: _DiarizationParams) -> None:
        self.id: str = uuid.uuid4().hex
        self.tmp_path: Path = tmp_path
        self.params: _DiarizationParams = params
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()


_JOB_QUEUE: asyncio.Queue[_Job] | None = None
_WORKER_TASKS: list[asyncio.Task[None]] = []


def _run_diarization_sync(tmp_path: Path, params: _DiarizationParams) -> DiarizeResponse:
    """Blocking diarization pipeline call; intended for asyncio.to_thread."""
    t0 = time.monotonic()
    waveform, sample_rate = torchaudio.load(str(tmp_path))
    if waveform.dim() != 2:
        raise HTTPException(status_code=400, detail={"error": "invalid_audio"})
    audio_duration = float(waveform.shape[1]) / float(sample_rate)
    AUDIO_DURATION_SECONDS.observe(audio_duration)
    if audio_duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "audio_too_long",
                "duration_seconds": audio_duration,
                "max_seconds": MAX_AUDIO_SECONDS,
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    infer_kwargs: dict[str, Any] = {}
    if params.num_speakers is not None:
        infer_kwargs["num_speakers"] = params.num_speakers
    if params.min_speakers is not None:
        infer_kwargs["min_speakers"] = params.min_speakers
    if params.max_speakers is not None:
        infer_kwargs["max_speakers"] = params.max_speakers

    t_infer0 = time.monotonic()
    raw_out = _pipeline({"waveform": waveform, "sample_rate": sample_rate}, **infer_kwargs)
    inference_seconds = time.monotonic() - t_infer0
    DIARIZATION_SECONDS.observe(inference_seconds)
    if audio_duration > 0:
        REALTIME_FACTOR.observe(inference_seconds / audio_duration)

    try:
        diarization = _extract_diarization_output(raw_out, exclusive=params.exclusive)
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "diarization_output_parse_failed"},
        ) from exc

    segments, speakers = _annotation_to_segments(diarization)
    return DiarizeResponse(
        duration_seconds=audio_duration,
        num_speakers=len(speakers),
        speakers=speakers,
        segments=segments,
        processing_time_seconds=time.monotonic() - t0,
        pyannote_version=_pyannote_version,
    )


async def _worker(worker_id: int) -> None:
    """Pull jobs off the shared queue and run the pyannote pipeline in a thread."""
    assert _JOB_QUEUE is not None
    while True:
        job = await _JOB_QUEUE.get()
        QUEUE_DEPTH.dec()
        ACTIVE_REQUESTS.inc()
        try:
            await job.events.put({"type": "status", "phase": "running", "worker": worker_id})
            logger.info("job_started job=%s worker=%s", job.id, worker_id)
            loop = asyncio.get_running_loop()
            infer_future = loop.run_in_executor(
                None, _run_diarization_sync, job.tmp_path, job.params
            )
            try:
                if INFERENCE_TIMEOUT_SECONDS > 0:
                    # asyncio.shield keeps a clean reference to the underlying
                    # executor future so we can attach a done-callback after a
                    # timeout. Cancelling the future would not stop the OS
                    # thread anyway -- Python cannot interrupt a running
                    # thread -- so we let it run to completion and just track
                    # it via the LEAKED_INFERENCE_THREADS gauge.
                    result = await asyncio.wait_for(
                        asyncio.shield(infer_future), timeout=INFERENCE_TIMEOUT_SECONDS
                    )
                else:
                    result = await infer_future
            except TimeoutError:
                LEAKED_INFERENCE_THREADS.inc()
                infer_future.add_done_callback(
                    lambda _f: LEAKED_INFERENCE_THREADS.dec()
                )
                logger.warning(
                    "diarization_timeout job=%s timeout_seconds=%s "
                    "(thread continues until pipeline returns; see "
                    "pyannote_leaked_inference_threads gauge)",
                    job.id,
                    INFERENCE_TIMEOUT_SECONDS,
                )
                await job.events.put(
                    {
                        "type": "error",
                        "status": 504,
                        "detail": {
                            "error": "diarization_timeout",
                            "timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
                        },
                    }
                )
            except HTTPException as exc:
                await job.events.put(
                    {"type": "error", "status": exc.status_code, "detail": exc.detail}
                )
            except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
                logger.exception("diarization_failed job=%s err=%s", job.id, exc)
                await job.events.put(
                    {"type": "error", "status": 500, "detail": {"error": "diarization_failed"}}
                )
            else:
                await job.events.put({"type": "result", "data": result.model_dump()})
                logger.info("job_finished job=%s worker=%s", job.id, worker_id)
        finally:
            ACTIVE_REQUESTS.dec()
            try:
                job.tmp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete temp upload %s: %s", job.tmp_path, exc)
            _JOB_QUEUE.task_done()


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


async def _enqueue_diarization_job(
    tmp_path: Path,
    params: _DiarizationParams,
    *,
    queue_full_status: int = 503,
) -> _Job:
    if _pipeline is None or _JOB_QUEUE is None:
        raise HTTPException(status_code=503, detail={"error": "pipeline_not_loaded"})

    if _JOB_QUEUE.full():
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as unlink_exc:
            logger.warning("Could not delete temp upload %s: %s", tmp_path, unlink_exc)
        logger.warning(
            "queue_full reject depth=%d max=%d",
            _JOB_QUEUE.qsize(),
            MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=queue_full_status,
            detail={"error": "queue_full", "max_queue_depth": MAX_QUEUE_DEPTH},
            headers={"Retry-After": "5"},
        )

    job = _Job(tmp_path=tmp_path, params=params)
    try:
        _JOB_QUEUE.put_nowait(job)
        QUEUE_DEPTH.inc()
    except asyncio.QueueFull as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as unlink_exc:
            logger.warning("Could not delete temp upload %s: %s", tmp_path, unlink_exc)
        logger.warning(
            "queue_full reject_after_prepare depth=%d max=%d",
            _JOB_QUEUE.qsize(),
            MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=queue_full_status,
            detail={"error": "queue_full", "max_queue_depth": MAX_QUEUE_DEPTH},
            headers={"Retry-After": "5"},
        ) from exc
    logger.info(
        "job_queued job=%s queue_depth=%d max=%d",
        job.id,
        _JOB_QUEUE.qsize(),
        MAX_QUEUE_DEPTH,
    )
    return job


async def _iter_diarization_sse(request: Request, job: _Job) -> AsyncIterator[bytes]:
    t0 = time.monotonic()
    phase = "queued"
    yield _sse_frame("status", {"job_id": job.id, "phase": phase})
    while True:
        if await request.is_disconnected():
            logger.info("client_disconnected job=%s phase=%s", job.id, phase)
            return
        try:
            event = await asyncio.wait_for(job.events.get(), timeout=SSE_HEARTBEAT_SECONDS)
        except TimeoutError:
            yield _sse_frame(
                "heartbeat",
                {
                    "job_id": job.id,
                    "phase": phase,
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                },
            )
            continue

        kind = event["type"]
        if kind == "status":
            phase = event["phase"]
            yield _sse_frame(
                "status",
                {"job_id": job.id, "phase": phase, "worker": event.get("worker")},
            )
        elif kind == "result":
            SSE_RESULTS_TOTAL.labels(outcome="success").inc()
            yield _sse_frame("result", {"job_id": job.id, **event["data"]})
            return
        elif kind == "error":
            SSE_RESULTS_TOTAL.labels(outcome="error").inc()
            yield _sse_frame(
                "error",
                {"job_id": job.id, "status": event["status"], "detail": event["detail"]},
            )
            return


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline, _pyannote_version, _JOB_QUEUE
    app.state.ready = False
    MODEL_LOADED.set(0)
    _pyannote_version = _load_pyannote_version()
    logger.info(
        "startup pyannote_version=%s torch_cuda_available=%s model_path=%s model_id=%s hf_home=%s",
        _pyannote_version,
        torch.cuda.is_available(),
        MODEL_PATH_RAW or "(none)",
        MODEL_ID,
        os.environ.get("HF_HOME", ""),
    )
    PYANNOTE_BUILD.info(
        {
            "version": _pyannote_version,
            "model_id": MODEL_ID,
            "torch_version": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()).lower(),
        }
    )
    _pipeline = _load_pipeline()
    _JOB_QUEUE = asyncio.Queue(maxsize=MAX_QUEUE_DEPTH)
    for i in range(DIARIZE_WORKERS):
        _WORKER_TASKS.append(asyncio.create_task(_worker(i), name=f"diarize-worker-{i}"))
    app.state.ready = True
    MODEL_LOADED.set(1)
    logger.info(
        'Attribution: "%s" by pyannote / pyannoteAI; license CC-BY-4.0 (%s); model card %s.',
        MODEL_ID,
        MODEL_LICENSE_URL,
        MODEL_CARD_URL,
    )
    logger.info(
        "Pipeline loaded; ready to serve. workers=%d max_queue_depth=%d heartbeat=%.1fs",
        DIARIZE_WORKERS,
        MAX_QUEUE_DEPTH,
        SSE_HEARTBEAT_SECONDS,
    )
    await chunked_upload.start_chunked_upload_background()
    yield
    await chunked_upload.stop_chunked_upload_background()
    MODEL_LOADED.set(0)
    app.state.ready = False
    for task in _WORKER_TASKS:
        task.cancel()
    await asyncio.gather(*_WORKER_TASKS, return_exceptions=True)
    _WORKER_TASKS.clear()
    _JOB_QUEUE = None
    _pipeline = None


app = FastAPI(title="pyannote speaker diarization (community-1)", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    _audit("rate_limited", request, limit=str(exc.detail))
    response = JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": str(exc.detail)},
    )
    response.headers["Retry-After"] = "1"
    return response


async def _require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER_SCHEME)],
) -> None:
    if credentials is None:
        reason = "missing_token"
    elif not _is_valid_api_key(credentials.credentials):
        reason = "invalid_token"
    else:
        return
    _audit("auth_failed", request, reason=reason)
    if AUTH_FAIL_DELAY_SECONDS > 0:
        await asyncio.sleep(AUTH_FAIL_DELAY_SECONDS)
    raise HTTPException(status_code=401, detail={"error": "unauthorized"})


@app.middleware("http")
async def count_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response = await call_next(request)
    status = str(response.status_code)
    REQUESTS_TOTAL.labels(endpoint=request.url.path, status=status).inc()
    return response


@app.get("/live")
@limiter.limit(RATE_LIMIT_LIVE)
async def live(request: Request, response: Response) -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
@limiter.limit(RATE_LIMIT_HEALTH)
async def health(request: Request, response: Response) -> JSONResponse:
    ready = bool(getattr(request.app.state, "ready", False))
    if not ready:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.get("/metrics")
@limiter.limit(RATE_LIMIT_METRICS)
async def metrics(request: Request, response: Response) -> Response:
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post("/diarize")
@limiter.limit(RATE_LIMIT_DIARIZE_IP, key_func=_rate_limit_ip_key)
@limiter.limit(RATE_LIMIT_DIARIZE)
async def diarize(
    _auth: Annotated[None, Depends(_require_api_key)],
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(..., description="Audio file")],
    num_speakers: Annotated[int | None, Query()] = None,
    min_speakers: Annotated[int | None, Query()] = None,
    max_speakers: Annotated[int | None, Query()] = None,
    exclusive: Annotated[bool, Query()] = False,
) -> StreamingResponse:
    if _pipeline is None or _JOB_QUEUE is None:
        raise HTTPException(status_code=503, detail={"error": "pipeline_not_loaded"})

    if _JOB_QUEUE.full():
        logger.warning(
            "queue_full reject endpoint=/diarize depth=%d max=%d",
            _JOB_QUEUE.qsize(),
            MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=503,
            detail={"error": "queue_full", "max_queue_depth": MAX_QUEUE_DEPTH},
            headers={"Retry-After": "5"},
        )

    suffix = Path(file.filename or "audio").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed manually below
        prefix="pyannote_upload_", suffix=suffix, delete=False
    )
    tmp_path = Path(tmp.name)
    bytes_written = 0
    upload_too_large = False
    try:
        chunk_size = 1024 * 1024
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                upload_too_large = True
                break
            tmp.write(chunk)
    finally:
        tmp.close()

    if upload_too_large:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as unlink_exc:
            logger.warning("Could not delete temp upload %s: %s", tmp_path, unlink_exc)
        _audit(
            "upload_too_large",
            request,
            received_bytes=bytes_written,
            limit_bytes=MAX_UPLOAD_BYTES,
        )
        raise HTTPException(
            status_code=413,
            detail={"error": "upload_too_large", "max_bytes": MAX_UPLOAD_BYTES},
        )

    params = _DiarizationParams(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=exclusive,
    )
    job = await _enqueue_diarization_job(tmp_path, params)

    return StreamingResponse(
        _iter_diarization_sse(request, job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


chunked_upload.configure_chunked_upload(
    enqueue_diarization=_enqueue_diarization_job,
    build_sse_stream=_iter_diarization_sse,
    diarization_params_type=_DiarizationParams,
)
app.include_router(
    chunked_upload.router,
    dependencies=[Depends(_require_api_key)],
)
