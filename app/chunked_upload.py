"""Chunked upload sessions for large WAV files (Cloudflare-safe multi-request upload)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import shutil
import time
import uuid
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

logger = logging.getLogger("pyannote_service")

ALLOWED_WAV_CONTENT_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})

CHUNKED_UPLOAD_MAX_BYTES = max(
    1,
    int(os.environ.get("CHUNKED_UPLOAD_MAX_BYTES", str(1024 * 1024 * 1024)).strip()),
)
MAX_CHUNK_SIZE_BYTES = max(
    1,
    int(os.environ.get("MAX_CHUNK_SIZE_BYTES", str(80 * 1024 * 1024)).strip()),
)
RECOMMENDED_CHUNK_SIZE_BYTES = max(
    1,
    int(os.environ.get("RECOMMENDED_CHUNK_SIZE_BYTES", str(64 * 1024 * 1024)).strip()),
)
SINGLE_UPLOAD_MAX_BYTES = max(
    1,
    int(os.environ.get("SINGLE_UPLOAD_MAX_BYTES", str(100 * 1024 * 1024)).strip()),
)
CHUNK_SESSION_TTL_SECONDS = max(
    60,
    int(os.environ.get("CHUNK_SESSION_TTL_SECONDS", "7200").strip() or "7200"),
)
MAX_CONCURRENT_CHUNKED_SESSIONS = max(
    1,
    int(os.environ.get("MAX_CONCURRENT_CHUNKED_SESSIONS", "32").strip() or "32"),
)
CHUNKED_UPLOAD_DISK_QUOTA_BYTES = max(
    CHUNKED_UPLOAD_MAX_BYTES,
    int(
        os.environ.get(
            "CHUNKED_UPLOAD_DISK_QUOTA_BYTES",
            str(2 * CHUNKED_UPLOAD_MAX_BYTES),
        ).strip()
    ),
)
CHUNKED_UPLOAD_BASE_DIR = Path(
    os.environ.get("CHUNKED_UPLOAD_DIR", "").strip()
    or os.path.join(os.environ.get("TMPDIR", "/tmp"), "pyannote_chunked_uploads")
)
CHUNKED_JANITOR_INTERVAL_SECONDS = max(
    30.0,
    float(os.environ.get("CHUNKED_JANITOR_INTERVAL_SECONDS", "300").strip() or "300"),
)
# Upper bound for client-supplied num/min/max speaker hints (mirrors MAX_SPEAKERS
# in main.py). Out-of-range values are rejected with 422 before enqueue.
MAX_SPEAKERS = max(1, int(os.environ.get("MAX_SPEAKERS", "50").strip() or "50"))

CHUNKED_SESSIONS_CREATED = Counter(
    "pyannote_chunked_sessions_created_total",
    "Chunked upload sessions created",
)
CHUNKED_CHUNKS_RECEIVED = Counter(
    "pyannote_chunked_chunks_received_total",
    "Chunked upload chunk PUT requests that stored new bytes",
)
CHUNKED_SESSIONS_COMPLETED = Counter(
    "pyannote_chunked_sessions_completed_total",
    "Chunked upload sessions that reached diarisation enqueue after reassembly",
)
CHUNKED_SESSIONS_FAILED = Counter(
    "pyannote_chunked_sessions_failed_total",
    "Chunked upload session terminal failures before or during complete",
    labelnames=("reason",),
)
CHUNKED_REASSEMBLY_BYTES = Counter(
    "pyannote_chunked_reassembly_bytes_total",
    "Bytes written while reassembling chunked uploads",
)
CHUNKED_REASSEMBLY_SECONDS = Histogram(
    "pyannote_chunked_reassembly_duration_seconds",
    "Wall time to concatenate chunk files into the final WAV",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
CHUNKED_SESSION_TO_RESULT_SECONDS = Histogram(
    "pyannote_chunked_session_to_result_seconds",
    "Wall time from session create to SSE result event",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0),
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class SessionState(str, Enum):
    OPEN = "open"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class _StoredChunk:
    size: int
    sha256_hex: str


@dataclass
class UploadSession:
    upload_id: str
    filename: str
    content_type: str
    total_size_bytes: int
    chunk_size_bytes: int
    expected_chunk_count: int
    content_sha256: str | None
    created_at: float
    expires_at: float
    state: SessionState
    session_dir: Path
    chunks: dict[int, _StoredChunk] = field(default_factory=dict)

    @property
    def bytes_received(self) -> int:
        return sum(c.size for c in self.chunks.values())

    @property
    def chunks_received_count(self) -> int:
        return len(self.chunks)

    def is_complete(self) -> bool:
        return (
            self.chunks_received_count == self.expected_chunk_count
            and self.bytes_received == self.total_size_bytes
        )


class CreateSessionRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=512)
    content_type: str
    total_size_bytes: int = Field(..., gt=0)
    chunk_size_bytes: int | None = Field(default=None, gt=0)
    content_sha256: str | None = None


class CreateSessionResponse(BaseModel):
    upload_id: str
    chunk_size_bytes: int
    max_chunk_size_bytes: int
    max_total_size_bytes: int
    expires_at: str
    expected_chunk_count: int


class ChunkProgressResponse(BaseModel):
    upload_id: str
    chunks_received: int
    chunks_expected: int
    bytes_received: int
    complete: bool


class CapabilitiesResponse(BaseModel):
    single_upload_max_bytes: int
    chunked_upload: bool
    chunked_upload_max_bytes: int
    max_chunk_size_bytes: int
    recommended_chunk_size_bytes: int
    session_ttl_seconds: int


_sessions: dict[str, UploadSession] = {}
_sessions_lock = asyncio.Lock()
_disk_bytes_reserved = 0
_janitor_task: asyncio.Task[None] | None = None


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if not _SHA256_HEX_RE.match(lowered):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_content_sha256"},
        )
    return lowered


def _chunk_path(session: UploadSession, chunk_index: int) -> Path:
    return session.session_dir / "chunks" / str(chunk_index)


async def _remove_session_unlocked(upload_id: str) -> None:
    global _disk_bytes_reserved
    session = _sessions.pop(upload_id, None)
    if session is None:
        return
    _disk_bytes_reserved = max(0, _disk_bytes_reserved - session.total_size_bytes)
    try:
        shutil.rmtree(session.session_dir, ignore_errors=True)
    except OSError as exc:
        logger.warning(
            "chunked_session_cleanup_failed upload_id=%s err=%s",
            upload_id,
            exc,
        )


async def _purge_expired_sessions() -> int:
    now = time.time()
    removed = 0
    async with _sessions_lock:
        expired_ids = [
            sid
            for sid, sess in _sessions.items()
            if sess.expires_at <= now and sess.state == SessionState.OPEN
        ]
        for sid in expired_ids:
            await _remove_session_unlocked(sid)
            removed += 1
            CHUNKED_SESSIONS_FAILED.labels(reason="expired").inc()
    if removed:
        logger.info("chunked_janitor_removed expired_sessions=%d", removed)
    return removed


async def _janitor_loop() -> None:
    while True:
        try:
            await asyncio.sleep(CHUNKED_JANITOR_INTERVAL_SECONDS)
            await _purge_expired_sessions()
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            logger.warning("chunked_janitor_error err=%s", exc)


async def start_chunked_upload_background() -> None:
    global _janitor_task
    CHUNKED_UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)
    _janitor_task = asyncio.create_task(_janitor_loop(), name="chunked-upload-janitor")


async def stop_chunked_upload_background() -> None:
    global _janitor_task, _disk_bytes_reserved
    if _janitor_task is not None:
        _janitor_task.cancel()
        await asyncio.gather(_janitor_task, return_exceptions=True)
        _janitor_task = None
    async with _sessions_lock:
        for sid in list(_sessions.keys()):
            await _remove_session_unlocked(sid)
        _disk_bytes_reserved = 0


async def _get_open_session(upload_id: str) -> UploadSession:
    now = time.time()
    async with _sessions_lock:
        session = _sessions.get(upload_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        if session.expires_at <= now:
            await _remove_session_unlocked(upload_id)
            raise HTTPException(status_code=404, detail={"error": "session_expired"})
        if session.state == SessionState.ABORTED:
            raise HTTPException(status_code=404, detail={"error": "session_aborted"})
        if session.state in (SessionState.COMPLETING, SessionState.COMPLETED):
            raise HTTPException(status_code=409, detail={"error": "session_not_open"})
        return session


def _validate_wav_file(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
    except wave.Error as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_wav"},
        ) from exc
    if nframes == 0 and path.stat().st_size > 44:
        raise HTTPException(status_code=422, detail={"error": "invalid_wav"})
    if channels != 1 or rate != 16000 or sampwidth != 2:
        logger.warning(
            "chunked_wav_format_mismatch path=%s channels=%s rate=%s sampwidth=%s",
            path,
            channels,
            rate,
            sampwidth,
        )


async def _reassemble_session(session: UploadSession) -> Path:
    t0 = time.monotonic()
    out_path = session.session_dir / "assembled.wav"
    expected_offset = 0
    with out_path.open("wb") as out_f:
        for index in range(session.expected_chunk_count):
            chunk_file = _chunk_path(session, index)
            if not chunk_file.is_file():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "incomplete_upload",
                        "chunks_received": session.chunks_received_count,
                        "chunks_expected": session.expected_chunk_count,
                    },
                )
            data = chunk_file.read_bytes()
            chunk_meta = session.chunks.get(index)
            if chunk_meta is None or len(data) != chunk_meta.size:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "chunk_size_mismatch", "chunk_index": index},
                )
            out_f.write(data)
            expected_offset += len(data)
    if expected_offset != session.total_size_bytes:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "size_mismatch",
                "bytes_assembled": expected_offset,
                "expected_bytes": session.total_size_bytes,
            },
        )
    CHUNKED_REASSEMBLY_BYTES.inc(expected_offset)
    CHUNKED_REASSEMBLY_SECONDS.observe(time.monotonic() - t0)
    return out_path


def _verify_content_sha256(path: Path, expected_hex: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    if digest.hexdigest() != expected_hex:
        CHUNKED_SESSIONS_FAILED.labels(reason="sha256_mismatch").inc()
        raise HTTPException(status_code=422, detail={"error": "content_sha256_mismatch"})


def _expected_chunk_byte_length(session: UploadSession, chunk_index: int) -> int:
    start = chunk_index * session.chunk_size_bytes
    if start >= session.total_size_bytes:
        return 0
    end = min(start + session.chunk_size_bytes, session.total_size_bytes)
    return end - start


async def _read_request_body(request: Request, max_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_content_length"},
            ) from exc
        if declared > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={"error": "chunk_too_large", "max_chunk_size_bytes": max_bytes},
            )
    # Read the body incrementally and abort as soon as the cap is crossed.
    # `request.body()` buffers the *entire* stream into memory before any size
    # check, so a request using chunked transfer-encoding (no Content-Length,
    # bypassing the check above) could exhaust memory before the 413 fires.
    chunks: list[bytes] = []
    received = 0
    async for part in request.stream():
        received += len(part)
        if received > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={"error": "chunk_too_large", "max_chunk_size_bytes": max_bytes},
            )
        chunks.append(part)
    return b"".join(chunks)


def _validate_content_range(
    session: UploadSession,
    chunk_index: int,
    body_len: int,
    content_range: str | None,
) -> None:
    if content_range is None:
        return
    match = re.match(
        r"^bytes (\d+)-(\d+)/(\d+)$",
        content_range.strip(),
    )
    if not match:
        raise HTTPException(status_code=400, detail={"error": "invalid_content_range"})
    start = int(match.group(1))
    end = int(match.group(2))
    total = int(match.group(3))
    expected_start = chunk_index * session.chunk_size_bytes
    expected_len = _expected_chunk_byte_length(session, chunk_index)
    if total != session.total_size_bytes:
        raise HTTPException(status_code=400, detail={"error": "content_range_total_mismatch"})
    if start != expected_start or (end - start + 1) != body_len or body_len != expected_len:
        raise HTTPException(status_code=400, detail={"error": "content_range_mismatch"})


_enqueue_diarization: Callable[..., Awaitable[Any]] | None = None
_build_sse_stream: Callable[..., AsyncIterator[bytes]] | None = None
_diarization_params_type: type[Any] | None = None

router = APIRouter(prefix="/diarize", tags=["chunked-upload"])


def configure_chunked_upload(
    *,
    enqueue_diarization: Callable[..., Awaitable[Any]],
    build_sse_stream: Callable[..., AsyncIterator[bytes]],
    diarization_params_type: type[Any],
) -> None:
    global _enqueue_diarization, _build_sse_stream, _diarization_params_type
    _enqueue_diarization = enqueue_diarization
    _build_sse_stream = build_sse_stream
    _diarization_params_type = diarization_params_type


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def diarize_capabilities(
    request: Request,
    response: Response,
) -> CapabilitiesResponse:
    return CapabilitiesResponse(
            single_upload_max_bytes=SINGLE_UPLOAD_MAX_BYTES,
            chunked_upload=True,
            chunked_upload_max_bytes=CHUNKED_UPLOAD_MAX_BYTES,
            max_chunk_size_bytes=MAX_CHUNK_SIZE_BYTES,
            recommended_chunk_size_bytes=RECOMMENDED_CHUNK_SIZE_BYTES,
            session_ttl_seconds=CHUNK_SESSION_TTL_SECONDS,
        )


@router.post("/sessions", status_code=201, response_model=CreateSessionResponse)
async def create_upload_session(
    request: Request,
    response: Response,
    body: CreateSessionRequest,
) -> CreateSessionResponse:
    content_type = body.content_type.strip().lower()
    if content_type not in ALLOWED_WAV_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_media_type",
                "allowed": sorted(ALLOWED_WAV_CONTENT_TYPES),
            },
        )
    if body.total_size_bytes > CHUNKED_UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "total_size_too_large",
                "max_total_size_bytes": CHUNKED_UPLOAD_MAX_BYTES,
            },
        )
    content_sha256 = _normalize_sha256(body.content_sha256)
    chunk_size = body.chunk_size_bytes or RECOMMENDED_CHUNK_SIZE_BYTES
    chunk_size = min(chunk_size, MAX_CHUNK_SIZE_BYTES)
    expected_chunk_count = math.ceil(body.total_size_bytes / chunk_size)
    if expected_chunk_count < 1:
        raise HTTPException(status_code=400, detail={"error": "invalid_chunk_count"})

    now = time.time()
    expires_at = now + CHUNK_SESSION_TTL_SECONDS
    upload_id = str(uuid.uuid4())
    session_dir = CHUNKED_UPLOAD_BASE_DIR / upload_id

    global _disk_bytes_reserved
    async with _sessions_lock:
        open_count = sum(1 for s in _sessions.values() if s.state == SessionState.OPEN)
        if open_count >= MAX_CONCURRENT_CHUNKED_SESSIONS:
            CHUNKED_SESSIONS_FAILED.labels(reason="too_many_sessions").inc()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "too_many_open_sessions",
                    "max_concurrent_sessions": MAX_CONCURRENT_CHUNKED_SESSIONS,
                },
                headers={"Retry-After": "30"},
            )
        if _disk_bytes_reserved + body.total_size_bytes > CHUNKED_UPLOAD_DISK_QUOTA_BYTES:
            CHUNKED_SESSIONS_FAILED.labels(reason="disk_quota").inc()
            raise HTTPException(
                status_code=503,
                detail={"error": "chunked_disk_quota_exceeded"},
                headers={"Retry-After": "60"},
            )
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "chunks").mkdir()
        session = UploadSession(
            upload_id=upload_id,
            filename=body.filename,
            content_type=content_type,
            total_size_bytes=body.total_size_bytes,
            chunk_size_bytes=chunk_size,
            expected_chunk_count=expected_chunk_count,
            content_sha256=content_sha256,
            created_at=now,
            expires_at=expires_at,
            state=SessionState.OPEN,
            session_dir=session_dir,
        )
        _sessions[upload_id] = session
        _disk_bytes_reserved += body.total_size_bytes

    CHUNKED_SESSIONS_CREATED.inc()
    logger.info(
        "chunked_session_created upload_id=%s total_size_bytes=%d chunk_count=%d",
        upload_id,
        body.total_size_bytes,
        expected_chunk_count,
    )
    return CreateSessionResponse(
        upload_id=upload_id,
        chunk_size_bytes=chunk_size,
        max_chunk_size_bytes=MAX_CHUNK_SIZE_BYTES,
        max_total_size_bytes=CHUNKED_UPLOAD_MAX_BYTES,
        expires_at=_utc_iso(expires_at),
        expected_chunk_count=expected_chunk_count,
    )


@router.put(
    "/sessions/{upload_id}/chunks/{chunk_index}",
    response_model=None,
)
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    request: Request,
    response: Response,
    return_progress: Annotated[bool, Query(alias="progress")] = False,
) -> Response:
    session = await _get_open_session(upload_id)
    if chunk_index < 0 or chunk_index >= session.expected_chunk_count:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_chunk_index",
                "chunk_index": chunk_index,
                "expected_chunk_count": session.expected_chunk_count,
            },
        )

    body = await _read_request_body(request, MAX_CHUNK_SIZE_BYTES)
    expected_len = _expected_chunk_byte_length(session, chunk_index)
    if len(body) != expected_len:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unexpected_chunk_size",
                "chunk_index": chunk_index,
                "expected_bytes": expected_len,
                "received_bytes": len(body),
            },
        )
    _validate_content_range(
        session,
        chunk_index,
        len(body),
        request.headers.get("content-range"),
    )

    digest = hashlib.sha256(body).hexdigest()
    chunk_file = _chunk_path(session, chunk_index)

    async with _sessions_lock:
        locked = _sessions.get(upload_id)
        if locked is None or locked.state != SessionState.OPEN:
            raise HTTPException(status_code=409, detail={"error": "session_not_open"})
        session = locked
        existing = session.chunks.get(chunk_index)
        if existing is not None:
            if existing.sha256_hex == digest and existing.size == len(body):
                progress = ChunkProgressResponse(
                    upload_id=upload_id,
                    chunks_received=session.chunks_received_count,
                    chunks_expected=session.expected_chunk_count,
                    bytes_received=session.bytes_received,
                    complete=session.is_complete(),
                )
                if return_progress:
                    return JSONResponse(content=progress.model_dump())
                return Response(status_code=204)
            raise HTTPException(status_code=409, detail={"error": "chunk_conflict"})

        chunk_file.write_bytes(body)
        session.chunks[chunk_index] = _StoredChunk(size=len(body), sha256_hex=digest)

    CHUNKED_CHUNKS_RECEIVED.inc()
    logger.info(
        "chunked_chunk_received upload_id=%s chunk_index=%d bytes=%d",
        upload_id,
        chunk_index,
        len(body),
    )

    progress = ChunkProgressResponse(
        upload_id=upload_id,
        chunks_received=session.chunks_received_count,
        chunks_expected=session.expected_chunk_count,
        bytes_received=session.bytes_received,
        complete=session.is_complete(),
    )
    if return_progress:
        return JSONResponse(content=progress.model_dump())
    return Response(status_code=204)


@router.post("/sessions/{upload_id}/complete")
async def complete_upload_session(
    upload_id: str,
    request: Request,
    response: Response,
    num_speakers: Annotated[int | None, Query(ge=1, le=MAX_SPEAKERS)] = None,
    min_speakers: Annotated[int | None, Query(ge=1, le=MAX_SPEAKERS)] = None,
    max_speakers: Annotated[int | None, Query(ge=1, le=MAX_SPEAKERS)] = None,
    exclusive: Annotated[bool, Query()] = False,
    return_embeddings: Annotated[bool, Query()] = False,
) -> StreamingResponse:
    if (
        _enqueue_diarization is None
        or _build_sse_stream is None
        or _diarization_params_type is None
    ):
        raise HTTPException(status_code=503, detail={"error": "pipeline_not_loaded"})

    session = await _get_open_session(upload_id)
    if not session.is_complete():
        CHUNKED_SESSIONS_FAILED.labels(reason="incomplete").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "incomplete_upload",
                "chunks_received": session.chunks_received_count,
                "chunks_expected": session.expected_chunk_count,
            },
        )

    async with _sessions_lock:
        locked = _sessions.get(upload_id)
        if locked is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        if locked.state != SessionState.OPEN:
            raise HTTPException(status_code=409, detail={"error": "session_not_open"})
        locked.state = SessionState.COMPLETING
        session = locked

    session_created_at = session.created_at
    try:
        assembled_path = await _reassemble_session(session)
        _validate_wav_file(assembled_path)
        if session.content_sha256 is not None:
            _verify_content_sha256(assembled_path, session.content_sha256)

        params = _diarization_params_type(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=exclusive,
            return_embeddings=return_embeddings,
        )
        job = await _enqueue_diarization(
            assembled_path,
            params,
            queue_full_status=429,
        )
    except HTTPException as exc:
        reason = "complete_failed"
        if isinstance(exc.detail, dict):
            reason = str(exc.detail.get("error", reason))
        if exc.status_code in {400, 422}:
            CHUNKED_SESSIONS_FAILED.labels(reason=reason).inc()
        async with _sessions_lock:
            if upload_id in _sessions:
                _sessions[upload_id].state = SessionState.OPEN
        raise

    CHUNKED_SESSIONS_COMPLETED.inc()
    logger.info(
        "chunked_session_complete upload_id=%s total_size_bytes=%d chunk_count=%d",
        upload_id,
        session.total_size_bytes,
        session.expected_chunk_count,
    )

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async for frame in _build_sse_stream(request, job):
                yield frame
                if frame.startswith(b"event: result\n"):
                    CHUNKED_SESSION_TO_RESULT_SECONDS.observe(
                        time.monotonic() - session_created_at
                    )
        finally:
            async with _sessions_lock:
                await _remove_session_unlocked(upload_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/sessions/{upload_id}", status_code=204)
async def abort_upload_session(
    upload_id: str,
    request: Request,
    response: Response,
) -> Response:
    async with _sessions_lock:
        session = _sessions.get(upload_id)
        if session is None:
            return Response(status_code=404)
        session.state = SessionState.ABORTED
        await _remove_session_unlocked(upload_id)
    logger.info("chunked_session_aborted upload_id=%s", upload_id)
    return Response(status_code=204)
