from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import time
import wave
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import chunked_upload
import main


def _streaming_request(
    parts: list[bytes], headers: list[tuple[bytes, bytes]] | None = None
) -> Request:
    """Build a Request whose body arrives as multiple ASGI stream messages."""
    queue = list(parts)

    async def receive() -> dict[str, Any]:
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "PUT",
        "path": "/",
        "headers": headers or [],
    }
    return Request(scope, receive)


def test_read_request_body_caps_streamed_body() -> None:
    """A body streamed without Content-Length is rejected once it crosses the
    cap, instead of being buffered whole (the OOM vector)."""
    req = _streaming_request([b"x" * 100, b"x" * 100, b"x" * 100])
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chunked_upload._read_request_body(req, max_bytes=150))
    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"] == "chunk_too_large"


def test_read_request_body_returns_body_under_cap() -> None:
    req = _streaming_request([b"ab", b"cd", b"ef"])
    body = asyncio.run(chunked_upload._read_request_body(req, max_bytes=100))
    assert body == b"abcdef"


def _parse_sse(stream_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for raw_block in stream_text.split("\n\n"):
        block = raw_block.strip()
        if not block:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def _silent_wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    n = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-integration-key"}


def _create_session(
    client: TestClient,
    wav_bytes: bytes,
    *,
    chunk_size: int | None = None,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "filename": "recording.wav",
        "content_type": "audio/wav",
        "total_size_bytes": len(wav_bytes),
    }
    if chunk_size is not None:
        body["chunk_size_bytes"] = chunk_size
    if content_sha256 is not None:
        body["content_sha256"] = content_sha256
    r = client.post("/diarize/sessions", headers=_auth_headers(), json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _upload_all_chunks(
    client: TestClient,
    session: dict[str, Any],
    wav_bytes: bytes,
) -> None:
    chunk_size = session["chunk_size_bytes"]
    total = len(wav_bytes)
    expected = session["expected_chunk_count"]
    for index in range(expected):
        start = index * chunk_size
        end = min(start + chunk_size, total)
        chunk = wav_bytes[start:end]
        r = client.put(
            f"/diarize/sessions/{session['upload_id']}/chunks/{index}",
            headers={**_auth_headers(), "Content-Type": "application/octet-stream"},
            content=chunk,
        )
        assert r.status_code == 204, r.text


def _complete_session(client: TestClient, upload_id: str) -> Any:
    return client.post(
        f"/diarize/sessions/{upload_id}/complete",
        headers={**_auth_headers(), "Accept": "text/event-stream"},
    )


def test_capabilities() -> None:
    with TestClient(main.app) as client:
        r = client.get("/diarize/capabilities", headers=_auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["chunked_upload"] is True
    assert body["max_chunk_size_bytes"] == chunked_upload.MAX_CHUNK_SIZE_BYTES


def test_chunked_upload_requires_auth() -> None:
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize/sessions",
            json={
                "filename": "a.wav",
                "content_type": "audio/wav",
                "total_size_bytes": 100,
            },
        )
    assert r.status_code == 401


def test_chunked_happy_path() -> None:
    wav = _silent_wav_bytes(0.5)
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=256)
        assert session["expected_chunk_count"] == math.ceil(len(wav) / 256)
        _upload_all_chunks(client, session, wav)
        r = _complete_session(client, session["upload_id"])
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert "result" in [name for name, _ in events]
    result = next(data for name, data in events if name == "result")
    assert "segments" in result


def test_chunked_idempotent_chunk_put() -> None:
    wav = _silent_wav_bytes()
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=len(wav))
        chunk = wav
        url = f"/diarize/sessions/{session['upload_id']}/chunks/0"
        headers = {**_auth_headers(), "Content-Type": "application/octet-stream"}
        assert client.put(url, headers=headers, content=chunk).status_code == 204
        assert client.put(url, headers=headers, content=chunk).status_code == 204
        r = _complete_session(client, session["upload_id"])
    assert r.status_code == 200
    assert "result" in [name for name, _ in _parse_sse(r.text)]


def test_chunked_conflict_on_different_bytes() -> None:
    wav = _silent_wav_bytes()
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=len(wav))
        url = f"/diarize/sessions/{session['upload_id']}/chunks/0"
        headers = {**_auth_headers(), "Content-Type": "application/octet-stream"}
        assert client.put(url, headers=headers, content=wav).status_code == 204
        r = client.put(url, headers=headers, content=wav[:-1] + b"\x01")
    assert r.status_code == 409


def test_chunked_complete_before_all_chunks() -> None:
    wav = _silent_wav_bytes()
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=max(1, len(wav) // 2))
        r = _complete_session(client, session["upload_id"])
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "incomplete_upload"


def test_chunked_invalid_wav_after_reassembly() -> None:
    garbage = b"not a wav file at all" * 10
    with TestClient(main.app) as client:
        session = _create_session(client, garbage, chunk_size=len(garbage))
        _upload_all_chunks(client, session, garbage)
        r = _complete_session(client, session["upload_id"])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_wav"


def test_chunked_sha256_mismatch() -> None:
    wav = _silent_wav_bytes()
    wrong_hash = "0" * 64
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=len(wav), content_sha256=wrong_hash)
        _upload_all_chunks(client, session, wav)
        r = _complete_session(client, session["upload_id"])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "content_sha256_mismatch"


def test_chunked_sha256_success() -> None:
    wav = _silent_wav_bytes()
    digest = hashlib.sha256(wav).hexdigest()
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=len(wav), content_sha256=digest)
        _upload_all_chunks(client, session, wav)
        r = _complete_session(client, session["upload_id"])
    assert r.status_code == 200
    assert "result" in [name for name, _ in _parse_sse(r.text)]


def test_chunked_session_expired() -> None:
    wav = _silent_wav_bytes()
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=len(wav))
        upload_id = session["upload_id"]
        chunked_upload._sessions[upload_id].expires_at = time.time() - 1
        r = client.put(
            f"/diarize/sessions/{upload_id}/chunks/0",
            headers={**_auth_headers(), "Content-Type": "application/octet-stream"},
            content=wav,
        )
    assert r.status_code == 404


def test_chunked_abort_cleans_session() -> None:
    wav = _silent_wav_bytes()
    with TestClient(main.app) as client:
        session = _create_session(client, wav, chunk_size=len(wav))
        upload_id = session["upload_id"]
        assert upload_id in chunked_upload._sessions
        r = client.delete(f"/diarize/sessions/{upload_id}", headers=_auth_headers())
        assert r.status_code == 204
        assert upload_id not in chunked_upload._sessions


def test_diarize_legacy_unchanged() -> None:
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers=_auth_headers(),
            files={"file": ("test.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 200
    assert "result" in [name for name, _ in _parse_sse(r.text)]
