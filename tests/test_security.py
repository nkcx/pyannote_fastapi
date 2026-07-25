from __future__ import annotations

import io
import json
import logging
import wave

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import main


def _silent_wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    n = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


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


def _fake_request(headers: dict[str, str], peer: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (peer, 1234),
    }
    return Request(scope)


# ---------- helper functions ----------


def test_client_ip_prefers_cf_connecting_ip() -> None:
    req = _fake_request(
        {"cf-connecting-ip": "1.2.3.4", "x-real-ip": "5.6.7.8", "x-forwarded-for": "9.9.9.9"}
    )
    assert main._client_ip(req) == "1.2.3.4"


def test_client_ip_falls_back_to_x_real_ip() -> None:
    req = _fake_request({"x-real-ip": "5.6.7.8", "x-forwarded-for": "9.9.9.9"})
    assert main._client_ip(req) == "5.6.7.8"


def test_client_ip_uses_first_xff_hop() -> None:
    req = _fake_request({"x-forwarded-for": "9.9.9.9, 10.10.10.10, 11.11.11.11"})
    assert main._client_ip(req) == "9.9.9.9"


def test_client_ip_falls_back_to_socket_peer() -> None:
    req = _fake_request({}, peer="172.16.0.5")
    assert main._client_ip(req) == "172.16.0.5"


def test_client_ip_ignores_headers_from_untrusted_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ipaddress

    monkeypatch.setattr(main, "TRUSTED_PROXIES", (ipaddress.ip_network("10.0.0.0/8"),))
    req = _fake_request({"cf-connecting-ip": "1.2.3.4"}, peer="203.0.113.5")
    # Peer is not a trusted proxy, so the spoofable header is ignored.
    assert main._client_ip(req) == "203.0.113.5"


def test_client_ip_honors_headers_from_trusted_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ipaddress

    monkeypatch.setattr(main, "TRUSTED_PROXIES", (ipaddress.ip_network("10.0.0.0/8"),))
    req = _fake_request({"cf-connecting-ip": "1.2.3.4"}, peer="10.1.2.3")
    assert main._client_ip(req) == "1.2.3.4"


def test_parse_trusted_proxies_skips_invalid() -> None:
    nets = main._parse_trusted_proxies("10.0.0.0/8, garbage, 192.168.1.5")
    assert len(nets) == 2


def test_key_prefix_redacts_token() -> None:
    assert main._key_prefix("supersecretkey") == "supe***"
    assert main._key_prefix("") == "-"


def test_is_valid_api_key_matches_and_rejects() -> None:
    assert main._is_valid_api_key("test-integration-key") is True
    assert main._is_valid_api_key("wrong-key") is False


def test_is_valid_api_key_handles_non_ascii_token() -> None:
    """Tokens arrive latin-1 decoded; a non-ASCII byte must yield a clean
    False, not a TypeError (which would surface as a 500 instead of a 401)."""
    assert main._is_valid_api_key("\x80not-a-key") is False


# ---------- auth ----------


def test_auth_failure_emits_audit_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="pyannote_service")
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer wrong-key", "User-Agent": "pytest-ua"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 401
    assert any(
        "audit_event=auth_failed" in rec.message and "reason=invalid_token" in rec.message
        for rec in caplog.records
    )


def test_auth_failure_missing_token_audit(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="pyannote_service")
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize", files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")}
        )
    assert r.status_code == 401
    assert any(
        "audit_event=auth_failed" in rec.message and "reason=missing_token" in rec.message
        for rec in caplog.records
    )


def test_auth_failure_delay_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 401 path awaits for AUTH_FAIL_DELAY_SECONDS; verify the sleep happens."""
    sleeps: list[float] = []

    real_sleep = main.asyncio.sleep

    async def _spy_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(main, "AUTH_FAIL_DELAY_SECONDS", 0.25)
    monkeypatch.setattr(main.asyncio, "sleep", _spy_sleep)

    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer nope"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 401
    assert 0.25 in sleeps


# ---------- rate limiting ----------


def test_rate_limit_live_returns_429_after_burst() -> None:
    with TestClient(main.app) as client:
        # RATE_LIMIT_LIVE = "10/minute" in test env.
        statuses = [client.get("/live").status_code for _ in range(12)]
    assert statuses[:10] == [200] * 10
    assert 429 in statuses[10:]


def test_rate_limit_429_payload_shape() -> None:
    with TestClient(main.app) as client:
        for _ in range(11):
            r = client.get("/live")
        assert r.status_code == 429
        body = r.json()
        assert body["error"] == "rate_limited"
        assert "detail" in body
        assert r.headers.get("Retry-After") == "1"


def test_rate_limit_diarize_per_key_caps_below_per_ip() -> None:
    """Per-key limit (3/min) should kick in before per-IP (5/min)
    when the same key is used from the same client."""
    headers = {"Authorization": "Bearer test-integration-key"}
    audio = _silent_wav_bytes()
    with TestClient(main.app) as client:
        statuses: list[int] = []
        for _ in range(5):
            r = client.post(
                "/diarize",
                headers=headers,
                files={"file": ("a.wav", audio, "audio/wav")},
            )
            statuses.append(r.status_code)
    # First 3 succeed (per-key cap), 4th and 5th hit per-key 429.
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429
    assert statuses[4] == 429


def test_rate_limit_diarize_per_ip_applies_across_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with a fresh (random) Bearer token on each call, the per-IP cap
    must trip — proving an attacker rotating tokens cannot bypass limits."""
    monkeypatch.setattr(main, "ALLOWED_API_KEYS", frozenset({"k1", "k2", "k3", "k4", "k5", "k6"}))
    audio = _silent_wav_bytes()
    statuses: list[int] = []
    with TestClient(main.app) as client:
        for i in range(1, 7):
            r = client.post(
                "/diarize",
                headers={"Authorization": f"Bearer k{i}"},
                files={"file": ("a.wav", audio, "audio/wav")},
            )
            statuses.append(r.status_code)
    assert statuses.count(200) == 5
    assert statuses[-1] == 429


def test_rate_limit_audit_log_emitted(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="pyannote_service")
    with TestClient(main.app) as client:
        for _ in range(11):
            client.get("/live")
    assert any("audit_event=rate_limited" in rec.message for rec in caplog.records)


def test_rate_limit_uses_cf_connecting_ip() -> None:
    """Different cf-connecting-ip values must get independent buckets."""
    with TestClient(main.app) as client:
        for _ in range(10):
            r = client.get("/live", headers={"cf-connecting-ip": "10.0.0.1"})
            assert r.status_code == 200
        # Same TCP peer but different cf-connecting-ip — should not be limited.
        r = client.get("/live", headers={"cf-connecting-ip": "10.0.0.2"})
        assert r.status_code == 200
        # 11th request from the first IP should now hit the limit.
        r = client.get("/live", headers={"cf-connecting-ip": "10.0.0.1"})
        assert r.status_code == 429
