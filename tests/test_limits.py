from __future__ import annotations

import io
import json
import logging
import time
import wave
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

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


# ---------- upload size cap ----------


def test_upload_too_large_returns_413(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 256)
    caplog.set_level(logging.WARNING, logger="pyannote_service")
    with TestClient(main.app) as client:
        big = b"\x00" * 4096
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("big.wav", big, "audio/wav")},
        )
    assert r.status_code == 413
    detail = r.json()["detail"]
    assert detail["error"] == "upload_too_large"
    assert detail["max_bytes"] == 256
    assert any("audit_event=upload_too_large" in rec.message for rec in caplog.records)


def test_upload_under_limit_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("ok.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 200


def test_upload_too_large_cleans_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the rejected upload's temp file is removed."""
    created: list[Path] = []
    real_named_tmp = main.tempfile.NamedTemporaryFile

    def _spy_named_tmp(*args, **kwargs):  # type: ignore[no-untyped-def]
        tmp = real_named_tmp(*args, **kwargs)
        created.append(Path(tmp.name))
        return tmp

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 256)
    monkeypatch.setattr(main.tempfile, "NamedTemporaryFile", _spy_named_tmp)

    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("big.wav", b"\x00" * 4096, "audio/wav")},
        )
    assert r.status_code == 413
    assert created, "expected the upload handler to create a temp file"
    assert not created[0].exists(), f"temp file {created[0]} was not cleaned up"


# ---------- audio duration cap ----------


def test_audio_too_long_emits_sse_413_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "MAX_AUDIO_SECONDS", 1.0)

    def _long_load(_path: str) -> tuple[torch.Tensor, int]:
        # 5 seconds at 16 kHz — comfortably above the 1 s cap.
        return torch.zeros(1, 80000), 16000

    monkeypatch.setattr(main.torchaudio, "load", _long_load)

    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    error = next(d for n, d in events if n == "error")
    assert error["status"] == 413
    assert error["detail"]["error"] == "audio_too_long"
    assert error["detail"]["max_seconds"] == 1.0
    assert error["detail"]["duration_seconds"] == 5.0


# ---------- inference timeout ----------


def test_inference_timeout_emits_sse_504(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT_SECONDS", 0.2)

    finished = False

    def _slow_inference(tmp_path: Path, params: main._DiarizationParams) -> main.DiarizeResponse:
        nonlocal finished
        time.sleep(1.5)
        finished = True
        return main.DiarizeResponse(
            duration_seconds=0.0,
            num_speakers=0,
            speakers=[],
            segments=[],
            processing_time_seconds=0.0,
            pyannote_version="test",
        )

    monkeypatch.setattr(main, "_run_diarization_sync", _slow_inference)

    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    error = next(d for n, d in events if n == "error")
    assert error["status"] == 504
    assert error["detail"]["error"] == "diarization_timeout"
    assert error["detail"]["timeout_seconds"] == 0.2


def test_inference_timeout_increments_leak_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT_SECONDS", 0.2)

    def _slow_inference(tmp_path: Path, params: main._DiarizationParams) -> main.DiarizeResponse:
        time.sleep(1.0)
        return main.DiarizeResponse(
            duration_seconds=0.0,
            num_speakers=0,
            speakers=[],
            segments=[],
            processing_time_seconds=0.0,
            pyannote_version="test",
        )

    monkeypatch.setattr(main, "_run_diarization_sync", _slow_inference)

    with TestClient(main.app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200
        # Right after the timed-out SSE error, the leak gauge should be > 0
        # (the OS thread is still sleeping).
        metrics = client.get("/metrics").text
        leaked_lines = [
            line
            for line in metrics.splitlines()
            if line.startswith("pyannote_leaked_inference_threads ")
        ]
        assert leaked_lines, "leaked-threads gauge missing from metrics"
        leaked_value = float(leaked_lines[0].split()[-1])
        assert leaked_value >= 1.0

    # Wait for the leaked thread to finish so the gauge decrements before the
    # next test runs.
    time.sleep(1.2)


# ---------- speaker-count bounds ----------


def test_speaker_count_out_of_range_rejected() -> None:
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize?num_speakers=1000000",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422


def test_speaker_count_negative_rejected() -> None:
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize?min_speakers=-1",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422


def test_speaker_count_within_range_accepted() -> None:
    with TestClient(main.app) as client:
        r = client.post(
            "/diarize?num_speakers=2",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("a.wav", _silent_wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 200
