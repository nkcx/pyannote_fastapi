from __future__ import annotations

import io
import json
import wave

from fastapi.testclient import TestClient

from main import app


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


def test_live_and_health() -> None:
    with TestClient(app) as client:
        r = client.get("/live")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json().get("status") == "ready"


def test_metrics_prometheus_text() -> None:
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        assert "pyannote_requests_total" in body


def test_diarize_requires_auth() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post("/diarize", files={"file": ("test.wav", audio, "audio/wav")})
        assert r.status_code == 401


def test_diarize_rejects_wrong_bearer_token() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer not-a-real-key"},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert r.status_code == 401


def test_diarize_success_with_bearer_token() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(r.text)
        event_names = [name for name, _ in events]
        assert "status" in event_names
        assert "result" in event_names
        result_payload = next(data for name, data in events if name == "result")
        assert "segments" in result_payload
        assert result_payload["num_speakers"] >= 1
        assert "job_id" in result_payload


def test_diarize_omits_embeddings_by_default() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("test.wav", _silent_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200
        result = next(data for name, data in _parse_sse(r.text) if name == "result")
        # Field is always present but empty when not requested.
        assert result["embeddings"] == {}


def test_diarize_returns_normalized_embeddings_when_requested() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/diarize?return_embeddings=true",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("test.wav", _silent_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200
        result = next(data for name, data in _parse_sse(r.text) if name == "result")
        embeddings = result["embeddings"]
        assert set(embeddings) == set(result["speakers"])
        vec = embeddings["SPEAKER_00"]
        # Dry-run centroid [3,4,0,0] L2-normalizes to [0.6,0.8,0,0].
        assert vec == [0.6, 0.8, 0.0, 0.0]
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-9


def test_extract_speaker_embeddings_normalizes_and_keys_by_label() -> None:
    import numpy as np
    from pyannote.core import Annotation, Segment

    import main

    ann = Annotation()
    ann[Segment(0.0, 1.0)] = "SPEAKER_00"
    ann[Segment(1.0, 2.0)] = "SPEAKER_01"

    class _Out:
        speaker_diarization = ann
        speaker_embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)

    result = main._extract_speaker_embeddings(_Out())
    assert result["SPEAKER_00"] == [0.6, 0.8]
    # Zero (padding) row is kept but left unnormalized.
    assert result["SPEAKER_01"] == [0.0, 0.0]


def test_extract_speaker_embeddings_handles_missing_embeddings() -> None:
    from pyannote.core import Annotation, Segment

    import main

    ann = Annotation()
    ann[Segment(0.0, 1.0)] = "SPEAKER_00"

    class _Out:
        speaker_diarization = ann
        speaker_embeddings = None

    assert main._extract_speaker_embeddings(_Out()) == {}
    # An output object with no embeddings attribute at all.
    assert main._extract_speaker_embeddings(object()) == {}
