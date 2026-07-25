# pyannote speaker diarization FastAPI Docker image

> ### 🍴 This is an actively maintained fork
>
> Forked from [**maximsachs/pyannote_fastapi**](https://github.com/maximsachs/pyannote_fastapi) specifically to add **per-speaker embedding extraction** to the `/diarize` endpoint — the embeddings let you match anonymous diarization labels (`SPEAKER_00`, `SPEAKER_01`, …) back to real people via cosine similarity, instead of the labels resetting every run. In the process, a handful of security and robustness issues were also fixed (constant-time API-key checks, unbounded-memory / rate-limit-bypass hardening, upload streaming caps, and more).
>
> **Please report issues with this fork on [this repository's issue tracker](https://github.com/nkcx/pyannote_fastapi/issues) — not against the upstream project.** Fixes here may be offered back upstream, but this fork is maintained independently.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Workflow: build-on-push](https://img.shields.io/github/actions/workflow/status/nkcx/pyannote_fastapi/build-on-push.yml?branch=main&label=build)](https://github.com/nkcx/pyannote_fastapi/actions/workflows/build-on-push.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-nkcx%2Fpyannote__fastapi-blue?logo=github)](https://github.com/nkcx/pyannote_fastapi/pkgs/container/pyannote_fastapi)

<!-- pyannote-version:start -->
**Latest published images build against `pyannote.audio` (not yet released by automation).**
<!-- pyannote-version:end -->

A minimal FastAPI service around [**pyannote/speaker-diarization-community-1**](https://huggingface.co/pyannote/speaker-diarization-community-1). The pipeline is loaded once at startup; diarization requests are queued to in-process workers and streamed back to the client as Server-Sent Events with periodic heartbeats and a final result frame.

**Images:** published to GHCR at [`ghcr.io/nkcx/pyannote_fastapi`](https://github.com/nkcx/pyannote_fastapi/pkgs/container/pyannote_fastapi) — CUDA `:latest`. This fork builds **CUDA images only** and publishes to **GHCR only** (no Docker Hub).

**Integrating a client?** See [`docs/API.md`](docs/API.md) for the full endpoint reference, every error code, the SSE event schema, and the performance-tuning knobs.

## Model access (weights are not bundled)

The upstream pipeline is **CC-BY-4.0** and **gated** on Hugging Face. This image ships application code only. At runtime you must either:

1. Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) from an account that has accepted the [model card](https://huggingface.co/pyannote/speaker-diarization-community-1) terms, **or**
2. Mount an offline checkout and set `MODEL_PATH` to its directory (must contain `config.yaml`).

## Environment variables

| Name | Required | Default | Notes |
| --- | --- | --- | --- |
| `API_KEYS` | yes | — | Comma-separated accepted keys; clients send one via `Authorization: Bearer <key>`. |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | yes\* | — | \*Unless `MODEL_PATH` is set. |
| `MODEL_PATH` | no | — | Absolute path to a local pipeline checkout. |
| `MODEL_ID` | no | `pyannote/speaker-diarization-community-1` | Hub repo id when `MODEL_PATH` is unset. |
| `HF_HOME` | no | `/opt/huggingface` | Hugging Face cache root (mount a volume here to persist). |
| `DIARIZE_WORKERS` | no | `1` | Concurrent diarization workers. Keep at `1` for single-GPU setups. |
| `MAX_QUEUE_DEPTH` | no | `64` | Max number of jobs that may be queued. Further requests are rejected with `503 {"error":"queue_full"}` and a `Retry-After: 5` header. |
| `SSE_HEARTBEAT_SECONDS` | no | `5` | Interval between SSE `heartbeat` frames while a job is queued or running. |
| `MAX_UPLOAD_BYTES` | no | `2147483648` (2 GiB) | Hard cap on the request body size for `/diarize`. Oversized uploads are aborted mid-stream with `413 {"error":"upload_too_large"}`. |
| `MAX_AUDIO_SECONDS` | no | `43200` (12 h) | Hard cap on decoded audio duration. Longer clips are rejected with `413 {"error":"audio_too_long"}` after decode. |
| `INFERENCE_TIMEOUT_SECONDS` | no | `7200` | Soft per-request inference timeout. On expiry the queue slot is freed and the client receives a `504 diarization_timeout` SSE event. The underlying thread keeps running until pyannote returns; track via `pyannote_leaked_inference_threads`. Set `0` to disable. |
| `AUTH_FAIL_DELAY_SECONDS` | no | `0.5` | Delay added to `401` responses to slow credential stuffing. |
| `RATE_LIMIT_DIARIZE` | no | `10/minute` | Per-API-key (or per-IP if no Bearer token) rate limit for `POST /diarize`. |
| `RATE_LIMIT_DIARIZE_IP` | no | `20/minute` | Per-IP rate limit for `POST /diarize`, applied **in addition** to `RATE_LIMIT_DIARIZE` (defends against attackers rotating Bearer tokens). |
| `RATE_LIMIT_LIVE` | no | `120/minute` | Per-IP rate limit for `GET /live`. |
| `RATE_LIMIT_HEALTH` | no | `120/minute` | Per-IP rate limit for `GET /health`. |
| `RATE_LIMIT_METRICS` | no | `60/minute` | Per-IP rate limit for `GET /metrics`. |
| `RATE_LIMIT_STORAGE_URI` | no | `memory://` | slowapi storage URI. Use e.g. `redis://host:6379` to share limits across replicas. |
| `TRUSTED_PROXY_IPS` | no | — | Comma-separated IPs/CIDRs of trusted proxies. Forwarded client-IP headers are honored **only** when the request arrives from one of these. Unset = legacy behavior (headers trusted from any peer; a startup warning is logged). |
| `MAX_SPEAKERS` | no | `50` | Upper bound for the `num_speakers` / `min_speakers` / `max_speakers` query params. Out-of-range values return `422`. |
| `LOG_LEVEL` | no | `INFO` | Python logging level. |
| `PYANNOTE_TELEMETRY` | no | `0` | Set to `1`/`true`/`yes` to opt in to upstream pyannote.audio anonymous usage telemetry. Disabled by default. |

### Client IP detection

Rate limits and audit logs key off the originating client IP, resolved in this priority order:

1. `cf-connecting-ip` (Cloudflare proxy / tunnel)
2. `x-real-ip`
3. First hop of `x-forwarded-for`
4. The direct socket peer

Set `TRUSTED_PROXY_IPS` to the egress IPs/CIDRs of your proxy so these headers are honored **only** when the request actually arrives from it; requests from any other peer fall back to the direct socket address. If `TRUSTED_PROXY_IPS` is left unset the legacy behavior applies — the headers are trusted from any peer and a startup warning is logged. Either way, if you deploy this image **without** a trusted proxy in front, anyone on the internet can spoof these headers, so also lock the origin down so only your reverse proxy / Cloudflare egress IPs can reach the pod.

### Audit logging

The service emits a `WARNING`-level structured log line for security-relevant events: `auth_failed`, `rate_limited`, and `upload_too_large`. Each line contains `ip`, `path`, `method`, `key=<first-4-chars>***`, `ua`, `cf_ray`, and `cf_country`, so you can correlate with Cloudflare logs.

## Quick start

```bash
docker run --rm -it --gpus all \
  -e API_KEYS="replace-me" \
  -e HF_TOKEN="replace-me" \
  -v pyannote_hf_cache:/opt/huggingface \
  -p 8000:8000 \
  ghcr.io/nkcx/pyannote_fastapi:latest
```

Submit a file and tail the SSE stream:

```bash
curl -N -fsS \
  -H "Authorization: Bearer replace-me" \
  -H "Accept: text/event-stream" \
  -F "file=@/path/to/audio.wav" \
  http://127.0.0.1:8000/diarize
```

`-N` disables curl's output buffering so you see each event as it arrives.

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/live` | Liveness. `200` while the process is up. |
| `GET` | `/health` | Readiness. `200 {"status":"ready"}` once the pipeline is loaded; `503 {"status":"not_ready"}` otherwise. |
| `GET` | `/metrics` | Prometheus exposition. |
| `POST` | `/diarize` | Submit audio, receive an SSE stream of `status` / `heartbeat` events ending in a `result` (or `error`) event. |

**See [`docs/API.md`](docs/API.md)** for the complete request/response schema, every error code, the full SSE event reference, and a client implementation checklist.

**Proxy note:** the SSE response sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`. If you front this with nginx, also set `proxy_buffering off` on the `/diarize` location and make sure idle timeouts on every hop are larger than `SSE_HEARTBEAT_SECONDS`.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export API_KEYS=dev-key
export PYANNOTE_TESTING=1
uvicorn main:app --app-dir app --reload --host 0.0.0.0 --port 8000
```

```bash
ruff check app tests
pytest -q
```

## License and attribution

Wrapper code is **MIT** ([`LICENSE`](LICENSE)). The model is **CC-BY-4.0**; the service logs an attribution line (model id, license URL, model card URL) at startup. Cached weights on the mounted volume remain gated material — treat the volume as sensitive.
