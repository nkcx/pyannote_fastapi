# API reference and integration guide

This document is the source of truth for clients integrating with the diarization service: every endpoint, every error code, every SSE event, and the tuning knobs that affect what clients should expect at runtime.

For setup, environment variables, and `docker run` examples see the [README](../README.md).

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/live` | none | Liveness probe. `200` while the process is running. |
| `GET` | `/health` | none | Readiness probe. `200` once the pipeline is loaded, `503` until then. |
| `GET` | `/metrics` | none | Prometheus exposition (text/plain). |
| `POST` | `/diarize` | Bearer | Submit an audio file and stream the diarization result over SSE. |
| `GET` | `/diarize/capabilities` | Bearer | Chunked-upload limits and feature flags. |
| `POST` | `/diarize/sessions` | Bearer | Create a chunked upload session. |
| `PUT` | `/diarize/sessions/{upload_id}/chunks/{chunk_index}` | Bearer | Upload one raw chunk (`application/octet-stream`). |
| `POST` | `/diarize/sessions/{upload_id}/complete` | Bearer | Reassemble WAV and stream diarization over SSE (same as `/diarize`). |
| `DELETE` | `/diarize/sessions/{upload_id}` | Bearer | Abort session and delete partial data. |

For large files behind a **~100 MB** request body limit (e.g. Cloudflare), use the chunked session flow. See [CHUNKED_UPLOAD.md](CHUNKED_UPLOAD.md).

## Authentication

All authenticated endpoints expect:

```
Authorization: Bearer <key>
```

`<key>` must match one of the comma-separated entries in the `API_KEYS` environment variable. There is no rotation / expiry logic — multiple keys are simply all valid, which lets you rotate by adding a new key, deploying, then removing the old one.

Unauthenticated calls return `401`:

```json
{ "detail": { "error": "unauthorized" } }
```

A small server-side delay (`AUTH_FAIL_DELAY_SECONDS`, default `0.5 s`) is added to every `401` response to slow credential-stuffing attacks. Each failure also emits a structured `audit_event=auth_failed` log line including the originating IP, the first 4 characters of the presented token (if any), the user-agent, and the Cloudflare `cf-ray` / `cf-ipcountry` headers.

## Rate limits

All endpoints are rate limited. The limit key is the API key for authenticated requests (`Authorization: Bearer …`) and the originating client IP otherwise. `POST /diarize` additionally enforces an independent per-IP cap that always applies, so an attacker rotating Bearer tokens cannot bypass the limit.

| Endpoint | Default limit | Env var(s) |
| --- | --- | --- |
| `POST /diarize` | `10/minute` per key + `20/minute` per IP | `RATE_LIMIT_DIARIZE`, `RATE_LIMIT_DIARIZE_IP` |
| `GET /live` | `120/minute` per IP | `RATE_LIMIT_LIVE` |
| `GET /health` | `120/minute` per IP | `RATE_LIMIT_HEALTH` |
| `GET /metrics` | `60/minute` per IP | `RATE_LIMIT_METRICS` |

When a limit is exceeded the server responds with:

- **Status:** `429`
- **Headers:** `Retry-After: 1`, plus the standard `X-RateLimit-Limit` / `X-RateLimit-Remaining` / `X-RateLimit-Reset` headers.
- **Body:** `{"error":"rate_limited","detail":"<limit string that was breached>"}`

The originating client IP is resolved from `cf-connecting-ip` → `x-real-ip` → first hop of `x-forwarded-for` → socket peer. Deploy behind a trusted proxy (Cloudflare, ingress, etc.) so these headers cannot be spoofed.

For multi-replica deployments set `RATE_LIMIT_STORAGE_URI=redis://…` so limits are shared across pods; the default `memory://` is per-process.

## `POST /diarize`

### Request

| Part | Where | Required | Description |
| --- | --- | --- | --- |
| `Authorization: Bearer <key>` | header | yes | API key (see above). |
| `Accept: text/event-stream` | header | recommended | Signals intent; the server returns SSE regardless. |
| `file` | multipart form field | yes | Audio file (any format `torchaudio` can decode — wav, flac, mp3, m4a, …). |
| `num_speakers` | query | no | Exact number of speakers (overrides min/max). Must be `1`–`MAX_SPEAKERS` (default `50`); out-of-range values return `422`. |
| `min_speakers` | query | no | Lower bound on speaker count. Must be `1`–`MAX_SPEAKERS` (default `50`); out-of-range values return `422`. |
| `max_speakers` | query | no | Upper bound on speaker count. Must be `1`–`MAX_SPEAKERS` (default `50`); out-of-range values return `422`. |
| `exclusive` | query | no | If `true`, return the pipeline's `exclusive_speaker_diarization` output (non-overlapping segments). Default `false`. |
| `return_embeddings` | query | no | If `true`, include a per-speaker centroid embedding vector in the `result` event under `embeddings`. Default `false` (the field is present but empty). |

### Successful response

- **Status:** `200`
- **Content-Type:** `text/event-stream`
- **Headers:** `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`
- **Body:** a sequence of SSE frames terminated by either a `result` event (success) or an `error` event (application-level failure). See the event reference below.

The HTTP status is `200` as soon as the request is accepted onto the queue. Application-level failures during processing surface as an SSE `error` event, **not** an HTTP error, because the response has already started streaming. Clients must inspect the event stream — checking only the HTTP status is not sufficient.

### Pre-acceptance HTTP errors

The following errors are returned before the SSE stream begins, with a normal JSON body. Their shape is FastAPI's default `{"detail": ...}`.

| Status | `detail.error` | Headers | Meaning | Recommended client action |
| --- | --- | --- | --- | --- |
| `401` | `unauthorized` | — | Missing or invalid `Authorization: Bearer` header. Response is delayed by `AUTH_FAIL_DELAY_SECONDS` (default `0.5 s`). | Fix the key. Do not retry. |
| `413` | `upload_too_large` | — | Request body exceeded `MAX_UPLOAD_BYTES` (default 2 GiB). The upload is aborted mid-stream. Detail includes `max_bytes`. | Re-encode / split the file. Do not retry as-is. |
| `429` | `rate_limited` | `Retry-After: 1`, `X-RateLimit-*` | Per-key or per-IP rate limit was exceeded. Detail field is the limit string that was breached. | Honour `Retry-After`. Back off exponentially on repeated 429s. |
| `503` | `pipeline_not_loaded` | — | The container is up but the pipeline is still initialising (cold start, model download). | Retry with exponential backoff; `/health` will be `200` once ready. |
| `503` | `queue_full` | `Retry-After: 5` | The in-process queue has reached `MAX_QUEUE_DEPTH` (default `64`). | Honour `Retry-After`, then retry. Detail payload includes `max_queue_depth`. |
| `422` | (FastAPI validation) | — | Missing `file` field, invalid query parameter type, etc. | Fix the request; do not retry as-is. |

## SSE event reference

Each frame is:

```
event: <name>
data: <single-line JSON>

```

Frames are separated by a blank line. The stream ends after a `result` or `error` event. Clients should also handle the underlying TCP connection closing without either of those (treat as transport failure and retry).

### `event: status`

Lifecycle transitions for the job. Emitted at least twice: once on enqueue (`phase: "queued"`) and once when a worker picks the job up (`phase: "running"`).

```json
{
  "job_id": "a1b2c3...",
  "phase": "queued" | "running",
  "worker": 0
}
```

- `job_id` — opaque identifier for this request; useful for log correlation.
- `worker` — only present on `phase: "running"`; zero-based worker index.

### `event: heartbeat`

Emitted every `SSE_HEARTBEAT_SECONDS` (default `5`) while the job is queued or running. Its only purpose is to keep the connection alive through proxies and to give the client a coarse progress indicator. **It is not a guarantee of forward progress** — the worker may be deep inside a synchronous pyannote call.

```json
{
  "job_id": "a1b2c3...",
  "phase": "queued" | "running",
  "elapsed_seconds": 12.5
}
```

`elapsed_seconds` is wall time since the SSE stream started (i.e. since the upload completed and the job was enqueued), not since the worker began processing.

### `event: result`

Terminal success event. The stream closes immediately after.

```json
{
  "job_id": "a1b2c3...",
  "duration_seconds": 123.45,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    { "start": 0.21, "end": 3.84, "speaker": "SPEAKER_00" }
  ],
  "processing_time_seconds": 4.2,
  "model": "pyannote/speaker-diarization-community-1",
  "pyannote_version": "<from installed pyannote.audio>",
  "embeddings": {
    "SPEAKER_00": [0.0123, -0.0456, "... (embedding dimension floats)"]
  }
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `duration_seconds` | float | Length of the input audio, derived from `torchaudio.load`. |
| `num_speakers` | int | Distinct speaker labels in `segments`. |
| `speakers` | string[] | Sorted list of distinct speaker labels. |
| `segments` | object[] | Time-ordered speech turns. `start` < `end`, both in seconds. |
| `processing_time_seconds` | float | Wall time of the inference call only (not including upload/queue wait). |
| `model` | string | Model id used for this run; mirrors the `MODEL_ID` env var (default `pyannote/speaker-diarization-community-1`). Useful for downstream auditing. |
| `pyannote_version` | string | Installed `pyannote.audio` version. |
| `embeddings` | object | Per-speaker L2-normalized centroid embedding vectors, keyed by speaker label. Empty `{}` unless `return_embeddings=true`. Each vector is the pipeline's clustering centroid for that speaker, suitable for cosine-similarity matching against reference embeddings. May be empty for a speaker the pipeline could not embed. |

### `event: error`

Terminal failure event during processing. The stream closes immediately after.

```json
{
  "job_id": "a1b2c3...",
  "status": 400 | 500,
  "detail": { "error": "<code>" }
}
```

| `status` | `detail.error` | Meaning | Recommended client action |
| --- | --- | --- | --- |
| `400` | `invalid_audio` | The uploaded file decoded to an unexpected shape (e.g. zero-dimensional tensor). | Validate the file locally before retrying. |
| `413` | `audio_too_long` | Decoded audio duration exceeded `MAX_AUDIO_SECONDS` (default 12 h). Detail includes `duration_seconds` and `max_seconds`. | Split the audio into shorter chunks. |
| `500` | `diarization_output_parse_failed` | The pyannote pipeline returned a structure this service did not recognise (typically a version skew). | Open an issue; retrying is unlikely to help. |
| `500` | `diarization_failed` | An unexpected runtime error from `torchaudio` / `pyannote` / `torch` (file decode failure, CUDA OOM, etc.). | Inspect server logs. Retry once after backoff; if persistent, treat as a server-side bug. |
| `504` | `diarization_timeout` | Inference exceeded `INFERENCE_TIMEOUT_SECONDS` (default `7200`). The queue slot is freed immediately so other requests proceed; the underlying thread keeps running until pyannote returns. Detail includes `timeout_seconds`. | Retry with shorter audio. Operators: alert on `pyannote_leaked_inference_threads`. |

The `status` field mirrors the HTTP status this error *would* have produced if it had happened before the stream started. It is informational; the actual HTTP status is always `200` for any response that reached the SSE phase.

## Performance and tuning

The service runs an in-process job queue: incoming requests are accepted onto an `asyncio.Queue`, and a fixed pool of workers pulls from it and runs pyannote inference in a thread (so the event loop stays free to emit heartbeats and accept new uploads).

Three environment variables control this behaviour:

### `DIARIZE_WORKERS` (default: `1`)

Number of background workers consuming the queue. **One diarization runs per worker at a time.**

- **Single GPU (the default and most common case): keep this at `1`.** pyannote saturates the GPU; two concurrent inferences contend on the same CUDA context, don't go faster, and frequently OOM on longer clips.
- **Multi-GPU host:** the current implementation always sends tensors to `cuda` (device 0). Raising `DIARIZE_WORKERS` will *not* automatically use other GPUs — both workers will fight over GPU 0. Multi-device support would need a worker→device mapping (not implemented; track in the issue tracker).
- **CPU-only image:** PyTorch already parallelises a single inference across cores via `OMP_NUM_THREADS`. Running multiple workers oversubscribes the cores unless you also reduce intra-op threads per worker.

### `MAX_QUEUE_DEPTH` (default: `64`)

Maximum number of jobs that may sit in the queue (excluding the one currently being processed). When the queue is full:

- The server rejects new `POST /diarize` calls with `503 {"detail":{"error":"queue_full","max_queue_depth":N}}` and `Retry-After: 5`.
- The check runs **before** the upload body is read, so flooding the endpoint does not waste disk I/O.
- A second check runs after the upload to close the race against other concurrent requests; the temp file is cleaned up on this path.

Sizing guidance: keep this small enough that the worst-case queue wait (`MAX_QUEUE_DEPTH * avg_processing_time / DIARIZE_WORKERS`) is shorter than your upstream client/LB timeout. With a 30 s average job and one worker, `MAX_QUEUE_DEPTH=64` implies up to ~32 minutes of queue wait for the unluckiest caller — set it lower if your clients are less patient.

### `MAX_UPLOAD_BYTES` (default: `2147483648` = 2 GiB)

Hard cap on the request body for `POST /diarize`. The check runs inside the chunked read loop, so an oversized upload is aborted as soon as the cap is crossed (it is not buffered to completion). The temp file is unlinked and the server responds with `413 {"error":"upload_too_large","max_bytes":N}`. Configure the same or stricter limit at your reverse proxy / Cloudflare.

### `MAX_AUDIO_SECONDS` (default: `43200` = 12 h)

Hard cap on decoded audio duration. Enforced after `torchaudio.load` returns, so a small but highly-compressed file that decodes into hours of audio will still be rejected. Returns the SSE error `{"status":504,"detail":{"error":"audio_too_long",...}}` once a worker picks the job up.

### `INFERENCE_TIMEOUT_SECONDS` (default: `7200` = 2 h, set `0` to disable)

Soft per-request inference timeout. On expiry:

- The queue slot is freed and `ACTIVE_REQUESTS` decrements, so new jobs can start.
- The client receives an SSE `error` event with `status: 504` and `detail.error: "diarization_timeout"`.
- The underlying OS thread continues running until pyannote returns naturally (Python cannot kill a running thread). The leaked thread is tracked by the `pyannote_leaked_inference_threads` gauge for the duration.

This is a deliberate trade-off: a true hard kill would require a `ProcessPoolExecutor`, which would multiply GPU VRAM usage per worker. For well-behaved input bounded by `MAX_AUDIO_SECONDS`, the soft timeout combined with leak observability + pod recycling on sustained leaks is sufficient.

### `SSE_HEARTBEAT_SECONDS` (default: `5`)

Interval between `heartbeat` frames. Must be lower than:

- The idle timeout of every proxy between the client and the service (nginx default `proxy_read_timeout` is `60s`; cloud load balancers typically `60`–`350s`).
- The client's own read timeout.

Lower values give snappier disconnect detection (the server polls `request.is_disconnected()` once per heartbeat tick) at the cost of marginally more bytes on the wire. `5` is a reasonable default for typical deployments.

## Client implementation checklist

1. **Always send `Authorization: Bearer <key>`.** Treat `401` as fatal — do not retry.
2. **Treat `503 queue_full` as backpressure.** Respect `Retry-After`. A naive retry loop without backoff will keep the queue saturated.
3. **Treat `503 pipeline_not_loaded` as a startup race.** Retry with exponential backoff; combine with a `GET /health` probe if you control the deployment lifecycle.
4. **Stream the response.** Do not buffer the full body before parsing — the heartbeats are the *point*. Use an HTTP client that exposes a byte/line stream (`fetch` + `ReadableStream`, `httpx.stream`, `requests` with `stream=True`, etc.). Browser `EventSource` cannot be used directly because it does not support `POST` with multipart bodies.
5. **Parse SSE properly.** Split on blank lines (`\n\n`), then extract `event:` and `data:` lines per block. The `data:` payload is always single-line JSON in this service.
6. **Distinguish stream events from transport errors.** A successful HTTP response that ends without a `result` or `error` event means the TCP connection was dropped (timeout, server crash, …); treat it as retryable. A `result` event means success even though the HTTP status was already `200` from the start. An `error` event means application-level failure — check the table above to decide whether to retry.
7. **Honour your own timeouts on `elapsed_seconds`.** The server enforces `INFERENCE_TIMEOUT_SECONDS` (default 2 h) and audio is capped at `MAX_AUDIO_SECONDS` (default 12 h), but you should still apply a tighter client-side ceiling appropriate to your UX. Closing the connection client-side is detected on the next heartbeat tick; pyannote inference is not cancellable, so the worker finishes the in-flight job but no result is delivered to you.
8. **Log `job_id`.** Every event carries it; pairing it with server logs (`grep <job_id>`) is the fastest way to debug a stuck or failing request.

## Metrics (Prometheus)

`GET /metrics` exposes the following series in addition to the standard `prometheus-client` process collectors:

| Name | Type | Labels | Description |
| --- | --- | --- | --- |
| `pyannote_requests_total` | counter | `endpoint`, `status` | HTTP requests processed. SSE responses always count as `status="200"` regardless of in-stream outcome — use `pyannote_sse_results_total` for application-level success/error rate. |
| `pyannote_sse_results_total` | counter | `outcome` (`success` \| `error`) | Terminal SSE events emitted by `/diarize`. Counts the `result` and `error` frames; client disconnects mid-stream are not counted. |
| `pyannote_diarization_duration_seconds` | histogram | — | Wall time of the pyannote inference call only (excludes upload and queue wait). Buckets: `0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800` s. |
| `pyannote_audio_duration_seconds` | histogram | — | Length of input audio. Buckets: `1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600` s. |
| `pyannote_realtime_factor` | histogram | — | `diarization_seconds / audio_seconds`. Values <1 mean faster than realtime; key capacity-planning metric. Buckets: `0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30`. |
| `pyannote_queue_depth` | gauge | — | Jobs currently waiting in the queue (excludes the one being processed). Compare against `MAX_QUEUE_DEPTH` to detect backpressure. |
| `pyannote_active_requests` | gauge | — | Jobs currently being processed by a worker. Sum with `pyannote_queue_depth` for total in-flight load. |
| `pyannote_leaked_inference_threads` | gauge | — | Inference threads still running after `INFERENCE_TIMEOUT_SECONDS` expired. Decrements when the underlying pyannote call eventually returns. Sustained non-zero values indicate the model is getting stuck — alert on this and recycle the pod. Suggested rules: warn if `> 0` for 5 min, restart if `>= DIARIZE_WORKERS` for 1 min. |
| `pyannote_model_loaded` | gauge | — | `1` after the pipeline is loaded, `0` during startup / shutdown. |
| `pyannote` | info | `version`, `model_id`, `torch_version`, `cuda_available` | Static build/runtime info. |
