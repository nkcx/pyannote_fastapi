# Chunked upload migration guide

Use the chunked session API when a WAV exceeds the Cloudflare **~100 MB** per-request limit. Small files should keep using `POST /diarize` unchanged.

## Three-step flow

1. **Create session** — `POST /diarize/sessions` with JSON metadata (`total_size_bytes`, `content_type: audio/wav`, optional `chunk_size_bytes`).
2. **Upload chunks** — `PUT /diarize/sessions/{upload_id}/chunks/{chunk_index}` with **raw** `application/octet-stream` body (one chunk per request, ≤ `max_chunk_size_bytes`, default 80 MiB).
3. **Complete** — `POST /diarize/sessions/{upload_id}/complete` with `Accept: text/event-stream`. Response matches single-shot `/diarize` (status events + `event: result`).

Optional: `DELETE /diarize/sessions/{upload_id}` to abort and free disk.

Discovery: `GET /diarize/capabilities` returns limits and recommended chunk size (default 64 MiB).

## Error reference

| HTTP | When | Client action |
| --- | --- | --- |
| `201` | Session created | Store `upload_id`, `chunk_size_bytes`, `expected_chunk_count` |
| `204` | Chunk stored (or idempotent re-PUT) | Continue until all indices `0..N-1` uploaded |
| `200` + SSE | Complete accepted | Parse SSE until `event: result` |
| `400` | Bad JSON, wrong chunk index/size, incomplete upload on complete | Fix request |
| `401` | Missing/invalid Bearer | Fix key, no retry |
| `409` | Chunk bytes conflict, session not open | Reconcile chunk or start new session |
| `413` | Total or chunk over server max | Split differently or reduce file |
| `415` | `content_type` not WAV | Use `audio/wav` |
| `422` | Invalid WAV after reassembly, size/SHA mismatch | Fix source file |
| `404` | Unknown or expired session | Create new session |
| `429` | Too many open sessions or queue full on complete | Backoff, retry complete only |
| `503` | Pipeline not ready or chunked disk quota | Retry with backoff |

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHUNKED_UPLOAD_MAX_BYTES` | 1 GiB | Max declared `total_size_bytes` |
| `MAX_CHUNK_SIZE_BYTES` | 80 MiB | Max per-chunk body |
| `RECOMMENDED_CHUNK_SIZE_BYTES` | 64 MiB | Default when client omits `chunk_size_bytes` |
| `CHUNK_SESSION_TTL_SECONDS` | 7200 | Session expiry |
| `MAX_CONCURRENT_CHUNKED_SESSIONS` | 32 | Open sessions cap |
| `CHUNKED_UPLOAD_DISK_QUOTA_BYTES` | 2× max upload | Partial upload disk budget |
| `CHUNKED_UPLOAD_DIR` | `$TMPDIR/pyannote_chunked_uploads` | Staging directory |
| `CHUNKED_JANITOR_INTERVAL_SECONDS` | 300 | How often the background janitor purges expired sessions (min 30) |
| `SINGLE_UPLOAD_MAX_BYTES` | 100 MiB | Advertised in `/diarize/capabilities` only |

## Example (curl)

```bash
API_KEY=your-key
BASE=https://diarize.example.com
WAV=recording.wav
SIZE=$(wc -c < "$WAV")
CHUNK=$((64 * 1024 * 1024))

SESSION=$(curl -sS -X POST "$BASE/diarize/sessions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"recording.wav\",\"content_type\":\"audio/wav\",\"total_size_bytes\":$SIZE,\"chunk_size_bytes\":$CHUNK}")

UPLOAD_ID=$(echo "$SESSION" | jq -r .upload_id)
EXPECTED=$(echo "$SESSION" | jq -r .expected_chunk_count)
CHUNK_SIZE=$(echo "$SESSION" | jq -r .chunk_size_bytes)

for i in $(seq 0 $((EXPECTED - 1))); do
  OFFSET=$((i * CHUNK_SIZE))
  dd if="$WAV" bs=1 skip="$OFFSET" count="$CHUNK_SIZE" 2>/dev/null | \
    curl -sS -X PUT "$BASE/diarize/sessions/$UPLOAD_ID/chunks/$i" \
      -H "Authorization: Bearer $API_KEY" \
      -H "Content-Type: application/octet-stream" \
      --data-binary @-
done

curl -sS -N -X POST "$BASE/diarize/sessions/$UPLOAD_ID/complete" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: text/event-stream"
```

Python integration script: [`scripts/test_chunked_upload.py`](../scripts/test_chunked_upload.py).
