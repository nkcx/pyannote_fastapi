#!/usr/bin/env bash
# End-to-end smoke test for a published pyannote_fastapi image.
#
# What it does:
#   1. Loads HF_TOKEN from .env (or current environment).
#   2. Pulls the image (default: ghcr.io/nkcx/pyannote_fastapi:latest).
#   3. Starts the container with a named volume for the HF cache.
#   4. Polls /live then /health until ready (first run downloads the model,
#      which can take several minutes; the CUDA image falls back to CPU when
#      the container is started without --gpus, which this script does).
#   5. Posts test-audio/test.wav to /diarize and prints the JSON.
#   6. Tears the container down.
#
# Usage:
#   scripts/test-image.sh                              # default image and audio
#   IMAGE=ghcr.io/nkcx/pyannote_fastapi:latest scripts/test-image.sh
#   AUDIO=path/to/clip.wav scripts/test-image.sh
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/nkcx/pyannote_fastapi:latest}"
AUDIO="${AUDIO:-test-audio/test.wav}"
CONTAINER_NAME="${CONTAINER_NAME:-pyannote-fastapi-smoke}"
HOST_PORT="${HOST_PORT:-8000}"
CACHE_VOLUME="${CACHE_VOLUME:-pyannote_hf_cache}"
API_KEY="${API_KEY:-test-key-$RANDOM}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-900}"
PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-5}"
QUEUE_SAMPLE_INTERVAL="${QUEUE_SAMPLE_INTERVAL:-0.5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set (export it or put it in .env)." >&2
  exit 1
fi

if [ ! -f "${AUDIO}" ]; then
  echo "ERROR: audio file not found: ${AUDIO}" >&2
  exit 1
fi

cleanup() {
  echo
  echo "==> Cleaning up container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "==> Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "==> Pulling ${IMAGE}"
docker pull "${IMAGE}"

echo "==> Starting container (cache volume: ${CACHE_VOLUME}, host port: ${HOST_PORT})"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e API_KEYS="${API_KEY}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${CACHE_VOLUME}:/opt/huggingface" \
  -p "${HOST_PORT}:8000" \
  "${IMAGE}" >/dev/null

echo "==> Waiting for /live ..."
deadline=$(( $(date +%s) + 60 ))
until curl -fsS "http://127.0.0.1:${HOST_PORT}/live" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "ERROR: /live never came up within 60s" >&2
    docker logs "${CONTAINER_NAME}" | tail -50 >&2
    exit 1
  fi
  sleep 1
done
echo "    /live OK"

echo "==> Waiting for /health (model load, up to ${READY_TIMEOUT_SECONDS}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
last_log_ts=0
until curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; do
  now=$(date +%s)
  if [ "${now}" -ge "${deadline}" ]; then
    echo "ERROR: /health did not become ready within ${READY_TIMEOUT_SECONDS}s" >&2
    docker logs "${CONTAINER_NAME}" | tail -80 >&2
    exit 1
  fi
  if [ $(( now - last_log_ts )) -ge 15 ]; then
    echo "    still loading... ($(( deadline - now ))s left)"
    last_log_ts=${now}
  fi
  sleep 2
done
echo "    /health OK"

echo "==> POST /diarize with ${AUDIO} (SSE)"
RESPONSE_FILE="$(mktemp)"
HTTP_CODE="$(curl -sS -N -o "${RESPONSE_FILE}" -w '%{http_code}' \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: text/event-stream" \
  -F "file=@${AUDIO}" \
  "http://127.0.0.1:${HOST_PORT}/diarize")"

echo "==> HTTP ${HTTP_CODE}"
echo "----- SSE stream -----"
cat "${RESPONSE_FILE}"
echo "----------------------"

if [ "${HTTP_CODE}" != "200" ]; then
  echo "ERROR: /diarize returned ${HTTP_CODE}" >&2
  docker logs "${CONTAINER_NAME}" | tail -50 >&2
  rm -f "${RESPONSE_FILE}"
  exit 1
fi

RESULT_JSON="$(awk '
  /^event: result$/ { in_result = 1; next }
  in_result && /^data: / { sub(/^data: /, ""); print; exit }
' "${RESPONSE_FILE}")"
rm -f "${RESPONSE_FILE}"

if [ -z "${RESULT_JSON}" ]; then
  echo "ERROR: SSE stream contained no 'result' event" >&2
  docker logs "${CONTAINER_NAME}" | tail -50 >&2
  exit 1
fi

echo "==> Result payload"
if command -v jq >/dev/null 2>&1; then
  echo "${RESULT_JSON}" | jq .
else
  echo "${RESULT_JSON}"
fi

echo "==> Firing ${PARALLEL_REQUESTS} parallel /diarize requests"
PARALLEL_DIR="$(mktemp -d)"
pids=()
for i in $(seq 1 "${PARALLEL_REQUESTS}"); do
  (
    code="$(curl -sS -N -o "${PARALLEL_DIR}/resp-${i}.sse" -w '%{http_code}' \
      -H "Authorization: Bearer ${API_KEY}" \
      -H "Accept: text/event-stream" \
      -F "file=@${AUDIO}" \
      "http://127.0.0.1:${HOST_PORT}/diarize")"
    printf '%s\n' "${code}" > "${PARALLEL_DIR}/code-${i}.txt"
  ) &
  pids+=("$!")
done

echo "==> Sampling queue depth while jobs run"
max_queue=0
max_active=0
while :; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=1
      break
    fi
  done

  sample="$(curl -sS "http://127.0.0.1:${HOST_PORT}/metrics" \
    | grep -E '^(pyannote_queue_depth|pyannote_active_requests) ' || true)"
  qd="$(printf '%s\n' "${sample}" | awk '/^pyannote_queue_depth /{print $2}')"
  ar="$(printf '%s\n' "${sample}" | awk '/^pyannote_active_requests /{print $2}')"
  qd="${qd:-0}"
  ar="${ar:-0}"
  qd_int="${qd%.*}"
  ar_int="${ar%.*}"
  [ "${qd_int}" -gt "${max_queue}" ] && max_queue="${qd_int}"
  [ "${ar_int}" -gt "${max_active}" ] && max_active="${ar_int}"
  echo "    [$(date -u +%H:%M:%S)] queue_depth=${qd} active=${ar}"

  if [ "${alive}" -eq 0 ]; then
    break
  fi
  sleep "${QUEUE_SAMPLE_INTERVAL}"
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "==> Per-request results"
parallel_fail=0
for i in $(seq 1 "${PARALLEL_REQUESTS}"); do
  code="$(cat "${PARALLEL_DIR}/code-${i}.txt")"
  if [ "${code}" != "200" ]; then
    echo "    request ${i}: HTTP ${code} (FAIL)"
    parallel_fail=1
    continue
  fi
  if grep -q '^event: result$' "${PARALLEL_DIR}/resp-${i}.sse"; then
    echo "    request ${i}: HTTP 200, result event received"
  else
    echo "    request ${i}: HTTP 200 but no result event (FAIL)"
    parallel_fail=1
  fi
done
rm -rf "${PARALLEL_DIR}"

echo "    observed max queue_depth=${max_queue}, max active=${max_active}"
if [ "${PARALLEL_REQUESTS}" -gt 1 ] && [ "${max_queue}" -eq 0 ]; then
  echo "WARNING: queue_depth never observed > 0; sampling may have been too slow," >&2
  echo "         or the worker drained jobs faster than the sample interval." >&2
fi

if [ "${parallel_fail}" -ne 0 ]; then
  echo "ERROR: at least one parallel /diarize call failed" >&2
  docker logs "${CONTAINER_NAME}" | tail -80 >&2
  exit 1
fi

echo "==> GET /metrics"
METRICS_FILE="$(mktemp)"
METRICS_CODE="$(curl -sS -o "${METRICS_FILE}" -w '%{http_code}' \
  "http://127.0.0.1:${HOST_PORT}/metrics")"

if [ "${METRICS_CODE}" != "200" ]; then
  echo "ERROR: /metrics returned ${METRICS_CODE}" >&2
  cat "${METRICS_FILE}" >&2
  rm -f "${METRICS_FILE}"
  exit 1
fi

REQUIRED_METRICS=(
  "pyannote_requests_total"
  "pyannote_sse_results_total"
  "pyannote_diarization_duration_seconds"
  "pyannote_audio_duration_seconds"
  "pyannote_realtime_factor"
  "pyannote_queue_depth"
  "pyannote_active_requests"
  "pyannote_model_loaded"
)
missing=0
for metric in "${REQUIRED_METRICS[@]}"; do
  if ! grep -q "^${metric}" "${METRICS_FILE}"; then
    echo "ERROR: /metrics missing series ${metric}" >&2
    missing=1
  fi
done

if ! grep -Eq '^pyannote_requests_total\{[^}]*endpoint="/diarize"[^}]*status="200"[^}]*\} [1-9][0-9]*(\.[0-9]+)?$' "${METRICS_FILE}"; then
  echo "ERROR: /metrics has no successful /diarize counter increment" >&2
  grep '^pyannote_requests_total' "${METRICS_FILE}" >&2 || true
  missing=1
fi

if ! grep -Eq '^pyannote_sse_results_total\{outcome="success"\} [1-9][0-9]*(\.[0-9]+)?$' "${METRICS_FILE}"; then
  echo "ERROR: /metrics has no successful SSE outcome counted" >&2
  grep '^pyannote_sse_results_total' "${METRICS_FILE}" >&2 || true
  missing=1
fi

if [ "${missing}" -ne 0 ]; then
  rm -f "${METRICS_FILE}"
  exit 1
fi

echo "    /metrics OK (expected series present, /diarize counter incremented)"
echo "----- final /metrics (pyannote_* series, _bucket lines suppressed) -----"
grep '^pyannote_' "${METRICS_FILE}" | grep -v '_bucket{' || true
echo "------------------------------------------------------------------------"
rm -f "${METRICS_FILE}"

echo
expected_total=$(( PARALLEL_REQUESTS + 1 ))
echo "==> Smoke test passed (1 sequential + ${PARALLEL_REQUESTS} parallel = ${expected_total} successful /diarize calls)."
