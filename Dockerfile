# syntax=docker/dockerfile:1

# pyannote.audio 4.x requires torch>=2.8 / torchaudio>=2.8. Keep the
# default base image at or above that floor so pip does not silently
# replace torch with a wheel built against a different CUDA toolkit.
ARG PYTORCH_BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_BASE}

# Omit build-arg or leave empty to install the latest pyannote.audio from PyPI at build time.
# CI passes the current PyPI release so CUDA/CPU images in one run match.
ARG PYTORCH_BASE
ARG PYANNOTE_VERSION
ARG BUILD_DATE
ARG VCS_REF

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/opt/huggingface

LABEL org.opencontainers.image.title="pyannote_fastapi (CUDA)" \
      org.opencontainers.image.description="FastAPI wrapper around pyannote/speaker-diarization-community-1. CUDA build; base image is resolved at build time from pyannote.audio's torch lower bound (org.opencontainers.image.base.name). Model weights are NOT baked in; supply HF_TOKEN at runtime or mount MODEL_PATH. Fork of https://github.com/maximsachs/pyannote_fastapi" \
      org.opencontainers.image.source="https://github.com/nkcx/pyannote_fastapi" \
      org.opencontainers.image.url="https://github.com/nkcx/pyannote_fastapi" \
      org.opencontainers.image.documentation="https://github.com/nkcx/pyannote_fastapi#readme" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="nkcx (fork of Maxim Sachs / @maximsachs)" \
      org.opencontainers.image.base.name="docker.io/${PYTORCH_BASE}" \
      org.opencontainers.image.version="${PYANNOTE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/huggingface

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && if [ -z "${PYANNOTE_VERSION}" ]; then \
         pip install --no-cache-dir "pyannote.audio"; \
       else \
         pip install --no-cache-dir "pyannote.audio==${PYANNOTE_VERSION}"; \
       fi

COPY app/ /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/live || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
