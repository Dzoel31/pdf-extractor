FROM ghcr.io/astral-sh/uv:0.13.10 AS uv-bin

FROM python:3.15-slim-bookworm AS builder

COPY --from=uv-bin /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy

WORKDIR /app

ARG TORCH_VARIANT=cpu

# Resolve dependencies from the lockfile before adding the application source,
# so code changes reuse the expensive dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra ${TORCH_VARIANT}

FROM python:3.15-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-ind

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app /app/app
COPY dashboard.py main.py /app/
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
RUN python -c "import cv2, torch, whisper; print('torch', torch.__version__, 'cuda wheel:', torch.version.cuda)"

EXPOSE 8501

CMD ["/entrypoint.sh", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
