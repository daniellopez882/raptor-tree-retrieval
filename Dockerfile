# syntax=docker/dockerfile:1
# Both stages start from the same image so the virtualenv's interpreter path is valid in both.
ARG PYTHON_IMAGE=python:3.12-slim

# Stage 1: install the locked runtime dependencies and the package into /venv with uv.
# Every dependency ships manylinux wheels, so no compiler is installed here either.
FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /build
# Dependencies first: this layer is only rebuilt when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: the virtualenv on a clean slim image, run as uid 10001. QuALITY, the ONNX models
# and built trees are not baked in; mount a volume at /data and run `raptree fetch` into it.
FROM ${PYTHON_IMAGE}
RUN groupadd --gid 10001 raptree \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/raptree \
        --shell /usr/sbin/nologin raptree \
    && install -d -o 10001 -g 10001 /data
COPY --from=build /venv /venv
COPY LICENSE /usr/share/doc/raptor-tree-retrieval/LICENSE
ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    RAPTREE_DATA_DIR=/data \
    HF_HOME=/data/hf \
    NUMBA_CACHE_DIR=/tmp/numba
USER 10001:10001
WORKDIR /home/raptree
# A one-shot command-line image (the TUI needs `docker run -it`); there is no server to probe.
HEALTHCHECK NONE
ENTRYPOINT ["raptree"]
CMD ["demo"]
