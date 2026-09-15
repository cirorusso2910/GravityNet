FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# static uv binary (manages its own Python -- no system python/pip needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN groupadd --gid 1000 gravitynet \
    && useradd --uid 1000 --gid gravitynet --create-home gravitynet

USER gravitynet
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy

# install dependencies first so they're cached across code-only changes
COPY --chown=gravitynet:gravitynet pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY --chown=gravitynet:gravitynet . .

ENTRYPOINT ["python", "GravityNet.py"]
