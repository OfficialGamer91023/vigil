# One image, two services, different commands (PLAN §9). Two images for one
# codebase would be ceremony — `worker` and `api` share every dependency and
# differ only in the process they start.

# --------------------------------------------------------------------------- #
# Stage 1 — build the virtualenv
# --------------------------------------------------------------------------- #
# uv's own image ships uv and CPython 3.12 together, so nothing here downloads an
# interpreter at build time. 3.12 is pinned in pyproject and pinned again here:
# `requires-python` would *fail* on a mismatch, but failing at build time with a
# clear tag is better than failing at `uv sync` with a resolver error.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Copy the venv rather than symlink into uv's cache, so stage 2 can take it whole.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# **Dependencies before source.** Layer caching is keyed on the files a layer
# copies, so installing from just the manifests means editing a .py file does not
# reinstall alpaca-py, sqlalchemy and the rest. `--no-install-project` is what
# makes that possible: it resolves and installs the dependency tree while
# deliberately skipping `vigil` itself, which is not present yet.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# `--frozen` refuses to update uv.lock. In an image build that is the whole point:
# a build that silently re-resolved a dependency would produce an artifact that
# does not match what the tests ran against. `--no-dev` keeps pytest, mypy, ruff
# and hypothesis out of the runtime image — they are PEP 735 dev dependencies and
# nothing in production imports them.
COPY src/ ./src/
RUN uv sync --frozen --no-dev


# --------------------------------------------------------------------------- #
# Stage 2 — the runtime
# --------------------------------------------------------------------------- #
# Plain python:3.12-slim, not the uv image: uv is a build tool and has no reason
# to exist in a container that holds live brokerage credentials.
FROM python:3.12-slim-bookworm AS runtime

# `America/New_York` because a human reading `docker logs` during a session
# should see 15:40 when the flatten fires. **Nothing in the code depends on it**
# — `vigil.clock` carries an explicit ZoneInfo and `KernelContext` refuses a
# naive datetime outright — so this is legibility, not correctness. A codebase
# that needed the container's TZ to be right would have the B5 bug class wide
# open.
ENV TZ=America/New_York \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

# tzdata for the TZ above; ca-certificates so TLS to Alpaca and OpenAI verifies.
# Both are absent from slim, and the second fails in a way that reads like a
# network outage rather than a missing package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root. This process holds paper-trading credentials and reaches the public
# internet; there is no reason for it to be able to write to its own image.
RUN useradd --create-home --uid 10001 vigil
WORKDIR /app

COPY --from=builder --chown=vigil:vigil /app/.venv /app/.venv
COPY --chown=vigil:vigil src/ ./src/
COPY --chown=vigil:vigil alembic/ ./alembic/
COPY --chown=vigil:vigil alembic.ini ./

# config/ is **baked in, deliberately**, and that includes `config/account.lock`.
# Mounting it would make the account lock editable without a rebuild, and hard
# rule #7 wants exactly the opposite: re-pointing the agent at a different
# account should require rebuilding and redeploying, not restarting.
COPY --chown=vigil:vigil config/ ./config/

USER vigil

# Overridden per service in docker-compose.yml. The default is the agent, so a
# bare `docker run` starts the thing this repository is for.
CMD ["python", "-m", "vigil.worker.runner"]
