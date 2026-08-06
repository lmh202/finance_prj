# AURORA backend — FastAPI service.
#
# Build from the REPO ROOT (the build context needs both backend/ and data/):
#   docker build -t aurora-backend .
#   docker run -p 8000:8000 -e AURORA_ALLOWED_ORIGINS=http://localhost:3000 aurora-backend
#
# The image is stateless. Clients own their portfolios and send them in each
# request (see CLAUDE.md, "The backend is stateless"), so there is nothing to
# mount a volume for: /app/data holds only regenerable caches.

FROM python:3.11-slim

# curl is the healthcheck; the rest are build-time only for any package that
# still needs a compiler, and are dropped again in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits don't invalidate the (slow) install layer.
# requirements-deploy.txt deliberately omits torch/transformers — see the
# comments in that file for what that costs and how to add them back.
COPY backend/requirements-deploy.txt ./backend/requirements-deploy.txt
RUN pip install --no-cache-dir -r backend/requirements-deploy.txt

COPY backend/ ./backend/

# Only what the request path actually reads. Everything else under data/
# (training sets, experimental candidates, the 400 MB FinBERT weights) is
# offline-only and stays out of the image.
#
#   sample_portfolio.csv  REQUIRED — served by GET /portfolio/sample so a new
#                         visitor can load a demo portfolio.
#   tickers.cs[v]         OPTIONAL — only a warm start for the symbol
#                         universe; load_ticker_universe() downloads the
#                         NASDAQ directory when it's absent and falls back to
#                         FALLBACK_UNIVERSE if that fails too. The bracket is
#                         a glob: Docker does not fail a COPY when a globbed
#                         source matches nothing, so an uncommitted or removed
#                         tickers.csv costs a slower first request instead of
#                         breaking the build. It shares this COPY with
#                         sample_portfolio.csv deliberately — a COPY whose
#                         sources ALL fail to match errors with "no source
#                         files were specified", so the line needs one
#                         guaranteed source to lean on.
COPY data/sample_portfolio.csv data/tickers.cs[v] ./data/

#   risk_model.json     REQUIRED — the one PROMOTED artifact. risk_engine has
#                       no deterministic fallback, so without it /risk/*
#                       answers 503 no_model and the daily recommendation
#                       degrades to its legacy path.
COPY data/processed/risk_model.json ./data/processed/risk_model.json

# The app writes caches here (refreshed ticker directory, RSS news store).
# They are regenerable, so this needs no volume — but it does need to be
# writable by the non-root user below.
RUN useradd --create-home --uid 10001 aurora \
    && mkdir -p /app/data/processed \
    && chown -R aurora:aurora /app/data
USER aurora

# Railway (and most PaaS) inject $PORT; default to 8000 for plain docker run.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/ping" || exit 1

# `sh -c` so $PORT expands, and `exec` so uvicorn REPLACES the shell and
# becomes PID 1 — otherwise SIGTERM on deploy/restart goes to the shell,
# uvicorn never hears it, and the platform waits out its kill timeout instead
# of shutting down gracefully.
#
# One worker: the market-data cache in data_loader is per-process, so extra
# workers multiply yfinance traffic instead of sharing it — scale with
# replicas only after moving that cache out of process.
CMD ["sh", "-c", "exec uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
