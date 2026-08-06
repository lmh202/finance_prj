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

# Only what the request path actually reads:
#   risk_model.json     the one PROMOTED artifact; risk_engine has no
#                       deterministic fallback, so without it /risk/* answers
#                       503 no_model and the daily recommendation degrades.
#   sample_portfolio.csv  served by GET /portfolio/sample for new visitors.
#   tickers.csv         seed for the symbol universe. Without it the first
#                       request downloads the NASDAQ directory instead — this
#                       just makes a cold container useful immediately.
# Everything else under data/ (training sets, experimental candidates, the
# 400 MB FinBERT weights) is offline-only and stays out of the image.
COPY data/sample_portfolio.csv ./data/sample_portfolio.csv
COPY data/tickers.csv ./data/tickers.csv
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
