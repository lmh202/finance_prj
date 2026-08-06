# Deploying AURORA

Two processes, two hosts:

```
Next.js frontend (Vercel)  ──https──▶  FastAPI backend (Railway)  ──▶  Yahoo Finance / RSS
   localStorage holds                     stateless, no volume
   each visitor's portfolio               shared caches only
```

The Streamlit app (`frontend/`) is a **local dev tool and is not deployed** —
it and the Next.js app keep separate portfolios.

## Why this split

The backend cannot run on Vercel, for reasons that are structural rather than
fixable with configuration:

| Blocker | Detail |
|---|---|
| Function size | Vercel Python functions cap at 250 MB unzipped. `torch` alone is ~4.3 GB installed on Windows, ~200 MB–2.5 GB on Linux. |
| Execution time | `/risk/portfolio` and `/recommendation/daily` take 15–25 s on a cold market-data cache (measured). |
| In-process cache | `data_loader`'s per-symbol TTL cache and `_common.load_benchmark_close()`'s 6 h memo assume a long-lived process. Serverless re-pays both on every cold start. |

The portfolio-storage problem that used to also block this is gone: clients
own their portfolios now (see `CLAUDE.md`, "The backend is stateless"), so the
backend needs no disk, session or database. That is what makes a small
single-container deployment enough.

## What the deployed image does NOT include

`backend/requirements-deploy.txt` drops `torch` + `transformers` (~4.4 GB).
They exist only for **live FinBERT news sentiment**, which is lazy and already
has a documented fallback — `finbert_sentiment._load()` raises
`ModelUnavailable` on ImportError, `analyzer.py:181` catches it and uses the
keyword scorer, and the degradation is logged:

```
UserWarning: FinBERT unavailable, using keyword sentiment fallback:
transformers/torch not installed: No module named 'torch'
```

Everything else is unaffected. Verified against a clean venv with only the
deploy requirements: `/recommendation/daily` still runs the **primary**
decision path (`production_mode=strategy_external_harx_news_risk`) with full
fusion explanations — not a fallback. The risk engine is untouched because its
promoted artifact (`risk_model.json`) uses a `linear_gamma_variance_ratio`
news overlay, which needs no torch.

**To deploy with live FinBERT instead:** add `torch` and `transformers` back to
`requirements-deploy.txt`, add `COPY data/models/finbert/ ./data/models/finbert/`
to the Dockerfile (bake the 438 MB of weights in — do not rely on a runtime
download), un-ignore that path in `.dockerignore`, and size the container for
~2 GB RAM.

## Deploy the backend to Railway

1. Push this repo to GitHub.
2. Railway → **New Project → Deploy from GitHub repo**. `railway.toml` selects
   the repo-root `Dockerfile` automatically; no build config needed.
3. **Variables** → set:

   | Variable | Value |
   |---|---|
   | `AURORA_ALLOWED_ORIGINS` | your Vercel origin, e.g. `https://aurora.vercel.app` — comma-separated for several, **no trailing slash** |
   | `DEEPSEEK_API_KEY` | optional; nicer text on the *fallback* decision path only |

   Do **not** set `PORT` — Railway injects it and the Dockerfile binds it.
   Do **not** set `AURORA_DATA_DIR` — the image already places `/app/data`.
4. **Settings → Networking → Generate Domain**. That HTTPS URL is what the
   frontend calls.
5. Check `https://<your-backend>/ping` returns `{"status":"ok"}` and
   `https://<your-backend>/docs` lists 22 endpoints.

Sizing, all measured on the built image rather than estimated:

| | |
|---|---|
| Image | **825 MB** |
| RAM, after serving the full engine path | **174 MB** — a 512 MB instance is plenty |
| Cold request (empty market cache) | 5–13 s |
| Warm request (TTL cache hit) | 0.0–5 s |

No volume — the container writes only regenerable caches (refreshed ticker
directory, RSS store); verified by restarting it and confirming it still
serves. Shutdown is graceful (1 s, `Application shutdown complete`), because
the Dockerfile `exec`s uvicorn into PID 1 so SIGTERM reaches it.

That 825 MB is after one non-obvious fix. The plain `xgboost` Linux wheel
depends on the NVIDIA CUDA runtime, which put **400 MB of `nvidia/*` plus a
333 MB GPU-enabled libxgboost** into a CPU-only container — more than half the
image, for a library whose only job here is satisfying a module-level import.
`requirements-deploy.txt` uses `xgboost-cpu` instead (same `xgboost` module, no
code change), which took the build from **2.11 GB → 825 MB**. If you ever
switch that line back to plain `xgboost`, expect the 2.11 GB.

Keep `numReplicas = 1`. The market-data cache is in-process, so a second
replica does not share it: it doubles yfinance traffic and moves you toward
rate limiting. Scale out only after that cache moves somewhere shared.

## Deploy the frontend to Vercel

1. Vercel → **New Project** → this repo.
2. **Root Directory: `frontendjs`** ← easy to miss; the build fails without it.
3. **Environment Variables** → `NEXT_PUBLIC_BACKEND_URL` = your Railway HTTPS
   URL (no trailing slash).

   `NEXT_PUBLIC_*` is inlined at **build time** — changing it later requires a
   redeploy, not just a restart. The tracked `frontendjs/.env.local` points at
   `localhost:8000` for local dev and is overridden by the project variable.
4. Deploy, then go back and confirm `AURORA_ALLOWED_ORIGINS` on Railway matches
   the Vercel origin exactly.

### One build-time coupling to be aware of

Railway builds from the GitHub repo, so the three files the Dockerfile `COPY`s
out of `data/` must be **committed**, not merely present on your machine:

```
data/sample_portfolio.csv     GET /portfolio/sample
data/tickers.csv              symbol-universe seed
data/processed/risk_model.json  the one promoted artifact
```

All three are tracked today. `.gitignore` lists `data/tickers.csv`, which stops
*future edits* from being staged but does not untrack the committed copy — so
the build works. It would break if anyone ran `git rm --cached` on one of them;
the symptom is a build failure at the COPY step, not a runtime error. Verify
with `git archive HEAD | tar -t | grep data/`.

## The three failure modes you will actually hit

1. **Every request fails, console says CORS.** `AURORA_ALLOWED_ORIGINS` doesn't
   match the frontend origin exactly — scheme, host and no trailing slash all
   matter. Preview deployments get their own origins; add them or use a stable
   custom domain. Never set it to `*`: it would let any page on the internet
   drive a visitor's browser against the API.
2. **Mixed content.** Vercel is HTTPS; a backend served over plain `http://`
   is blocked by the browser. Railway's generated domain is HTTPS, so this only
   bites on a self-hosted VPS without a certificate.
3. **Frontend loads, data doesn't.** Check `NEXT_PUBLIC_BACKEND_URL` was set
   *before* the last build. If you set it after, redeploy.

## Verifying a deployment

```bash
# 1. liveness
curl https://<backend>/ping

# 2. CORS: your origin allowed, others refused
curl -i -X OPTIONS https://<backend>/health/report \
  -H "Origin: https://<your-frontend>" \
  -H "Access-Control-Request-Method: POST" | grep -i "access-control-allow-origin"

# 3. a real decision (needs >=5 holdings for the primary path)
curl -s https://<backend>/portfolio/sample \
  | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({'holdings':d['holdings'],'cash':d['cash']}))" \
  | curl -s -X POST https://<backend>/recommendation/daily \
      -H "Content-Type: application/json" -d @- \
  | python -c "import json,sys; d=json.load(sys.stdin); print(d['decision_meta']['production_mode'])"
# expect: strategy_external_harx_news_risk
```

`smoke_http.py` in the repo's job scratch runs all of this; point it at any
base URL.

## Notes and known limits

- **First request after a cold start is slow** (15–25 s) — it downloads market
  data. Subsequent requests hit the TTL cache (6 h bars / 15 min quotes) and
  are near-instant. Railway does not sleep paid services; Render's free tier
  does, which makes the first visit after idle painful.
- **`/recommendation/daily` needs ≥5 holdings** to run the primary
  risk-controlled path — `gated_news` raises "fewer than five held assets have
  formal risk" below that and the response falls back to
  `legacy_signal_fallback`. That is the engine's own guard, not a deployment
  problem.
- **Portfolios are per-browser.** Clearing site data or switching device loses
  them; CSV export on `/portfolio` is the backup path. Worth saying in the UI.
- **Mainland China access:** `*.vercel.app` is unreliable there. A custom
  domain helps; a domestic VPS is the reliable option.
