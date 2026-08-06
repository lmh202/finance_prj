"""AURORA backend — FastAPI service over the engine library in src/.

Run from the repo root:
    uvicorn main:app --app-dir backend --reload --port 8000

Interactive docs: http://localhost:8000/docs
The Streamlit frontend (frontend/) talks to this API via HTTP only — no
streamlit anywhere under backend/, no `src` imports anywhere under frontend/.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analysis, health, market, news, portfolio, recommendation, risk, strategy

app = FastAPI(title="AURORA API", version="0.1.0")

# The Next.js frontend calls this API directly from the browser, so its exact
# origin has to be allow-listed. Local dev is the default; a deployment sets
# AURORA_ALLOWED_ORIGINS to its real frontend origin(s), comma-separated.
# Keep this a specific list, never "*" — a wildcard would let any page on the
# internet drive a user's browser against this API.
DEFAULT_ORIGINS = "http://localhost:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("AURORA_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping", tags=["meta"])
def ping() -> dict:
    return {"status": "ok"}


app.include_router(analysis.router)
app.include_router(market.router)
app.include_router(portfolio.router)
app.include_router(health.router)
app.include_router(strategy.router)
app.include_router(news.router)
app.include_router(recommendation.router)
app.include_router(risk.router)
