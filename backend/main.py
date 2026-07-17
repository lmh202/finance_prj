"""AURORA backend — FastAPI service over the engine library in src/.

Run from the repo root:
    uvicorn main:app --app-dir backend --reload --port 8000

Interactive docs: http://localhost:8000/docs
The Streamlit frontend (frontend/) talks to this API via HTTP only — no
streamlit anywhere under backend/, no `src` imports anywhere under frontend/.
"""

from fastapi import FastAPI

from routers import health, market, news, portfolio, recommendation, strategy

app = FastAPI(title="AURORA API", version="0.1.0")


@app.get("/ping", tags=["meta"])
def ping() -> dict:
    return {"status": "ok"}


app.include_router(market.router)
app.include_router(portfolio.router)
app.include_router(health.router)
app.include_router(strategy.router)
app.include_router(news.router)
app.include_router(recommendation.router)
