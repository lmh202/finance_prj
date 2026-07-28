# AURORA

**Real Time Market & Global Event Intelligence for Healthier Portfolios**

[📰 Newsprint Poster](aurora_poster/GROUP-8-POSTER(AURORA).pdf)

[🎬 Video Demonstration](https://drive.google.com/drive/folders/1U5nQa_rUdQF210iREnKHaEfUrywlrZE8)

## What is AURORA

*“We don't predict the future. We illuminate what's coming next.”*

The Aurora Borealis is one of nature's earliest lights, illuminating the sky before sunrise.

Inspired by this phenomenon, AURORA is an AI-powered financial assistant designed to illuminate market opportunities and risks before the majority of investors react.

Rather than simply tracking stock prices, AURORA **continuously** analyzes financial news, forecasts market movements, and recommends intelligent portfolio adjustments helping investors make informed decisions ahead of the market.

*“Helping investors stay one step ahead”*

## Why is AURORA Different?

*"Information is everywhere. Insight is rare. AURORA delivers both."*

Existing Investment Platforms

- Display stock prices
- Provide market news
- Require manual analysis
- Depend on user interpretation

AURORA

✓ Understands financial news automatically

✓ Predicts possible market impact

✓ Calculates portfolio-specific risk

✓ Suggests intelligent portfolio adjustments

✓ Keeps the investor in control by providing recommendations rather than *executing trades automatically*

## Quick start

Two processes: the FastAPI backend, and the Next.js frontend. The frontend talks to the backend over HTTP only (`frontend/api_client.py`); nothing under `backend/` imports streamlit.
### Install all dependencies
```bash
# backend + Streamlit frontend
pip install -r requirements.txt
# Next.js frontend
cd frontendjs && npm install             
```
### Run processes
```bash
# 
# terminal 1: backend
uvicorn main:app --app-dir backend --reload --port 8000   
# for development only: Streamlit UI (port 8501)
# streamlit run frontend/app.py
# terminal 2: Next.js UI (port 3000)
cd frontendjs && npm run dev                              
```
```bash
# Standalone RSS collector (grows data/news_raw.json — run regularly)
python backend/src/news_intelligence/collector.py

# API docs (when backend is running)
# http://localhost:8000/docs
```

## Layout

```
backend/                     FastAPI service (no streamlit anywhere)
  main.py                    API entry point — uvicorn main:app --app-dir backend
  serialize.py               dataclass/DataFrame -> JSON helpers
  routers/                   one thin router per engine + market/portfolio
  src/
    interfaces.py            THE CONTRACT — frozen (custodian: Developer 4)
    config.py                DATA_DIR resolution (AURORA_DATA_DIR override)
    data_loader.py           shared kernel: universe + prices + history
    portfolio.py             shared kernel: holdings + valuation
    portfolio_health/        Developer 1  (engine.py, README.md)
    daily_strategy/          Developer 2  (engine.py, README.md)
    news_intelligence/       Developer 3  (engine.py, collector.py, README.md)
    recommendation/          Developer 4  (engine.py, README.md)
frontend/                    Streamlit UI (backend access via HTTP only)
  app.py                     Home — portfolio builder (Developer 4)
  api_client.py              typed wrappers over the backend API
  views/                     one view per engine page (owned per developer)
  pages/                     routing shims only (DO NOT EDIT)
scripts/dev.ps1              start backend + frontend together
data/                        caches + user portfolio (gitignored) + sample
  processed/                 Dev 3's historical sentiment feature table
```

## Data sources

- **Ticker universe**: official NASDAQ Trader symbol directory (all NASDAQ /
  NYSE / NYSE American / Arca / Cboe / IEX listings), cached 7 days.
- **Prices**: Yahoo Finance via yfinance (latest + daily history; expand the
  asset universe here for ML training).
- **Live news**: RSS feeds (MarketWatch, CNBC, Yahoo Finance) + Anthropic LLM
  API for classification/sentiment/summaries.
- **Historical news**: FNSPID or a Kaggle financial-news corpus, scored
  locally (FinBERT/VADER) into `sentiment_features` — no look-ahead.


## Responsible AI

* AURORA is a **decision-support system, not an autonomous trading agent.** Final investment decisions remain with the user. The system provides recommendations, explanations, and risk analysis—**it never executes trades automatically**.

* **Classical quantitative rules remain the primary decision maker.** Market direction is determined by transparent technical indicators and portfolio rules. AI components (e.g., FinBERT sentiment analysis and HAR-X risk forecasting) provide additional evidence rather than replacing deterministic decision logic.

* **Predictions are probabilistic rather than guarantees.** Sentiment analysis, volatility forecasts, and risk estimates are subject to uncertainty and should be interpreted as decision support, not promises of future market performance.

* Known limitations are disclosed rather than hidden. The current prototype has been evaluated on historical data and demonstration portfolios, **but it has not undergone live-market deployment, regulatory certification, or comprehensive fairness and robustness audits.**

## References

[1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307–327, 1986. *(Introduced the GARCH model, laying the foundation for financial volatility modelling.)*

[2] T. G. Andersen, T. Bollerslev, F. X. Diebold, and P. Labys, "Modeling and forecasting realized volatility," *Econometrica*, vol. 71, no. 2, pp. 579–625, 2003. *(Established realised volatility as a more informative measure for volatility forecasting.)*

[3] F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility," *Journal of Financial Econometrics*, vol. 7, no. 2, pp. 174–196, 2009. *(__Proposed the HAR model, capturing long-memory effects in market volatility and forming the basis of HAR-X.__)*

[4] X. Li, H. Xie, L. Chen, J. Wang, and X. Deng, "News Impact on Stock Price Return via Sentiment Analysis," in *Proc. 2014 Int. Conf. on Cloud Computing and Big Data*, 2014, pp. 1–8. *(Showed that incorporating news sentiment significantly improves stock prediction compared with traditional bag-of-words models.)*

[5] Y. Zhou, S. Liu, and X. Hu, "Trade the Event: Corporate Events Detection for News-Based Stock Prediction," 2021. *(Demonstrated that identifying specific corporate events, such as earnings, mergers, and lawsuits, provides stronger predictive signals than sentiment alone.)*

[6] D. Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models," *arXiv preprint arXiv:1908.10063*, 2019. *(Adapted BERT to financial text, substantially improving financial sentiment classification accuracy.)*

[7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, Minneapolis, MN, USA, 2019, pp. 4171–4186. *(Introduced contextual language representations that became the foundation for FinBERT and modern NLP.)*
