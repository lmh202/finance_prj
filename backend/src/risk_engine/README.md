# Risk Engine — volatility-based downside risk (per-stock + portfolio)

**Mission:** turn price history and causal news attention into decision-ready
risk numbers, online, for a **formula-based** (untrained) decision layer. The
risk output must therefore be high-quality on its own — calibrated, fat-tail
aware, and interpretable.

## Two-stage design (train offline, infer online)

- **Offline (in `scripts/`, already done):**
  - `optimize_risk_engine.py` — runs the embargoed 2018–2023 nested
    walk-forward ceiling search for 5/20-session realised volatility. It
    writes a multi-artifact candidate to
    `data/processed/risk_model_candidate/` and diagnostics to
    `reports/risk_engine_optimization/`; it never replaces the schema-v1
    checkpoint without component gates and backend parity.
  - `train_risk_engine_v2.py` — fits pooled HAR on log realized vol, selects
    components under a min-gain ablation (leverage/news rejected OOS), and
    serializes coefficients to `data/processed/risk_model.json`.
  - `risk_measures.py` — builds the Filtered-Historical-Simulation tail
    (empirical standardized-return quantiles, ≤2020) for VaR/ES/band, plus the
    portfolio EWMA-covariance tail, and backtests them (Kupiec / Christoffersen).
    Writes the `fhs` block into `risk_model.json`.
- **Online (this module):** pure inference — a dot product for σ̂ + a table
  lookup for the FHS quantiles. No training, no per-stock refitting.

## Public functions (`engine.py`)

```python
model_available() -> bool
risk_estimate(symbol, ohlc, h) -> RiskEstimate
risk_estimates(ohlc_by_symbol, horizons=(5,)) -> List[RiskEstimate]
portfolio_risk(ohlc_by_symbol, weights, h) -> Optional[PortfolioRisk]
```

The formal five-session output is the integrated HAR-News model. Every call
uses a news-attention input: callers may provide it explicitly, otherwise the
engine causally derives `log_count` from `data/news_raw.json`. An observed zero
count is a real news state and still passes through the joint model. The
twenty-session price-only estimate remains available only as an explicit
auxiliary diagnostic because news did not improve it out of sample.

Run the complete deterministic search from the repository root with:

```powershell
python scripts/optimize_risk_engine.py
```

`RiskEstimate` (per symbol × horizon): sigma_daily, sigma_h, var_95, var_99,
es_95, band_lo, band_hi, **risk_level (0-100)**, as_of, has_history,
model_version, news_applied, news_quality.
`PortfolioRisk` (per horizon): sigma_h, var_95, var_99, es_95,
diversification_ratio, as_of.

## How the numbers are formed (the conversion)

```
σ̂_price = exp(coef · price_features + intercept) · smearing
news_ratio = exp(news_coef · scaled_log_count)
σ̂_daily = σ̂_price · sqrt(news_ratio)       (formal HAR-News model)
σ̂_h     = σ̂_daily · √h
VaR_α    = σ̂_h · Q_α(z)          Q_α = empirical quantile of standardized returns
ES_α     = σ̂_h · mean(z | z≤Q_α)  (fat-tailed, left-skewed — no Gaussian assumption)
risk_level = percentile of current σ̂_h vs the stock's own history  (0-100)
Portfolio: σ_p = √(wᵀ · diag(σ̂) · R_ewma · diag(σ̂) · w),  VaR = σ_p·√h · Q_α^port
```

## Inputs / dependencies

- OHLC history from the shared kernel: `data_loader.get_ohlc_history(symbols)`
  (the Parkinson feature needs high/low — `get_history` is close-only).
- The artifact `data/processed/risk_model.json`. If it is missing,
  `model_available()` is False and the router returns a `no_model` marker.

## Consumers

- `routers/risk.py` → `GET /risk/estimates`, `GET /risk/portfolio`.
- The recommendation engine can feed `risk_level / 100` into its §7
  `market_volatility` factor (replacing the current binary regime proxy).

## Validation (proved offline)

Against the same price HAR base, adding causal news attention improves
five-session OOF QLIKE by 5.24% over 2018–2023, with 95% moving-block
bootstrap interval +2.38% to +8.83% and DM p=0.0011. It improves 5/6 years,
85.7% of stocks, and high-volatility QLIKE by 13.56%. VaR-95 breach is 4.32%,
95% band coverage 96.34%, and ES ratio 0.983.

## Rules

- Read-only w.r.t. the shared kernel; imports no other engine.
- The model is retrained offline; do not fit anything at request time.
