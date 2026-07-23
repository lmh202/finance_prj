# Risk Engine — volatility-based downside risk (per-stock + portfolio)

**Mission:** turn the offline-trained HAR volatility model into decision-ready
risk numbers, online, for a **formula-based** (untrained) decision layer. The
risk output must therefore be high-quality on its own — calibrated, fat-tail
aware, and interpretable.

## Two-stage design (train offline, infer online)

- **Offline (in `scripts/`, already done):**
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
risk_estimates(ohlc_by_symbol, horizons=(5, 20)) -> List[RiskEstimate]
portfolio_risk(ohlc_by_symbol, weights, h) -> Optional[PortfolioRisk]
```

`RiskEstimate` (per symbol × horizon): sigma_daily, sigma_h, var_95, var_99,
es_95, band_lo, band_hi, **risk_level (0-100)**, as_of, has_history.
`PortfolioRisk` (per horizon): sigma_h, var_95, var_99, es_95,
diversification_ratio, as_of.

## How the numbers are formed (the conversion)

```
σ̂_daily = exp(coef · price_features + intercept) · smearing      (HAR)
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

## Validation (proved offline, see `reports/risk_engine/`)

HAR beats naive/GARCH/EGARCH/GBM on QLIKE (DM p<0.01). Per-stock VaR-95 breach
5.4% (target 5%), no clustering (15/16 Christoffersen), ES ratio 0.89. Portfolio
VaR-95 breach 4.1% (Kupiec p=0.26) after the EWMA-correlation fix.

## Rules

- Read-only w.r.t. the shared kernel; imports no other engine.
- The model is retrained offline; do not fit anything at request time.
