/* =========================================================================
 * figures.js — every number the AURORA poster prints, in one object.
 *
 * PROVENANCE. Nothing here is estimated or rounded by eye. Each block names
 * the file in the repo it was read out of, on 2026-07-26:
 *
 *   corpus    data/processed/fnspid_top20_2013_2023.csv
 *             data/news_raw.json
 *   engine    backend/src/rule_fusion/README.md + scripts/fusion_selfcheck.py
 *   ledger    reports/decision_layer_ledger_fusion/{report.md,manifest.json}
 *   repair    backend/src/rule_fusion/README.md, "symmetry repair" section
 *   risk      reports/risk_engine_presentation/model_comparison_report.md
 *
 * These are frozen snapshots, not a live view. Re-running any of those studies
 * changes the source and NOT this file — re-syncing is manual.
 *
 * poster.html ALSO hardcodes the same values inline as fallback text, on
 * purpose: if this script fails to load, the poster degrades to a correct but
 * static page rather than a page full of blanks. CHANGE A NUMBER HERE AND
 * CHANGE IT IN poster.html TOO.
 * ========================================================================= */
window.FIGURES = {
  /* ---- The flood the product exists to absorb ------------------------- */
  corpus_headlines: 74073, // FNSPID top-20 slice, 2013-01-02 .. 2023-12-29
  corpus_symbols: 21,
  corpus_years: 11,
  rss_cached: 1252, // data/news_raw.json, the live RSS cache

  /* ---- The engine itself ---------------------------------------------- */
  n_signals: 4,
  n_steps: 4,
  n_engines: 6, // backend/src/, four original + risk_engine + rule_fusion
  n_invariants: 5,
  selfcheck_cases: 8064, // largest grid in scripts/fusion_selfcheck.py (INV-4/5)
  n_tests: 48, // pytest tests/
  critical_day_pct: 0.55, // % of symbol-days with a critical headline in trailing 5

  /* ---- Outcome: the staged ledger, ablated ----------------------------
     Locked test 2021-01-01 .. 2023-12-31, 21 symbols, 25 bps, rebalance every
     5 sessions, 732 sessions. `sharpe` is the ranked quantity; `cash` is the
     average idle weight, which is what the risk step actually moves.
     ------------------------------------------------------------------- */
  ledger_period: "2021-2023 locked test",
  benchmark_cagr: 0.0846,
  ledger: [
    { name: "Strategy only", sharpe: 0.649, cagr: 0.1477, cash: 0.0253, kind: "ref" },
    { name: "Equal weight", sharpe: 0.5922, cagr: 0.1271, cash: 0.0447, kind: "ref" },
    { name: "Staged ledger, full", sharpe: 0.5795, cagr: 0.1056, cash: 0.1451, kind: "full" },
    { name: "…without health", sharpe: 0.5819, cagr: 0.1059, cash: 0.1504, kind: "abl" },
    { name: "…without news", sharpe: 0.5645, cagr: 0.1011, cash: 0.1552, kind: "abl" },
    { name: "…without risk", sharpe: 0.4251, cagr: 0.0689, cash: 0.1831, kind: "abl-bad" },
  ],

  /* ---- Outcome: the Step-1 symmetry repair ----------------------------
     Reading daily_strategy's HOLD band literally produced a ~13:1 sell/buy
     ratchet that drained the backtested book to cash. Same rule table, one
     mirrored threshold, measured on the same panel.
     ------------------------------------------------------------------- */
  repair: {
    before: { ratio: 13.0, cash: 0.83, cagr: 0.024, sharpe: 0.36 },
    after: { ratio: 0.84, cash: 0.145, cagr: 0.106, sharpe: 0.58 },
  },
  raw_hold_pct: 97, // share of 2018-2023 symbol-days labelled raw HOLD

  /* ---- Accuracy: the one learned model in the stack --------------------
     Future 5-session realised volatility. Expanding annual walk-forward with
     a 5-session embargo, stock-equal QLIKE, 27,027 stock-date forecasts,
     outer test years 2018-2023. Lower QLIKE is better.
     ------------------------------------------------------------------- */
  n_forecasts: 27027,
  risk_models: [
    { name: "XGBoost Gamma", qlike: 0.657, gain: 0.1031, calib: 0.892, dm: 0.001, live: false },
    { name: "HAR-X + News", qlike: 0.694, gain: 0.0524, calib: 0.981, dm: 0.0011, live: true },
    { name: "Residual MLP", qlike: 0.71, gain: 0.031, calib: 0.913, dm: 0.8394, live: false },
    { name: "HAR-X", qlike: 0.733, gain: 0.0, calib: 1.057, dm: null, live: false },
  ],
  risk_news_gain: 0.0524,
  risk_news_ci: [0.0238, 0.0883],
  risk_news_dm_p: 0.001087,
  risk_years_won: [5, 6],
  risk_stocks_won: 0.857,
  risk_highvol_gain: 0.1356,
  /* Calibration is the reason HAR-X + News ships and XGBoost does not: the
     shipped model is the one whose intervals are honest, not the one with the
     lowest point-forecast error. */
  calibration: {
    var_breach: 0.0432, // realised VaR-95 breach rate; nominal 5%
    var_nominal: 0.05,
    band_cov: 0.9634, // realised 95% risk-band coverage; nominal 95%
    band_nominal: 0.95,
    es_ratio: 0.983, // realised / expected shortfall; 1.0 is perfect
  },
};
