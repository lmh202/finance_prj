# AURORA Risk Engine — Progress Report

*Status as of 2026-07-22. All numbers below are reproduced from the saved
outputs in `reports/` and the re-run logged this session; nothing is quoted
from memory.*

---

## TL;DR

We set out to predict what news does to a portfolio. The honest finding — arrived
at through pre-registered, model-agnostic tests, not after-the-fact rationalizing —
is a clean division:

1. **Direction is not predictable**, and FinBERT news sentiment does not change that.
2. **Sentiment polarity is *priced-in*** — it echoes moves that already happened.
3. **News *intensity* (volume + disagreement) carries a small, real volatility signal**
   that survives a proper GARCH(1,1) control.
4. **Risk *is* predictable.** A pooled HAR model forecasts realized volatility and
   **beats naïve, GARCH, EGARCH, and gradient-boosted trees significantly** on a
   held-out 2021–2023 test set.
5. **News adds a small but statistically significant increment to the risk forecast —
   but only at the 5-day horizon** (QLIKE −1.4%, p = 1×10⁻⁴), not at 20 days.

The product consequence: **AURORA is a risk engine, not a return predictor.** News
enters the system through the *volatility* channel (setting position-sizing band
width), never as a direction signal multiplying position size.

---

## 1. The research question

The original hope was "news → better trade direction." Before building anything
elaborate, we tested the more fundamental question the model-agnostic way:

> *Is there a detectable relationship between news and subsequent price behaviour,
> in the raw data, for the stocks that actually have news?*

Scope for every news test: the **12 stocks with ≥50 % news coverage**
(`QCOM, MU, GLD, COST, PEP, TXN, ADBE, AMD, NVDA, INTC, SLV, ASML`; ~20.8 k pooled
news-days). No-look-ahead is baked in — news is shifted to the next trading session
before it can predict anything, and all validation is walk-forward (never a random
shuffle).

---

## 2. What we built

| Component | Script | Purpose |
|---|---|---|
| Training dataset | [build_training_dataset.py](../scripts/build_training_dataset.py) | FinBERT scoring → sentiment aggregation (no look-ahead) → price features → merge |
| Direction ablation | [train_ablation.py](../scripts/train_ablation.py) | Walk-forward GBM, price-only vs price+news, ROC-AUC |
| Sentiment validation | [validate_sentiment_signal.py](../scripts/validate_sentiment_signal.py) | 6 pre-registered tests of polarity → returns |
| News-metric validation | [validate_news_metrics.py](../scripts/validate_news_metrics.py) | Volume / disagreement → forward volatility, partial correlation |
| GARCH control | [garch_control_check.py](../scripts/garch_control_check.py) | Re-test the vol signal against a proper GARCH(1,1) baseline |
| **Risk engine** | [train_risk_engine.py](../scripts/train_risk_engine.py) | **Pooled HAR vs 4 benchmarks, honestly split & DM-tested** |
| Worked example | [har_example.py](../scripts/har_example.py) | One-stock, one-day walkthrough + forecast-vs-realized chart |

---

## 3. Findings

### Finding 1 — Direction is not predictable (sentiment null)

Pooled Spearman correlation of FinBERT sentiment vs **forward return** at horizons
h ∈ {1, 3, 5, 10, 20}:

| horizon | Spearman ρ | p | FDR-significant? |
|---|---|---|---|
| 1 | +0.0008 | 0.90 | No |
| 3 | +0.0011 | 0.88 | No |
| 5 | −0.0056 | 0.41 | No |
| 10 | −0.0107 | 0.11 | No |
| 20 | −0.0095 | 0.16 | No |

Every horizon is within noise of zero; the only sub-0.05 raw p-value (Pearson, h=10)
is **negatively** signed — the wrong direction for a real effect. Magnitude, extreme-
sentiment, and sentiment-surprise arms are all null too. With ~21.8 k observations
this is a **well-powered null**, not an underpowered one — a directional effect of
usable size would have shown up.

The GBM direction ablation ([train_ablation.py](../scripts/train_ablation.py)) agrees:
price-only was already ≈ coin-flip out-of-sample, and adding news moved AUC by ≈ 0.

### Finding 2 — Sentiment polarity is priced-in

The same sentiment correlates strongly and positively with the **past** 20-day return:

| scope | ρ (sentiment vs past 20d return) | p |
|---|---|---|
| **POOLED** | **+0.1297** | 9×10⁻⁸³ |
| SLV | +0.216 | 2×10⁻⁸ |
| GLD | +0.201 | 3×10⁻²² |
| NVDA | +0.162 | 1×10⁻¹⁰ |
| ASML | +0.156 | 4×10⁻⁹ |
| … | *(all 12 stocks positive & significant)* | |

This is the mechanism behind Finding 1: FinBERT sentiment **echoes moves that already
happened**. It is backward-looking, so it cannot predict forward returns — and it
directly informs Dev 4's `priced_in` factor (the system must not react to a story
the price has already absorbed). Consistent with Tetlock (2007).

### Finding 3 — News *intensity* predicts volatility (and survives a GARCH control)

Switching the target from direction to **magnitude**, and the predictor from polarity
to **volume / disagreement**, a real signal appears. Partial Spearman controls for
past 20-day realized vol, so the metric only "works" if it predicts future vol
*beyond* what past vol already tells you:

| metric | target | raw ρ | partial ρ | FDR-sig? |
|---|---|---|---|---|
| log news count | vol 5d | +0.181 | **+0.034** | ✅ |
| log news count | vol 20d | +0.199 | **+0.033** | ✅ |
| disagreement | vol 5d | +0.120 | **+0.034** | ✅ |
| disagreement | vol 10d | +0.116 | +0.020 | ✅ |
| disagreement | vol 20d | +0.114 | +0.011 | ✗ (fades) |

`log_count` is FDR-significant at **all four** horizons; `disagreement` at three.
The raw correlations are large but mostly reflect clustering (log_count vs *past*
vol = +0.240) — the partial ρ of ~0.02–0.03 is the honest incremental size. It
**survived replacement of the crude linear control with a proper GARCH(1,1)**
([garch_control_check.py](../scripts/garch_control_check.py)) at roughly half
magnitude — i.e. the signal is real, not residual volatility clustering the linear
control missed. Direction, again, is null for these metrics too.

This replicates Antweiler & Frank (2004) and Audrino, Sigrist & Ballinari (2020):
attention/volume predicts volatility; the effect is real but economically small.

### Finding 4 — Risk IS predictable: HAR beats every benchmark

Pooled HAR on log realized vol (realized vol at 5/22/66d + Parkinson range at 5/22d
+ |rₜ|). Time-split, identical calendar boundaries for all stocks, 20-day embargo at
each seam. **Held-out TEST set (2021–2023):**

**h = 5 days**

| model | QLIKE ↓ | R²(log) ↑ | MZ b | coverage | vs HAR (DM) |
|---|---|---|---|---|---|
| naïve (rv22) | 0.4824 | 0.342 | 0.818 | 0.932 | worse, p=5×10⁻¹² |
| GARCH(1,1) | 0.4632 | 0.331 | 0.904 | 0.935 | worse, p=4×10⁻⁵ |
| EGARCH(1,1,1) | 0.4636 | 0.346 | 0.886 | 0.938 | worse, p=3×10⁻⁵ |
| **HAR** | **0.4294** | **0.460** | 1.086 | 0.925 | — |
| GBM (trees) | 0.4468 | 0.447 | 0.986 | 0.925 | worse, p=0.002 |

**h = 20 days**

| model | QLIKE ↓ | R²(log) ↑ | MZ b | coverage | vs HAR (DM) |
|---|---|---|---|---|---|
| naïve (rv22) | 0.2582 | 0.552 | 0.783 | 0.923 | worse, p=2×10⁻¹⁵ |
| GARCH(1,1) | 0.2362 | 0.549 | 0.863 | 0.934 | worse, p=3×10⁻¹¹ |
| EGARCH(1,1,1) | 0.2220 | 0.573 | 0.861 | 0.934 | worse, p=4×10⁻⁹ |
| **HAR** | **0.1645** | **0.665** | 1.081 | 0.946 | — |
| GBM (trees) | 0.1786 | 0.657 | 0.962 | 0.940 | worse, p=0.010 |

**HAR wins on QLIKE and R²(log) at both horizons, and beats all four benchmarks with
Diebold-Mariano p < 0.01.** Notably, the GBM — which was handed *every* feature
(HAR terms + GARCH + EGARCH + news) — still loses. The bottleneck is **not** model
capacity; the signal is a slow, persistent, near-log-linear component that the
linear model captures as well as anything.

*Caveats worth stating in the write-up:* on the **validation** set at h=5, GARCH
actually edges HAR (DM t=−1.81, p=0.07, not significant) — HAR's dominance is not
uniform across every split. And EGARCH blows up on validation (QLIKE 3×10¹⁰) due to
a numerical collapse on the thin SLV series; the test set is unaffected because SLV
data ends in 2020.

### Finding 5 — News helps the risk forecast, but only at h=5

The cleanest news test in the project: **model class held fixed** (linear vs linear),
same 12 stocks, DM-tested. Features = `log_count`, `disagreement`, `has_news`
(polarity deliberately excluded, per Findings 1–3):

| horizon | HAR QLIKE | HAR+news QLIKE | Δ | DM t | p | verdict |
|---|---|---|---|---|---|---|
| **h=5** | 0.4459 | **0.4399** | −0.0061 | **−3.86** | **1.1×10⁻⁴** | **news helps (sig)** |
| h=20 | 0.1709 | 0.1709 | −0.0000 | −0.11 | 0.91 | no gain |

A −1.4 % relative QLIKE improvement at h=5, p ≈ 10⁻⁴. Small, but that is the honest
size of a real news effect — and the h=5-only pattern is mechanistically sensible:
a burst of headlines today informs *next week's* volatility, not next month's.

---

## 4. Honest limitations

1. **Selection bias on the news features.** `log_count` and `disagreement` were
   chosen using `validate_news_metrics.py`, whose correlations were computed over the
   **full 2013–2023 sample, including the test period**. The DM test itself is clean
   (models fit on train only), but the *feature choice* saw test data. **Action:**
   re-run the screen on train+val (≤2020) only and confirm the same two features win.
2. **Coverage / calibration is a short-horizon problem.** HAR's point forecasts are
   well-calibrated (MZ slope ≈ 1), but the ±1.96σ interval is slightly too narrow at
   h=5 (coverage 0.925 vs 0.95 target; NVDA 0.901). At h=20 it's already at target
   (0.946). This is a fat-tail *distribution* issue, not a level issue.
3. **The news increment is print-only.** `train_risk_engine.py` prints the increment
   but does not persist it to `risk_engine_results.csv` — it must be re-run to
   reproduce. **Action:** write it to disk.
4. **EGARCH numerical fragility** on thin series (SLV) needs a guard before the
   validation table is presentable.

---

## 5. Recommended next steps

**In priority order.**

1. **Close the selection-bias hole (cheap, do first).** Re-run the news-feature
   screen restricted to ≤2020 and confirm `log_count` + `disagreement` still come out
   on top. Converts the best news result from "suggestive" to "clean."

2. **Build the self-built 4th model, "HAR-X", targeting h=5** (plan in
   [plan.md](../plan.md)). Findings 2 and 5 converge on the same target: news helps at
   h=5, and the coverage gap is at h=5. Components:
   - a **leverage / asymmetry term** (down-days lift future vol more — EGARCH's sound
     intuition without its numerical fragility);
   - a **Student-t interval band** fit on validation (fixes the h=5 under-coverage
     without any new predictor — likely the single biggest usable win);
   - the already-validated **news features** (`log_count`, `disagreement`).
   Acceptance criterion is pre-registered in `plan.md`; if it fails, report the null.

3. **Wire the risk engine into the strategy engine.** The forecast feeds
   inverse-volatility position sizing (Moreira & Muir 2017): `weight ∝ 1 / σ̂√h`.
   News reaches sizing **only** through σ̂ (heavier news flow → wider band → smaller
   position, regardless of polarity), never as a direction multiplier.

4. **Persist the news increment** to CSV and add the EGARCH numerical guard, so the
   full results table is reproducible in one run.

---

*Scripts: `scripts/`. Raw outputs: `reports/{risk_engine,news_metrics,sentiment_validation}/`.
Design plan for the next model: `plan.md`.*
