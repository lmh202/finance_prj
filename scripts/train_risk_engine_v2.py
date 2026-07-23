"""HAR-X — the self-built risk model, trained HERE for backend online inference.

Trains on 2013-2018, SELECTS components on 2019-2020 validation under a disciplined
minimum-gain rule, tests ONCE on 2021-2023 against the three existing baselines
(HAR / GARCH(1,1) / EGARCH), and SERIALIZES the fitted linear model to
data/processed/risk_model.json.

HONEST FINDING (see the run): HAR is extremely hard to beat. Candidate extensions
— a leverage term (downside vol) and news features — were tested and REJECTED:
leverage gave only a sub-1% validation gain that reversed out-of-sample; pooled
news does not lower validation QLIKE. So the deployed model is HAR itself, and the
self-built contribution ("HAR-X") is what vanilla HAR lacks: an EMPIRICALLY
CALIBRATED risk band (HAR's Gaussian ±1.96σ under-covers at h=5; the calibrated
band hits ~95%). That band — not the point forecast — is what the decision layer
needs (it sizes positions and standardizes news impact).

Why linear/HAR and not something fancier: gradient-boosted trees handed every
feature still LOSE to HAR out-of-sample. The bonus: inference is a dot product on
price features, no per-stock refitting — the backend scores any stock from ~70d of
OHLC.  A component is adopted ONLY if it cuts validation QLIKE by >= MIN_VAL_GAIN.

Run: python scripts/train_risk_engine_v2.py
"""

import json
import sys
import warnings
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from build_training_dataset import trading_days
from train_risk_engine import (EMBARGO, FEATS_BASE, HORIZONS, TRAIN_END, VAL_END,
                               build_stock, dm_test, evaluate, log_feats, qlike,
                               stocks)
from validate_news_metrics import news_features

PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "risk_engine"
MODEL_PATH = ROOT / "data" / "processed" / "risk_model.json"
NEWS_FEATS = ["log_count", "disagreement", "has_news"]
MIN_VAL_GAIN = 0.01   # a component must cut val QLIKE by >=1% to be adopted (else it's val noise)
EPS = 1e-6


# ----------------------------------------------------------- extra features
def leverage_feature(sym: str) -> pd.DataFrame:
    """Downside realized vol over 5d (semivariance) — the candidate leverage term.
    Past returns only (no look-ahead)."""
    px = pd.read_csv(PRICES / f"{sym}.csv", usecols=["date", "adj close"])
    px["date"] = pd.to_datetime(px["date"])
    c = px.sort_values("date").set_index("date")["adj close"]
    ret = c.pct_change()
    down = ret.where(ret < 0, 0.0)
    rv5_down = np.sqrt((down ** 2).rolling(5).mean())
    return pd.DataFrame({"date": c.index, "symbol": sym, "rv5_down": rv5_down.values})


def build_panel() -> pd.DataFrame:
    print("building panel (GARCH/EGARCH fit on TRAIN only; ~few min) …")
    panel = log_feats(pd.concat([build_stock(s) for s in stocks()], ignore_index=True))
    lev = pd.concat([leverage_feature(s) for s in stocks()], ignore_index=True)
    panel = panel.merge(lev, on=["symbol", "date"], how="left")
    panel["l_rv5_down"] = np.log(panel["rv5_down"] + EPS)
    nf = news_features(trading_days())[["symbol", "date", "log_count", "disagreement"]]
    panel = panel.merge(nf, on=["symbol", "date"], how="left")
    panel["has_news"] = panel["log_count"].notna().astype(int)
    panel[["log_count", "disagreement"]] = panel[["log_count", "disagreement"]].fillna(0.0)
    return panel


# ------------------------------------------------------------ fit / predict
def fit_har(tr, va, feats):
    m = LinearRegression().fit(tr[feats], tr["ly"])
    smear = float(np.exp(np.var(va["ly"] - m.predict(va[feats]), ddof=1) / 2))
    return m, smear


def predict(m, smear, d, feats):
    return np.exp(m.predict(d[feats])) * smear


def calibrate_band(fc_val, fret_val, h):
    """95% two-sided band multiplier k: P(|return| <= k·σ̂·√h)=0.95 on validation;
    also a Student-t dof for reporting."""
    z = np.asarray(fret_val, float) / (np.asarray(fc_val, float) * np.sqrt(h))
    z = z[np.isfinite(z)]
    k = float(np.quantile(np.abs(z), 0.95))
    try:
        dof = float(stats.t.fit(z, floc=0)[0])
    except Exception:
        dof = float("nan")
    return k, dof


def coverage(fc, fret, h, k):
    m = np.isfinite(fc) & np.isfinite(fret) & (fc > 0)
    return float(np.mean(np.abs(fret[m]) <= k * fc[m] * np.sqrt(h)))


# ------------------------------------------------------------------- main
def main():
    panel = build_panel()
    serial = {"model": "HAR-X (HAR point forecast + empirically calibrated band)",
              "created": date.today().isoformat(),
              "train_end": TRAIN_END, "val_end": VAL_END,
              "min_val_gain_to_adopt": MIN_VAL_GAIN,
              "feature_spec": {"rv_windows": [5, 22, 66], "park_windows": [5, 22],
                               "inputs": FEATS_BASE, "note": "needs ~70d daily OHLC per symbol"},
              "rejected_components": {},
              "horizons": {}}
    all_rows = []
    emb = pd.Timedelta(days=int(EMBARGO * 1.5))

    for h in HORIZONS:
        y, g, eg = f"y_{h}", f"garch_{h}", f"egarch_{h}"
        need = FEATS_BASE + ["l_rv5_down", y, g, eg, f"fret_{h}"]
        d = panel.dropna(subset=need).copy()
        d["ly"] = np.log(d[y] + EPS)
        tr = d[d.date <= pd.Timestamp(TRAIN_END) - emb]
        va = d[(d.date > TRAIN_END) & (d.date <= pd.Timestamp(VAL_END) - emb)]
        te = d[d.date > VAL_END]
        print(f"\n{'='*78}\nHORIZON h={h}d   train {len(tr):,} | val {len(va):,} | test {len(te):,}")

        # ---- component ablation on VALIDATION (adopt only if >= MIN_VAL_GAIN vs base) ----
        cands = {"HAR base": FEATS_BASE, "+leverage": FEATS_BASE + ["l_rv5_down"]}
        if h == 5:
            cands["+news"] = FEATS_BASE + NEWS_FEATS
            cands["+lev+news"] = FEATS_BASE + ["l_rv5_down"] + NEWS_FEATS
        val_q = {n: qlike(va[y].to_numpy(), predict(*fit_har(tr, va, f), va, f))
                 for n, f in cands.items()}
        base_q = val_q["HAR base"]
        adopted = [n for n in cands if n != "HAR base" and val_q[n] <= base_q * (1 - MIN_VAL_GAIN)]
        best = min(adopted, key=val_q.get) if adopted else "HAR base"
        print("  val QLIKE:  " + "  ".join(
            f"{n} {q:.4f}{'*' if n == best else ''}" for n, q in val_q.items()))
        print(f"  -> deploy '{best}'  (need >={MIN_VAL_GAIN:.0%} val gain to add a component; "
              f"best non-base gain = {(base_q-min(val_q[n] for n in cands if n!='HAR base'))/base_q:+.2%})")
        serial["rejected_components"][str(h)] = [n for n in cands if n not in ("HAR base", best)]

        deploy_feats = cands[best]
        m_dep, s_dep = fit_har(tr, va, deploy_feats)
        k, dof = calibrate_band(predict(m_dep, s_dep, va, deploy_feats),
                                va[f"fret_{h}"].to_numpy(), h)

        # ---- single TEST evaluation: 3 baselines + the deployed HAR-X, plus the
        #      rejected leverage extension (shown for honesty) ----
        fr, yv = te[f"fret_{h}"].to_numpy(), te[y].to_numpy()
        m_lev, s_lev = fit_har(tr, va, FEATS_BASE + ["l_rv5_down"])
        cand_fc = {
            "naive (rv22)": te["rv22"].to_numpy(),
            "GARCH(1,1)": te[g].to_numpy(),
            "EGARCH(1,1,1)": te[eg].to_numpy(),
            "HAR-X (deployed)": predict(m_dep, s_dep, te, deploy_feats),
            "HAR+lev (rejected)": predict(m_lev, s_lev, te, FEATS_BASE + ["l_rv5_down"]),
        }
        losses, rows = {}, []
        for nm, fc in cand_fc.items():
            row, ls = evaluate(nm, yv, fc, fr, h)
            row["split"], row["horizon"] = "TEST", h
            rows.append(row); losses[nm] = ls
        res = pd.DataFrame(rows); all_rows.append(res)

        print(f"\n  [TEST] h={h}d — lower QLIKE better; MZ ideal a=0 b=1")
        print(f"  {'model':>20} | {'QLIKE':>7} | {'R2_log':>7} | {'MZ a':>6} {'MZ b':>6}")
        print("  " + "-" * 58)
        for _, r in res.iterrows():
            print(f"  {r.model:>20} | {r.qlike:>7.4f} | {r.r2_log:>7.3f} | {r.mz_a:>6.3f} {r.mz_b:>6.3f}")
        for other in ["naive (rv22)", "GARCH(1,1)", "EGARCH(1,1,1)", "HAR+lev (rejected)"]:
            t, p = dm_test(losses["HAR-X (deployed)"], losses[other], lag=h)
            verd = "HAR-X better" if t < 0 else "HAR-X worse"
            print(f"    DM  HAR-X vs {other:<18}: t={t:+.2f} p={p:.3g} -> {verd}"
                  f"{' (sig)' if p < 0.05 else ''}")

        # ---- the calibrated band: the actual decision-layer payoff ----
        fc_dep = cand_fc["HAR-X (deployed)"]
        cov_g, cov_c = coverage(fc_dep, fr, h, 1.96), coverage(fc_dep, fr, h, k)
        print(f"\n  RISK BAND  Gaussian k=1.96 -> coverage {cov_g:.3f}  |  "
              f"calibrated k={k:.2f} (t-dof≈{dof:.1f}) -> coverage {cov_c:.3f}  (target 0.95)")

        serial["horizons"][str(h)] = {
            "features": deploy_feats,
            "coef": {f: float(c) for f, c in zip(deploy_feats, m_dep.coef_)},
            "intercept": float(m_dep.intercept_),
            "smearing": s_dep, "band_k": k, "t_dof": dof,
            "test_qlike": float(res.set_index("model").loc["HAR-X (deployed)", "qlike"]),
            "test_r2_log": float(res.set_index("model").loc["HAR-X (deployed)", "r2_log"]),
            "test_coverage_gauss": cov_g, "test_coverage_calibrated": cov_c,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows).to_csv(OUT / "risk_engine_v2_results.csv", index=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(serial, indent=2))
    print(f"\n{'='*78}\nserialized HAR-X -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"results          -> {(OUT/'risk_engine_v2_results.csv').relative_to(ROOT)}")
    _inference_demo(panel, serial)


# ----------------------------------------------- backend-ready inference core
def har_x_forecast(feat_row: dict, model_h: dict, h: int) -> dict:
    """Pure inference: feature row + one horizon's serialized model -> vol forecast
    and calibrated risk band. Exactly what the backend calls online (a dot product)."""
    z = model_h["intercept"] + sum(model_h["coef"][f] * feat_row[f] for f in model_h["features"])
    sigma_daily = float(np.exp(z) * model_h["smearing"])
    sigma_h = sigma_daily * np.sqrt(h)
    return {"sigma_daily": sigma_daily, "sigma_h": sigma_h,
            "band_95": model_h["band_k"] * sigma_h}


def _inference_demo(panel, serial):
    print("\nINFERENCE DEMO (what the backend does online, from the serialized JSON):")
    for h in HORIZONS:
        mh = serial["horizons"][str(h)]
        row = (panel.dropna(subset=mh["features"])
               .query("symbol == 'NVDA'").sort_values("date").iloc[-1])
        fc = har_x_forecast(row[mh["features"]].to_dict(), mh, h)
        print(f"  NVDA h={h}d on {row.date.date()}: σ̂={fc['sigma_daily']*100:.2f}%/day, "
              f"σ̂_{h}d={fc['sigma_h']*100:.1f}%, 95% band ±{fc['band_95']*100:.1f}% over {h}d")


if __name__ == "__main__":
    main()
