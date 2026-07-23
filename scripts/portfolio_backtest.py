"""Equal-weighted top-10 Nasdaq backtest + technical dashboard (strategy system).

A single-file, buy-and-hold-with-rebalancing backtester for an equal-weighted
(10 % each) basket of the ten largest Nasdaq names, over a ~10-year horizon on
yfinance adjusted prices. It reports the headline performance metrics, a per-stock
comparison table, and two matplotlib figures (a 3-panel portfolio dashboard and a
5x2 per-stock price grid).

WHERE THIS SITS
  This is research/strategy tooling, deliberately kept in scripts/ rather than
  inside backend/src/daily_strategy/. It is NOT the frozen contract backtest()
  (that one is Dev 2's walk-forward ML backtest, with a fixed signature in
  interfaces.py). This is a simpler, transparent portfolio-analytics backtest —
  a baseline the ML strategy must beat. It can later feed daily_strategy as a
  benchmark, but it must not overwrite the engine's contract.

WHAT IT COMPUTES
  indicators (portfolio AND each stock): EMA-20, EMA-50, RSI-14 (Wilder),
                                         MACD 12/26/9 (line, signal, histogram)
  metrics:   total return, CAGR, annualized Sharpe (rf=2 %, configurable),
             max drawdown, per-stock table, optional QQQ benchmark

KNOWN LIMITATIONS (surfaced, and partly parameterized for the optimize phase)
  - Survivorship bias: today's top 10 are chosen with hindsight; a stock that
    fell out of the top 10 is not here. Printed as a disclaimer, not corrected.
  - Rebalancing: daily by default (REBALANCE); weekly/monthly/quarterly/none
    are supported to study the drag/turnover trade-off.
  - Transaction costs: modeled as COST_BPS on rebalance turnover; default 0.0
    (i.e. the classic "no costs" assumption) but wired so it can be switched on.
  - Benchmark: QQQ comparison included (BENCHMARK; set None to disable).

Run:  python scripts/portfolio_backtest.py
      python scripts/portfolio_backtest.py --years 10 --rf 0.02 --rebalance M
      python scripts/portfolio_backtest.py --no-benchmark --cost-bps 5
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------- configuration
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
           "META", "AVGO", "TSLA", "COST", "NFLX"]
BENCHMARK = "QQQ"          # set to None to disable the benchmark comparison
YEARS = 10                 # horizon in years
RF_RATE = 0.02             # annual risk-free rate for Sharpe
REBALANCE = "D"            # D / W / M / Q / Y / none
COST_BPS = 0.0             # per-side transaction cost (bps of turnover); 0 = none
TRADING_DAYS = 252

OUT = ROOT / "reports" / "strategy"


# ============================================================ technical indicators
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (EWM with alpha=1/period)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd line, signal line, histogram)."""
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def indicators(s: pd.Series) -> dict:
    """All indicators for one price/value series (portfolio or a single stock)."""
    m_line, m_sig, m_hist = macd(s)
    return dict(ema20=ema(s, 20), ema50=ema(s, 50), rsi=rsi(s),
                macd=m_line, macd_signal=m_sig, macd_hist=m_hist)


# ============================================================ data
def load_prices(tickers, years: int) -> pd.DataFrame:
    """Adjusted daily closes (auto_adjust=True -> 'Close' is adjusted)."""
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True,
                      progress=False, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close[[t for t in tickers if t in close.columns]]
    return close.dropna(how="any").sort_index()


# ============================================================ backtest
def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set:
    if freq == "D":
        return set(dates)
    if freq.lower() in ("none", "bh", "hold"):
        return set()                      # buy-and-hold: never rebalance after t0
    rule = {"W": "W", "M": "M", "Q": "Q", "Y": "A"}[freq.upper()]
    last = pd.Series(dates, index=dates).groupby(dates.to_period(rule)).max()
    return set(last.values)


def backtest(prices: pd.DataFrame, rebalance: str = "D", cost_bps: float = 0.0) -> pd.Series:
    """Equal-weight portfolio value (start=1.0), rebalanced on `rebalance`, with
    `cost_bps` charged on turnover at each rebalance."""
    rets = prices.pct_change().fillna(0.0).values
    dates = prices.index
    n = prices.shape[1]
    w_target = np.full(n, 1.0 / n)
    reb = _rebalance_dates(dates, rebalance)

    alloc = w_target.copy()               # dollar allocation, total = 1.0 at t0
    values = np.empty(len(dates))
    for i, dt in enumerate(dates):
        if i > 0:
            alloc = alloc * (1.0 + rets[i])
        value = alloc.sum()
        if i > 0 and dt in reb:           # rebalance back to equal weight
            target = value * w_target
            turnover = np.abs(target - alloc).sum() / value
            value -= turnover * cost_bps / 1e4
            alloc = value * w_target
        values[i] = value
    return pd.Series(values, index=dates, name="portfolio")


# ============================================================ metrics
def perf_metrics(value: pd.Series, rf: float = RF_RATE, periods: int = TRADING_DAYS) -> dict:
    rets = value.pct_change().dropna()
    years = (value.index[-1] - value.index[0]).days / 365.25
    total = value.iloc[-1] / value.iloc[0] - 1.0
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1.0 / years) - 1.0
    sharpe = (rets.mean() - rf / periods) / rets.std(ddof=1) * np.sqrt(periods)
    mdd = (value / value.cummax() - 1.0).min()
    return dict(total_return=total, cagr=cagr, sharpe=sharpe,
                max_drawdown=mdd, vol_ann=rets.std(ddof=1) * np.sqrt(periods))


def per_stock_table(prices: pd.DataFrame, rf: float) -> pd.DataFrame:
    rows = []
    for t in prices.columns:
        v = prices[t].dropna()
        m = perf_metrics(v, rf)
        m["symbol"] = t
        m["rsi_now"] = rsi(v).iloc[-1]
        rows.append(m)
    df = pd.DataFrame(rows).set_index("symbol")
    return df.sort_values("total_return", ascending=False)


# ============================================================ reporting
def _fmt_pct(x):
    return f"{x*100:>8.1f}%"


def print_report(port_v, port_m, bench_v, bench_m, stock_tbl, cfg):
    print("=" * 76)
    print("EQUAL-WEIGHTED TOP-10 NASDAQ BACKTEST")
    print("=" * 76)
    print(f"  horizon      : {port_v.index[0].date()} -> {port_v.index[-1].date()} "
          f"({(port_v.index[-1]-port_v.index[0]).days/365.25:.1f}y, {len(port_v)} days)")
    print(f"  basket       : {', '.join(cfg['tickers'])}  (equal weight, "
          f"{100/len(cfg['tickers']):.0f}% each)")
    print(f"  rebalance    : {cfg['rebalance']}   cost: {cfg['cost_bps']:.1f} bps   "
          f"rf: {cfg['rf']*100:.1f}%")
    print()
    print(f"  {'':<14} {'total':>10} {'CAGR':>9} {'Sharpe':>8} {'maxDD':>9} {'vol':>8}")
    print("  " + "-" * 62)
    print(f"  {'PORTFOLIO':<14} {_fmt_pct(port_m['total_return'])} {_fmt_pct(port_m['cagr'])} "
          f"{port_m['sharpe']:>8.2f} {_fmt_pct(port_m['max_drawdown'])} {_fmt_pct(port_m['vol_ann'])}")
    if bench_m:
        print(f"  {cfg['benchmark']:<14} {_fmt_pct(bench_m['total_return'])} {_fmt_pct(bench_m['cagr'])} "
              f"{bench_m['sharpe']:>8.2f} {_fmt_pct(bench_m['max_drawdown'])} {_fmt_pct(bench_m['vol_ann'])}")
        edge = port_m['cagr'] - bench_m['cagr']
        print(f"  {'-> vs '+cfg['benchmark']:<14} {'':>10} {edge*100:>+8.1f}% CAGR "
              f"{'(basket wins)' if edge > 0 else '(benchmark wins)'}")

    print("\n  PER-STOCK COMPARISON (sorted by total return)")
    print(f"  {'symbol':<8} {'total':>10} {'CAGR':>9} {'Sharpe':>8} {'maxDD':>9} {'RSI':>6}")
    print("  " + "-" * 56)
    for t, r in stock_tbl.iterrows():
        print(f"  {t:<8} {_fmt_pct(r['total_return'])} {_fmt_pct(r['cagr'])} "
              f"{r['sharpe']:>8.2f} {_fmt_pct(r['max_drawdown'])} {r['rsi_now']:>6.0f}")

    print("\n  DISCLAIMER: survivorship bias — the top-10 are chosen with today's")
    print("  hindsight; names that dropped out of the top 10 are excluded. Daily")
    print("  rebalancing assumed; transaction costs default to zero.")


# ============================================================ plotting
def plot_portfolio(port_v, ind, bench_v, cfg):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1.3]})

    # -- panel 1: value (log) + EMAs + benchmark --
    ax[0].semilogy(port_v.index, port_v, color="#111", lw=1.6, label="Portfolio")
    ax[0].semilogy(port_v.index, ind["ema20"], color="#2e86de", lw=1.0, label="EMA 20")
    ax[0].semilogy(port_v.index, ind["ema50"], color="#e67e22", lw=1.0, label="EMA 50")
    if bench_v is not None:
        bn = bench_v / bench_v.iloc[0] * port_v.iloc[0]
        ax[0].semilogy(bn.index, bn, color="#c0392b", lw=1.1, ls="--",
                       label=f"{cfg['benchmark']} (norm.)")
    ax[0].set_ylabel("growth of $1 (log)")
    ax[0].set_title(f"Equal-weighted top-10 Nasdaq — {port_v.index[0].date()} to "
                    f"{port_v.index[-1].date()}  ·  rebalance={cfg['rebalance']}")
    ax[0].legend(fontsize=8, ncol=2); ax[0].grid(True, which="both", alpha=.15)

    # -- panel 2: RSI --
    ax[1].plot(port_v.index, ind["rsi"], color="#8e44ad", lw=1.0)
    ax[1].axhline(70, color="#c0392b", lw=.7, ls="--")
    ax[1].axhline(30, color="#27ae60", lw=.7, ls="--")
    ax[1].axhline(50, color="#888", lw=.5)
    ax[1].fill_between(port_v.index, 30, 70, color="#8e44ad", alpha=.05)
    ax[1].set_ylabel("RSI-14"); ax[1].set_ylim(0, 100)

    # -- panel 3: MACD --
    ax[2].plot(port_v.index, ind["macd"], color="#2e86de", lw=1.0, label="MACD")
    ax[2].plot(port_v.index, ind["macd_signal"], color="#e67e22", lw=1.0, label="signal")
    colors = np.where(ind["macd_hist"] >= 0, "#27ae60", "#c0392b")
    ax[2].bar(port_v.index, ind["macd_hist"], color=colors, width=2.0, alpha=.5)
    ax[2].axhline(0, color="#888", lw=.5)
    ax[2].set_ylabel("MACD 12/26/9"); ax[2].legend(fontsize=8)
    ax[2].set_xlabel("date")

    fig.tight_layout()
    p = OUT / "fig_portfolio.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_stock_grid(prices, cfg):
    n = prices.shape[1]
    fig, axes = plt.subplots(5, 2, figsize=(13, 16), sharex=True)
    for ax, t in zip(axes.ravel(), prices.columns):
        s = prices[t]
        ax.semilogy(s.index, s, color="#111", lw=1.1)
        ax.semilogy(s.index, ema(s, 20), color="#2e86de", lw=.8, label="EMA20")
        ax.semilogy(s.index, ema(s, 50), color="#e67e22", lw=.8, label="EMA50")
        tot = s.iloc[-1] / s.iloc[0] - 1
        ax.set_title(f"{t}   ({tot*100:+.0f}% total)", fontsize=10)
        ax.grid(True, which="both", alpha=.12)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per-stock adjusted price (log) with EMA-20 / EMA-50", y=1.0, fontsize=13)
    fig.tight_layout()
    p = OUT / "fig_stock_grid.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


# ============================================================ main
def parse_args():
    ap = argparse.ArgumentParser(description="Equal-weighted top-10 Nasdaq backtest")
    ap.add_argument("--years", type=int, default=YEARS)
    ap.add_argument("--rf", type=float, default=RF_RATE)
    ap.add_argument("--rebalance", default=REBALANCE, help="D / W / M / Q / Y / none")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    ap.add_argument("--no-benchmark", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    benchmark = None if a.no_benchmark else BENCHMARK
    cfg = dict(tickers=TICKERS, benchmark=benchmark, rf=a.rf,
               rebalance=a.rebalance, cost_bps=a.cost_bps)
    OUT.mkdir(parents=True, exist_ok=True)

    dl = TICKERS + ([benchmark] if benchmark else [])
    print(f"downloading {len(dl)} symbols from yfinance ({a.years}y adjusted) …")
    prices_all = load_prices(dl, a.years)
    prices = prices_all[TICKERS]
    bench_v = prices_all[benchmark] if benchmark else None

    port_v = backtest(prices, rebalance=a.rebalance, cost_bps=a.cost_bps)
    ind = indicators(port_v)
    port_m = perf_metrics(port_v, a.rf)
    bench_m = perf_metrics(bench_v, a.rf) if bench_v is not None else None
    stock_tbl = per_stock_table(prices, a.rf)

    print_report(port_v, port_m, bench_v, bench_m, stock_tbl, cfg)

    p1 = plot_portfolio(port_v, ind, bench_v, cfg)
    p2 = plot_stock_grid(prices, cfg)

    # persist tidy outputs
    summ = pd.DataFrame({"portfolio": port_m, **({benchmark: bench_m} if bench_m else {})}).T
    summ.to_csv(OUT / "metrics_summary.csv")
    stock_tbl.to_csv(OUT / "per_stock.csv")
    port_v.to_frame().assign(**{k: ind[k] for k in ind}).to_csv(OUT / "portfolio_series.csv")

    print(f"\n  charts -> {p1.relative_to(ROOT)}")
    print(f"           {p2.relative_to(ROOT)}")
    print(f"  tables -> {(OUT/'metrics_summary.csv').relative_to(ROOT)} · "
          f"{(OUT/'per_stock.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
