# AURORA System Architecture

## **AURORA — AI-Powered Portfolio Intelligence Copilot**

AURORA should have **three intelligence engines** connected to one recommendation layer:

1. **Portfolio Intelligence Engine** — monitors the user’s portfolio daily
2. **Market Strategy Engine** — identifies market conditions and generates normal-day recommendations
3. **Event Intelligence Engine** — filters essential news and evaluates whether reacting is risky

---

## 1. High-Level Architecture

```text
 ┌───────────────────────────────────────────────────────────────┐
 │                         DATA SOURCES                          │
 ├─────────────────────┬─────────────────────┬───────────────────┤
 │ Market Data         │ Portfolio Data      │ News Data         │
 │                     │                     │                   │
 │ Stocks              │ User holdings       │ Global news       │
 │ ETFs                │ Quantities          │ Financial news    │
 │ Gold                │ Purchase prices     │ Economic news     │
 │ Silver              │ Current weights     │ Geopolitical news │
 │ Benchmark index     │ Sample portfolio    │ Company news      │
 └──────────┬──────────┴──────────┬──────────┴─────────┬─────────┘
            │                     │                    │
            ▼                     ▼                    ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                    DATA PROCESSING LAYER                      │
 │                                                               │
 │ Price cleaning          Portfolio validation                  │
 │ Missing-value handling  News deduplication                    │
 │ Return calculation      Timestamp alignment                   │
 │ Feature engineering     Asset and sector mapping              │
 └──────────┬─────────────────────┬────────────────────┬──────────┘
            │                     │                    │
            ▼                     ▼                    ▼
 ┌───────────────────┐ ┌────────────────────┐ ┌───────────────────┐
 │ PORTFOLIO ENGINE  │ │ MARKET STRATEGY    │ │ EVENT ENGINE      │
 │                   │ │ ENGINE             │ │                   │
 │ Health score      │ │ Regime detection   │ │ Essential news    │
 │ Sharpe ratio      │ │ Opportunity score  │ │ Event category    │
 │ Sortino ratio     │ │ Technical signals  │ │ Sentiment         │
 │ Drawdown          │ │ Daily strategy     │ │ Relevance         │
 │ Volatility        │ │ Allocation signal  │ │ Asset impact      │
 │ Diversification   │ │                    │ │ Reaction risk     │
 └─────────┬─────────┘ └──────────┬─────────┘ └─────────┬─────────┘
           │                      │                     │
           └──────────────────────┼─────────────────────┘
                                  ▼
                  ┌────────────────────────────────┐
                  │ RECOMMENDATION & RISK ENGINE   │
                  │                                │
                  │ Combine daily and event scores │
                  │ Apply portfolio constraints    │
                  │ Compare possible actions       │
                  │ Calculate confidence           │
                  │ Generate explanations          │
                  └───────────────┬────────────────┘
                                  ▼
                  ┌────────────────────────────────┐
                  │       STREAMLIT DASHBOARD      │
                  │                                │
                  │ Portfolio overview             │
                  │ Portfolio health               │
                  │ Market regime                  │
                  │ Daily recommendations          │
                  │ Essential news                 │
                  │ Risk of reacting               │
                  │ Suggested allocation           │
                  │ Benchmark comparison           │
                  └────────────────────────────────┘
```

---

# 2. Data Sources

## A. Market Data

Use daily historical prices for:

* Stocks
* ETFs
* Gold
* Silver
* Broad-market benchmark

For the prototype, your universe could contain approximately:

* 6–8 stocks
* 4–6 ETFs
* Gold
* Silver
* One benchmark such as the S&P 500

Possible data fields:

```text
Date
Open
High
Low
Close
Adjusted Close
Volume
```

For commodities, you can use tradable proxies such as:

* Gold ETF
* Silver ETF

This is easier than handling physical spot-market data.

---

## B. Portfolio Data

Initially, use the sample portfolio provided by your team.

Example:

```text
Asset       Category       Current Weight

AAPL        Stock               12%
MSFT        Stock               10%
SPY         ETF                 20%
XLV         ETF                 12%
Gold ETF    Commodity           15%
Silver ETF  Commodity            8%
Cash        Cash                23%
```

Later, the dashboard can allow a user to:

* Select an asset
* Enter quantity
* Enter purchase price
* Upload portfolio CSV

For the capstone, the sample portfolio is enough.

---

## C. News Data

The news collector gathers:

* Global economic news
* Geopolitical news
* Interest-rate news
* Inflation news
* Commodity news
* Company-specific news
* Regulatory news

The news engine should not display everything. It should filter the feed into a maximum of approximately **five essential events per day**.

---

# 3. Data Processing Layer

This layer prepares all raw data before analysis.

## Market-data processing

```text
Download prices
      ↓
Sort by date
      ↓
Handle missing values
      ↓
Calculate daily returns
      ↓
Calculate rolling indicators
      ↓
Align all assets to common dates
```

Features can include:

* Daily and cumulative returns
* 20-day volatility
* 20-day momentum
* 50-day SMA
* 200-day SMA
* RSI
* Maximum drawdown
* Asset correlations
* Beta against benchmark

---

## News-data processing

```text
Collect recent articles
       ↓
Remove duplicate headlines
       ↓
Remove irrelevant stories
       ↓
Classify the event
       ↓
Measure sentiment and importance
       ↓
Map the event to affected assets
```

Example event categories:

```text
Interest rates
Inflation
Economic growth
Geopolitical conflict
Energy and commodities
Technology regulation
Corporate earnings
Financial-system stress
```

---

# 4. Engine One: Portfolio Intelligence

This engine evaluates the portfolio itself.

It answers:

> “How healthy is the user’s current portfolio?”

## Inputs

* Portfolio holdings
* Current asset prices
* Historical returns
* Benchmark returns

## Calculations

* Annualized return
* Annualized volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Beta
* Asset correlation
* Sector concentration
* Single-asset concentration
* Commodity and defensive exposure

---

## Portfolio Health Score

Create one score from 0 to 100.

Example structure:

```text
Risk-adjusted return       25%
Diversification            20%
Drawdown control           20%
Volatility                 15%
Concentration              10%
Benchmark performance      10%
```

Example output:

```text
Portfolio Health: 78/100

Strengths:
• Good diversification
• Moderate drawdown
• Positive Sharpe ratio

Weaknesses:
• Technology concentration is high
• Silver has increased portfolio volatility
```

The health score should be a communication tool, while the dashboard also shows the underlying metrics.

---

# 5. Engine Two: Daily Market Strategy

This is the dimension that keeps AURORA useful even when there is no important news.

It answers:

> “Given normal market conditions, should the portfolio allocation change?”

## Recommended daily strategy: Regime-Aware Momentum

This is manageable within two weeks and provides a clear strategy.

### Daily indicators

Use around five indicators:

1. **Short-term momentum**
2. **Price relative to SMA50**
3. **SMA50 relative to SMA200**
4. **Rolling volatility**
5. **Sharpe or risk-adjusted momentum**

Optional:

* RSI
* Correlation with benchmark
* Recent drawdown

---

## Market regime classification

AURORA can classify the market into four regimes:

### Bullish

```text
Benchmark above SMA50
SMA50 above SMA200
Positive momentum
Moderate volatility
```

### Bearish

```text
Benchmark below SMA50
SMA50 below SMA200
Negative momentum
Elevated drawdown
```

### High-volatility

```text
Rolling volatility above historical threshold
Large daily price movements
Increasing asset correlation
```

### Sideways or uncertain

```text
Weak momentum
Mixed moving-average signals
Moderate volatility
```

---

## Strategy response

| Market regime   | Portfolio behaviour                                              |
| --------------- | ---------------------------------------------------------------- |
| Bullish         | Increase strong-momentum stocks and growth ETFs                  |
| Bearish         | Reduce risky positions and increase gold, defensive ETFs or cash |
| High volatility | Lower concentration and increase defensive allocation            |
| Sideways        | Maintain balanced allocation and reduce unnecessary turnover     |

---

## Daily asset score

Each asset receives a score:

```text
Asset Score =
30% Momentum
+ 25% Trend
+ 20% Sharpe Ratio
- 15% Volatility
- 10% Drawdown
```

The score is then used to:

* Rank assets
* Identify weakening positions
* Detect opportunities
* Recommend small weight adjustments

Example:

```text
Gold ETF
Daily score: 81/100
Signal: Increase slightly

AAPL
Daily score: 59/100
Signal: Hold

Silver ETF
Daily score: 42/100
Signal: Reduce
```

---

# 6. Engine Three: Event Intelligence

This engine is activated when important news appears.

It answers:

> “What happened, which holdings may be affected, and how risky would it be to react?”

## News filtering pipeline

```text
Raw news
   ↓
Relevance filter
   ↓
Duplicate removal
   ↓
Event classification
   ↓
Importance score
   ↓
Portfolio exposure mapping
   ↓
Reaction-risk calculation
```

---

## Essential News Score

Each story can receive an importance score based on:

```text
Source credibility
Number of sources reporting it
Portfolio relevance
Event severity
Recency
Expected market impact
```

Only high-scoring stories appear under **Essential News**.

---

## Asset-impact mapping

Example:

```text
News:
Unexpected increase in interest rates

Potential impact:
Technology stocks        Negative
Banking stocks           Mixed/positive
Long-duration ETFs       Negative
Gold                     Mixed
Cash                     Positive defensively
```

The user should see the logic rather than only a buy or sell instruction.

---

# 7. Reaction-Risk Engine

This is one of AURORA’s strongest differentiators.

It calculates:

> “How risky is it to change the portfolio because of this event?”

## Reaction-risk factors

### 1. News uncertainty

Is the report confirmed by several credible sources?

### 2. Market already moved

Has the affected asset already reacted strongly?

### 3. Technical disagreement

Does the news recommendation contradict existing market signals?

### 4. Current volatility

Are market conditions unusually unstable?

### 5. Portfolio concentration

Would the proposed trade increase concentration?

### 6. Event relevance

Does the event directly affect the user’s holdings?

---

## Example formula

```text
Reaction Risk =
25% News uncertainty
+ 20% Technical disagreement
+ 20% Market volatility
+ 15% Already-priced-in risk
+ 10% Concentration impact
+ 10% Event ambiguity
```

Output:

```text
Risk of reacting: 67%

Reasons:
• Technology prices already fell by 5%
• Only two sources confirm the event
• Existing trend indicators are mixed
• The portfolio is already heavily exposed to technology

Suggested response:
Wait for additional confirmation
```

---

# 8. Recommendation Engine

This is where the three engines come together.

```text
Portfolio health
        +
Daily asset scores
        +
Market regime
        +
Event impact
        +
Reaction risk
        ↓
Final recommendation
```

There should be two types of recommendation.

## A. Normal daily recommendation

Example:

```text
Suggested normal-day adjustment:

Reduce AAPL by 2%
Increase Gold ETF by 1%
Increase Healthcare ETF by 1%

Reason:
Technology concentration is high and healthcare momentum is improving.
```

## B. Event-driven recommendation

Example:

```text
Event:
Unexpected geopolitical escalation

Possible adjustment:
Increase Gold ETF by 3%
Reduce cyclical ETF by 2%
Maintain 1% additional cash

Risk of reacting:
41%

Decision:
Moderate action may be considered
```

The system should not execute the trade. The final decision remains with the user.

---

# 9. Portfolio Constraints

Before showing a final recommendation, apply safety rules.

Examples:

```text
Maximum individual stock weight: 20%
Maximum sector allocation: 35%
Maximum commodity allocation: 25%
Maximum weight change per recommendation: 5%
Minimum number of holdings: 5
No rebalance when proposed change is below 1%
```

These constraints stop the system from making unrealistic recommendations.

---

# 10. Dashboard Architecture

## Page 1: Home

Display:

* Portfolio value
* Portfolio Health Score
* Current market regime
* Current risk level
* Number of essential news events
* Main recommendation

---

## Page 2: Portfolio Health

Display:

* Sharpe ratio
* Sortino ratio
* Volatility
* Maximum drawdown
* Beta
* Diversification
* Correlation heatmap
* Concentration warnings

---

## Page 3: Daily Strategy

Display:

* Current market regime
* Asset rankings
* Indicator values
* Current versus recommended weights
* Explanation for each recommendation

---

## Page 4: Essential News

Display:

* Top five essential stories
* Event category
* Importance score
* Affected holdings
* Sentiment
* AI-generated summary

---

## Page 5: Should I React?

The user selects an event and sees:

```text
Potential impact
Confidence score
Risk of reacting
Current technical alignment
Suggested action
Reasons
```

Give three choices:

```text
Do nothing
Moderate adjustment
Aggressive adjustment
```

Show the estimated risk beside each choice.

---

## Page 6: Performance and Benchmark

Compare:

1. Buy-and-hold sample portfolio
2. Equal-weight portfolio
3. Daily-strategy portfolio
4. Daily strategy plus selected event adjustments

Metrics:

* Cumulative return
* Annualized return
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Volatility
* Turnover
* Transaction costs

---

# 11. Suggested Technical Architecture

```text
Frontend:
Streamlit

Core language:
Python

Data handling:
Pandas
NumPy

Market data:
yfinance or another approved source

Machine learning:
scikit-learn

NLP:
Transformers, FinBERT or a simpler sentiment model

Visualizations:
Matplotlib
Plotly

Storage:
CSV files or SQLite

Development:
Jupyter Notebook + Python scripts

Deployment:
Streamlit Community Cloud or local demonstration
```

---

# 12. Suggested Project Folder

```text
aurora/
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── portfolio_health.py
│   │   ├── daily_strategy.py
│   │   ├── essential_news.py
│   │   ├── reaction_risk.py
│   │   └── performance.py
│
├── data/
│   ├── sample_portfolio.csv
│   ├── market_data/
│   ├── news_data/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_portfolio_metrics.ipynb
│   ├── 03_daily_strategy.ipynb
│   ├── 04_news_engine.ipynb
│   ├── 05_reaction_risk.ipynb
│   └── 06_backtesting.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── portfolio_engine.py
│   ├── regime_engine.py
│   ├── strategy_engine.py
│   ├── news_engine.py
│   ├── reaction_risk.py
│   ├── recommendation_engine.py
│   └── backtester.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── recommendations/
│
├── requirements.txt
└── README.md
```

---

# 13. Work Allocation for Four Members

## Member 1 — Portfolio Intelligence

* Portfolio data
* Returns
* Sharpe and Sortino
* Drawdown
* Health Score
* Correlation and concentration

## Member 2 — Daily Strategy

* Indicators
* Market regime
* Asset ranking
* Daily recommendations
* Backtest

## Member 3 — News and AI

* News collection
* Essential-news filtering
* Sentiment
* Event classification
* Asset-impact mapping

## Member 4 — Risk and Dashboard

* Reaction-risk model
* Portfolio constraints
* Streamlit application
* Integration
* Visual presentation

All four members should contribute to testing, documentation and final presentation.

---

# 14. Recommended MVP

Because you have only two weeks, the minimum working version should do these seven things:

1. Load your sample portfolio.
2. Calculate portfolio health and risk metrics.
3. Classify the current market regime.
4. Rank assets using five indicators.
5. Retrieve and filter essential news.
6. Calculate the risk of reacting to a selected story.
7. Display normal-day and event-driven recommendations in Streamlit.

Do not initially attempt:

* Automatic brokerage execution
* Hundreds of securities
* Deep reinforcement learning
* Real-time tick data
* Perfect historical news matching
* Complex multi-agent systems

The most important architecture decision is to keep **daily strategy recommendations** and **event-driven recommendations** separate. AURORA should first explain what it would recommend under normal market conditions, then clearly show how a major event changes—or does not change—that recommendation.
