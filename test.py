import pandas as pd

df = pd.read_csv("FNSPID/Stock_news/nasdaq_exteral_data.csv")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

tickers = [
    "NVDA","AAPL","GOOGL","MSFT","AMZN","AVGO","META","TSLA",
    "ASML","NFLX","COST","AMD","MU","CSCO","PEP","ADBE",
    "INTC","QCOM","TXN","INTU","GLD","SLV"
]

df_small = df[
    df["Stock_symbol"].isin(tickers) &
    (df["Date"] >= "2013-01-01") &
    (df["Date"] <= "2023-12-31")
]

print(df_small["Stock_symbol"].value_counts())

df_small.to_csv("FNSPID/Stock_news/news_top20_gold_silver_2013_2023.csv", index=False)
print("Saved!")