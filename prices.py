import pandas as pd
import os

src_dir = "FNSPID/Stock_price/full_history"
dst_dir = "FNSPID/Stock_price/reduced"
os.makedirs(dst_dir, exist_ok=True)

tickers = [
    "NVDA","AAPL","GOOGL","MSFT","AMZN","AVGO","META","TSLA",
    "ASML","NFLX","COST","AMD","MU","CSCO","PEP","ADBE",
    "INTC","QCOM","TXN","INTU","GLD","SLV"
]

# See what's actually in there first
all_files = os.listdir(src_dir)
print(f"Total files in folder: {len(all_files)}")
print("Sample filenames:", all_files[:10])

for t in tickers:
    src_file = os.path.join(src_dir, f"{t}.csv")
    if os.path.exists(src_file):
        df = pd.read_csv(src_file)
        
        # Figure out the date column name (varies by dataset version)
        date_col = None
        for candidate in ["Date", "date", "DATE"]:
            if candidate in df.columns:
                date_col = candidate
                break
        
        if date_col is None:
            print(f"⚠️ {t}: no date column found — columns are: {df.columns.tolist()}")
            continue
        
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df_filtered = df[(df[date_col] >= "2013-01-01") & (df[date_col] <= "2023-12-31")]
        
        df_filtered.to_csv(os.path.join(dst_dir, f"{t}.csv"), index=False)
        print(f"✅ {t}: {len(df_filtered)} rows saved")
    else:
        print(f"❌ {t}.csv not found in folder")

print("\nDone! Reduced files saved to:", dst_dir)