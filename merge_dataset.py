import os
import shutil
# Final consolidated output folder
final_dir = "FNSPID/final_dataset"
os.makedirs(os.path.join(final_dir, "prices"), exist_ok=True)
# Copy the news file
shutil.copy(
    "FNSPID/Stock_news/news_top20_gold_silver_2013_2023.csv",
    os.path.join(final_dir, "news_top20_gold_silver_2013_2023.csv")
)
# Copy all reduced price files
price_src = "FNSPID/Stock_price/reduced"
for f in os.listdir(price_src):
    shutil.copy(os.path.join(price_src, f), os.path.join(final_dir, "prices", f))
print("Done! Final dataset structure:")
for root, dirs, files in os.walk(final_dir):
    level = root.replace(final_dir, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        print(f"{indent}  {file}")