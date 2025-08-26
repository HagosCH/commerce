import os, pandas as pd
from zipfile import ZipFile

slug = "suvroo/personalized-recommendations-for-e-commerce"
data_dir = r"C:\Users\JustMe\commerce\data"
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

print(f"Downloading: {slug}")
res = os.system(f"kaggle datasets download -d {slug} --force")
if res != 0:
    print("Download failed. Check your kaggle.json or slug.")
    exit()
zip_name = slug.split("/")[-1] + ".zip"
with ZipFile(zip_name, 'r') as z:
    z.extractall(data_dir)
    print("Extraction complete.")

csv = next((f for f in os.listdir(data_dir) if f.endswith(".csv")), None)
if csv:
    df = pd.read_csv(csv)
    print(f"Loaded {csv}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
else:
    print("No CSV found.")
