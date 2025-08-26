import os
import pandas as pd
import numpy as np

# === SETTINGS ===
data_dir = r"C:\Users\JustMe\commerce\data"
output_dir = r"C:\Users\JustMe\commerce\processed"
os.makedirs(output_dir, exist_ok=True)

# === SAFE CSV READER ===
def safe_read_csv(path):
    """Try reading CSV with multiple separators and encodings, return None if not valid."""
    if os.stat(path).st_size == 0:
        return None
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8", "latin1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 0:
                    return df
            except pd.errors.EmptyDataError:
                return None
            except Exception:
                continue
    return None

# === STEP 1: Load & Combine CSVs ===
dfs = []
for file in os.listdir(data_dir):
    if file.lower().endswith(".csv"):
        df = safe_read_csv(os.path.join(data_dir, file))
        if df is None:
            print(f"[WARNING] Skipping file with no valid CSV data: {file}")
            continue
        df["source_file"] = file
        dfs.append(df)

if not dfs:
    raise RuntimeError("No valid CSV files found.")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Combined dataset shape: {combined_df.shape}")

# === STEP 2: Basic Cleaning ===
combined_df = combined_df.drop_duplicates()

for col in combined_df.columns:
    if combined_df[col].dtype in ["int64", "float64"]:
        if combined_df[col].notna().any():
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        else:
            combined_df[col] = combined_df[col].fillna(0)
    else:
        combined_df[col] = combined_df[col].fillna("unknown")

# Standardize text columns
for col in combined_df.select_dtypes(include=["object"]).columns:
    combined_df[col] = combined_df[col].astype(str).str.strip().str.lower()

# Convert date columns
for col in combined_df.columns:
    if "date" in col.lower():
        combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")

print(f"[INFO] Cleaned dataset shape: {combined_df.shape}")

# === STEP 3: Ensure IDs Exist ===
if "customer_id" not in combined_df.columns:
    possible_ids = [c for c in combined_df.columns if "customer" in c.lower() or "user" in c.lower()]
    if possible_ids:
        combined_df.rename(columns={possible_ids[0]: "customer_id"}, inplace=True)
    else:
        combined_df["customer_id"] = [f"c{1000+i}" for i in range(len(combined_df))]

if "order_id" not in combined_df.columns:
    possible_orders = [c for c in combined_df.columns if "order" in c.lower() or "invoice" in c.lower()]
    if possible_orders:
        combined_df.rename(columns={possible_orders[0]: "order_id"}, inplace=True)
    else:
        combined_df["order_id"] = np.arange(1, len(combined_df) + 1)

# === STEP 4: Price Column ===
price_cols = [c for c in combined_df.columns if "price" in c.lower()]
if price_cols:
    combined_df["total_price"] = pd.to_numeric(combined_df[price_cols[0]], errors="coerce").fillna(0)
else:
    combined_df["total_price"] = 0

# === STEP 5: RFM Segmentation ===
if "order_date" in combined_df.columns:
    rfm = combined_df.groupby("customer_id").agg({
        "order_date": lambda x: (pd.Timestamp.today() - x.max()).days,
        "order_id": "count",
        "total_price": "sum"
    }).rename(columns={
        "order_date": "Recency",
        "order_id": "Frequency",
        "total_price": "Monetary"
    })
else:
    rfm = combined_df.groupby("customer_id").agg({
        "order_id": "count",
        "total_price": "sum"
    }).rename(columns={
        "order_id": "Frequency",
        "total_price": "Monetary"
    })
    rfm["Recency"] = np.nan

# === STEP 6: Calculate AOV and CLV ===
rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]
rfm["CLV"] = rfm["AOV"] * rfm["Frequency"] * 12  # Assuming 12 months projection

# === STEP 7: Save Cleaned Data and RFM Table ===
combined_df.to_csv(os.path.join(output_dir, "combined_cleaned_ecommerce.csv"), index=False)
rfm.to_csv(os.path.join(output_dir, "rfm_segmentation.csv"))

# === STEP 8: Top 100 Customers by CLV ===
top_100 = rfm.sort_values("CLV", ascending=False).head(100)
top_100.to_csv(os.path.join(output_dir, "top_100_high_value_customers.csv"))

# === STEP 9: Print Top 5 for Quick Insight ===
print("\n=== TOP 5 CUSTOMERS BY CLV ===")
print(top_100.head())

print(f"\n[SUCCESS] Cleaned data saved to: {os.path.join(output_dir, 'combined_cleaned_ecommerce.csv')}")
print(f"[SUCCESS] RFM segmentation saved to: {os.path.join(output_dir, 'rfm_segmentation.csv')}")
print(f"[SUCCESS] Top 100 high-value customers saved to: {os.path.join(output_dir, 'top_100_high_value_customers.csv')}")


import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 10: Create Visual Charts ===
sns.set(style="whitegrid")
charts_dir = os.path.join(output_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

# --- 1. RFM Segment Distribution (based on CLV percentiles) ---
rfm["Segment"] = pd.qcut(rfm["CLV"].rank(method="first"), q=3, labels=["Low", "Medium", "High"])

plt.figure(figsize=(8, 5))
sns.countplot(data=rfm, x="Segment", palette="coolwarm")
plt.title("Customer Segments by CLV")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "segment_distribution.png"))
plt.close()

# --- 2. Recency Histogram ---
if rfm["Recency"].notna().any():
    plt.figure(figsize=(8, 5))
    sns.histplot(rfm["Recency"].dropna(), bins=30, kde=True, color="skyblue")
    plt.title("Recency Distribution")
    plt.xlabel("Recency (Days)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "recency_distribution.png"))
    plt.close()

# --- 3. Frequency Histogram ---
plt.figure(figsize=(8, 5))
sns.histplot(rfm["Frequency"], bins=30, kde=True, color="orange")
plt.title("Frequency Distribution")
plt.xlabel("Number of Orders")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "frequency_distribution.png"))
plt.close()

# --- 4. Monetary Histogram ---
plt.figure(figsize=(8, 5))
sns.histplot(rfm["Monetary"], bins=30, kde=True, color="green")
plt.title("Monetary Value Distribution")
plt.xlabel("Total Spend")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "monetary_distribution.png"))
plt.close()

# --- 5. CLV Bar Chart (Top 10 Customers) ---
top_10 = top_100.head(10).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x="customer_id", y="CLV", palette="viridis")
plt.title("Top 10 Customers by CLV")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "top_10_clv.png"))
plt.close()

print(f"[SUCCESS] Charts saved to: {charts_dir}")
