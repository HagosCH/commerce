import os
import pandas as pd
import numpy as np

# === SETTINGS ===
data_dir = r"C:\Users\JustMe\commerce\data"
output_dir = r"C:\Users\JustMe\commerce\processed"
os.makedirs(output_dir, exist_ok=True)

# === SAFE CSV READER ===
def safe_read_csv(path):
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

# Drop unnamed/junk columns
combined_df = combined_df.loc[:, ~combined_df.columns.str.contains("^Unnamed")]

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

# === STEP 3: Detect and Parse Date Column ===
possible_date_cols = [c for c in combined_df.columns if "date" in c.lower() or "time" in c.lower()]
order_date_col = None

for col in possible_date_cols:
    try:
        parsed = pd.to_datetime(combined_df[col], errors="coerce", infer_datetime_format=True)
        if parsed.notna().sum() > len(combined_df) * 0.5:  # at least 50% valid dates
            combined_df[col] = parsed
            order_date_col = col
            break
    except Exception:
        continue

if order_date_col:
    print(f"[INFO] Using '{order_date_col}' as order date column.")
else:
    print("[WARNING] No valid date column found — Recency will be NaN.")

# === STEP 4: Ensure IDs Exist ===
if "customer_id" not in combined_df.columns:
    possible_ids = [c for c in combined_df.columns if "customer" in c.lower() or "user" in c.lower()]
    if possible_ids:
        combined_df.rename(columns={possible_ids[0]: "customer_id"}, inplace=True)
    else:
        combined_df["customer_id"] = ["c" + str(i) for i in range(1, len(combined_df) + 1)]

if "order_id" not in combined_df.columns:
    possible_orders = [c for c in combined_df.columns if "order" in c.lower() or "invoice" in c.lower()]
    if possible_orders:
        combined_df.rename(columns={possible_orders[0]: "order_id"}, inplace=True)
    else:
        combined_df["order_id"] = np.arange(1, len(combined_df) + 1)

# === STEP 5: Price Column ===
price_cols = [c for c in combined_df.columns if "price" in c.lower() or "amount" in c.lower()]
if price_cols:
    combined_df["total_price"] = pd.to_numeric(combined_df[price_cols[0]], errors="coerce").fillna(0)
else:
    combined_df["total_price"] = 0

# === STEP 6: RFM Segmentation ===
if order_date_col:
    rfm = combined_df.groupby("customer_id").agg({
        order_date_col: lambda x: (pd.Timestamp.today() - x.max()).days,
        "order_id": "count",
        "total_price": "sum"
    }).rename(columns={
        order_date_col: "Recency",
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

# Calculate RFM score
rfm["R_rank"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=range(5, 0, -1)) if rfm["Recency"].notna().any() else 3
rfm["F_rank"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=range(1, 6))
rfm["M_rank"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=range(1, 6))
rfm["RFM_score"] = rfm[["R_rank", "F_rank", "M_rank"]].sum(axis=1)

# Assign segment
rfm["Segment"] = pd.cut(rfm["RFM_score"], bins=[0,5,10,15], labels=["Low", "Medium", "High"])

print(f"[INFO] RFM table shape: {rfm.shape}")

# === STEP 7: Save Files ===
combined_df.to_csv(os.path.join(output_dir, "combined_cleaned_ecommerce.csv"), index=False)
rfm.to_csv(os.path.join(output_dir, "rfm_segmentation.csv"), index=True)

# === STEP 8: Print Top 5 Customers ===
print("\n=== TOP 5 CUSTOMERS BY RFM SCORE ===")
print(rfm.sort_values("RFM_score", ascending=False).head())

print(f"\n[SUCCESS] Cleaned data saved to: {os.path.join(output_dir, 'combined_cleaned_ecommerce.csv')}")
print(f"[SUCCESS] RFM segmentation saved to: {os.path.join(output_dir, 'rfm_segmentation.csv')}")




# import os
# import pandas as pd
# import numpy as np

# # === SETTINGS ===
# data_dir = r"C:\Users\HChe\Dr_Girma_AWS\commerce\data"
# output_dir = r"C:\Users\HChe\Dr_Girma_AWS\commerce\processed"
# os.makedirs(output_dir, exist_ok=True)

# # === SAFE CSV READER ===
# def safe_read_csv(path):
#     """Try reading CSV with multiple separators and encodings, return None if invalid."""
#     if os.stat(path).st_size == 0:
#         return None
#     for sep in [",", ";", "\t", "|"]:
#         for enc in ["utf-8", "latin1"]:
#             try:
#                 df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
#                 if df.shape[1] > 0:
#                     return df
#             except pd.errors.EmptyDataError:
#                 return None
#             except Exception:
#                 continue
#     return None

# # === STEP 1: Load & Combine CSVs ===
# dfs = []
# for file in os.listdir(data_dir):
#     if file.lower().endswith(".csv"):
#         df = safe_read_csv(os.path.join(data_dir, file))
#         if df is None:
#             print(f"[WARNING] Skipping file with no valid CSV data: {file}")
#             continue
#         df["source_file"] = file
#         dfs.append(df)

# if not dfs:
#     raise RuntimeError("No valid CSV files found.")

# combined_df = pd.concat(dfs, ignore_index=True)
# print(f"[INFO] Combined dataset shape: {combined_df.shape}")

# # === STEP 2: Basic Cleaning ===
# combined_df = combined_df.drop_duplicates()

# for col in combined_df.columns:
#     if combined_df[col].dtype in ["int64", "float64"]:
#         if combined_df[col].notna().any():
#             combined_df[col] = combined_df[col].fillna(combined_df[col].median())
#         else:
#             combined_df[col] = combined_df[col].fillna(0)
#     else:
#         combined_df[col] = combined_df[col].fillna("unknown")

# # Standardize text columns
# for col in combined_df.select_dtypes(include=["object"]).columns:
#     combined_df[col] = combined_df[col].astype(str).str.strip().str.lower()

# # Convert only valid date columns
# for col in combined_df.columns:
#     if "date" in col.lower():
#         sample_values = combined_df[col].dropna().astype(str).head(20)
#         if any(char.isdigit() for val in sample_values for char in val):
#             combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")

# print(f"[INFO] Cleaned dataset shape: {combined_df.shape}")

# # === STEP 3: Ensure IDs Exist ===
# if "customer_id" not in combined_df.columns:
#     possible_ids = [c for c in combined_df.columns if "customer" in c.lower() or "user" in c.lower()]
#     if possible_ids:
#         combined_df.rename(columns={possible_ids[0]: "customer_id"}, inplace=True)
#     else:
#         combined_df["customer_id"] = np.arange(1, len(combined_df) + 1)

# if "order_id" not in combined_df.columns:
#     possible_orders = [c for c in combined_df.columns if "order" in c.lower() or "invoice" in c.lower()]
#     if possible_orders:
#         combined_df.rename(columns={possible_orders[0]: "order_id"}, inplace=True)
#     else:
#         combined_df["order_id"] = np.arange(1, len(combined_df) + 1)

# # === STEP 4: Price Column ===
# price_cols = [c for c in combined_df.columns if "price" in c.lower()]
# if price_cols:
#     combined_df["total_price"] = pd.to_numeric(combined_df[price_cols[0]], errors="coerce").fillna(0)
# else:
#     combined_df["total_price"] = 0

# # === STEP 5: RFM Segmentation ===
# if "order_date" in combined_df.columns:
#     rfm = combined_df.groupby("customer_id").agg({
#         "order_date": lambda x: (pd.Timestamp.today() - x.max()).days,
#         "order_id": "count",
#         "total_price": "sum"
#     }).rename(columns={
#         "order_date": "Recency",
#         "order_id": "Frequency",
#         "total_price": "Monetary"
#     })
# else:
#     rfm = combined_df.groupby("customer_id").agg({
#         "order_id": "count",
#         "total_price": "sum"
#     }).rename(columns={
#         "order_id": "Frequency",
#         "total_price": "Monetary"
#     })
#     rfm["Recency"] = np.nan

# # === STEP 6: Calculate RFM Score ===
# rfm["R_rank"] = rfm["Recency"].rank(ascending=False)
# rfm["F_rank"] = rfm["Frequency"].rank(ascending=True)
# rfm["M_rank"] = rfm["Monetary"].rank(ascending=True)
# rfm["RFM_score"] = rfm["R_rank"] + rfm["F_rank"] + rfm["M_rank"]

# # === STEP 7: Add Segmentation Labels ===
# def segment_customer(row):
#     if row["Recency"] <= 30 and row["Frequency"] > 10 and row["Monetary"] > 1000:
#         return "VIP"
#     elif row["Frequency"] > 10:
#         return "Loyal"
#     elif row["Recency"] > 90 and row["Frequency"] > 5:
#         return "At Risk"
#     elif row["Recency"] <= 30 and row["Frequency"] <= 2:
#         return "New Customer"
#     else:
#         return "Others"

# rfm["Segment"] = rfm.apply(segment_customer, axis=1)

# # === STEP 8: Merge Customer Names (if available) ===
# name_cols = [c for c in combined_df.columns if "name" in c.lower()]
# if name_cols:
#     customer_names = combined_df.groupby("customer_id")[name_cols[0]].first().reset_index()
#     rfm = rfm.reset_index().merge(customer_names, on="customer_id", how="left")
# else:
#     rfm = rfm.reset_index()

# # === STEP 9: Show Top 5 Customers ===
# top_5 = rfm.sort_values("RFM_score", ascending=False).head(5)
# print("\n=== TOP 5 CUSTOMERS BY RFM SCORE ===")
# print(top_5[["customer_id", "Recency", "Frequency", "Monetary", "RFM_score", "Segment"] + ([name_cols[0]] if name_cols else [])])

# print(f"[INFO] RFM table shape: {rfm.shape}")

# # === STEP 10: Save Files ===
# combined_df.to_csv(os.path.join(output_dir, "combined_cleaned_ecommerce.csv"), index=False)
# rfm.to_csv(os.path.join(output_dir, "rfm_segmentation.csv"), index=False)

# print(f"[SUCCESS] Cleaned data saved to: {os.path.join(output_dir, 'combined_cleaned_ecommerce.csv')}")
# print(f"[SUCCESS] RFM segmentation saved to: {os.path.join(output_dir, 'rfm_segmentation.csv')}")
