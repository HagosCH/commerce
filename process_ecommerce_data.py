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

# Convert only valid date columns
for col in combined_df.columns:
    if "date" in col.lower():
        sample_values = combined_df[col].dropna().astype(str).head(20)
        if any(char.isdigit() for val in sample_values for char in val):
            combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")

print(f"[INFO] Cleaned dataset shape: {combined_df.shape}")

# === STEP 3: Ensure IDs Exist ===
if "customer_id" not in combined_df.columns:
    possible_ids = [c for c in combined_df.columns if "customer" in c.lower() or "user" in c.lower()]
    if possible_ids:
        combined_df.rename(columns={possible_ids[0]: "customer_id"}, inplace=True)
    else:
        combined_df["customer_id"] = np.arange(1, len(combined_df) + 1)

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

print(f"[INFO] RFM table shape: {rfm.shape}")

# === STEP 6: Save Files ===
combined_df.to_csv(os.path.join(output_dir, "combined_cleaned_ecommerce.csv"), index=False)
rfm.to_csv(os.path.join(output_dir, "rfm_segmentation.csv"), index=False)

print(f"[SUCCESS] Cleaned data saved to: {os.path.join(output_dir, 'combined_cleaned_ecommerce.csv')}")
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
#     """Try reading CSV with multiple separators, return None if not valid."""
#     if os.stat(path).st_size == 0:
#         return None
#     for sep in [",", ";", "\t", "|"]:
#         try:
#             df = pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
#             if df.shape[1] > 0:
#                 return df
#         except pd.errors.EmptyDataError:
#             return None
#         except UnicodeDecodeError:
#             try:
#                 df = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
#                 if df.shape[1] > 0:
#                     return df
#             except Exception:
#                 continue
#         except Exception:
#             continue
#     return None

# # === STEP 1: Load & Combine CSVs ===
# csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
# dfs = []

# for file in csv_files:
#     file_path = os.path.join(data_dir, file)
#     df = safe_read_csv(file_path)
#     if df is None:
#         print(f"[WARNING] Skipping file with no valid CSV data: {file}")
#         continue
#     df["source_file"] = file  # Keep track of original dataset
#     dfs.append(df)

# if not dfs:
#     raise RuntimeError("No valid CSV files found in the data directory.")

# # Merge all datasets into one
# combined_df = pd.concat(dfs, ignore_index=True)
# print(f"[INFO] Combined dataset shape: {combined_df.shape}")

# # === STEP 2: Basic Cleaning ===

# # Remove exact duplicates
# combined_df = combined_df.drop_duplicates()

# # Fill missing values properly
# for col in combined_df.columns:
#     if combined_df[col].dtype in ["int64", "float64"]:
#         median_value = combined_df[col].median()
#         combined_df[col] = combined_df[col].fillna(median_value)
#     else:
#         combined_df[col] = combined_df[col].fillna("unknown")

# # Standardize text columns
# for col in combined_df.select_dtypes(include=["object"]).columns:
#     combined_df[col] = combined_df[col].astype(str).str.strip().str.lower()

# # Convert date columns
# date_cols = [c for c in combined_df.columns if "date" in c.lower()]
# for col in date_cols:
#     combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")

# print(f"[INFO] Cleaned dataset shape: {combined_df.shape}")

# # === STEP 3: Extract Features for Segmentation & Personalization ===

# # Ensure customer_id exists
# if "customer_id" not in combined_df.columns:
#     combined_df["customer_id"] = combined_df.groupby(level=0).cumcount() + 1

# # Ensure order_id exists
# if "order_id" not in combined_df.columns:
#     combined_df["order_id"] = np.arange(1, len(combined_df) + 1)

# # Ensure price column exists
# price_cols = [c for c in combined_df.columns if "price" in c.lower()]
# if price_cols:
#     combined_df["total_price"] = pd.to_numeric(combined_df[price_cols[0]], errors="coerce").fillna(0)
# else:
#     combined_df["total_price"] = 0

# # === STEP 4: RFM Segmentation ===
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
#     rfm["Recency"] = np.nan  # No date column found

# print(f"[INFO] RFM table shape: {rfm.shape}")

# # === STEP 5: Save Cleaned Data ===
# cleaned_path = os.path.join(output_dir, "combined_cleaned_ecommerce.csv")
# rfm_path = os.path.join(output_dir, "rfm_segmentation.csv")

# combined_df.to_csv(cleaned_path, index=False)
# rfm.to_csv(rfm_path)

# print(f"[SUCCESS] Cleaned data saved to: {cleaned_path}")
# print(f"[SUCCESS] RFM segmentation saved to: {rfm_path}")
