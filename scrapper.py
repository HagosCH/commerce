import os, math, datetime, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kaggle.api.kaggle_api_extended import KaggleApi

# ===================== CONFIG =====================
KAGGLE_DATASETS = [
    "carrie1/ecommerce-data",                  # Online Retail (UCI)
    "promptcloud/amazon-products-dataset",     # Amazon product catalog
    "karkavelrajaj/amazon-sales-dataset",      # Amazon sales transactions
    "thedevastator/online-retail-transactions-dataset"  # Large transaction dataset
]

PROJECT_BASE = Path(r"C:\Users\JustMe\commerce")
DATA_BASE = PROJECT_BASE / "data"

MAX_PRODUCT_ROWS = None
MAX_TXN_ROWS = None
K = 10
# =====================================================

DATA_BASE.mkdir(parents=True, exist_ok=True)

# 1️⃣ Kaggle Download
api = KaggleApi()
api.authenticate()

print("📥 Downloading datasets from Kaggle...")
for slug in KAGGLE_DATASETS:
    print(f"  - {slug}")
    api.dataset_download_files(slug, path=DATA_BASE, unzip=True)
print("✅ All datasets downloaded.")

# 2️⃣ Load and Merge
product_frames = []
txn_frames = []

for csv_file in DATA_BASE.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file, encoding="ISO-8859-1", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
        except Exception as e:
            print(f"⚠️ Could not read {csv_file}: {e}")
            continue

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect product table
    if any(col in df.columns for col in ["product_id", "stockcode", "asin"]):
        prod_df = df.copy()
        if "product_id" not in prod_df.columns:
            if "stockcode" in prod_df.columns:
                prod_df["product_id"] = prod_df["stockcode"]
            elif "asin" in prod_df.columns:
                prod_df["product_id"] = prod_df["asin"]
            else:
                prod_df["product_id"] = prod_df.index.astype(str)
        prod_df["product_name"] = prod_df.get("description", prod_df.get("name", ""))
        prod_df["category"] = prod_df.get("category", "")
        prod_df["brand"] = prod_df.get("brand", "")
        product_frames.append(prod_df[["product_id", "product_name", "category", "brand"]])

    # Detect transaction table
    if any(col in df.columns for col in ["order_id", "invoice", "transaction_id"]) and \
       any(col in df.columns for col in ["customer_id", "customerid"]) and \
       any(col in df.columns for col in ["quantity", "qty", "count"]):
        txn_df = df.copy()
        if "customer_id" not in txn_df.columns:
            if "customerid" in txn_df.columns:
                txn_df["customer_id"] = txn_df["customerid"].astype(str)
        if "product_id" not in txn_df.columns:
            if "stockcode" in txn_df.columns:
                txn_df["product_id"] = txn_df["stockcode"].astype(str)
            elif "asin" in txn_df.columns:
                txn_df["product_id"] = txn_df["asin"].astype(str)
        if "quantity" not in txn_df.columns:
            for qcol in ["qty", "count"]:
                if qcol in txn_df.columns:
                    txn_df["quantity"] = pd.to_numeric(txn_df[qcol], errors="coerce").fillna(1)
        txn_frames.append(txn_df[["customer_id", "product_id", "quantity"]])

# Merge and clean
prod_dim = pd.concat(product_frames, ignore_index=True).drop_duplicates()
txn = pd.concat(txn_frames, ignore_index=True).drop_duplicates()

if MAX_PRODUCT_ROWS:
    prod_dim = prod_dim.head(MAX_PRODUCT_ROWS)
if MAX_TXN_ROWS:
    txn = txn.head(MAX_TXN_ROWS)

# 3️⃣ Build user-item mappings
pairs = txn[["customer_id", "product_id"]].dropna().drop_duplicates()
shuffled = pairs.sample(frac=1.0, random_state=1).reset_index(drop=True)
split = int(0.8*len(shuffled))
train_pairs = shuffled.iloc[:split]
test_pairs = shuffled.iloc[split:]

train_user_items = train_pairs.groupby("customer_id")["product_id"].apply(set).to_dict()
test_user_items = test_pairs.groupby("customer_id")["product_id"].apply(set).to_dict()

# 4️⃣ Popularity model
popularity = train_pairs["product_id"].value_counts()
def recommend_popular(k=10):
    return list(popularity.head(k).index)

# 5️⃣ Content-Based
prod_dim["text"] = prod_dim["product_name"].fillna("") + " " + prod_dim["category"].fillna("") + " " + prod_dim["brand"].fillna("")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(prod_dim["text"].astype(str))
sim = cosine_similarity(X)
pid_to_idx = {pid:i for i, pid in enumerate(prod_dim["product_id"].astype(str))}
idx_to_pid = {i:pid for pid,i in pid_to_idx.items()}

def cb_recommend_for_user(user_id, k=10):
    items = list(train_user_items.get(user_id, []))
    idxs = [pid_to_idx[i] for i in items if i in pid_to_idx]
    if not idxs: return []
    scores = np.zeros(sim.shape[0])
    for ix in idxs:
        scores = np.maximum(scores, sim[ix])
    for ix in idxs:
        scores[ix] = -1.0
    top = np.argpartition(scores, -k)[-k:]
    ranked = top[np.argsort(scores[top])[::-1]]
    return [idx_to_pid[i] for i in ranked]

# 6️⃣ Item-CF
co = {}
item_freq = train_pairs["product_id"].value_counts().to_dict()
for _, grp in train_pairs.groupby("customer_id"):
    items = list(set(grp["product_id"].tolist()))
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            a,b = items[i], items[j]
            co.setdefault(a, {}).setdefault(b, 0)
            co.setdefault(b, {}).setdefault(a, 0)
            co[a][b] += 1
            co[b][a] += 1

def cf_similar(pid, topn=50):
    neigh = co.get(pid, {})
    sims = []
    for j,cij in neigh.items():
        denom = math.sqrt(item_freq.get(pid,1)*item_freq.get(j,1))
        sims.append((j, cij/denom if denom>0 else 0))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topn]

def cf_recommend_for_user(user_id, k=10):
    seen = set(train_user_items.get(user_id, []))
    scores = {}
    for pid in list(seen)[:50]:
        for j,s in cf_similar(pid, topn=50):
            if j in seen: continue
            scores[j] = scores.get(j,0.0) + s
    if not scores: return []
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [p for p,_ in ranked]

# 7️⃣ Hybrid
def hybrid_recommend_for_user(user_id, k=10, w_cb=0.5, w_cf=0.5):
    cb = cb_recommend_for_user(user_id, k=50)
    cf = cf_recommend_for_user(user_id, k=50)
    pop = recommend_popular(k=50)
    scores = {}
    for rank, pid in enumerate(cb): scores[pid] = scores.get(pid,0)+w_cb/(rank+1)
    for rank, pid in enumerate(cf): scores[pid] = scores.get(pid,0)+w_cf/(rank+1)
    for rank, pid in enumerate(pop): scores[pid] = scores.get(pid,0)+0.1/(rank+1)
    seen = set(train_user_items.get(user_id, []))
    ranked = [p for p,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True) if p not in seen]
    return ranked[:k]

# 8️⃣ Evaluation
def precision_recall_at_k(fn, k=10, max_users=200):
    users = [u for u, items in test_user_items.items() if len(items)>0]
    users = users[:max_users]
    precisions, recalls = [], []
    for u in users:
        recs = set(fn(u,k))
        truth = test_user_items.get(u, set())
        tp = len(recs & truth)
        if len(recs)>0: precisions.append(tp/len(recs))
        if len(truth)>0: recalls.append(tp/len(truth))
    return float(np.nanmean(precisions)), float(np.nanmean(recalls)), len(users)

p_pop, r_pop, n = precision_recall_at_k(lambda u,k: recommend_popular(k), k=K)
p_cb, r_cb, _ = precision_recall_at_k(cb_recommend_for_user, k=K)
p_cf, r_cf, _ = precision_recall_at_k(cf_recommend_for_user, k=K)
p_h, r_h, _ = precision_recall_at_k(lambda u,k: hybrid_recommend_for_user(u,k,0.5,0.5), k=K)

metrics = pd.DataFrame([
    {"model":"Popularity@10", "precision": p_pop, "recall": r_pop, "users_evaluated": n},
    {"model":"ContentBased@10", "precision": p_cb, "recall": r_cb, "users_evaluated": n},
    {"model":"ItemCF@10", "precision": p_cf, "recall": r_cf, "users_evaluated": n},
    {"model":"Hybrid@10", "precision": p_h, "recall": r_h, "users_evaluated": n},
])
metrics_path = DATA_BASE / "recommender_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# 9️⃣ Save recommendations
recs_rows = []
for u in list(train_user_items.keys())[:20]:
    for rank, pid in enumerate(hybrid_recommend_for_user(u, k=K), start=1):
        name = ""
        s = prod_dim.loc[prod_dim["product_id"]==pid, "product_name"]
        name = s.iloc[0] if len(s)>0 else ""
        recs_rows.append({"customer_id": u, "rank": rank, "product_id": pid, "product_name": name})
recs_df = pd.DataFrame(recs_rows)
recs_path = DATA_BASE / "sample_hybrid_recommendations.csv"
recs_df.to_csv(recs_path, index=False)

print(f"✅ Metrics saved to {metrics_path}")
print(f"✅ Recommendations saved to {recs_path}")








# # e_commerce_pipeline.py
# # End-to-end: Kaggle download -> cleaning -> unified schema -> recommenders -> metrics & recs -> reports

# import os, sys, shutil, zipfile, subprocess, datetime, math, json, glob, platform
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from typing import List, Dict, Tuple, Optional

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ===================== CONFIG =====================

# # >>>>>>>>>>>>>>>>>>> EDIT THIS <<<<<<<<<<<<<<<<<<<
# # Provide one or more Kaggle dataset slugs (from the dataset URL)
# KAGGLE_DATASETS = [
#     # examples (replace with the exact ones you need):
#     # "carrie1/ecommerce-data",              # UCI Online Retail mirror
#     # "hellbuoy/online-retail-customer-segmentation", 
#     # "promptcloud/amazon-products-dataset", 
# ]
# # Where to place Kaggle credentials (change if needed)
# KAGGLE_CONFIG_DIR = str(Path.home() / ".kaggle")

# # Base project folder (change if you like)
# BASE = Path.cwd() / "ecomm_runs"

# # Sample caps (set to None to disable sampling)
# MAX_TXN_ROWS = 200_000
# MAX_PRODUCT_ROWS = 200_000

# K = 10  # Top-K for recommendations and evaluation

# # =================================================

# def ensure_dirs(*paths: Path):
#     for p in paths:
#         p.mkdir(parents=True, exist_ok=True)

# def run_cmd(cmd: List[str], cwd: Optional[Path] = None):
#     print(">>>", " ".join(cmd))
#     result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
#     if result.returncode != 0:
#         print(result.stdout)
#         print(result.stderr)
#         raise RuntimeError(f"Command failed: {' '.join(cmd)}")
#     return result.stdout

# def download_kaggle_dataset(slug: str, out_dir: Path):
#     """
#     Uses Kaggle CLI to download and unzip a dataset into out_dir/slug
#     """
#     target = out_dir / slug.replace("/", "_")
#     ensure_dirs(target)
#     print(f"[KAGGLE] Downloading: {slug} -> {target}")
#     run_cmd(["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"])
#     # Some datasets come zipped inside; extract nested zips too
#     for z in target.glob("*.zip"):
#         with zipfile.ZipFile(z, "r") as zf:
#             zf.extractall(target)
#         z.unlink(missing_ok=True)
#     return target

# def robust_read_csv(path: Path) -> Optional[pd.DataFrame]:
#     attempts = [
#         dict(sep=None, engine="python"),
#         dict(sep=",", engine="python"),
#         dict(sep=",", encoding="latin1", engine="python"),
#         dict(sep=",", encoding="ISO-8859-1", engine="python"),
#         dict(sep=";", engine="python"),
#         dict(sep="\t", engine="python"),
#     ]
#     for args in attempts:
#         try:
#             return pd.read_csv(path, on_bad_lines="skip", **args)
#         except Exception:
#             continue
#     print(f"[WARN] Could not read {path.name}")
#     return None

# def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
#     return df

# def ensure_str(s: pd.Series) -> pd.Series:
#     try:
#         return s.astype(str)
#     except Exception:
#         return s.apply(lambda x: "" if pd.isna(x) else str(x))

# PRODUCT_ID_CANDS = ["product_id","asin","sku","stockcode","stock_code","unique_id","uniqe_id","item_id","sku_id","stockcode"]
# PRODUCT_NAME_CANDS = ["product_name","title","name","description","item_name","producttitle","product_title"]
# CATEGORY_CANDS = ["category","main_category","product_category","productgroup","product_group","product_category_tree","cat"]
# BRAND_CANDS = ["brand","brand_name","manufacturer"]

# ORDER_ID_CANDS = ["order_id","invoiceno","invoice_no","transaction_id","orderid","invoice","order_number"]
# CUSTOMER_ID_CANDS = ["customer_id","customerid","user_id","buyer_id","userid","clientid","customer"]
# TXN_PID_CANDS = PRODUCT_ID_CANDS + ["product","productname","product_name","stock_code"]
# QTY_CANDS = ["quantity","qty","count","num_items","item_count"]

# def pick_first(df: pd.DataFrame, candidates: List[str], default=None) -> pd.Series:
#     for c in candidates:
#         if c in df.columns:
#             return ensure_str(df[c])
#     if default is None:
#         return pd.Series([""] * len(df))
#     if callable(default):
#         return ensure_str(default(df))
#     if isinstance(default, (list, np.ndarray, pd.Series)):
#         return ensure_str(pd.Series(default))
#     return pd.Series([default] * len(df))

# def detect_tables(csv_dir: Path) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
#     product_tables, txn_tables = [], []
#     for f in csv_dir.rglob("*.csv"):
#         df = robust_read_csv(f)
#         if df is None or df.empty: 
#             continue
#         df = norm_cols(df)
#         cols = set(df.columns)
#         # product-like
#         if (cols & set(PRODUCT_ID_CANDS)) or (cols & set(PRODUCT_NAME_CANDS)) or (cols & set(CATEGORY_CANDS)):
#             product_tables.append(df)
#         # transaction-like
#         if ((cols & set(ORDER_ID_CANDS)) or (cols & set(CUSTOMER_ID_CANDS))) and ((cols & set(TXN_PID_CANDS)) or (cols & set(QTY_CANDS))):
#             txn_tables.append(df)
#     return product_tables, txn_tables

# def build_product_dim(product_tables: List[pd.DataFrame]) -> pd.DataFrame:
#     if not product_tables:
#         return pd.DataFrame(columns=["product_id","product_name","category","brand"])
#     prod = pd.concat(product_tables, ignore_index=True)
#     prod = norm_cols(prod).drop_duplicates()
#     if MAX_PRODUCT_ROWS and len(prod) > MAX_PRODUCT_ROWS:
#         prod = prod.sample(MAX_PRODUCT_ROWS, random_state=42)
#     out = pd.DataFrame()
#     out["product_id"]   = pick_first(prod, PRODUCT_ID_CANDS, default=lambda d: np.arange(len(d)).astype(str))
#     out["product_name"] = pick_first(prod, PRODUCT_NAME_CANDS, default="")
#     out["category"]     = pick_first(prod, CATEGORY_CANDS, default="")
#     out["brand"]        = pick_first(prod, BRAND_CANDS, default="")
#     out = out.dropna().drop_duplicates()
#     return out

# def build_transactions(txn_tables: List[pd.DataFrame]) -> pd.DataFrame:
#     if not txn_tables:
#         return pd.DataFrame(columns=["customer_id","product_id","quantity"])
#     rows = []
#     for df in txn_tables:
#         df = norm_cols(df)
#         order_id = pick_first(df, ORDER_ID_CANDS, default=lambda d: np.arange(len(d)).astype(str))
#         cust_id  = pick_first(df, CUSTOMER_ID_CANDS, default="unknown")
#         if any(c in df.columns for c in TXN_PID_CANDS):
#             pid = pick_first(df, TXN_PID_CANDS)
#         elif any(c in df.columns for c in PRODUCT_NAME_CANDS):
#             pid = pick_first(df, PRODUCT_NAME_CANDS)
#         else:
#             pid = pd.Series(np.arange(len(df)).astype(str))
#         if any(c in df.columns for c in QTY_CANDS):
#             q = pick_first(df, QTY_CANDS)
#             try:
#                 q = pd.to_numeric(q, errors="coerce").fillna(1).astype(int)
#             except Exception:
#                 q = pd.Series([1] * len(df))
#         else:
#             q = pd.Series([1] * len(df))
#         tmp = pd.DataFrame({"order_id": ensure_str(order_id),
#                             "customer_id": ensure_str(cust_id),
#                             "product_id": ensure_str(pid),
#                             "quantity": q})
#         rows.append(tmp)
#     txn = pd.concat(rows, ignore_index=True)
#     txn = txn[["customer_id","product_id","quantity"]].dropna()
#     txn["quantity"] = pd.to_numeric(txn["quantity"], errors="coerce").fillna(1).astype(int)
#     txn = txn.drop_duplicates()
#     if MAX_TXN_ROWS and len(txn) > MAX_TXN_ROWS:
#         txn = txn.sample(MAX_TXN_ROWS, random_state=42)
#     return txn

# def basic_eda(prod: pd.DataFrame, txn: pd.DataFrame) -> Dict:
#     eda = {
#         "n_products": int(len(prod)),
#         "n_transactions": int(len(txn)),
#         "n_users": int(txn["customer_id"].nunique() if not txn.empty else 0),
#         "n_items": int(txn["product_id"].nunique() if not txn.empty else 0),
#         "avg_items_per_user": float(txn.groupby("customer_id")["product_id"].nunique().mean()) if not txn.empty else 0.0,
#         "top_categories": prod["category"].value_counts().head(10).to_dict() if "category" in prod else {},
#         "top_brands": prod["brand"].value_counts().head(10).to_dict() if "brand" in prod else {},
#     }
#     return eda

# # ------------------ Recommenders ------------------

# def fit_content_sim(prod_dim: pd.DataFrame):
#     if prod_dim.empty:
#         return None, {}, {}
#     text = (prod_dim["product_name"].fillna("") + " " + 
#             prod_dim["category"].fillna("") + " " + 
#             prod_dim["brand"].fillna(""))
#     if text.str.strip().eq("").all():
#         return None, {}, {}
#     vec = TfidfVectorizer(max_features=5000, stop_words="english")
#     X = vec.fit_transform(text)
#     sim = cosine_similarity(X)
#     pid_to_idx = {pid:i for i, pid in enumerate(prod_dim["product_id"].astype(str))}
#     idx_to_pid = {i:pid for pid,i in pid_to_idx.items()}
#     return sim, pid_to_idx, idx_to_pid

# def cb_recommend_for_user(user_id: str, train_user_items: Dict[str, set], sim, pid_to_idx, idx_to_pid, k=10):
#     if sim is None or not train_user_items.get(user_id):
#         return []
#     items = list(train_user_items.get(user_id, []))
#     idxs = [pid_to_idx[i] for i in items if i in pid_to_idx]
#     if not idxs:
#         return []
#     scores = np.zeros(sim.shape[0])
#     for ix in idxs:
#         scores = np.maximum(scores, sim[ix])
#     for ix in idxs:
#         scores[ix] = -1.0
#     k_eff = min(k, scores.shape[0])
#     top = np.argpartition(scores, -k_eff)[-k_eff:]
#     ranked = top[np.argsort(scores[top])[::-1]]
#     return [idx_to_pid[i] for i in ranked]

# def build_item_co(train_pairs: pd.DataFrame) -> Tuple[Dict, Dict]:
#     co = {}
#     item_freq = train_pairs["product_id"].value_counts().to_dict()
#     for _, grp in train_pairs.groupby("customer_id"):
#         items = list(set(grp["product_id"].tolist()))
#         if len(items) > 100:
#             items = items[:100]
#         for i in range(len(items)):
#             for j in range(i+1, len(items)):
#                 a,b = items[i], items[j]
#                 co.setdefault(a, {}).setdefault(b, 0)
#                 co.setdefault(b, {}).setdefault(a, 0)
#                 co[a][b] += 1
#                 co[b][a] += 1
#     return co, item_freq

# def cf_similar(pid: str, co: Dict, item_freq: Dict, topn=100):
#     neigh = co.get(pid, {})
#     sims = []
#     for j,cij in neigh.items():
#         denom = math.sqrt(item_freq.get(pid,1)*item_freq.get(j,1))
#         sims.append((j, cij/denom if denom>0 else 0))
#     sims.sort(key=lambda x: x[1], reverse=True)
#     return sims[:topn]

# def cf_recommend_for_user(user_id: str, train_user_items: Dict[str, set], co, item_freq, k=10):
#     seen = set(train_user_items.get(user_id, []))
#     scores = {}
#     for pid in list(seen)[:100]:
#         for j,s in cf_similar(pid, co, item_freq, topn=100):
#             if j in seen:
#                 continue
#             scores[j] = scores.get(j,0.0) + s
#     if not scores:
#         return []
#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
#     return [p for p,_ in ranked]

# def popularity_top(popularity: pd.Series, k=10) -> List[str]:
#     return list(popularity.head(k).index)

# def hybrid_recommend_for_user(user_id: str, train_user_items: Dict[str,set],
#                               sim, pid_to_idx, idx_to_pid, co, item_freq,
#                               popularity: pd.Series, k=10, w_cb=0.5, w_cf=0.5, w_pop=0.1):
#     cb = cb_recommend_for_user(user_id, train_user_items, sim, pid_to_idx, idx_to_pid, k=50)
#     cf = cf_recommend_for_user(user_id, train_user_items, co, item_freq, k=50)
#     pop = popularity_top(popularity, k=50)
#     scores = {}
#     for rank, pid in enumerate(cb):
#         scores[pid] = scores.get(pid,0) + w_cb/(rank+1)
#     for rank, pid in enumerate(cf):
#         scores[pid] = scores.get(pid,0) + w_cf/(rank+1)
#     for rank, pid in enumerate(pop):
#         scores[pid] = scores.get(pid,0) + w_pop/(rank+1)
#     if not scores:
#         return pop[:k]
#     seen = set(train_user_items.get(user_id, []))
#     ranked = [p for p,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True) if p not in seen]
#     return ranked[:k]

# def precision_recall_at_k(fn, users: List[str], truth_map: Dict[str,set], k=10) -> Tuple[float,float,int]:
#     if not users:
#         return float("nan"), float("nan"), 0
#     precisions, recalls = [], []
#     for u in users:
#         recs = set(fn(u,k))
#         truth = truth_map.get(u, set())
#         tp = len(recs & truth)
#         if len(recs) > 0:
#             precisions.append(tp/len(recs))
#         if len(truth) > 0:
#             recalls.append(tp/len(truth))
#     return float(np.nanmean(precisions)), float(np.nanmean(recalls)), len(users)

# def main():
#     if not KAGGLE_DATASETS:
#         print("Please set KAGGLE_DATASETS with one or more dataset slugs (e.g., 'carrie1/ecommerce-data').")
#         sys.exit(1)

#     # Prepare run folders
#     run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = BASE / f"run_{run_ts}"
#     raw_dir = run_dir / "raw"
#     intermediate_dir = run_dir / "intermediate"
#     output_dir = run_dir / "outputs"
#     ensure_dirs(run_dir, raw_dir, intermediate_dir, output_dir)

#     # Download datasets
#     os.environ["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR
#     for slug in KAGGLE_DATASETS:
#         download_kaggle_dataset(slug, raw_dir)

#     # Detect & collect CSVs
#     product_tables, txn_tables = detect_tables(raw_dir)
#     prod_dim = build_product_dim(product_tables)
#     txn = build_transactions(txn_tables)

#     # Save cleaned/unified copies
#     prod_path = intermediate_dir / "products_unified.csv"
#     txn_path  = intermediate_dir / "transactions_unified.csv"
#     prod_dim.to_csv(prod_path, index=False)
#     txn.to_csv(txn_path, index=False)

#     # EDA summary
#     eda = basic_eda(prod_dim, txn)
#     (run_dir / "eda_summary.json").write_text(json.dumps(eda, indent=2))
#     print("\n=== EDA SUMMARY ===")
#     print(json.dumps(eda, indent=2))

#     # Train/test split on user-item pairs
#     pairs = txn[["customer_id","product_id"]].dropna().drop_duplicates()
#     if pairs.empty:
#         print("No interactions found. Check your dataset slugs or mapping rules.")
#         sys.exit(0)

#     shuffled = pairs.sample(frac=1.0, random_state=1).reset_index(drop=True)
#     split = int(0.8*len(shuffled))
#     train_pairs = shuffled.iloc[:split]
#     test_pairs  = shuffled.iloc[split:]

#     train_user_items = train_pairs.groupby("customer_id")["product_id"].apply(set).to_dict()
#     test_user_items  = test_pairs.groupby("customer_id")["product_id"].apply(set).to_dict()

#     popularity = train_pairs["product_id"].value_counts()

#     # Content-based similarity
#     sim, pid_to_idx, idx_to_pid = fit_content_sim(prod_dim)

#     # Item-Item co-occurrence
#     co, item_freq = build_item_co(train_pairs)

#     # Define wrappers for evaluation
#     pop_fn = lambda u,k: popularity_top(popularity, k)
#     cb_fn  = lambda u,k: cb_recommend_for_user(u, train_user_items, sim, pid_to_idx, idx_to_pid, k)
#     cf_fn  = lambda u,k: cf_recommend_for_user(u, train_user_items, co, item_freq, k)
#     hy_fn  = lambda u,k: hybrid_recommend_for_user(u, train_user_items, sim, pid_to_idx, idx_to_pid, co, item_freq, popularity, k)

#     users_eval = [u for u, items in test_user_items.items() if len(items)>0][:500]
#     p_pop, r_pop, n = precision_recall_at_k(pop_fn, users_eval, test_user_items, k=K)
#     p_cb,  r_cb,  _ = precision_recall_at_k(cb_fn,  users_eval, test_user_items, k=K)
#     p_cf,  r_cf,  _ = precision_recall_at_k(cf_fn,  users_eval, test_user_items, k=K)
#     p_h,   r_h,   _ = precision_recall_at_k(hy_fn,  users_eval, test_user_items, k=K)

#     metrics = pd.DataFrame([
#         {"model":f"Popularity@{K}",   "precision": p_pop, "recall": r_pop, "users_evaluated": n},
#         {"model":f"ContentBased@{K}", "precision": p_cb,  "recall": r_cb,  "users_evaluated": n},
#         {"model":f"ItemCF@{K}",       "precision": p_cf,  "recall": r_cf,  "users_evaluated": n},
#         {"model":f"Hybrid@{K}",       "precision": p_h,   "recall": r_h,   "users_evaluated": n},
#     ])
#     metrics_path = output_dir / "recommender_metrics.csv"
#     metrics.to_csv(metrics_path, index=False)

#     # Sample recommendations (first 50 users in train)
#     recs_rows = []
#     for u in list(train_user_items.keys())[:50]:
#         items = hy_fn(u, K)
#         for rank, pid in enumerate(items, start=1):
#             name = ""
#             if not prod_dim.empty:
#                 s = prod_dim.loc[prod_dim["product_id"]==pid, "product_name"]
#                 if len(s)>0: name = s.iloc[0]
#             recs_rows.append({"customer_id": u, "rank": rank, "product_id": pid, "product_name": name})
#     recs_df = pd.DataFrame(recs_rows)
#     recs_path = output_dir / "sample_hybrid_recommendations.csv"
#     recs_df.to_csv(recs_path, index=False)

#     # Export simple data dictionary
#     dict_rows = []
#     dict_rows += [{"table":"products_unified","column":c,"dtype":str(prod_dim[c].dtype)} for c in prod_dim.columns]
#     dict_rows += [{"table":"transactions_unified","column":c,"dtype":str(txn[c].dtype)} for c in txn.columns]
#     data_dict = pd.DataFrame(dict_rows)
#     data_dict.to_csv(output_dir / "data_dictionary.csv", index=False)

#     # Lightweight run report (Markdown)
#     md = []
#     md.append(f"# E-commerce Recommender Run — {run_ts}")
#     md.append("## Data Overview")
#     md.append(f"- Datasets: {', '.join(KAGGLE_DATASETS)}")
#     md.append(f"- Products: {eda['n_products']:,}")
#     md.append(f"- Transactions: {eda['n_transactions']:,}")
#     md.append(f"- Users: {eda['n_users']:,} | Items: {eda['n_items']:,}")
#     md.append(f"- Avg unique items per user: {eda['avg_items_per_user']:.2f}")
#     md.append("\n## Models & Metrics")
#     md.append(metrics.to_markdown(index=False))
#     md.append("\n## Sample Hybrid Recommendations (first 10 rows)")
#     md.append(recs_df.head(10).to_markdown(index=False))
#     (output_dir / "run_report.md").write_text("\n".join(md), encoding="utf-8")

#     print("\n=== OUTPUTS ===")
#     print(f"- Clean products:      {prod_path}")
#     print(f"- Clean transactions:  {txn_path}")
#     print(f"- Metrics:             {metrics_path}")
#     print(f"- Recommendations:     {recs_path}")
#     print(f"- Data dictionary:     {output_dir / 'data_dictionary.csv'}")
#     print(f"- Mini report:         {output_dir / 'run_report.md'}")
#     print("\nDone.")

# if __name__ == "__main__":
#     main()
