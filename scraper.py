import os, math, datetime, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CONFIG =====================
KAGGLE_DATASETS = [
    "carrie1/ecommerce-data",                  # Online Retail (UCI)
    "promptcloud/amazon-products-dataset",     # Amazon product catalog
    "karkavelrajaj/amazon-sales-dataset",      # Amazon sales transactions
    "thedevastator/online-retail-transactions-dataset"  # Large transaction dataset
]

PROJECT_BASE = Path(r"C:\Users\Just\commerce")
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
