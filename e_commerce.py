# e_commerce.py
import os, math, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use your local Windows-friendly data folder
BASE = r"C:\Users\JustMe\commerce\data"
os.makedirs(BASE, exist_ok=True)

# ---- Helper function to read CSV robustly ----
def robust_read(path):
    for args in [
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "python"},
        {"sep": ",", "encoding": "latin1", "engine": "python"},
        {"sep": ",", "encoding": "ISO-8859-1", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
    ]:
        try:
            return pd.read_csv(path, **args)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {path}")

# ---- Load datasets ----
paths = {
    "products": os.path.join(BASE, "product_details.csv"),
    "transactions": os.path.join(BASE, "transactions.csv"),
}

dfs = {}
for k, p in paths.items():
    if os.path.exists(p):
        dfs[k] = robust_read(p)
        dfs[k].columns = [c.strip().lower().replace(" ", "_") for c in dfs[k].columns]

# ---- Prepare product dimension ----
prod = dfs.get("products", pd.DataFrame())
if not prod.empty:
    cand = [c for c in ["product_id","asin","stockcode","unique_id","uniqe_id"] if c in prod.columns]
    if cand:
        prod["product_id"] = prod[cand[0]].astype(str)
    else:
        prod = prod.reset_index().rename(columns={"index":"product_id"})
        prod["product_id"] = prod["product_id"].astype(str)
    name_cols = [c for c in ["product_name","description","name","title"] if c in prod.columns]
    prod["product_name"] = prod[name_cols[0]] if name_cols else ""
    cat_cols = [c for c in ["category","main_category","product_category"] if c in prod.columns]
    prod["category"] = prod[cat_cols[0]] if cat_cols else ""
    brand_cols = [c for c in ["brand_name","brand"] if c in prod.columns]
    prod["brand"] = prod[brand_cols[0]] if brand_cols else ""
    prod_dim = prod[["product_id","product_name","category","brand"]].drop_duplicates()
else:
    prod_dim = pd.DataFrame(columns=["product_id","product_name","category","brand"])

# ---- Prepare transactions ----
txn = dfs.get("transactions", pd.DataFrame())
if not txn.empty:
    if len(txn) > 10000:
        txn = txn.sample(10000, random_state=42)
    order_col = [c for c in ["order_id","transaction_id","invoiceno"] if c in txn.columns]
    cust_col = [c for c in ["customer_id","customerid"] if c in txn.columns]
    prod_col = [c for c in ["product_id","stockcode","asin","uniqeid","product_name"] if c in txn.columns]
    qty_col = [c for c in ["quantity","qty","count"] if c in txn.columns]
    txn["order_id"] = txn[order_col[0]] if order_col else np.arange(len(txn)).astype(str)
    txn["customer_id"] = txn[cust_col[0]].astype(str) if cust_col else "unknown"
    txn["product_id"] = txn[prod_col[0]].astype(str) if prod_col else txn.get("product_name","unknown").astype(str)
    txn["quantity"] = pd.to_numeric(txn[qty_col[0]], errors="coerce").fillna(1) if qty_col else 1
    pairs = txn[["customer_id","product_id"]].dropna().drop_duplicates()
else:
    pairs = pd.DataFrame(columns=["customer_id","product_id"])

# ---- Train/Test split ----
shuffled = pairs.sample(frac=1.0, random_state=1).reset_index(drop=True)
split = int(0.8*len(shuffled))
train_pairs = shuffled.iloc[:split]
test_pairs = shuffled.iloc[split:]

train_user_items = train_pairs.groupby("customer_id")["product_id"].apply(set).to_dict()
test_user_items = test_pairs.groupby("customer_id")["product_id"].apply(set).to_dict()

# ---- Popularity model ----
popularity = train_pairs["product_id"].value_counts()
def recommend_popular(k=10):
    return list(popularity.head(k).index)

# ---- Content-based model ----
def build_text(df):
    def txt(row):
        return " ".join([str(row.get("product_name","")), str(row.get("category","")), str(row.get("brand",""))])
    df = df.copy()
    df["text"] = df.apply(txt, axis=1)
    return df

if not prod_dim.empty:
    prod_dim = build_text(prod_dim)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(prod_dim["text"])
    sim = cosine_similarity(X)
    pid_to_idx = {pid:i for i, pid in enumerate(prod_dim["product_id"].astype(str))}
    idx_to_pid = {i:pid for pid,i in pid_to_idx.items()}
else:
    sim = None
    pid_to_idx = {}
    idx_to_pid = {}

def cb_recommend_for_user(user_id, k=10):
    if sim is None: return []
    items = list(train_user_items.get(user_id, []))
    idxs = [pid_to_idx[i] for i in items if i in pid_to_idx]
    if not idxs: return []
    scores = np.zeros(sim.shape[0])
    for ix in idxs:
        scores = np.maximum(scores, sim[ix])
    for ix in idxs: scores[ix] = -1.0
    top = np.argpartition(scores, -k)[-k:]
    ranked = top[np.argsort(scores[top])[::-1]]
    return [idx_to_pid[i] for i in ranked]

# ---- Collaborative Filtering (item-based co-occurrence) ----
co = {}
item_freq = train_pairs["product_id"].value_counts().to_dict()
for _, grp in train_pairs.groupby("customer_id"):
    items = list(set(grp["product_id"].tolist()))
    if len(items) > 50:
        items = items[:50]
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

# ---- Hybrid model ----
def hybrid_recommend_for_user(user_id, k=10, w_cb=0.5, w_cf=0.5):
    cb = cb_recommend_for_user(user_id, k=50)
    cf = cf_recommend_for_user(user_id, k=50)
    pop = recommend_popular(k=50)
    scores = {}
    for rank, pid in enumerate(cb): scores[pid] = scores.get(pid,0)+w_cb/(rank+1)
    for rank, pid in enumerate(cf): scores[pid] = scores.get(pid,0)+w_cf/(rank+1)
    for rank, pid in enumerate(pop): scores[pid] = scores.get(pid,0)+0.1/(rank+1)
    if not scores: return pop[:k]
    seen = set(train_user_items.get(user_id, []))
    ranked = [p for p,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True) if p not in seen]
    return ranked[:k]

# ---- Evaluation ----
def precision_recall_at_k(fn, k=10, max_users=200):
    users = [u for u, items in test_user_items.items() if len(items)>0]
    users = users[:max_users]
    if not users: return np.nan, np.nan, 0
    precisions, recalls = [], []
    for u in users:
        recs = set(fn(u,k))
        truth = test_user_items.get(u, set())
        tp = len(recs & truth)
        if len(recs)>0: precisions.append(tp/len(recs))
        if len(truth)>0: recalls.append(tp/len(truth))
    return float(np.nanmean(precisions)), float(np.nanmean(recalls)), len(users)

p_pop, r_pop, n = precision_recall_at_k(lambda u,k: recommend_popular(k), k=10)
p_cb, r_cb, _ = precision_recall_at_k(cb_recommend_for_user, k=10)
p_cf, r_cf, _ = precision_recall_at_k(cf_recommend_for_user, k=10)
p_h, r_h, _ = precision_recall_at_k(lambda u,k: hybrid_recommend_for_user(u,k,0.5,0.5), k=10)

metrics = pd.DataFrame([
    {"model":"Popularity@10", "precision": p_pop, "recall": r_pop, "users_evaluated": n},
    {"model":"ContentBased@10", "precision": p_cb, "recall": r_cb, "users_evaluated": n},
    {"model":"ItemCF@10", "precision": p_cf, "recall": r_cf, "users_evaluated": n},
    {"model":"Hybrid@10", "precision": p_h, "recall": r_h, "users_evaluated": n},
])

# ---- Generate recommendations for sample users ----
recs_rows = []
for u in list(train_user_items.keys())[:20]:
    for rank, pid in enumerate(hybrid_recommend_for_user(u, k=10), start=1):
        name = ""
        if not prod_dim.empty:
            s = prod_dim.loc[prod_dim["product_id"]==pid, "product_name"]
            name = s.iloc[0] if len(s)>0 else ""
        recs_rows.append({"customer_id": u, "rank": rank, "product_id": pid, "product_name": name})
recs_df = pd.DataFrame(recs_rows)

# ---- Save outputs ----
metrics_path = os.path.join(BASE, "recommender_metrics_light.csv")
recs_path = os.path.join(BASE, "sample_hybrid_recommendations_light.csv")
metrics.to_csv(metrics_path, index=False)
recs_df.to_csv(recs_path, index=False)

print(f"✅ Saved metrics to: {metrics_path}")
print(f"✅ Saved recommendations to: {recs_path}")
