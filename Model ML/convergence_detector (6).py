"""
Convergence Event Detector v3
==============================
Calibrated on 3 known convergences:
  - H01M x B60L (Electric vehicles, 2012)    -> Jaccard x4.1, I_conv=0.726
  - H04W x G06F (Smartphones, 2007)          -> Jaccard x5.6, I_conv=0.858
  - C12N x G01N (Biotech diagnostics, 1997)  -> Jaccard x1.69, I_conv=0.188

The model learns the convergence signature from these cases and generalizes
to all ~180k CPC pairs to detect emerging convergences and predict ~2026.

Threshold: Jaccard x2.0 (conservative — catches VE and Smartphone clearly,
borderline Biotech which is classified as co-evolution, not convergence).

Usage:
    pip install pandas pyarrow networkx scikit-learn matplotlib scipy
    python3 convergence_detector.py

Authors: Benchetrit Q., Faye S., Lesne J. — ENSAE 2A — StatApp 2025-2026
Supervisor: Antonin Bergeaud (HEC Paris)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")

# ===========================================================================
# PARAMETERS
# ===========================================================================
PARQUET_PATH = "/home/onyxia/work/Stat-App/patents_merged.parquet"
OUTPUT_DIR   = "/home/onyxia/work/Stat-App"

WINDOW       = 5
HORIZON      = 5
N_CLUSTERS   = 10
PCA_DIM      = 20
TOP_K_PAIRS  = 800

# Convergence event threshold calibrated on 3 cases:
#   VE: x4.1, Smartphone: x5.6, Biotech: x1.69
# x2.0 is conservative: captures real convergences (VE, Smartphone)
# while correctly excluding co-evolutions (Biotech at x1.69 < 2.0)
JACCARD_MULT_THRESHOLD = 2.0

# Symmetry ratio thresholds (from Method 4, validated on 3 cases)
#   > 0.6 = symmetric bilateral convergence (Biotech: 0.747)
#   0.3 - 0.6 = asymmetric bilateral convergence (VE: 0.42, Smartphone: 0.41)
#   < 0.3 = GPT integration, not real convergence
SYMMETRY_THRESHOLD_GPT = 0.3
SYMMETRY_THRESHOLD_SYM = 0.6

# Known convergence cases for backtest validation
# Format: (code1, code2, emergence_year, name, jaccard_mult, i_conv)
KNOWN_CASES = [
    ("H01M", "B60L", 2012, "Electric vehicles",     4.1, 0.726),
    ("H04W", "G06F", 2007, "Smartphones",            5.6, 0.858),
    ("C12N", "G01N", 1997, "Biotech diagnostics",    1.69, 0.188),
]

# Backtest cutoffs: test at various points before known emergences
# 2007: 5 yrs before VE, same year as Smartphone
# 2003: 4 yrs before Smartphone, 9 yrs before VE
# 2010: 2 yrs before VE, post-Smartphone validation
# 1993: 4 yrs before Biotech
BACKTEST_CUTOFFS = [1993, 2003, 2007, 2010]


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_data(path):
    print("[1/8] Loading patent data...")
    t0 = time.time()
    try:
        import polars as pl
        df = pl.read_parquet(path, columns=["publication_date", "cpc4_list"])
        df = df.with_columns((pl.col("publication_date") // 10000).alias("year"))
        df = df.filter(pl.col("cpc4_list").list.len() >= 2)
        df = df.select(["year", "cpc4_list"]).to_pandas()
    except ImportError:
        df = pd.read_parquet(path, columns=["publication_date", "cpc4_list"])
        df["year"] = df["publication_date"] // 10000
        df = df[df["cpc4_list"].apply(len) >= 2][["year", "cpc4_list"]].reset_index(drop=True)

    # exclude Y codes (transversal EPO tags, not real convergence)
    def filter_y(codes):
        return [c for c in codes if not c.startswith("Y")]
    df["cpc4_list"] = df["cpc4_list"].apply(filter_y)
    df = df[df["cpc4_list"].apply(len) >= 2].reset_index(drop=True)
    print(f"   {len(df):,} patents | {int(df['year'].min())}-{int(df['year'].max())} | {time.time()-t0:.1f}s")
    return df


# ===========================================================================
# 2. SINGLE-PASS PRE-COMPUTATION
# ===========================================================================
def precompute_all(df):
    """
    One pass over all patents to build:
      - weighted pair records (for graphs)
      - annual code counts (for fast correlation on diff2)
      - annual pair counts (for Jaccard without iterrows)
    """
    print("[2/8] Pre-computing all data structures (single pass)...")
    t0 = time.time()
    patents_per_year = df.groupby("year").size().to_dict()
    mean_patents = np.mean(list(patents_per_year.values()))

    pair_c1, pair_c2, pair_year, pair_weight = [], [], [], []
    annual_code_counts = defaultdict(lambda: defaultdict(int))
    annual_pair_counts = defaultdict(lambda: defaultdict(int))

    years_arr, cpc_arr = df["year"].values, df["cpc4_list"].values
    n = len(df)
    report = max(n // 10, 1)

    for idx in range(n):
        codes = cpc_arr[idx]
        if not isinstance(codes, (list, np.ndarray)):
            continue
        codes = sorted(set(codes))
        m = len(codes)
        if m < 2:
            continue
        yr = int(years_arr[idx])
        w = (2.0 / (m * (m - 1))) * (mean_patents / patents_per_year.get(yr, mean_patents))

        for c in codes:
            annual_code_counts[yr][c] += 1
        for c1, c2 in combinations(codes, 2):
            pair_c1.append(c1)
            pair_c2.append(c2)
            pair_year.append(yr)
            pair_weight.append(w)
            annual_pair_counts[yr][(c1, c2)] += 1

        if (idx + 1) % report == 0:
            print(f"   {(idx+1)/n*100:.0f}% — {time.time()-t0:.0f}s")

    pairs_df = pd.DataFrame({"c1": pair_c1, "c2": pair_c2, "year": pair_year, "weight": pair_weight})
    print(f"   {len(pairs_df):,} pair-records | {time.time()-t0:.0f}s")
    return pairs_df, patents_per_year, mean_patents, dict(annual_code_counts), dict(annual_pair_counts)


# ===========================================================================
# 3. GRAPHS + JACCARD (vectorized)
# ===========================================================================
def build_graphs_and_jaccard(pairs_df, annual_code_counts, annual_pair_counts, year_min, year_max):
    print("[3/8] Building graphs and Jaccard matrices...")
    t0 = time.time()
    half = WINDOW // 2
    centers = list(range(year_min + half, year_max - half + 1))

    graphs, jaccards, code_counts_w = {}, {}, {}

    for i, center in enumerate(centers):
        start, end = center - half, center + half

        # weighted graph
        mask = (pairs_df["year"] >= start) & (pairs_df["year"] <= end)
        agg = pairs_df.loc[mask].groupby(["c1", "c2"], sort=False)["weight"].sum()
        G = nx.Graph()
        for (c1, c2), w in agg.items():
            G.add_edge(c1, c2, weight=w)
        graphs[center] = G

        # aggregate counts over window for Jaccard
        cc = defaultdict(int)
        pc = defaultdict(int)
        for yr in range(start, end + 1):
            for code, cnt in annual_code_counts.get(yr, {}).items():
                cc[code] += cnt
            for pair, cnt in annual_pair_counts.get(yr, {}).items():
                pc[pair] += cnt
        code_counts_w[center] = dict(cc)

        jacc = {}
        for (c1, c2), n_both in pc.items():
            denom = cc.get(c1, 0) + cc.get(c2, 0) - n_both
            jacc[(c1, c2)] = n_both / denom if denom > 0 else 0
        jaccards[center] = jacc

        if (i + 1) % 10 == 0 or i == 0:
            print(f"   {i+1}/{len(centers)} | {time.time()-t0:.0f}s")

    print(f"   {len(graphs)} periods | {time.time()-t0:.0f}s")
    return graphs, jaccards, code_counts_w


# ===========================================================================
# 4. CLUSTERING + ABSORPTION (cached, aligned)
# ===========================================================================
_cluster_cache = {}
_pca_vectors_cache = {}
_absorption_cache = {}

def compute_clustering_full(G, year):
    if year in _cluster_cache:
        return _cluster_cache[year], _pca_vectors_cache[year]
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n < N_CLUSTERS + 1:
        _cluster_cache[year] = {nd: 0 for nd in nodes}
        _pca_vectors_cache[year] = {nd: np.zeros(PCA_DIM) for nd in nodes}
        return _cluster_cache[year], _pca_vectors_cache[year]

    adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    M = adj / rs
    nc = min(PCA_DIM, n - 1)
    if nc < 2:
        _cluster_cache[year] = {nd: 0 for nd in nodes}
        _pca_vectors_cache[year] = {nd: np.zeros(PCA_DIM) for nd in nodes}
    else:
        reduced = PCA(n_components=nc).fit_transform(M)
        labs = KMeans(n_clusters=min(N_CLUSTERS, n), n_init=10, random_state=42).fit_predict(reduced)
        _cluster_cache[year] = dict(zip(nodes, labs))
        _pca_vectors_cache[year] = dict(zip(nodes, reduced))
    return _cluster_cache[year], _pca_vectors_cache[year]

def align_clusters(labels_prev, labels_curr):
    if not labels_prev or not labels_curr:
        return labels_curr
    common = set(labels_prev.keys()) & set(labels_curr.keys())
    if not common:
        return labels_curr
    k = max(max(labels_prev.values()), max(labels_curr.values())) + 1
    conf = np.zeros((k, k), dtype=int)
    for nd in common:
        conf[labels_prev[nd], labels_curr[nd]] += 1
    _, col_ind = linear_sum_assignment(-conf)
    mapping = {col_ind[i]: i for i in range(len(col_ind))}
    return {nd: mapping.get(lab, lab) for nd, lab in labels_curr.items()}

def compute_absorption(graphs, year):
    if year in _absorption_cache:
        return _absorption_cache[year]
    py = year - 1
    if py not in _cluster_cache or year not in _cluster_cache:
        _absorption_cache[year] = {}
        return {}
    lp, lc = _cluster_cache[py], _cluster_cache[year]
    common = set(lp.keys()) & set(lc.keys())
    if not common:
        _absorption_cache[year] = {}
        return {}
    k = max(max(lc.values()), max(lp.values())) + 1
    T = np.zeros((k, k), dtype=int)
    for nd in common:
        T[lp[nd], lc[nd]] += 1
    absorb = {}
    for c in range(k):
        tot = T[c, :].sum()
        persist = T[c, c] / tot if tot > 0 else 0
        entries = T[:, c].copy(); entries[c] = 0; te = entries.sum()
        h_in = -np.sum((entries[entries > 0] / te) * np.log(entries[entries > 0] / te + 1e-12)) if te > 0 else 0
        absorb[c] = h_in * persist
    _absorption_cache[year] = absorb
    return absorb


# ===========================================================================
# 5. FAST ANNUAL ARRAYS (for correlation)
# ===========================================================================
_annual_arrays = {}
_all_years = []

def precompute_annual_arrays(annual_code_counts, year_min, year_max):
    global _annual_arrays, _all_years
    print("   Building annual count arrays...")
    _all_years = list(range(year_min, year_max + 1))
    all_codes = set()
    for yr_data in annual_code_counts.values():
        all_codes.update(yr_data.keys())
    for code in all_codes:
        _annual_arrays[code] = np.array(
            [annual_code_counts.get(y, {}).get(code, 0) for y in _all_years], dtype=float)

def fast_corr_diff2(c1, c2, year, window=5):
    """Correlation on second differences — pure numpy, no iterrows."""
    if c1 not in _annual_arrays or c2 not in _annual_arrays:
        return 0.0
    yr0 = _all_years[0]
    si = max(0, (year - window - 2) - yr0)
    ei = min(len(_all_years), year - yr0 + 1)
    if ei - si < 5:
        return 0.0
    d2a = np.diff(_annual_arrays[c1][si:ei], n=2)
    d2b = np.diff(_annual_arrays[c2][si:ei], n=2)
    if len(d2a) < 3 or np.std(d2a) < 1e-10 or np.std(d2b) < 1e-10:
        return 0.0
    return float(np.corrcoef(d2a, d2b)[0, 1])


# ===========================================================================
# 6. FEATURE EXTRACTION — convergence signature from 3 case studies
# ===========================================================================
FEATURE_COLS = [
    # M1 — Centrality (precursor signal T-1 to T-6 years, from 3 cases)
    "strength_slope_c1",        # strength trend (T-5 to T)
    "strength_slope_c2",
    "strength_product",         # joint structural importance
    # M2 — Clustering (absorption peaks at T-1 to T-7, from 3 cases)
    "same_cluster",             # VE: never, Smartphone: always, Biotech: partial
    "absorption_max",           # peak values: 0.78 (VE), 0.37 (Smart), 0.33 (Bio)
    "acp_distance",             # euclidean in PCA-20 space
    "acp_distance_slope",       # universal range: -20% to -32% decrease
    # M3 — Time series (correlation on diff2, validated I(0))
    "rolling_corr_diff2",       # lag 0 optimal for all 3 cases
    # M4 — Jaccard + conditional probabilities
    "jaccard_current",          # direct co-occurrence intensity
    "jaccard_slope",            # pre-convergence trend
    "jaccard_acceleration",     # second derivative of Jaccard
    "symmetry_ratio",           # <0.3=GPT, 0.3-0.6=asym, >0.6=sym
    "cond_prob_max",            # max(P(j|i), P(i|j))
    "cond_prob_min",            # min — low value + high max = asymmetric
]

def get_strength(graphs, code, year):
    if year in graphs and code in graphs[year]:
        return graphs[year].degree(code, weight="weight")
    return 0.0

def extract_features(graphs, jaccards, code_counts, c1, c2, year):
    f = {}
    G = graphs[year]
    yb = [year - i for i in range(5) if (year - i) in graphs]

    # === M1 — Centrality: slope captures precursor acceleration ===
    if len(yb) >= 3:
        ybs = sorted(yb)
        tv = np.arange(len(ybs), dtype=float)
        f["strength_slope_c1"] = np.polyfit(tv, [get_strength(graphs, c1, y) for y in ybs], 1)[0]
        f["strength_slope_c2"] = np.polyfit(tv, [get_strength(graphs, c2, y) for y in ybs], 1)[0]
    else:
        f["strength_slope_c1"] = f["strength_slope_c2"] = 0
    f["strength_product"] = get_strength(graphs, c1, year) * get_strength(graphs, c2, year)

    # === M2 — Clustering: absorption + ACP distance ===
    labels, vecs = compute_clustering_full(G, year)
    f["same_cluster"] = int(labels.get(c1, -1) == labels.get(c2, -1))

    absorb = compute_absorption(graphs, year)
    cl1, cl2 = labels.get(c1, 0), labels.get(c2, 0)
    f["absorption_max"] = max(absorb.get(cl1, 0), absorb.get(cl2, 0))

    v1 = vecs.get(c1, np.zeros(PCA_DIM))
    v2 = vecs.get(c2, np.zeros(PCA_DIM))
    f["acp_distance"] = float(np.linalg.norm(v1 - v2))

    # ACP distance slope: universal -20% to -32% decrease across 3 cases
    dh = []
    for y in sorted(yb[:3]):
        if y in _pca_vectors_cache:
            dh.append(float(np.linalg.norm(
                _pca_vectors_cache[y].get(c1, np.zeros(PCA_DIM)) -
                _pca_vectors_cache[y].get(c2, np.zeros(PCA_DIM)))))
    f["acp_distance_slope"] = np.polyfit(np.arange(len(dh), dtype=float), dh, 1)[0] if len(dh) >= 2 else 0

    # === M3 — Correlation on diff2 (lag 0 optimal for all 3 cases) ===
    f["rolling_corr_diff2"] = fast_corr_diff2(c1, c2, year)

    # === M4 — Jaccard + symmetry (key classifier from 3 cases) ===
    pk = (min(c1, c2), max(c1, c2))
    f["jaccard_current"] = jaccards.get(year, {}).get(pk, 0)

    jh = [jaccards.get(y, {}).get(pk, 0) for y in sorted(yb)]
    if len(jh) >= 3:
        tv = np.arange(len(jh), dtype=float)
        f["jaccard_slope"] = np.polyfit(tv, jh, 1)[0]
        f["jaccard_acceleration"] = float(np.mean(np.diff(np.diff(jh)))) if len(jh) >= 4 else 0
    else:
        f["jaccard_slope"] = f["jaccard_acceleration"] = 0

    # conditional probabilities — both directions
    cc = code_counts.get(year, {})
    n1, n2 = cc.get(c1, 0), cc.get(c2, 0)
    w_edge = G[c1][c2]["weight"] if G.has_edge(c1, c2) else 0
    p_ji = w_edge / n1 if n1 > 0 else 0  # P(c2|c1)
    p_ij = w_edge / n2 if n2 > 0 else 0  # P(c1|c2)
    p_max, p_min = max(p_ji, p_ij), min(p_ji, p_ij)
    f["symmetry_ratio"] = p_min / p_max if p_max > 0 else 0
    f["cond_prob_max"] = p_max
    f["cond_prob_min"] = p_min

    return f


# ===========================================================================
# 7. BUILD DATASET
# ===========================================================================
def build_dataset(graphs, jaccards, code_counts):
    print("[4/8] Building classification dataset...")
    t0 = time.time()

    sy = sorted(graphs.keys())
    valid = [y for y in sy if (y + HORIZON) in jaccards]
    print(f"   {len(valid)} years ({min(valid)}-{max(valid)})")

    # clustering alignment
    print("   Aligning clusters across all years...")
    prev = None
    for y in sy:
        compute_clustering_full(graphs[y], y)
        if prev is not None:
            _cluster_cache[y] = align_clusters(prev, _cluster_cache[y])
        prev = _cluster_cache[y]
    for y in sy:
        compute_absorption(graphs, y)

    records = []
    for i, year in enumerate(valid):
        G = graphs[year]
        edges = sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:TOP_K_PAIRS]
        jn = jaccards.get(year, {})
        jf = jaccards.get(year + HORIZON, {})

        for c1, c2, _ in edges:
            pk = (min(c1, c2), max(c1, c2))
            j_now = jn.get(pk, 0)
            j_fut = jf.get(pk, 0)

            # target: convergence event
            if j_now < 1e-6:
                target = 1 if j_fut > 0.01 else 0
            else:
                target = 1 if (j_fut / j_now) >= JACCARD_MULT_THRESHOLD else 0

            feats = extract_features(graphs, jaccards, code_counts, c1, c2, year)
            feats["year"] = year
            feats["c1"] = c1
            feats["c2"] = c2
            feats["target"] = target
            records.append(feats)

        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(valid)} | {time.time()-t0:.0f}s")

    ds = pd.DataFrame(records)
    np_ = ds["target"].sum()
    print(f"   {len(ds):,} samples | {np_} events ({np_/len(ds)*100:.1f}%) | {time.time()-t0:.0f}s")
    return ds


# ===========================================================================
# 8. TRAIN
# ===========================================================================
def train_model(dataset, max_train_year=None):
    if max_train_year is None:
        uy = sorted(dataset["year"].unique())
        max_train_year = uy[int(len(uy) * 0.75)]

    tr = dataset["year"] <= max_train_year
    te = dataset["year"] > max_train_year
    Xtr, ytr = dataset.loc[tr, FEATURE_COLS].values, dataset.loc[tr, "target"].values
    Xte, yte = dataset.loc[te, FEATURE_COLS].values, dataset.loc[te, "target"].values

    print(f"   Train: {tr.sum():,} (<= {max_train_year}), {ytr.sum()} events ({ytr.mean()*100:.1f}%)")
    if te.sum() > 0:
        print(f"   Test:  {te.sum():,} (> {max_train_year}), {yte.sum()} events ({yte.mean()*100:.1f}%)")

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=15, random_state=42)
    model.fit(Xtr, ytr)

    if te.sum() > 0 and yte.sum() > 0:
        yp = model.predict_proba(Xte)[:, 1]
        ypred = model.predict(Xte)
        print(f"\n   AUC={roc_auc_score(yte, yp):.4f} "
              f"Prec={precision_score(yte, ypred, zero_division=0):.4f} "
              f"Rec={recall_score(yte, ypred, zero_division=0):.4f} "
              f"F1={f1_score(yte, ypred, zero_division=0):.4f}")

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n   Feature importances:")
    for feat, val in imp.items():
        print(f"   {feat:25s} {val:.4f}  {'#' * int(val * 60)}")
    return model, imp


# ===========================================================================
# 9. BACKTEST — validate on known convergences
# ===========================================================================
def run_backtest(dataset, graphs, jaccards, code_counts, cutoff):
    target_yr = cutoff + HORIZON
    print(f"\n{'='*65}")
    print(f"  BACKTEST: train <= {cutoff}, predict convergences by {target_yr}")
    print(f"{'='*65}")

    td = dataset[dataset["year"] + HORIZON <= cutoff].copy()
    if len(td) < 100 or td["target"].sum() < 3:
        print(f"   SKIP: {len(td)} samples, {td['target'].sum()} events")
        return None

    model, imp = train_model(td, max_train_year=cutoff)

    if cutoff not in graphs:
        print(f"   SKIP: no graph at {cutoff}")
        return None

    G = graphs[cutoff]
    edges = sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:TOP_K_PAIRS]

    preds = []
    for c1, c2, d in edges:
        feats = extract_features(graphs, jaccards, code_counts, c1, c2, cutoff)
        X = np.array([[feats[col] for col in FEATURE_COLS]])
        prob = model.predict_proba(X)[0][1]

        pk = (min(c1, c2), max(c1, c2))
        jn = jaccards.get(cutoff, {}).get(pk, 0)
        jf = jaccards.get(target_yr, {}).get(pk, 0)
        ac = (1 if (jf / jn) >= JACCARD_MULT_THRESHOLD else 0) if jn > 1e-6 else (1 if jf > 0.01 else 0)

        # symmetry classification
        sym = feats.get("symmetry_ratio", 0)
        if sym < SYMMETRY_THRESHOLD_GPT:
            conv_type = "GPT"
        elif sym < SYMMETRY_THRESHOLD_SYM:
            conv_type = "ASYM"
        else:
            conv_type = "SYM"

        preds.append({
            "cpc_1": c1, "cpc_2": c2,
            "convergence_prob": round(prob, 4),
            "actually_converged": ac,
            "jaccard_cutoff": round(jn, 6),
            "jaccard_target": round(jf, 6),
            "convergence_type": conv_type,
        })

    pdf = pd.DataFrame(preds).sort_values("convergence_prob", ascending=False)

    # check known cases
    print(f"\n   === KNOWN CASE DETECTION ===")
    for c1k, c2k, emerg, name, jmult, iconv in KNOWN_CASES:
        # only relevant if cutoff < emergence
        if cutoff >= emerg:
            print(f"   {name}: cutoff {cutoff} >= emergence {emerg}, skip (post-hoc)")
            continue
        match = pdf[((pdf["cpc_1"] == c1k) & (pdf["cpc_2"] == c2k)) |
                     ((pdf["cpc_1"] == c2k) & (pdf["cpc_2"] == c1k))]
        if len(match) > 0:
            row = match.iloc[0]
            rank = pdf.reset_index(drop=True)
            rank_idx = rank[(rank["cpc_1"] == row["cpc_1"]) & (rank["cpc_2"] == row["cpc_2"])].index[0] + 1
            pct = rank_idx / len(pdf) * 100
            status = "DETECTED" if row["convergence_prob"] > 0.5 else f"prob={row['convergence_prob']:.3f}"
            print(f"   {name} ({c1k}<->{c2k}, will emerge ~{emerg}):")
            print(f"     rank {rank_idx}/{len(pdf)} (top {pct:.1f}%) | prob={row['convergence_prob']:.4f} | [{status}]")
            print(f"     J: {row['jaccard_cutoff']:.5f} -> {row['jaccard_target']:.5f} | converged={bool(row['actually_converged'])}")
        else:
            print(f"   {name}: NOT in top-{TOP_K_PAIRS} at {cutoff}")

    # top 15
    print(f"\n   === TOP 15 PREDICTIONS ===")
    for _, r in pdf.head(15).iterrows():
        hit = "HIT" if r["actually_converged"] else "---"
        print(f"   {r['cpc_1']:5s}<->{r['cpc_2']:5s} p={r['convergence_prob']:.3f} [{hit}] {r['convergence_type']}")

    for k in [20, 50, 100]:
        if len(pdf) >= k:
            print(f"   Precision@{k}: {pdf.head(k)['actually_converged'].mean():.1%}")

    return pdf, imp


# ===========================================================================
# 10. FUTURE PREDICTIONS
# ===========================================================================
def predict_future(model, graphs, jaccards, code_counts):
    last = max(graphs.keys())
    target = last + HORIZON
    print(f"\n[7/8] Future predictions: {last} -> ~{target}...")

    G = graphs[last]
    edges = sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:TOP_K_PAIRS]
    labels = _cluster_cache.get(last, {})

    rows = []
    for c1, c2, d in edges:
        feats = extract_features(graphs, jaccards, code_counts, c1, c2, last)
        X = np.array([[feats[col] for col in FEATURE_COLS]])
        prob = model.predict_proba(X)[0][1]

        sym = feats.get("symmetry_ratio", 0)
        if sym < SYMMETRY_THRESHOLD_GPT:
            ctype = "GPT"
        elif sym < SYMMETRY_THRESHOLD_SYM:
            ctype = "ASYM"
        else:
            ctype = "SYM"

        rows.append({
            "cpc_1": c1, "cpc_2": c2,
            "convergence_prob": round(prob, 4),
            "horizon": f"{last}-{target}",
            "convergence_type": ctype,
            "jaccard_current": round(feats["jaccard_current"], 6),
            "jaccard_slope": round(feats["jaccard_slope"], 8),
            "symmetry_ratio": round(feats["symmetry_ratio"], 4),
            "acp_distance": round(feats["acp_distance"], 4),
            "absorption_max": round(feats["absorption_max"], 4),
            "corr_diff2": round(feats["rolling_corr_diff2"], 4),
            "same_cluster": int(labels.get(c1, -1) == labels.get(c2, -1)),
        })

    res = pd.DataFrame(rows).sort_values("convergence_prob", ascending=False)

    print(f"\n   === TOP 30 PREDICTED CONVERGENCES by ~{target} ===")
    for _, r in res.head(30).iterrows():
        inter = " [INTER]" if not r["same_cluster"] else ""
        print(f"   {r['cpc_1']:5s}<->{r['cpc_2']:5s} "
              f"p={r['convergence_prob']:.3f} {r['convergence_type']:4s} "
              f"sym={r['symmetry_ratio']:.2f} "
              f"J_slope={r['jaccard_slope']:+.6f}{inter}")
    return res


# ===========================================================================
# 11. PLOTS
# ===========================================================================
def plot_results(imp, future, bt_results):
    print("[8/8] Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Convergence Event Detector — calibrated on VE + Smartphone + Biotech",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(range(len(imp)), imp.values, color=colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Which signal best predicts convergence?")
    ax.invert_yaxis()

    ax = axes[1]
    top30 = future.head(30)
    labs = [f"{r['cpc_1']}<->{r['cpc_2']}" for _, r in top30.iterrows()]
    cmap = {"GPT": "#95a5a6", "ASYM": "#e74c3c", "SYM": "#3498db"}
    cb = [cmap.get(r["convergence_type"], "#3498db") for _, r in top30.iterrows()]
    ax.barh(range(len(top30)), top30["convergence_prob"].values, color=cb)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(labs, fontsize=7)
    h = future["horizon"].iloc[0] if len(future) > 0 else "?"
    ax.set_xlabel(f"P(convergence by ~{h})")
    ax.set_title("Top 30 (red=asymmetric, blue=symmetric, gray=GPT)")
    ax.invert_yaxis()
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/convergence_detector_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    if bt_results:
        n = len(bt_results)
        fig2, axes2 = plt.subplots(1, n, figsize=(6.5 * n, 6))
        if n == 1: axes2 = [axes2]
        for idx, (cut, (pdf, _)) in enumerate(bt_results.items()):
            ax = axes2[idx]
            t20 = pdf.head(20).reset_index(drop=True)
            labs = [f"{r['cpc_1']}<->{r['cpc_2']}" for _, r in t20.iterrows()]
            cols = ["#27ae60" if r["actually_converged"] else "#e74c3c" for _, r in t20.iterrows()]
            ax.barh(range(len(t20)), t20["convergence_prob"].values, color=cols)
            ax.set_yticks(range(len(t20)))
            ax.set_yticklabels(labs, fontsize=7)
            ax.set_title(f"Backtest {cut}->{cut+HORIZON}\ngreen=converged")
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/convergence_detector_backtest.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"   Saved to {OUTPUT_DIR}/")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 65)
    print("  CONVERGENCE EVENT DETECTOR v3")
    print(f"  Calibrated on 3 cases | Jaccard x{JACCARD_MULT_THRESHOLD} | Horizon {HORIZON}y")
    print("=" * 65)

    df = load_data(PARQUET_PATH)
    ymin, ymax = int(df["year"].min()), int(df["year"].max())
    pairs_df, _, _, acc, apc = precompute_all(df)
    del df

    graphs, jaccards, cc = build_graphs_and_jaccard(pairs_df, acc, apc, ymin, ymax)
    del pairs_df
    precompute_annual_arrays(acc, ymin, ymax)

    dataset = build_dataset(graphs, jaccards, cc)

    print("\n[5/8] Training full model...")
    model, imp = train_model(dataset)

    print("\n[6/8] Backtests on known cases...")
    bt = {}
    for cut in BACKTEST_CUTOFFS:
        r = run_backtest(dataset, graphs, jaccards, cc, cut)
        if r:
            bt[cut] = r
            r[0].to_csv(f"{OUTPUT_DIR}/detector_backtest_{cut}.csv", index=False)

    future = predict_future(model, graphs, jaccards, cc)
    future.to_csv(f"{OUTPUT_DIR}/detector_future_predictions.csv", index=False)
    dataset.to_csv(f"{OUTPUT_DIR}/detector_dataset.csv", index=False)
    plot_results(imp, future, bt)

    total = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  DONE in {total/60:.1f} min")
    print(f"  detector_future_predictions.csv  (convergences by ~{ymax + HORIZON})")
    for c in bt: print(f"  detector_backtest_{c}.csv")
    print(f"  detector_dataset.csv")
    print(f"  convergence_detector_results.png")
    print(f"  convergence_detector_backtest.png")
    print(f"{'='*65}")
    return model, future, dataset

if __name__ == "__main__":
    model, future, dataset = main()
