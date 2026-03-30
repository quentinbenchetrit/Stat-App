"""
Predictive Technological Convergence Index
===========================================
Temporal Link Prediction using Gradient Boosting on CPC co-occurrence networks.

We predict the future co-occurrence intensity between CPC code pairs at horizon
t+5, using features derived from four complementary axes: centrality, clustering,
topological proximity, and temporal dynamics. The aggregated predictions form our
synthetic convergence index.

Data: European patents from Google Patents (1980-2023), CPC codes at 4-digit level.

Usage:
    pip install pandas pyarrow networkx scikit-learn matplotlib
    python3 convergence_index.py

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

WINDOW_SIZE = 5        # sliding window width (years)
HORIZON = 5            # prediction horizon (years)
N_CLUSTERS = 10        # number of CPC clusters
TOP_K_PAIRS = 500      # most active pairs per period for ML dataset
MIN_COOCCURRENCES = 5  # minimum edge weight to consider a pair
PCA_COMPONENTS = 20    # dimensions kept before KMeans

# Set to True for a quick test run (~5 min instead of ~20 min)
FAST_MODE = True
FAST_MODE_YEAR_CUTOFF = 2000


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_data(path):
    """Load parquet, extract year, keep only patents with >= 2 CPC codes."""
    print("[1/7] Loading patent data...")
    t0 = time.time()

    try:
        import polars as pl
        df = pl.read_parquet(path, columns=["publication_date", "cpc4_list"])
        df = df.with_columns((pl.col("publication_date") // 10000).alias("year"))
        df = df.filter(pl.col("cpc4_list").list.len() >= 2)
        df = df.select(["year", "cpc4_list"]).to_pandas()
        print("   (loaded with polars)")
    except ImportError:
        df = pd.read_parquet(path, columns=["publication_date", "cpc4_list"])
        df["year"] = df["publication_date"] // 10000
        df = df[df["cpc4_list"].apply(len) >= 2][["year", "cpc4_list"]].reset_index(drop=True)
        print("   (loaded with pandas + pyarrow)")

    if FAST_MODE:
        df = df[df["year"] >= FAST_MODE_YEAR_CUTOFF].reset_index(drop=True)
        print(f"   >> FAST_MODE: keeping patents >= {FAST_MODE_YEAR_CUTOFF}")

    print(f"   {len(df):,} patents | {int(df['year'].min())}-{int(df['year'].max())}"
          f" | {time.time()-t0:.1f}s")
    return df


# ===========================================================================
# 2. PRE-COMPUTE ALL PAIRWISE CO-OCCURRENCES (done ONCE)
# ===========================================================================
def precompute_pairs(df, patents_per_year, mean_patents):
    """
    Explode every patent into weighted CPC pairs.

    Instead of re-iterating over all patents for each temporal window,
    we do a single pass and store (c1, c2, year, weight). Then each
    window is just a groupby filter — orders of magnitude faster.

    Weighting follows the two-step normalization from Mariani et al.:
      - intra-patent: 1 / C(m, 2) to avoid large patents dominating
      - temporal:     N_mean / N_year to correct for patent volume inflation
    """
    print("[2/7] Pre-computing all CPC pairs (single pass over patents)...")
    t0 = time.time()

    # pre-allocate lists for speed (append is O(1) amortized)
    pair_c1 = []
    pair_c2 = []
    pair_year = []
    pair_weight = []

    years_arr = df["year"].values
    cpc_arr = df["cpc4_list"].values

    n_total = len(df)
    report_every = max(n_total // 10, 1)

    for idx in range(n_total):
        codes = cpc_arr[idx]
        if not isinstance(codes, (list, np.ndarray)):
            continue
        codes = sorted(set(codes))
        m = len(codes)
        if m < 2:
            continue

        yr = int(years_arr[idx])
        w_patent = 1.0 / (m * (m - 1) / 2)
        n_yr = patents_per_year.get(yr, mean_patents)
        w = w_patent * (mean_patents / n_yr) if n_yr > 0 else w_patent

        for c1, c2 in combinations(codes, 2):
            pair_c1.append(c1)
            pair_c2.append(c2)
            pair_year.append(yr)
            pair_weight.append(w)

        if (idx + 1) % report_every == 0:
            pct = (idx + 1) / n_total * 100
            print(f"   {pct:5.0f}% ({idx+1:,}/{n_total:,}) — "
                  f"{len(pair_c1):,} pairs so far — {time.time()-t0:.0f}s")

    pairs_df = pd.DataFrame({
        "c1": pair_c1, "c2": pair_c2,
        "year": pair_year, "weight": pair_weight
    })

    print(f"   Done: {len(pairs_df):,} pair records in {time.time()-t0:.0f}s")
    return pairs_df


# ===========================================================================
# 3. BUILD TEMPORAL GRAPHS FROM PRE-COMPUTED PAIRS
# ===========================================================================
def build_temporal_graphs(pairs_df, year_min, year_max):
    """
    For each sliding window centered on year t, aggregate pair weights
    using a fast groupby instead of re-iterating over patents.
    """
    print("[3/7] Building temporal co-occurrence graphs...")
    t0 = time.time()

    half = WINDOW_SIZE // 2
    centers = list(range(year_min + half, year_max - half + 1))
    graphs = {}

    for i, center in enumerate(centers):
        start, end = center - half, center + half

        # fast vectorized filter + groupby
        mask = (pairs_df["year"] >= start) & (pairs_df["year"] <= end)
        agg = pairs_df.loc[mask].groupby(["c1", "c2"], sort=False)["weight"].sum()

        G = nx.Graph()
        for (c1, c2), w in agg.items():
            G.add_edge(c1, c2, weight=w)
        graphs[center] = G

        if (i + 1) % 10 == 0 or i == 0:
            print(f"   {i+1}/{len(centers)} graphs | "
                  f"{len(G.nodes())} nodes, {len(G.edges())} edges | "
                  f"{time.time()-t0:.0f}s")

    print(f"   Done: {len(graphs)} graphs in {time.time()-t0:.0f}s")
    return graphs


# ===========================================================================
# 4. CENTRALITY AND CLUSTERING HELPERS
# ===========================================================================
def compute_centrality(G):
    """Strength (weighted degree) and PageRank for all nodes."""
    return {
        "strength": dict(G.degree(weight="weight")),
        "pagerank": nx.pagerank(G, weight="weight", max_iter=100)
    }


def compute_clusters(G):
    """PCA on adjacency matrix + KMeans to assign each CPC to a cluster."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n < N_CLUSTERS + 1:
        return {nd: 0 for nd in nodes}

    adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    n_comp = min(PCA_COMPONENTS, n - 1)
    if n_comp < 2:
        return {nd: 0 for nd in nodes}

    reduced = PCA(n_components=n_comp).fit_transform(adj)
    labels = KMeans(
        n_clusters=min(N_CLUSTERS, n), n_init=10, random_state=42
    ).fit_predict(reduced)
    return dict(zip(nodes, labels))


# Cache to avoid recomputing centrality/clusters for the same year
_cache_centr = {}
_cache_clust = {}

def get_centrality(graphs, year):
    if year not in _cache_centr:
        _cache_centr[year] = compute_centrality(graphs[year])
    return _cache_centr[year]

def get_clusters(graphs, year):
    if year not in _cache_clust:
        _cache_clust[year] = compute_clusters(graphs[year])
    return _cache_clust[year]


# ===========================================================================
# 5. FEATURE EXTRACTION FOR A SINGLE PAIR (c1, c2)
# ===========================================================================
FEATURE_COLS = [
    "weight_current",
    "strength_c1", "strength_c2", "strength_product", "strength_diff",
    "pagerank_c1", "pagerank_c2", "pagerank_product",
    "same_cluster",
    "common_neighbors", "adamic_adar", "jaccard_topo", "top10_overlap",
    "weight_lag1", "weight_lag2", "weight_growth", "weight_acceleration",
]


def extract_pair_features(G, c1, c2, centr, clusters, prev_graphs):
    """
    17 features per pair, covering our four axes of analysis:
      - Centrality:  strength and PageRank of both nodes + products
      - Clustering:  same_cluster indicator
      - Topology:    common neighbors, Adamic-Adar, Jaccard, top-10 overlap
      - Dynamics:    lagged weights, growth rate, acceleration
    """
    f = {}

    # -- current edge weight --
    w = G[c1][c2]["weight"] if G.has_edge(c1, c2) else 0.0
    f["weight_current"] = w

    # -- node centralities --
    s, pr = centr["strength"], centr["pagerank"]
    s1, s2 = s.get(c1, 0), s.get(c2, 0)
    p1, p2 = pr.get(c1, 0), pr.get(c2, 0)
    f["strength_c1"] = s1
    f["strength_c2"] = s2
    f["strength_product"] = s1 * s2
    f["strength_diff"] = abs(s1 - s2)
    f["pagerank_c1"] = p1
    f["pagerank_c2"] = p2
    f["pagerank_product"] = p1 * p2

    # -- cluster membership --
    f["same_cluster"] = int(clusters.get(c1, -1) == clusters.get(c2, -1))

    # -- neighborhood structure --
    if c1 in G and c2 in G:
        n1 = set(G.neighbors(c1))
        n2 = set(G.neighbors(c2))
        common = n1 & n2
        union = n1 | n2

        f["common_neighbors"] = len(common)
        f["jaccard_topo"] = len(common) / max(len(union), 1)

        # Adamic-Adar: high when shared neighbors are themselves specialized
        aa = 0.0
        for cn in common:
            deg = G.degree(cn, weight="weight")
            if deg > 1:
                aa += 1.0 / np.log(deg)
        f["adamic_adar"] = aa

        # overlap among top-10 strongest neighbors
        def top_neighbors(node, k=10):
            nbrs = sorted(G[node].items(), key=lambda x: -x[1]["weight"])[:k]
            return set(n for n, _ in nbrs)
        t1, t2 = top_neighbors(c1), top_neighbors(c2)
        f["top10_overlap"] = len(t1 & t2) / max(len(t1 | t2), 1)
    else:
        f["common_neighbors"] = 0
        f["jaccard_topo"] = 0
        f["adamic_adar"] = 0
        f["top10_overlap"] = 0

    # -- temporal lags and dynamics --
    pw = []
    for pG in prev_graphs:
        pw.append(pG[c1][c2]["weight"] if pG.has_edge(c1, c2) else 0.0)

    lag1 = pw[0] if len(pw) >= 1 else 0.0
    lag2 = pw[1] if len(pw) >= 2 else 0.0
    f["weight_lag1"] = lag1
    f["weight_lag2"] = lag2
    f["weight_growth"] = (w - lag1) / lag1 if lag1 > 0 else 0.0
    f["weight_acceleration"] = ((w - lag1) - (lag1 - lag2)) if len(pw) >= 2 else 0.0

    return f


# ===========================================================================
# 6. BUILD ML DATASET
# ===========================================================================
def build_dataset(graphs):
    """
    For each valid year t (where t+HORIZON also exists in the graphs),
    extract features for the top-K most active pairs and record the
    target: actual edge weight at t+HORIZON.
    """
    print("[4/7] Building ML dataset...")
    t0 = time.time()

    sorted_years = sorted(graphs.keys())
    valid = [y for y in sorted_years if (y + HORIZON) in graphs]
    print(f"   {len(valid)} usable years "
          f"({min(valid)}-{max(valid)} -> targets {min(valid)+HORIZON}-{max(valid)+HORIZON})")

    records = []

    for i, year in enumerate(valid):
        G = graphs[year]
        G_target = graphs[year + HORIZON]
        centr = get_centrality(graphs, year)
        clusters = get_clusters(graphs, year)
        prev = [graphs[year - l] for l in [1, 2] if (year - l) in graphs]

        # select top-K edges by weight
        edges = [(c1, c2, d["weight"])
                 for c1, c2, d in G.edges(data=True)
                 if d["weight"] >= MIN_COOCCURRENCES]
        edges.sort(key=lambda x: -x[2])

        for c1, c2, _ in edges[:TOP_K_PAIRS]:
            feats = extract_pair_features(G, c1, c2, centr, clusters, prev)
            feats["year"] = year
            feats["c1"] = c1
            feats["c2"] = c2
            feats["target_weight"] = (
                G_target[c1][c2]["weight"] if G_target.has_edge(c1, c2) else 0.0
            )
            records.append(feats)

        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(valid)} years processed ({time.time()-t0:.0f}s)")

    ds = pd.DataFrame(records)
    print(f"   {len(ds):,} samples, {len(FEATURE_COLS)} features | {time.time()-t0:.0f}s")
    return ds


# ===========================================================================
# 7. MODEL TRAINING
# ===========================================================================
def train_model(dataset):
    """
    Gradient Boosting Regressor with strict temporal train/test split.
    We never use future data to predict the past.
    """
    print("[5/7] Training Gradient Boosting model...")

    X = dataset[FEATURE_COLS].values
    y = dataset["target_weight"].values
    years = dataset["year"].values

    # temporal split at 75th percentile of available years
    unique_years = sorted(dataset["year"].unique())
    split_year = unique_years[int(len(unique_years) * 0.75)]
    tr = years <= split_year
    te = years > split_year

    print(f"   Train: {tr.sum():,} samples (years <= {split_year})")
    print(f"   Test:  {te.sum():,} samples (years > {split_year})")

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model.fit(X[tr], y[tr])

    # evaluate
    for label, mask in [("Train", tr), ("Test ", te)]:
        if mask.sum() == 0:
            continue
        pred = model.predict(X[mask])
        rmse = np.sqrt(mean_squared_error(y[mask], pred))
        r2 = r2_score(y[mask], pred)
        print(f"   {label} -> RMSE: {rmse:8.2f} | R2: {r2:.4f}")

    # feature importances
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print("\n   Feature importances:")
    for feat, val in imp.head(10).items():
        bar = "#" * int(val * 80)
        print(f"     {feat:25s} {val:.4f}  {bar}")

    return model, imp


# ===========================================================================
# 8. CONVERGENCE PREDICTIONS ON LAST AVAILABLE PERIOD
# ===========================================================================
def compute_predictions(model, graphs, last_year):
    """
    Apply the trained model to the most recent period to predict which
    CPC pairs will converge most in the next HORIZON years.
    convergence_index = (predicted_weight - current_weight) / current_weight
    """
    print(f"\n[6/7] Predicting convergence from {last_year} to ~{last_year+HORIZON}...")

    G = graphs[last_year]
    centr = get_centrality(graphs, last_year)
    clusters = get_clusters(graphs, last_year)
    prev = [graphs[last_year - l] for l in [1, 2] if (last_year - l) in graphs]

    edges = [(c1, c2, d["weight"])
             for c1, c2, d in G.edges(data=True)
             if d["weight"] >= MIN_COOCCURRENCES]
    edges.sort(key=lambda x: -x[2])

    rows = []
    for c1, c2, w in edges[:TOP_K_PAIRS]:
        feats = extract_pair_features(G, c1, c2, centr, clusters, prev)
        X = np.array([[feats[col] for col in FEATURE_COLS]])
        w_pred = max(model.predict(X)[0], 0)
        conv = (w_pred - w) / w if w > 0 else 0.0

        rows.append({
            "cpc_1": c1, "cpc_2": c2,
            "weight_current": round(w, 2),
            "weight_predicted": round(w_pred, 2),
            "convergence_index": round(conv, 4),
            "same_cluster": int(clusters.get(c1, -1) == clusters.get(c2, -1)),
        })

    res = pd.DataFrame(rows).sort_values("convergence_index", ascending=False)

    print(f"\n   Top 20 predicted convergences:")
    for _, r in res.head(20).iterrows():
        tag = " ** INTER-CLUSTER" if not r["same_cluster"] else ""
        print(f"   {r['cpc_1']:5s} <-> {r['cpc_2']:5s}  "
              f"now: {r['weight_current']:7.1f}  "
              f"pred: {r['weight_predicted']:7.1f}  "
              f"delta: {r['convergence_index']:+.1%}{tag}")
    return res


# ===========================================================================
# 9. RETROSPECTIVE ANNUAL INDEX
# ===========================================================================
def compute_annual_index(model, graphs):
    """
    Compute the aggregated convergence index for each historical year.
    This lets us trace the overall "convergence potential" of the
    technological network over time, and compare it with actual outcomes.
    """
    print("[7/7] Computing retrospective annual index...")
    t0 = time.time()

    rows = []
    sorted_years = sorted(graphs.keys())

    for year in sorted_years:
        G = graphs[year]
        centr = get_centrality(graphs, year)
        clusters = get_clusters(graphs, year)
        prev = [graphs[year - l] for l in [1, 2] if (year - l) in graphs]

        edges = [(c1, c2, d["weight"])
                 for c1, c2, d in G.edges(data=True)
                 if d["weight"] >= MIN_COOCCURRENCES]
        edges.sort(key=lambda x: -x[2])

        scores = []
        inter_scores = []

        for c1, c2, w in edges[:TOP_K_PAIRS]:
            feats = extract_pair_features(G, c1, c2, centr, clusters, prev)
            X = np.array([[feats[col] for col in FEATURE_COLS]])
            w_pred = max(model.predict(X)[0], 0)
            if w > 0:
                s = (w_pred - w) / w
                scores.append(s)
                if clusters.get(c1, -1) != clusters.get(c2, -1):
                    inter_scores.append(s)

        rows.append({
            "year": year,
            "conv_mean": np.mean(scores) if scores else 0,
            "conv_median": np.median(scores) if scores else 0,
            "conv_p90": np.percentile(scores, 90) if scores else 0,
            "inter_cluster_conv": np.mean(inter_scores) if inter_scores else 0,
            "pct_growing": np.mean([s > 0 for s in scores]) if scores else 0,
            "n_pairs": len(scores),
        })

    df = pd.DataFrame(rows)
    print(f"   {len(df)} years computed in {time.time()-t0:.0f}s")
    return df


# ===========================================================================
# 10. PLOTS
# ===========================================================================
def plot_results(df_index, importances, predictions):
    print("   Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Predictive Technological Convergence Index",
                 fontsize=15, fontweight="bold")

    # aggregated index over time
    ax = axes[0, 0]
    ax.plot(df_index["year"], df_index["conv_mean"],
            "b-o", ms=3, lw=1.5, label="Mean")
    ax.fill_between(df_index["year"],
                    df_index["conv_median"], df_index["conv_p90"],
                    alpha=0.2, label="Median to P90")
    ax.set_xlabel("Year (window center)")
    ax.set_ylabel("Predicted convergence index")
    ax.set_title("Aggregate convergence index over time")
    ax.legend(); ax.grid(alpha=0.3)

    # inter-cluster vs global
    ax = axes[0, 1]
    ax.plot(df_index["year"], df_index["inter_cluster_conv"],
            "r-s", ms=3, lw=1.5, label="Inter-cluster")
    ax.plot(df_index["year"], df_index["conv_mean"],
            "b--", alpha=0.5, label="Global")
    ax.set_xlabel("Year")
    ax.set_title("Inter-cluster convergence vs global")
    ax.legend(); ax.grid(alpha=0.3)

    # feature importances
    ax = axes[1, 0]
    top = importances.head(12)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Feature importances (Gradient Boosting)")
    ax.invert_yaxis()

    # prediction distribution
    ax = axes[1, 1]
    vals = predictions["convergence_index"].clip(-2, 5)
    ax.hist(vals, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.5, label="No change")
    ax.axvline(vals.median(), color="orange", lw=1.5,
               label=f"Median = {vals.median():.2f}")
    ax.set_xlabel("Predicted convergence index")
    ax.set_title("Distribution of pair-level predictions")
    ax.legend()

    plt.tight_layout()
    p1 = f"{OUTPUT_DIR}/convergence_results.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # top 30 bar chart
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    top30 = predictions.head(30)
    labels = [f"{r['cpc_1']} <-> {r['cpc_2']}" for _, r in top30.iterrows()]
    colors = ["#e74c3c" if not r["same_cluster"] else "#3498db"
              for _, r in top30.iterrows()]
    ax2.barh(range(len(top30)), top30["convergence_index"].values, color=colors)
    ax2.set_yticks(range(len(top30)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Predicted convergence (relative 5-year growth)")
    ax2.set_title("Top 30 converging pairs (red = inter-cluster, blue = intra-cluster)")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    p2 = f"{OUTPUT_DIR}/top30_convergences.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Saved: {p1}")
    print(f"   Saved: {p2}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    total_t0 = time.time()
    print("=" * 65)
    print("  PREDICTIVE TECHNOLOGICAL CONVERGENCE INDEX")
    print("  Temporal Link Prediction via Gradient Boosting")
    print("=" * 65)

    # --- load ---
    df = load_data(PARQUET_PATH)
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())

    # --- pre-compute pairs (single pass) ---
    patents_per_year = df.groupby("year").size().to_dict()
    mean_patents = np.mean(list(patents_per_year.values()))
    pairs_df = precompute_pairs(df, patents_per_year, mean_patents)

    # free memory: we no longer need the raw patent dataframe
    del df

    # --- build graphs from pre-computed pairs ---
    graphs = build_temporal_graphs(pairs_df, year_min, year_max)
    del pairs_df  # free memory

    # --- ML pipeline ---
    dataset = build_dataset(graphs)
    model, importances = train_model(dataset)

    last_year = max(graphs.keys())
    predictions = compute_predictions(model, graphs, last_year)
    df_index = compute_annual_index(model, graphs)

    # --- save everything ---
    predictions.to_csv(f"{OUTPUT_DIR}/predictions_convergence.csv", index=False)
    df_index.to_csv(f"{OUTPUT_DIR}/indice_convergence_annuel.csv", index=False)
    dataset.to_csv(f"{OUTPUT_DIR}/dataset_features.csv", index=False)
    plot_results(df_index, importances, predictions)

    total = time.time() - total_t0
    print(f"\n{'=' * 65}")
    print(f"  DONE in {total/60:.1f} min — output in {OUTPUT_DIR}/")
    print(f"    predictions_convergence.csv    (pair-level scores)")
    print(f"    indice_convergence_annuel.csv   (yearly aggregate index)")
    print(f"    dataset_features.csv            (full ML dataset)")
    print(f"    convergence_results.png         (4-panel summary)")
    print(f"    top30_convergences.png          (top predicted pairs)")
    print(f"{'=' * 65}")

    return model, predictions, df_index


if __name__ == "__main__":
    model, predictions, df_index = main()
