import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score
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
TOP_K_PAIRS  = 800

# Seuils Multi-Classes (Soft Labeling)
JACCARD_MULT_COEVOL = 1.2
JACCARD_MULT_RUPTURE = 2.0

# Seuils Symétrie M4 (Vérification du type)
SYMMETRY_THRESHOLD_GPT = 0.3
SYMMETRY_THRESHOLD_SYM = 0.6

# Colonnes de Features (Sélectionnées pour leur pertinence prouvée)
FEATURE_COLS = [
    "strength_slope_c1", "strength_slope_c2", "strength_product", 
    "rolling_corr_diff2",
    "jaccard_current", "jaccard_slope", "jaccard_acceleration",
    "cond_prob_max", "cond_prob_min",
    "shared_applicants_5y"  # Nouveau signal industriel
]

# ===========================================================================
# 1. DATA LOADING (Enrichi avec Applicants)
# ===========================================================================
def load_data(path):
    print("[1/8] Chargement des données (Brevets + Déposants)...")
    t0 = time.time()
    try:
        import polars as pl
        df = pl.read_parquet(path, columns=["publication_date", "cpc4_list", "app_name_list"])
        df = df.with_columns((pl.col("publication_date") // 10000).alias("year"))
        df = df.filter(pl.col("cpc4_list").list.len() >= 2)
        df = df.select(["year", "cpc4_list", "app_name_list"]).to_pandas()
    except ImportError:
        df = pd.read_parquet(path, columns=["publication_date", "cpc4_list", "app_name_list"])
        df["year"] = df["publication_date"] // 10000
        df = df[df["cpc4_list"].apply(len) >= 2].reset_index(drop=True)

    # Filtrage des codes Y et nettoyage des applicants
    def filter_y(codes): return [c for c in codes if not c.startswith("Y")]
    df["cpc4_list"] = df["cpc4_list"].apply(filter_y)
    df["app_name_list"] = df["app_name_list"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    
    df = df[df["cpc4_list"].apply(len) >= 2].reset_index(drop=True)
    print(f"   {len(df):,} brevets | {time.time()-t0:.1f}s")
    return df

# ===========================================================================
# 2. PRE-COMPUTATION (Applicants + Poids)
# ===========================================================================
def precompute_all(df):
    print("[2/8] Pré-calcul des structures (Passage unique)...")
    t0 = time.time()
    patents_per_year = df.groupby("year").size().to_dict()
    mean_patents = np.mean(list(patents_per_year.values()))

    pair_c1, pair_c2, pair_year, pair_weight = [], [], [], []
    annual_code_counts = defaultdict(lambda: defaultdict(int))
    annual_pair_counts = defaultdict(lambda: defaultdict(int))
    annual_cpc_applicants = defaultdict(lambda: defaultdict(set))

    years_arr, cpc_arr, app_arr = df["year"].values, df["cpc4_list"].values, df["app_name_list"].values
    n = len(df)

    for idx in range(n):
        codes = sorted(set(cpc_arr[idx]))
        apps = set(app_arr[idx])
        yr = int(years_arr[idx])
        m = len(codes)
        w = (2.0 / (m * (m - 1))) * (mean_patents / patents_per_year.get(yr, mean_patents))

        for c in codes:
            annual_code_counts[yr][c] += 1
            if apps: annual_cpc_applicants[yr][c].update(apps)
            
        for c1, c2 in combinations(codes, 2):
            pair_c1.append(c1); pair_c2.append(c2); pair_year.append(yr); pair_weight.append(w)
            annual_pair_counts[yr][(c1, c2)] += 1

    pairs_df = pd.DataFrame({"c1": pair_c1, "c2": pair_c2, "year": pair_year, "weight": pair_weight})
    return pairs_df, annual_code_counts, annual_pair_counts, annual_cpc_applicants

# ===========================================================================
# 3. STRUCTURES TEMPORELLES
# ===========================================================================
def build_graphs_and_jaccard(pairs_df, annual_code_counts, annual_pair_counts, year_min, year_max):
    print("[3/8] Construction des graphes et Jaccard...")
    half = WINDOW // 2
    centers = list(range(year_min + half, year_max - half + 1))
    graphs, jaccards, code_counts_w = {}, {}, {}

    for center in centers:
        start, end = center - half, center + half
        mask = (pairs_df["year"] >= start) & (pairs_df["year"] <= end)
        agg = pairs_df.loc[mask].groupby(["c1", "c2"], sort=False)["weight"].sum()
        G = nx.Graph()
        for (c1, c2), w in agg.items(): G.add_edge(c1, c2, weight=w)
        graphs[center] = G

        cc, pc = defaultdict(int), defaultdict(int)
        for yr in range(start, end + 1):
            for code, cnt in annual_code_counts.get(yr, {}).items(): cc[code] += cnt
            for pair, cnt in annual_pair_counts.get(yr, {}).items(): pc[pair] += cnt
        code_counts_w[center] = dict(cc)
        
        jaccards[center] = {(c1, c2): n / (cc[c1] + cc[c2] - n) for (c1, c2), n in pc.items() if (cc[c1] + cc[c2] - n) > 0}

    return graphs, jaccards, code_counts_w

# ===========================================================================
# 4. FEATURES ET CORRELATION (Vectorisé)
# ===========================================================================
_annual_arrays = {}
_all_years = []

def precompute_annual_arrays(annual_code_counts, year_min, year_max):
    global _annual_arrays, _all_years
    _all_years = list(range(year_min, year_max + 1))
    all_codes = set(c for yr in annual_code_counts.values() for c in yr.keys())
    for code in all_codes:
        _annual_arrays[code] = np.array([annual_code_counts.get(y, {}).get(code, 0) for y in _all_years], dtype=float)

def fast_corr_diff2(c1, c2, year):
    if c1 not in _annual_arrays or c2 not in _annual_arrays: return 0.0
    si, ei = max(0, (year - 7) - _all_years[0]), min(len(_all_years), year - _all_years[0] + 1)
    if ei - si < 5: return 0.0
    d2a, d2b = np.diff(_annual_arrays[c1][si:ei], n=2), np.diff(_annual_arrays[c2][si:ei], n=2)
    if np.std(d2a) < 1e-10 or np.std(d2b) < 1e-10: return 0.0
    return float(np.corrcoef(d2a, d2b)[0, 1])

def extract_features(graphs, jaccards, code_counts, annual_apps, c1, c2, year):
    f = {}
    yb = [year - i for i in range(5) if (year - i) in graphs]
    
    # M1 - Centralité (Pente = accélération précurseur)
    if len(yb) >= 3:
        ybs = sorted(yb)
        f["strength_slope_c1"] = np.polyfit(np.arange(len(ybs)), [graphs[y].degree(c1, weight="weight") if c1 in graphs[y] else 0 for y in ybs], 1)[0]
        f["strength_slope_c2"] = np.polyfit(np.arange(len(ybs)), [graphs[y].degree(c2, weight="weight") if c2 in graphs[y] else 0 for y in ybs], 1)[0]
    else: f["strength_slope_c1"] = f["strength_slope_c2"] = 0
    f["strength_product"] = (graphs[year].degree(c1, weight="weight") if c1 in graphs[year] else 0) * \
                            (graphs[year].degree(c2, weight="weight") if c2 in graphs[year] else 0)

    # M3 - Corrélation
    f["rolling_corr_diff2"] = fast_corr_diff2(c1, c2, year)

    # M4 - Jaccard & Probabilités
    pk = (min(c1, c2), max(c1, c2))
    f["jaccard_current"] = jaccards.get(year, {}).get(pk, 0)
    jh = [jaccards.get(y, {}).get(pk, 0) for y in sorted(yb)]
    f["jaccard_slope"] = np.polyfit(np.arange(len(jh)), jh, 1)[0] if len(jh) >= 3 else 0
    f["jaccard_acceleration"] = float(np.mean(np.diff(np.diff(jh)))) if len(jh) >= 4 else 0

    cc = code_counts.get(year, {})
    n1, n2 = cc.get(c1, 0), cc.get(c2, 0)
    w = graphs[year][c1][c2]["weight"] if graphs[year].has_edge(c1, c2) else 0
    p_ji, p_ij = w / n1 if n1 > 0 else 0, w / n2 if n2 > 0 else 0
    f["cond_prob_max"], f["cond_prob_min"] = max(p_ji, p_ij), min(p_ji, p_ij)

    # Signal Industriel - Cross-pollination
    apps_c1 = set().union(*(annual_apps.get(y, {}).get(c1, set()) for y in yb))
    apps_c2 = set().union(*(annual_apps.get(y, {}).get(c2, set()) for y in yb))
    f["shared_applicants_5y"] = len(apps_c1.intersection(apps_c2))

    return f

# ===========================================================================
# 5. DATASET MULTI-CLASSES (Soft Labeling)
# ===========================================================================
def build_dataset(graphs, jaccards, cc, annual_apps):
    print("[4/8] Création du dataset (Classes : 0=Stagnation, 1=Co-evolution, 2=Rupture)...")
    sy = sorted(graphs.keys())
    valid = [y for y in sy if (y + HORIZON) in jaccards]
    records = []
    
    for year in valid:
        G = graphs[year]
        edges = sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:TOP_K_PAIRS]
        jn, jf = jaccards.get(year, {}), jaccards.get(year + HORIZON, {})

        for c1, c2, _ in edges:
            pk = (min(c1, c2), max(c1, c2))
            j_now, j_fut = jn.get(pk, 0), jf.get(pk, 0)
            
            if j_now < 1e-6: target = 2 if j_fut > 0.01 else 0
            else:
                ratio = j_fut / j_now
                if ratio >= JACCARD_MULT_RUPTURE: target = 2
                elif ratio >= JACCARD_MULT_COEVOL: target = 1
                else: target = 0

            feats = extract_features(graphs, jaccards, cc, annual_apps, c1, c2, year)
            feats.update({"year": year, "c1": c1, "c2": c2, "target": target})
            records.append(feats)

    return pd.DataFrame(records)

# ===========================================================================
# 6. ENTRAINEMENT ET PREDICTIONS FUTURES
# ===========================================================================
def train_and_predict(dataset, graphs, jaccards, cc, annual_apps, year_max):
    print("[5/8] Entraînement avec rééquilibrage des classes...")
    max_train = year_max - HORIZON - 2
    tr = dataset["year"] <= max_train
    Xtr, ytr = dataset.loc[tr, FEATURE_COLS], dataset.loc[tr, "target"]
    
    weights = compute_sample_weight(class_weight='balanced', y=ytr)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(Xtr, ytr, sample_weight=weights)

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    
    print(f"[6/8] Prédictions vers {year_max + HORIZON}...")
    last_y = max(graphs.keys())
    G_last = graphs[last_y]
    future_edges = sorted(G_last.edges(data=True), key=lambda x: -x[2]["weight"])[:TOP_K_PAIRS]
    
    future_preds = []
    for c1, c2, _ in future_edges:
        feats = extract_features(graphs, jaccards, cc, annual_apps, c1, c2, last_y)
        probas = model.predict_proba(pd.DataFrame([feats])[FEATURE_COLS])[0]
        prob_rupture = probas[2] if len(probas) > 2 else 0
        
        ratio = feats["cond_prob_min"] / feats["cond_prob_max"] if feats["cond_prob_max"] > 0 else 0
        ctype = "GPT" if ratio < 0.3 else ("ASYM" if ratio < 0.6 else "SYM")
        
        future_preds.append({
            "cpc_1": c1, "cpc_2": c2, "prob_rupture": round(prob_rupture, 4),
            "convergence_type": ctype, "acteurs_communs": feats["shared_applicants_5y"]
        })
    
    return imp, pd.DataFrame(future_preds).sort_values("prob_rupture", ascending=False)

# ===========================================================================
# 7. BACKTESTS HISTORIQUES V4
# ===========================================================================
def run_backtest_v4(dataset, graphs, jaccards, cc, annual_apps, cutoff):
    target_yr = cutoff + HORIZON
    print(f"\n--- BACKTEST : Train <= {cutoff}, Prédiction ~{target_yr} ---")
    
    td = dataset[dataset["year"] + HORIZON <= cutoff].copy()
    if len(td) < 50 or (td["target"] == 2).sum() < 2:
        print(f"   SKIP: Pas assez de données de 'Rupture' avant {cutoff}")
        return None

    Xtr, ytr = td[FEATURE_COLS], td["target"]
    weights = compute_sample_weight(class_weight='balanced', y=ytr)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(Xtr, ytr, sample_weight=weights)

    if cutoff not in graphs: return None
    G = graphs[cutoff]
    edges = sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:TOP_K_PAIRS]

    preds = []
    for c1, c2, _ in edges:
        feats = extract_features(graphs, jaccards, cc, annual_apps, c1, c2, cutoff)
        probas = model.predict_proba(pd.DataFrame([feats])[FEATURE_COLS])[0]
        prob_rupture = probas[2] if len(probas) > 2 else 0

        # Vérifier ce qui s'est VRAIMENT passé
        pk = (min(c1, c2), max(c1, c2))
        jn, jf = jaccards.get(cutoff, {}).get(pk, 0), jaccards.get(target_yr, {}).get(pk, 0)
        actual = 0
        if jn < 1e-6: actual = 2 if jf > 0.01 else 0
        elif (jf / jn) >= JACCARD_MULT_RUPTURE: actual = 2
        
        preds.append({
            "cpc_1": c1, "cpc_2": c2, "convergence_prob": round(prob_rupture, 4),
            "actually_converged": 1 if actual == 2 else 0 # 1 si vraie rupture pour le graphique
        })

    pdf = pd.DataFrame(preds).sort_values("convergence_prob", ascending=False)
    return pdf

# ===========================================================================
# 8. GÉNÉRATION DES GRAPHIQUES
# ===========================================================================
def plot_results_v4(imp, future, bt_results):
    print("\n[7/8] Génération des graphiques...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Convergence Detector V4 — Modèle Industriel", fontsize=14, fontweight="bold")

    # Importance des features
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    axes[0].barh(range(len(imp)), imp.values, color=colors)
    axes[0].set_yticks(range(len(imp)))
    axes[0].set_yticklabels(imp.index, fontsize=10)
    axes[0].set_xlabel("Importance")
    axes[0].set_title("Quels signaux prédisent la rupture ?")
    axes[0].invert_yaxis()

    # Top 30 Futur
    top30 = future.head(30)
    labs = [f"{r['cpc_1']}<->{r['cpc_2']}" for _, r in top30.iterrows()]
    cmap = {"GPT": "#95a5a6", "ASYM": "#e74c3c", "SYM": "#3498db"}
    cb = [cmap.get(r["convergence_type"], "#3498db") for _, r in top30.iterrows()]
    axes[1].barh(range(len(top30)), top30["prob_rupture"].values, color=cb)
    axes[1].set_yticks(range(len(top30)))
    axes[1].set_yticklabels(labs, fontsize=8)
    axes[1].set_xlabel("Probabilité de Rupture")
    axes[1].set_title("Top 30 Futur (Rouge=ASYM, Bleu=SYM, Gris=GPT)")
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/convergence_detector_v4_results.png", dpi=150)
    plt.close()

    # Graphiques de Backtest
    if bt_results:
        n = len(bt_results)
        fig2, axes2 = plt.subplots(1, n, figsize=(6.5 * n, 6))
        if n == 1: axes2 = [axes2]
        for idx, (cut, pdf) in enumerate(bt_results.items()):
            ax = axes2[idx]
            t20 = pdf.head(20).reset_index(drop=True)
            labs_bt = [f"{r['cpc_1']}<->{r['cpc_2']}" for _, r in t20.iterrows()]
            cols = ["#27ae60" if r["actually_converged"] else "#e74c3c" for _, r in t20.iterrows()]
            ax.barh(range(len(t20)), t20["convergence_prob"].values, color=cols)
            ax.set_yticks(range(len(t20)))
            ax.set_yticklabels(labs_bt, fontsize=8)
            ax.set_title(f"Backtest {cut}->{cut+HORIZON}\nVert = Vraie Rupture")
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/convergence_detector_v4_backtest.png", dpi=150)
        plt.close()

# ===========================================================================
# 9. MAIN
# ===========================================================================
if __name__ == "__main__":
    t_start = time.time()
    df = load_data(PARQUET_PATH)
    ymin, ymax = df["year"].min(), df["year"].max()
    
    pairs_df, acc, apc, a_apps = precompute_all(df); del df
    graphs, jaccards, cc_w = build_graphs_and_jaccard(pairs_df, acc, apc, ymin, ymax); del pairs_df
    precompute_annual_arrays(acc, ymin, ymax)
    
    dataset = build_dataset(graphs, jaccards, cc_w, a_apps)
    imp, future = train_and_predict(dataset, graphs, jaccards, cc_w, a_apps, ymax)
    
    # Exécuter les backtests
    print("\n[8/8] Lancement des Backtests Historiques...")
    bt_results = {}
    for cut in [1993, 2003, 2007, 2010]:
        pdf = run_backtest_v4(dataset, graphs, jaccards, cc_w, a_apps, cut)
        if pdf is not None:
            bt_results[cut] = pdf
            pdf.to_csv(f"{OUTPUT_DIR}/detector_v4_backtest_{cut}.csv", index=False)

    # Créer les images et sauvegarder
    plot_results_v4(imp, future, bt_results)
    future.to_csv(f"{OUTPUT_DIR}/detector_future_v4.csv", index=False)
    dataset.to_csv(f"{OUTPUT_DIR}/detector_dataset_v4.csv", index=False)
    
    print(f"\n[Terminé] Fichiers mis à jour dans {OUTPUT_DIR}/")
    print(f"Temps d'exécution total : {(time.time()-t_start)/60:.1f} min.")