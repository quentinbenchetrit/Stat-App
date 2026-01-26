#!/usr/bin/env python3
# exemple.py

from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt


# ====== PARAMS À MODIFIER SI BESOIN ======
PARQUET_PATH = Path("patents_merged (1).parquet")  # ton fichier dans WORK
CPC_A = "C12N"   # biotech typique
CPC_B = "A61K"   # pharma / préparations médicales
YEAR_MIN = 1980  # optionnel
MIN_COUNT = 30   # ignore les années où il y a trop peu d'observations
JACCARD_THRESHOLD = 0.20  # seuil "convergence"
OUT_PNG = Path(f"figure_convergence_{CPC_A}_{CPC_B}.png")
# =========================================


def load_yearly(parquet_path: Path, cpc_a: str, cpc_b: str, year_min: int | None) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")

    where_year = f"WHERE year >= {int(year_min)}" if year_min is not None else ""

    q = f"""
    WITH base AS (
      SELECT
        CAST(SUBSTR(CAST(publication_date AS VARCHAR), 1, 4) AS INTEGER) AS year,
        cpc4_list,
        list_contains(cpc4_list, '{cpc_a}') AS has_a,
        list_contains(cpc4_list, '{cpc_b}') AS has_b
      FROM read_parquet('{str(parquet_path)}')
    )
    SELECT
      year,
      SUM(CASE WHEN has_a THEN 1 ELSE 0 END) AS n_a,
      SUM(CASE WHEN has_b THEN 1 ELSE 0 END) AS n_b,
      SUM(CASE WHEN has_a AND has_b THEN 1 ELSE 0 END) AS n_ab
    FROM base
    {where_year}
    GROUP BY year
    ORDER BY year;
    """
    df = con.execute(q).df()
    con.close()

    # métriques
    df["union"] = df["n_a"] + df["n_b"] - df["n_ab"]
    df["jaccard"] = df["n_ab"] / df["union"].where(df["union"] != 0, pd.NA)
    df["p_b_given_a"] = df["n_ab"] / df["n_a"].where(df["n_a"] != 0, pd.NA)
    df["p_a_given_b"] = df["n_ab"] / df["n_b"].where(df["n_b"] != 0, pd.NA)

    # option: lisser un peu pour une figure plus “jolie”
    for col in ["jaccard", "p_b_given_a", "p_a_given_b"]:
        df[col + "_ma3"] = df[col].rolling(3, min_periods=1).mean()

    return df


def first_convergence_year(df: pd.DataFrame, min_count: int, thr: float) -> int | None:
    stable = df[(df["n_a"] >= min_count) & (df["n_b"] >= min_count)]
    hit = stable[stable["jaccard"] >= thr]
    if hit.empty:
        return None
    return int(hit.iloc[0]["year"])


def make_pretty_plot(df: pd.DataFrame, cpc_a: str, cpc_b: str, out_png: Path,
                     min_count: int, thr: float) -> None:
    conv_year = first_convergence_year(df, min_count=min_count, thr=thr)

    plt.figure(figsize=(11, 6))
    plt.plot(df["year"], df["jaccard_ma3"], marker="o", markersize=3, linewidth=2, label="Jaccard (MA3)")
    plt.plot(df["year"], df["p_b_given_a_ma3"], marker="o", markersize=3, linewidth=2, label=f"P({cpc_b} | {cpc_a}) (MA3)")
    plt.plot(df["year"], df["p_a_given_b_ma3"], marker="o", markersize=3, linewidth=2, label=f"P({cpc_a} | {cpc_b}) (MA3)")

    plt.title(f"Convergence technologique : {cpc_a} vs {cpc_b}")
    plt.xlabel("Année")
    plt.ylabel("Indice de convergence")
    plt.grid(True, alpha=0.25)
    plt.legend()

    # annotation convergence
    if conv_year is not None:
        y = df.loc[df["year"] == conv_year, "jaccard_ma3"].iloc[0]
        plt.axvline(conv_year, linestyle="--", linewidth=2)
        plt.annotate(
            f"Convergence (Jaccard ≥ {thr})\n{conv_year}",
            xy=(conv_year, y),
            xytext=(conv_year + 1, float(y) + 0.05 if pd.notna(y) else 0.3),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    # petit sous-texte “data quality”
    plt.text(
        0.01, 0.01,
        f"Filtre stabilité: n({cpc_a})≥{min_count} et n({cpc_b})≥{min_count} (par an) | Lissage MA(3)",
        transform=plt.gca().transAxes,
        fontsize=9,
        alpha=0.8
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"✅ Figure sauvegardée : {out_png.resolve()}")
    if conv_year is None:
        print("ℹ️ Pas de convergence détectée selon tes seuils (tu peux baisser JACCARD_THRESHOLD ou MIN_COUNT).")
    else:
        print(f"🎯 Première année de convergence (selon seuils): {conv_year}")


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {PARQUET_PATH}\n"
            "➡️ Vérifie que tu exécutes bien le script dans le dossier WORK."
        )

    df = load_yearly(PARQUET_PATH, CPC_A, CPC_B, YEAR_MIN)
    if df.empty:
        raise ValueError("Aucune ligne après filtrage (YEAR_MIN trop élevé ?).")

    make_pretty_plot(df, CPC_A, CPC_B, OUT_PNG, MIN_COUNT, JACCARD_THRESHOLD)


if __name__ == "__main__":
    main()
