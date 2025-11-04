# à mettre dans le terminal
# pip install gdown
# gdown https://drive.google.com/uc?id=1PkZYDUtgH-dRBPssB1U6OzQBikV6mVCb

import pandas as pd

# Lecture des 5 premières lignes du fichier
df = pd.read_csv("list_cpc_patent_EP.csv", nrows=10)
print(df)
# la table des brevets contient 29 779 374 lignes

df = pd.read_csv("list_cpc_patent_EP.csv")
#on compte le nb de lignes dans la table
n_avant = len(df)

# Étape 1 : on supprime les doublons exacts (si même brevet + même CPC apparaissent plusieurs fois)
df_unique = df.drop_duplicates(subset=["publication_number", "CPC4"])

n_apres = len(df_unique)nb_doublons = n_avant - n_apres

print(f"Nombre de lignes avant suppression des doublons : {n_avant}")
print(f"Nombre de lignes après suppression des doublons  : {n_apres}")
print(f"Nombre de doublons supprimés                     : {nb_doublons}")

# Nombre de lignes avant suppression des doublons : 29 779 373
# Nombre de lignes après suppression des doublons  : 5 401 265
# Nombre de doublons supprimés                     : 24 378 108
# Nombre de brevets avec plusieurs CPC : 1524689

#  Étape 2 : on compte le nombre de codes CPC distincts par brevet
cpc_counts = df_unique.groupby("publication_number")["CPC4"].nunique().reset_index(name="nb_cpc")

# Étape 3 : on filtre les brevets avec plusieurs codes CPC
multi_cpc = cpc_counts[cpc_counts["nb_cpc"] > 1]

# Étape 4 : on fait une jointure pour voir le détail des codes CPC correspondants
multi_cpc_details = df_unique[df_unique["publication_number"].isin(multi_cpc["publication_number"])]

# afficher un aperçu
print("Nombre de brevets avec plusieurs CPC :", multi_cpc.shape[0])
print(multi_cpc_details.head(10))
# Nombre de brevets avec plusieurs CPC : 1 524 689

# On regroupe les codes CPC par brevet sous forme de liste
cpc_list = (
    df_unique
    .groupby("publication_number")["CPC4"]
    .apply(list)
    .reset_index(name="liste_CPC")
)

# On garde uniquement les brevets avec plusieurs CPC
multi_cpc_list = cpc_list[cpc_list["liste_CPC"].apply(lambda x: len(x) > 1)]

print("Nombre de brevets avec plusieurs CPC :", multi_cpc_list.shape[0])
print(multi_cpc_list.head(10))
multi_cpc_list.to_csv("brevets_multicpc.csv", index=False)


df_multi_cpc = pd.read_csv("brevets_multicpc.csv", nrows=20)
print(df_multi_cpc)
n_multi = len(df_multi_cpc)
print(n_multi)

filename = "brevets_multicpc.csv"

with open(filename, "r", encoding="utf-8") as f:
    n_lines = sum(1 for _ in f) - 1  # on enlève 1 pour l'en-tête

print(f"Nombre de brevets multiclassés : {n_lines}")

# Nombre de brevets multiclassés : 1 524 689


# Analyse statistique de la répartition des sections parmi les brevets

# On part de df_unique qui contient les brevets uniques par (publication_number, CPC4)

# 1. Extraire la section (première lettre du code CPC4)
df_unique["section"] = df_unique["CPC4"].str[0]

# 2. Compter le nombre de brevets par section
counts = df_unique["section"].value_counts().sort_index()

# 3. Calculer le pourcentage
percentages = 100 * counts / counts.sum()

# 4. Afficher le résultat
result = pd.DataFrame({"count": counts, "percentage": percentages})
print(result)

# graphes


import matplotlib.pyplot as plt


# Graphique en camembert
plt.figure(figsize=(8, 8))
plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.tab20.colors
)
plt.title("Répartition des sections CPC (camembert)")
plt.axis("equal")  # Pour que le cercle soit rond
plt.show()

# Histogramme 
plt.figure(figsize=(10, 6))
counts.plot(kind="bar", color="skyblue")
plt.title("Nombre de brevets par section CPC")
plt.xlabel("Section CPC")
plt.ylabel("Nombre de brevets")
plt.xticks(rotation=0)
plt.show()


# Analyse des combinaisons fréquentes de codes CPC à partir de brevets_multicpc.csv

import ast

# On charge le fichier
df = pd.read_csv("brevets_multicpc.csv")

# La colonne liste_CPC est en format string (ex : "['A61K', 'A61P', 'C07D']")
# On la convertit en liste Python
df['liste_CPC'] = df['liste_CPC'].apply(ast.literal_eval)

print(df.head())

# Pour commencer, on peut extraire la classe CPC (lettre + 2 chiffres) pour réduire la granularité :
def extract_classe(cpc_code):
    # Exemple: 'A61K' -> 'A61', 'G06F' -> 'G06'
    return cpc_code[:3]

df['liste_classes'] = df['liste_CPC'].apply(lambda codes: list(set(extract_classe(c) for c in codes)))

print(df[['liste_CPC', 'liste_classes']].head())

# Construction de la matrice d’occurrence des classes CPC
# On veut savoir quelles classes apparaissent souvent ensemble.
# Pour cela, on va : 
# 1) lister toutes les classes distinctes
# 2) pour chaque brevet, noter la présence ou absence de chaque classe
# construire une matrice binaire (brevet x classe)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['liste_classes'])

df_matrix = pd.DataFrame(X, columns=mlb.classes_)

print(df_matrix.head())

# Calcul de la matrice de co-occurrence
# La matrice de co-occurrence est une matrice carrée où l’élément (i,j) indique
#  combien de brevets ont à la fois la classe i et la classe j
cooccurrence = df_matrix.T.dot(df_matrix)

print(cooccurrence)

# on identifie les associations fréquentes
# Les valeurs diagonales indiquent le nombre de brevets contenant chaque classe.
# Les valeurs hors diagonale indiquent le nombre de brevets contenant les deux classes ensemble.
# On peut par exemple extraire les paires (i,j) avec un seuil minimal pour repérer les associations fortes :

threshold = 1000  # Ajuste ce seuil selon la taille de ta base

cooccurrence_pairs = []

for i, classe_i in enumerate(cooccurrence.index):
    for j, classe_j in enumerate(cooccurrence.columns):
        if i < j:  # pour ne pas doubler les paires (i,j) et (j,i)
            val = cooccurrence.iloc[i,j]
            if val >= threshold:
                cooccurrence_pairs.append((classe_i, classe_j, val))

# Trier par co-occurrence décroissante
cooccurrence_pairs = sorted(cooccurrence_pairs, key=lambda x: x[2], reverse=True)

# Afficher les 10 premières associations fréquentes
print("Top 10 des associations fréquentes de classes CPC :")
for c1, c2, val in cooccurrence_pairs[:10]:
    print(f"{c1} & {c2} : {val} brevets")
