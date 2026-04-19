# Stat-App
Analyse de brevets



explication de l'algo de ML : 

## Ce que fait concrètement l'algorithme, étape par étape

---

### Le problème qu'on résout

On a ~600 codes CPC et donc ~180 000 paires possibles. Pour chacune de vos quatre méthodes (centralité, clustering, corrélation, Jaccard), vous savez analyser manuellement une paire donnée — vous l'avez fait sur H01M×B60L, H04W×G06F et C12N×G01N. Mais vous ne pouvez pas inspecter visuellement 180 000 paires. L'algorithme fait exactement ça : il prend les signaux que vous avez identifiés manuellement sur 3 cas, les encode sous forme de 14 variables numériques, et laisse un Gradient Boosting apprendre quelle combinaison de ces signaux prédit une convergence. Ensuite il applique ce qu'il a appris à toutes les paires.

---

### Étape 1 — Chargement et pré-calcul (fonctions `load_data` + `precompute_all`)

Le script charge le parquet (~1M brevets), filtre ceux qui ont au moins 2 codes CPC, et supprime les codes Y (transversaux EPO). Ensuite, en un seul passage sur tous les brevets, il pré-calcule trois structures de données. Pour chaque brevet, il génère toutes les paires de codes avec leur poids (normalisation intra-brevet 2/m(m-1) × normalisation temporelle N̄/Nₜ). Il compte aussi, pour chaque année, combien de brevets contiennent chaque code individuellement (pour la corrélation) et combien contiennent chaque paire (pour le Jaccard). Tout est stocké en mémoire pour que les étapes suivantes n'aient jamais à reboucler sur les brevets bruts.

---

### Étape 2 — Construction des graphes et du Jaccard (fonction `build_graphs_and_jaccard`)

Pour chaque année t de 1990 à 2020 (avec fenêtres glissantes de 5 ans centrées sur t), le script construit deux choses. D'abord un graphe pondéré NetworkX où chaque nœud est un code CPC et chaque arête porte le poids total des co-occurrences pondérées sur la fenêtre — c'est exactement votre réseau de co-occurrence de la Méthode 1. Ensuite, pour chaque paire, il calcule le Jaccard brut à partir des comptages de brevets : nombre de brevets contenant les deux codes divisé par le nombre contenant au moins l'un des deux. Ça donne ~30 graphes et ~30 matrices de Jaccard, un par année.

---

### Étape 3 — Clustering et absorption (fonctions `compute_clustering_full`, `align_clusters`, `compute_absorption`)

Pour chaque année, le script construit la matrice de probabilités conditionnelles M[i,j] = P(j|i), la réduit à 20 dimensions par ACP, puis applique un KMeans à 10 clusters. Les labels sont alignés d'une année à l'autre par l'algorithme hongrois (pour que le cluster 3 en 2005 corresponde bien au cluster 3 en 2006). Ensuite il calcule le score d'absorption de chaque cluster : entropie des entrées × persistance. Tout est caché pour ne jamais recalculer deux fois la même année.

---

### Étape 4 — Extraction des 14 features pour chaque paire (fonction `extract_features`)

C'est le cœur de l'algorithme. Pour une paire (c1, c2) à une année t, il calcule exactement ceci.

**Depuis la Méthode 1 (centralité) — 3 features :**
- `strength_slope_c1` : la pente de la régression linéaire du strength de c1 sur les 5 dernières années. Si c'est positif et fort, ça veut dire que ce code est en train de devenir plus central — le signal précurseur que vous avez observé sur H01M en 2011 (un an avant l'émergence VE).
- `strength_slope_c2` : pareil pour l'autre code. Sur vos 3 cas, c'est le code "donneur" (H01M, H04W, C12N) qui a le signal précurseur, pas le receveur.
- `strength_product` : le produit des deux strengths à l'instant t. C'est un proxy de l'importance structurelle conjointe des deux domaines.

**Depuis la Méthode 2 (clustering) — 4 features :**
- `same_cluster` : 1 si les deux codes sont dans le même cluster, 0 sinon. Sur vos 3 cas, c'est très variable : jamais pour le VE, toujours pour le Smartphone, partiel pour la Biotech. Donc le modèle apprend que cette variable n'est pas discriminante seule.
- `absorption_max` : le max du score d'absorption entre le cluster de c1 et celui de c2. Les pics étaient 0.783 (VE, 2011), 0.369 (Smartphone, 2000), 0.330 (Biotech, 1995) — tous 1 à 7 ans avant l'émergence.
- `acp_distance` : la distance euclidienne entre c1 et c2 dans l'espace ACP à 20 dimensions. Si elle est faible, les deux codes ont des profils de co-occurrence similaires — ils "fréquentent" les mêmes partenaires.
- `acp_distance_slope` : la pente de cette distance sur les 3 dernières années. Une valeur négative signifie que les deux codes se rapprochent structurellement. Sur vos 3 cas, la baisse totale était de -20% à -32%.

**Depuis la Méthode 3 (séries temporelles) — 1 feature :**
- `rolling_corr_diff2` : la corrélation de Pearson entre les différences secondes des séries annuelles de brevets de c1 et c2, sur une fenêtre de 5 ans. C'est le seul niveau statistiquement valide (les deux séries sont I(0) en diff2). Sur vos 3 cas, le lag optimal est toujours 0 et les valeurs montent à 0.593 (VE), 0.844 (Smartphone), 0.727 (Biotech) autour de l'émergence.

**Depuis la Méthode 4 (Jaccard + probabilités conditionnelles) — 6 features :**
- `jaccard_current` : le Jaccard brut de la paire à l'instant t. C'est la mesure directe d'intensité de co-occurrence.
- `jaccard_slope` : la pente du Jaccard sur les 5 dernières années. Une valeur positive et croissante, c'est la "pré-convergence" que vous observez dans l'Acte 1 de votre séquence narrative (2005-2010 pour le VE).
- `jaccard_acceleration` : la dérivée seconde du Jaccard. Si elle est positive, la pré-convergence s'accélère — le signal que la rupture approche.
- `symmetry_ratio` : min(P(c2|c1), P(c1|c2)) / max(P(c2|c1), P(c1|c2)). C'est la feature de classification la plus importante de votre Méthode 4. En dessous de 0.3 c'est une GPT (pas de vraie convergence), entre 0.3 et 0.6 c'est une convergence asymétrique (VE : 0.42, Smartphone : 0.41), au-dessus de 0.6 c'est une convergence symétrique (Biotech : 0.747).
- `cond_prob_max` : la plus grande des deux probabilités conditionnelles. P(H01M|B60L) = 21.8% pour le VE, P(G06F|H04W) = 25.8% pour le Smartphone.
- `cond_prob_min` : la plus petite. La combinaison max élevé + min quasi nul signale une GPT et non une convergence.

---

### Étape 5 — Construction du dataset d'entraînement (fonction `build_dataset`)

Pour chaque année t où on connaît aussi t+5, et pour les 800 paires les plus actives à l'année t, le script calcule les 14 features ci-dessus et regarde ce qui s'est réellement passé 5 ans plus tard : est-ce que le Jaccard a au moins doublé ? Si oui, target = 1 (événement de convergence), sinon target = 0. Le seuil de ×2 est calibré sur les 3 cas : il capte le VE (×4.1) et le Smartphone (×5.6) comme positifs, et classe correctement la Biotech (×1.69) comme négatif — cohérent avec votre indice I_conv qui donne 0.188 à la Biotech (co-évolution, pas convergence).

Le dataset final contient environ 800 paires × 25 années utilisables = ~20 000 observations, chacune avec 14 features et un label binaire.

---

### Étape 6 — Entraînement du Gradient Boosting (fonction `train_model`)

Le modèle est un GradientBoostingClassifier (300 arbres, profondeur 4, learning rate 0.05). Il apprend, à partir des 20 000 exemples, quelle combinaison de vos 14 signaux prédit le mieux un événement de convergence. La validation se fait par split temporel strict : on entraîne sur les 75% d'années les plus anciennes et on teste sur les 25% les plus récentes. Les métriques sont l'AUC-ROC (capacité à distinguer convergences de non-convergences), la precision (parmi les paires qu'on prédit comme convergentes, combien le sont vraiment), le recall (parmi les vraies convergences, combien on détecte), et le F1.

L'importance des features est le résultat le plus intéressant pour votre rapport : elle dit quel signal de vos 4 méthodes prédit le mieux la convergence future. Si `jaccard_slope` domine, ça signifie que la pré-convergence progressive (Acte 1 de votre séquence) est le meilleur prédicteur. Si `absorption_max` pèse lourd, c'est le signal précurseur du clustering (Acte 2) qui compte le plus.

---

### Étape 7 — Backtests (fonction `run_backtest`)

Pour chaque cutoff (1993, 2003, 2007, 2010), le script entraîne un modèle frais uniquement sur les données antérieures, puis score toutes les paires actives au cutoff. Ensuite il vérifie nommément : est-ce que H01M×B60L apparaît dans le top des prédictions en 2007 (5 ans avant son émergence) ? Est-ce que H04W×G06F apparaît en 2003 ? Est-ce que C12N×G01N apparaît en 1993 ? Il donne le rang, la probabilité, et si la convergence a effectivement eu lieu. Il ignore les cas déjà passés (par exemple au cutoff 2010, la Biotech de 1997 est ignorée car déjà émergée).

---

### Étape 8 — Prédictions futures (fonction `predict_future`)

Le modèle entraîné sur toutes les données score les 800 paires les plus actives à la dernière année disponible (~2021). Chaque paire reçoit une probabilité de convergence à horizon ~2026 et un label de type (GPT, ASYM, SYM). C'est le "tableau des convergences émergentes" promis dans votre introduction.

---

### Les fichiers produits

**`detector_future_predictions.csv`** — Le fichier principal. Une ligne par paire avec : les deux codes CPC, la probabilité de convergence à 5 ans, l'horizon temporel, le type de convergence (GPT/ASYM/SYM), le Jaccard actuel, sa pente, le ratio de symétrie, la distance ACP, le score d'absorption, la corrélation diff2, et l'indicateur same_cluster. C'est ce que vous commentez dans le rapport pour identifier les "prochains véhicules électriques".

**`detector_backtest_YYYY.csv`** (un par cutoff) — Le même format mais avec en plus une colonne `actually_converged` qui dit si la convergence a réellement eu lieu. C'est ce qui crédibilise le modèle : vous pouvez montrer que la paire H01M×B60L était dans le top N des prédictions en 2007.

**`detector_dataset.csv`** — Les ~20 000 observations avec les 14 features, l'année, les codes, et le label. Permet de refaire tourner d'autres modèles ou d'explorer les corrélations entre features.

**`convergence_detector_results.png`** — Deux graphiques : à gauche l'importance des 14 features (quel signal de vos méthodes compte le plus), à droite le top 30 des prédictions futures coloré par type (rouge = asymétrique, bleu = symétrique, gris = GPT).

**`convergence_detector_backtest.png`** — Un graphique par cutoff montrant les 20 meilleures prédictions avec en vert celles qui ont effectivement convergé et en rouge celles qui n'ont pas convergé. C'est la figure la plus parlante pour votre soutenance.
