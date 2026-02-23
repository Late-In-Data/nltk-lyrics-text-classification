# Classification de paroles (Rap vs Variété Française) - NLTK & scikit-learn

## Objectif
Construire un pipeline de **text mining / NLP** permettant de **classer des paroles de chansons** en deux catégories :
- **Rap**
- **Variété Française (VF)**

Le projet couvre :
- une **analyse exploratoire (EDA)**,
- un pipeline de **prétraitement NLP** (NLTK + stopwords personnalisés),
- des features **TF-IDF** et des **modèles supervisés** (scikit-learn),
- l’**évaluation** et l’**interprétation** (erreurs, limites).

## Données
Les données sont fournies sous forme de fichiers CSV par artiste, puis consolidées en un dataset unique.

- **2 633 chansons**
- **VF : 1 718** (≈ 65%)
- **Rap : 915** (≈ 35%)

Structure attendue :
```
data/
  chansons/
    Rap/*.csv
    VF/*.csv
  stopword.txt
```

Chaque CSV contient notamment une colonne `lyrics` (paroles).

## Notebooks
- `notebooks/01_eda_rap_vs_vf.ipynb`  
  Analyse exploratoire : qualité des données, distributions, longueur des paroles, lexique, wordclouds, insights & limites.

- `notebooks/02_modeling_rap_vs_vf.ipynb`  
  Modélisation : pipeline TF‑IDF + modèles scikit‑learn, **cross‑validation**, optimisation, matrices de confusion et analyse d’erreurs.

## Méthodologie (résumé)
- **Prétraitement** : normalisation (minuscule, espace vide), stopwords (NLTK + liste custom)
- **Vectorisation** : TF‑IDF (mots / n‑grams)
- **Évaluation** :
  - **Cross‑validation** sur le train (métrique principale : **F1 macro**)
  - **Test final** sur un jeu séparé
- **Interprétation** :
  - analyse d’erreurs (faux positifs / faux négatifs)

## Résultats
Meilleur modèle sélectionné via CV (F1 macro) : **RandomForest** sur features TF‑IDF.

- **Test**
  - **F1 macro : 0.9632**
  - **F1 (Rap) : 0.9511**
  - **Accuracy : 0.9696**

> Les classes étant déséquilibrées (VF > Rap), **F1 macro** est privilégiée en complément de l’accuracy.

## Exécution
Depuis la racine du projet :
```bash
jupyter notebook
```
Puis ouvrir les notebooks dans `notebooks/`.

## Limites & pistes d’amélioration
- **Biais artiste** : le modèle peut apprendre des signatures d’artistes plutôt que des signatures de genre.
- **Biais de longueur** : les paroles de Rap sont souvent plus longues, ce qui peut faciliter la classification.
- **Généralisation** : à renforcer via un split/validation **par artiste** (GroupKFold).

Pistes :
- validation stricte par artiste, calibration des scores, et comparaison avec des modèles linéaires régularisés.
- (optionnel) représentation plus riche (char n‑grams, modèles transformer type CamemBERT).

## Références
- NLTK : https://www.nltk.org/
- scikit-learn : https://scikit-learn.org/
