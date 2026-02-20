# Classification de paroles (Rap vs Variété Française) — NLTK & scikit-learn

## Objectif
Construire un pipeline de **text mining / NLP** permettant de **classer des paroles de chansons** en deux catégories :
- **Rap**
- **Variété Française (VF)**

Le projet couvre :
- une **analyse exploratoire (EDA)** structurée avec visualisations et insights,
- un pipeline de **prétraitement NLP** (NLTK + stopwords personnalisés),
- des features **TF-IDF** et des **modèles supervisés** (scikit-learn),
- l’**évaluation** et l’**interprétation** (erreurs, variables importantes, limites).

## Données
Les données sont organisées par artiste et stockées sous forme de fichiers CSV.

Structure attendue :
```
data/
  chansons/
    Rap/*.csv
    VF/*.csv
  stopword.txt
```

Chaque CSV contient notamment une colonne `lyrics` (paroles).

