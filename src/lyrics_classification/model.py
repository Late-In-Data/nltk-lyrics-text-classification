"""Entraînement et sauvegarde du modèle de classification (Rap vs Variété Française)."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .data import PROJECT_ROOT, load_dataset
from .text import load_stopwords, preprocessor

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

# Meilleurs hyperparamètres trouvés par BayesSearchCV (notebooks/02_modeling_rap_vs_vf.ipynb,
# CV F1 macro groupée par artiste : 0.9198 ± 0.0423).
BEST_PARAMS = {
    "tfidf__min_df": 4,
    "model__n_estimators": 178,
    "model__max_depth": 29,
    "model__max_features": "log2",
    "model__min_samples_split": 7,
    "model__min_samples_leaf": 2,
}


def build_pipeline() -> Pipeline:
    """Reconstruit le pipeline TF-IDF + RandomForest avec les hyperparamètres retenus."""
    vec = TfidfVectorizer(preprocessor=preprocessor, stop_words=list(load_stopwords()))
    pipe = Pipeline([
        ("tfidf", vec),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=11)),
    ])
    pipe.set_params(**BEST_PARAMS)
    return pipe


def train_and_save(model_path: Path = MODEL_PATH) -> Pipeline:
    """Entraîne le pipeline sur tout le dataset consolidé et le sauvegarde (pour la démo)."""
    df = load_dataset()
    X = df["lyrics"]
    y = df["genre"].astype(str).map({"VF": 0, "Rap": 1}).astype(int)

    pipe = build_pipeline()
    pipe.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    return pipe


if __name__ == "__main__":
    train_and_save()
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
