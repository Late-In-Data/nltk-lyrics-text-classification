"""Prétraitement de texte partagé entre l'EDA et la modélisation."""

import re
from pathlib import Path

from .data import STOPWORD_FILE

PONCTUATION = [",", "-", ":", "!", "?", ".", "...", "'", "’", "(", ")"]


def tokenize(texte: str) -> list[str]:
    """Tokenisation simple pour l'EDA : minuscules, retrait de la ponctuation, split sur espaces."""
    texte = texte.lower()
    for p in PONCTUATION:
        texte = texte.replace(p, " ")
    return [t for t in texte.split() if t.strip() != ""]


def preprocessor(text) -> str:
    """Normalisation légère utilisée dans le TfidfVectorizer (la tokenisation est déléguée à sklearn)."""
    text = "" if text is None else str(text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def load_custom_stopwords(extra_path: Path = STOPWORD_FILE) -> set[str]:
    """Charge la liste de stopwords custom du projet.

    Le fichier est une seule ligne de mots séparés par des virgules (pas des espaces) :
    utiliser `.split()` dessus ne renverrait qu'un seul "mot" (toute la ligne).
    """
    return {w.strip().lower() for w in extra_path.read_text(encoding="utf-8").split(",") if w.strip()}


def load_stopwords(extra_path: Path = STOPWORD_FILE) -> set[str]:
    """Union des stopwords NLTK (français) et de la liste custom du projet.

    Nécessite le corpus NLTK `stopwords` (`nltk.download("stopwords")`).
    """
    from nltk.corpus import stopwords as nltk_stopwords

    return set(nltk_stopwords.words("french")) | load_custom_stopwords(extra_path)
