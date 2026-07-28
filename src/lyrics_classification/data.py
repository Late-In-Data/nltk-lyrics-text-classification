"""Chargement des données de paroles (Rap vs Variété Française)."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAP_DIR = DATA_DIR / "chansons" / "Rap"
VF_DIR = DATA_DIR / "chansons" / "VF"
STOPWORD_FILE = DATA_DIR / "stopword.txt"
CONSOLIDATED_CSV = DATA_DIR / "data_song.csv"


def load_genre_dir(path: Path, label: str) -> pd.DataFrame:
    """Charge tous les CSV d'un dossier (un CSV par artiste) et les concatène.

    Chaque CSV est enrichi d'une colonne 'auteur' (nom du fichier) et 'genre' (label).
    Les CSV vides (échec de collecte pour un artiste) sont ignorés.
    """
    dfs = []
    for csv_path in sorted(path.glob("*.csv")):
        songs = pd.read_csv(csv_path)
        if songs.empty:
            continue
        songs = songs.loc[:, ~songs.columns.str.contains(r"^Unnamed")]
        songs["auteur"] = csv_path.stem
        songs["genre"] = label
        dfs.append(songs.drop(columns=["time"], errors="ignore"))
    return pd.concat(dfs, ignore_index=True)


def build_dataset(rap_dir: Path = RAP_DIR, vf_dir: Path = VF_DIR) -> pd.DataFrame:
    """Reconstruit le dataset consolidé (Rap + VF) à partir des CSV par artiste."""
    rap = load_genre_dir(rap_dir, "Rap")
    vf = load_genre_dir(vf_dir, "VF")
    return pd.concat([rap, vf], ignore_index=True)


def load_dataset(csv_path: Path = CONSOLIDATED_CSV) -> pd.DataFrame:
    """Charge le dataset consolidé déjà écrit sur disque (`data_song.csv`)."""
    return pd.read_csv(csv_path)
