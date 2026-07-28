"""Démo Streamlit : classification de paroles (Rap vs Variété Française).

Lancer avec : streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from lyrics_classification.data import load_dataset
from lyrics_classification.model import MODEL_PATH
from lyrics_classification.text import tokenize

st.set_page_config(page_title="Rap vs Variété française", page_icon=":material/mic:", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_dataset():
    return load_dataset()[["lyrics", "genre"]]


def pick_real_example(genre: str) -> str:
    """Tire un vrai extrait au hasard dans le dataset (pas de paroles inventées)."""
    df = get_dataset()
    row = df.loc[df["genre"] == genre].sample(1).iloc[0]
    lyrics = row["lyrics"]
    return lyrics[:600] + "…" if len(lyrics) > 600 else lyrics


model = load_model()

st.title(":material/mic: Rap ou variété française ?")
st.caption(
    "Classification de paroles par TF-IDF + RandomForest - colle les paroles d'une chanson en français "
    "(un texte très court est moins fiable)."
)

with st.container(horizontal=True):
    st.metric("F1 macro (test)", "0.965", border=True)
    st.metric("Accuracy (test)", "97 %", border=True)
    st.metric("Chansons dans le dataset", "2 633", border=True)

st.space("small")

if "lyrics_input" not in st.session_state:
    st.session_state.lyrics_input = ""

col_input, col_output = st.columns(2, gap="medium")

with col_input:
    with st.container(border=True):
        st.subheader("Paroles")
        with st.container(horizontal=True):
            if st.button(
                "Exemple réel : Rap",
                icon=":material/lightbulb:",
                help="Extrait aléatoire d'une vraie chanson Rap du dataset",
            ):
                st.session_state.lyrics_input = pick_real_example("Rap")
            if st.button(
                "Exemple réel : VF",
                icon=":material/lightbulb:",
                help="Extrait aléatoire d'une vraie chanson VF du dataset",
            ):
                st.session_state.lyrics_input = pick_real_example("VF")
        st.text_area(
            "Paroles",
            height=260,
            placeholder="Colle ici des paroles en français...",
            key="lyrics_input",
            label_visibility="collapsed",
        )
        predict_clicked = st.button("Prédire le genre", type="primary", icon=":material/search:")

with col_output:
    with st.container(border=True):
        st.subheader("Résultat")
        lyrics = st.session_state.lyrics_input

        if predict_clicked and lyrics.strip():
            pred = model.predict([lyrics])[0]
            proba = model.predict_proba([lyrics])[0]
            label = "Rap" if pred == 1 else "Variété française"
            badge_color = "blue" if pred == 1 else "orange"

            tokens = tokenize(lyrics)

            st.badge(label, icon=":material/label:", color=badge_color)
            with st.container(horizontal=True):
                st.metric("Confiance", f"{proba[pred]:.1%}", border=True)
                st.metric("P(Rap)", f"{proba[1]:.1%}", border=True)
                st.metric("Mots", len(tokens), border=True)
                st.metric("Mots différents", len(set(tokens)), border=True)

            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["model"]
            feature_names = tfidf.get_feature_names_out()
            vec = tfidf.transform([lyrics]).toarray()[0]
            weighted = clf.feature_importances_ * vec
            top_idx = weighted.argsort()[::-1][:30]
            freqs = {feature_names[i]: float(weighted[i]) for i in top_idx if weighted[i] > 0}

            if freqs:
                tab_cloud, tab_bars = st.tabs(["Wordcloud", "Graphique en barres"])
                with tab_cloud:
                    wc = WordCloud(
                        width=800,
                        height=350,
                        mode="RGBA",
                        background_color=None,
                        colormap="plasma",
                        random_state=11,
                    ).generate_from_frequencies(freqs)
                    st.image(wc.to_image(), width="stretch")
                with tab_bars:
                    chart_df = pd.DataFrame(freqs.items(), columns=["mot", "poids"]).set_index("mot")
                    st.bar_chart(chart_df.head(15), horizontal=True, color=badge_color)
        elif predict_clicked:
            st.warning("Colle des paroles avant de prédire.", icon=":material/warning:")
        else:
            st.caption("Colle des paroles à gauche puis clique sur Prédire pour voir le résultat ici.")
