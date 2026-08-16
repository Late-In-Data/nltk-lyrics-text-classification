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

st.html("""
<style>
[data-testid="stMetricValue"] {
    font-family: ui-monospace, "SF Mono", "Cascadia Mono", "JetBrains Mono", Consolas, monospace;
    font-variant-numeric: tabular-nums;
}
.st-key-paroles_module, .st-key-resultat_module {
    position: relative;
    overflow: hidden;
}
.st-key-paroles_module::before, .st-key-resultat_module::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #D9A441;
    opacity: 0.65;
}
.st-key-paroles_module [data-testid="stHeading"] p,
.st-key-resultat_module [data-testid="stHeading"] p {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #A69B8A;
}
[data-testid="stBaseButton-secondary"] {
    border-radius: 999px;
}
.app-waveform {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 28px;
    margin: 0.6rem 0 1.2rem;
    opacity: 0.8;
}
.app-waveform span {
    flex: 1;
    background: #8A6A33;
    border-radius: 1px;
    max-width: 4px;
}
.app-waveform span.peak { background: #D9A441; }
.app-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #3A3226;
    font-size: 12px;
    color: #776B5A;
    text-align: right;
}
.app-footer a { color: #A69B8A; }
</style>
""")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_dataset():
    return load_dataset()[["lyrics", "genre"]]


def pick_real_example(genre: str) -> str:
    """Tire un vrai extrait au hasard dans le dataset (pas de paroles inventées).

    Troncature à 3000 caractères (~500-600 mots), pas 600 caractères (~100 mots) comme
    avant : un extrait trop court prive le modèle de signal et biaise les prédictions
    vers la classe majoritaire (VF). Vérifié empiriquement : à 600 caractères, 7% des
    chansons Rap sont prédites correctement ; à 3000 caractères, 97% (identique à la
    chanson entière).
    """
    df = get_dataset()
    row = df.loc[df["genre"] == genre].sample(1).iloc[0]
    lyrics = row["lyrics"]
    return lyrics[:3000] + "…" if len(lyrics) > 3000 else lyrics


model = load_model()

st.title(":material/mic: Rap ou variété française ?")
st.caption(
    "Classification de paroles par TF-IDF + RandomForest - colle les paroles d'une chanson en français "
    "(un texte très court est moins fiable)."
)
st.html("""
<div class="app-waveform" aria-hidden="true">
<span style="height:30%"></span><span style="height:55%"></span><span style="height:40%" class="peak"></span>
<span style="height:70%"></span><span style="height:35%"></span><span style="height:90%" class="peak"></span>
<span style="height:50%"></span><span style="height:25%"></span><span style="height:60%"></span>
<span style="height:80%" class="peak"></span><span style="height:45%"></span><span style="height:30%"></span>
<span style="height:65%"></span><span style="height:20%"></span><span style="height:55%" class="peak"></span>
<span style="height:38%"></span><span style="height:72%"></span><span style="height:28%"></span>
<span style="height:48%"></span><span style="height:85%" class="peak"></span><span style="height:33%"></span>
<span style="height:58%"></span><span style="height:42%"></span><span style="height:66%"></span>
</div>
""")

with st.container(horizontal=True):
    st.metric("F1 macro (test)", "0.965", border=True)
    st.metric("Accuracy (test)", "97 %", border=True)
    st.metric("Chansons dans le dataset", "2 633", border=True)

st.space("small")

if "lyrics_input" not in st.session_state:
    st.session_state.lyrics_input = ""

col_input, col_output = st.columns(2, gap="medium")

with col_input:
    with st.container(border=True, key="paroles_module"):
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
    with st.container(border=True, key="resultat_module"):
        st.subheader("Résultat")
        lyrics = st.session_state.lyrics_input

        if predict_clicked and lyrics.strip():
            pred = model.predict([lyrics])[0]
            proba = model.predict_proba([lyrics])[0]
            label = "Rap" if pred == 1 else "Variété française"
            badge_color = "orange" if pred == 1 else "violet"

            tokens = tokenize(lyrics)
            stop_words = model.named_steps["tfidf"].get_stop_words() or frozenset()
            useful_tokens = [t for t in tokens if t not in stop_words]

            st.badge(label, icon=":material/label:", color=badge_color)
            with st.container(horizontal=True):
                st.metric("Confiance", f"{proba[pred]:.1%}", border=True)
                st.metric("P(Rap)", f"{proba[1]:.1%}", border=True)
            with st.container(horizontal=True):
                st.metric("Mots (bruts)", len(tokens), border=True, help="Après nettoyage léger (minuscules, ponctuation retirée), stopwords inclus (le, la, de...)")
                st.metric("Mots différents (bruts)", len(set(tokens)), border=True)
            with st.container(horizontal=True):
                st.metric("Mots utiles", len(useful_tokens), border=True, help="Hors stopwords — ce que le modèle utilise réellement pour prédire")
                st.metric("Mots différents utiles", len(set(useful_tokens)), border=True)

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
                        background_color="#2B241C",
                        max_words=100,
                        width=900,
                        height=450,
                        colormap="autumn",
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

st.html("""
<div class="app-footer">
    Laté Lawson · <a href="mailto:latejeanjacques@gmail.com">latejeanjacques@gmail.com</a>
</div>
""")
