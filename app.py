"""Streamlit visual dashboard for the sentiment analysis project."""

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from config import BASELINE_MODEL_PATH, LABEL_NAMES, OUTPUTS_DIR, TEXT_COLUMN, TRANSFORMER_MODEL_DIR  # noqa: E402
from data_utils import clean_text, dataset_to_dataframe, load_sentiment_dataset  # noqa: E402


st.set_page_config(
    page_title="Analyse de sentiments - Projet LLM",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #68758d;
            --blue: #2563eb;
            --teal: #0f766e;
            --pink: #db2777;
            --amber: #d97706;
            --green: #16a34a;
            --red: #dc2626;
            --panel: rgba(255, 255, 255, 0.88);
            --line: rgba(30, 41, 59, 0.12);
        }

        .stApp {
            background:
                linear-gradient(120deg, rgba(37, 99, 235, 0.10), rgba(20, 184, 166, 0.09), rgba(219, 39, 119, 0.08)),
                linear-gradient(180deg, #f8fbff 0%, #eef6f4 45%, #fff7ed 100%);
            color: var(--ink);
            overflow-x: hidden;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            overflow-x: hidden;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1120px;
            overflow-x: hidden;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 34px 34px 30px;
            border: 1px solid var(--line);
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.94), rgba(15, 118, 110, 0.90) 55%, rgba(219, 39, 119, 0.86));
            color: white;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.16);
            animation: rise-in 700ms ease-out both;
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: auto -20% -35% -20%;
            height: 120px;
            background: linear-gradient(90deg, rgba(255,255,255,0.20), rgba(255,255,255,0.02));
            transform: rotate(-2deg);
            animation: shimmer 5s ease-in-out infinite alternate;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.24);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 18px 0 10px;
            font-size: clamp(2.1rem, 5vw, 4.8rem);
            line-height: 1.02;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 760px;
            font-size: 1.08rem;
            line-height: 1.65;
            color: rgba(255, 255, 255, 0.92);
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 18px 0 4px;
        }

        .metric-card, .info-card, .prediction-card {
            width: 100%;
            box-sizing: border-box;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel);
            padding: 18px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            animation: rise-in 650ms ease-out both;
            overflow-wrap: anywhere;
        }

        .metric-card:nth-child(2), .info-card:nth-child(2) {
            animation-delay: 80ms;
        }

        .metric-card:nth-child(3), .info-card:nth-child(3) {
            animation-delay: 160ms;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 800;
        }

        .metric-value {
            margin-top: 8px;
            color: var(--ink);
            font-size: clamp(1.45rem, 2.4vw, 1.8rem);
            line-height: 1;
            font-weight: 900;
        }

        .metric-detail {
            margin-top: 10px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .section-title {
            margin: 34px 0 12px;
            padding-left: 12px;
            border-left: 5px solid var(--blue);
            font-size: 1.55rem;
            font-weight: 900;
        }

        .step-row {
            display: grid;
            grid-template-columns: repeat(5, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 14px;
        }

        .step {
            padding: 14px;
            border-radius: 12px;
            background: white;
            border: 1px solid var(--line);
            min-height: 102px;
            overflow-wrap: anywhere;
        }

        [data-testid="stDataFrame"] {
            max-width: 100%;
        }

        [data-testid="stVegaLiteChart"] {
            max-width: 100%;
            overflow: hidden;
            border-radius: 12px;
        }

        .step-number {
            width: 28px;
            height: 28px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            color: white;
            font-weight: 800;
            background: linear-gradient(135deg, var(--blue), var(--pink));
        }

        .score-track {
            height: 10px;
            overflow: hidden;
            border-radius: 999px;
            background: #e5e7eb;
            margin-top: 10px;
        }

        .score-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--teal), var(--green));
            animation: fill-in 900ms ease-out both;
        }

        .sentiment-positive {
            border-left: 6px solid var(--green);
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.13), rgba(255,255,255,0.92));
        }

        .sentiment-negative {
            border-left: 6px solid var(--red);
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.13), rgba(255,255,255,0.92));
        }

        @keyframes rise-in {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes shimmer {
            from { transform: translateX(-2%) rotate(-2deg); }
            to { transform: translateX(4%) rotate(-1deg); }
        }

        @keyframes fill-in {
            from { width: 0; }
        }

        @media (max-width: 860px) {
            .card-grid, .step-row {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 26px 22px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_preview_data(sample_size: int = 800) -> pd.DataFrame:
    dataset = load_sentiment_dataset()
    df = dataset_to_dataframe(dataset["train"])
    return df.sample(n=min(sample_size, len(df)), random_state=42)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sentiment_name(label: int) -> str:
    return LABEL_NAMES.get(label, str(label)).capitalize()


def render_metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-detail">{detail}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Projet Large Language Models</div>
            <h1>Analyse de sentiments avec Hugging Face</h1>
            <p>
                Une demonstration complete pour classifier automatiquement des avis
                de films en francais : exploration, nettoyage, baseline, Transformer,
                evaluation et prediction interactive.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Dataset", "Allocine", "180 000 avis en francais")
    with cols[1]:
        render_metric_card("Tache", "Sentiment", "Classification binaire")
    with cols[2]:
        render_metric_card("Sortie", "Positif / Negatif", "Prediction + score")


def render_dataset_section(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">1. Exploration des donnees</div>', unsafe_allow_html=True)

    label_counts = (
        df["label"]
        .map(sentiment_name)
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="nombre")
    )
    lengths = df[TEXT_COLUMN].str.split().map(len)
    length_df = pd.DataFrame(
        {
            "index": range(len(lengths)),
            "longueur_moyenne_mobile": lengths.reset_index(drop=True).rolling(25).mean(),
        }
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Distribution des sentiments")
        chart = (
            alt.Chart(label_counts)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("sentiment:N", title="Sentiment"),
                y=alt.Y("nombre:Q", title="Nombre d'avis"),
                color=alt.Color(
                    "sentiment:N",
                    scale=alt.Scale(range=["#dc2626", "#16a34a"]),
                    legend=None,
                ),
                tooltip=["sentiment", "nombre"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Longueur des avis")
        line = (
            alt.Chart(length_df)
            .mark_line(color="#2563eb", strokeWidth=3)
            .encode(
                x=alt.X("index:Q", title="Exemples"),
                y=alt.Y("longueur_moyenne_mobile:Q", title="Moyenne mobile"),
            )
            .properties(height=320)
        )
        st.altair_chart(line, use_container_width=True)

    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Longueur moyenne", f"{lengths.mean():.1f}", "mots par avis")
    with cols[1]:
        render_metric_card("Avis le plus court", str(int(lengths.min())), "mot")
    with cols[2]:
        render_metric_card("Avis le plus long", str(int(lengths.max())), "mots")

    st.subheader("Exemples d'avis")
    examples = df.sample(n=min(5, len(df)), random_state=7)[[TEXT_COLUMN, "label"]].copy()
    examples["sentiment"] = examples["label"].map(sentiment_name)
    st.dataframe(examples[[TEXT_COLUMN, "sentiment"]], use_container_width=True, hide_index=True)


def render_methodology() -> None:
    st.markdown('<div class="section-title">2. Methodologie</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="step-row">
            <div class="step"><div class="step-number">1</div><b>Donnees</b><br>Chargement du dataset Allocine depuis Hugging Face.</div>
            <div class="step"><div class="step-number">2</div><b>Nettoyage</b><br>Normalisation des espaces et suppression de balises HTML.</div>
            <div class="step"><div class="step-number">3</div><b>Baseline</b><br>TF-IDF + regression logistique pour une reference solide.</div>
            <div class="step"><div class="step-number">4</div><b>Transformer</b><br>Fine-tuning d'un modele Hugging Face leger.</div>
            <div class="step"><div class="step-number">5</div><b>Evaluation</b><br>Accuracy, precision, recall, F1-score et matrice de confusion.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results() -> None:
    st.markdown('<div class="section-title">3. Resultats experimentaux</div>', unsafe_allow_html=True)

    baseline_metrics = load_json(OUTPUTS_DIR / "baseline_metrics.json")
    transformer_metrics = load_json(OUTPUTS_DIR / "transformer_metrics.json")

    if not baseline_metrics and not transformer_metrics:
        st.info(
            "Aucun resultat n'est encore disponible. Lance d'abord "
            "`python src/train_baseline.py`, puis `python src/train_transformer.py`."
        )
        return

    rows = []
    if baseline_metrics:
        rows.append(
            {
                "modele": "TF-IDF + Regression logistique",
                "accuracy": baseline_metrics.get("accuracy"),
                "precision": baseline_metrics.get("precision_weighted"),
                "recall": baseline_metrics.get("recall_weighted"),
                "f1": baseline_metrics.get("f1_weighted"),
            }
        )

    if transformer_metrics:
        rows.append(
            {
                "modele": "BERT tiny fine-tune",
                "accuracy": transformer_metrics.get("eval_accuracy"),
                "precision": transformer_metrics.get("eval_precision"),
                "recall": transformer_metrics.get("eval_recall"),
                "f1": transformer_metrics.get("eval_f1"),
            }
        )

    results_df = pd.DataFrame(rows)
    best_f1 = results_df.sort_values("f1", ascending=False).iloc[0]

    cols = st.columns(4)
    for index, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        value = best_f1[metric]
        with cols[index]:
            render_metric_card(metric.upper(), f"{value:.3f}", f"meilleur modele : {best_f1['modele']}")

    long_df = results_df.melt(id_vars="modele", var_name="metrique", value_name="score")
    chart = (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("metrique:N", title="Metrique", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("modele:N"),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "modele:N",
                title="Modele",
                scale=alt.Scale(range=["#2563eb", "#db2777"]),
                legend=alt.Legend(orient="bottom", columns=1),
            ),
            tooltip=["modele", "metrique", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=330)
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(results_df.round(3), use_container_width=True, hide_index=True)

    confusion_matrix_path = OUTPUTS_DIR / "baseline_confusion_matrix.png"
    if confusion_matrix_path.exists():
        st.subheader("Matrice de confusion de la baseline")
        st.image(str(confusion_matrix_path), width=520)


def render_prediction() -> None:
    st.markdown('<div class="section-title">4. Prediction interactive</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.8])
    with left:
        text = st.text_area(
            "Avis a analyser",
            value="Ce film est vraiment excellent, les acteurs sont remarquables.",
            height=140,
        )
        model_choice = st.radio("Modele", ["Baseline TF-IDF", "Transformer"], horizontal=True)

    with right:
        st.markdown(
            """
            <div class="info-card">
                <b>Mode demonstration</b><br>
                Ecris un avis, choisis un modele, puis observe le sentiment predit
                et le niveau de confiance associe.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Analyser le sentiment", type="primary", use_container_width=True):
        cleaned = clean_text(text)

        if model_choice == "Baseline TF-IDF":
            if not BASELINE_MODEL_PATH.exists():
                st.warning("La baseline n'est pas encore entrainee. Lance `python src/train_baseline.py`.")
                return

            model = joblib.load(BASELINE_MODEL_PATH)
            prediction = int(model.predict([cleaned])[0])
            score = float(model.predict_proba([cleaned])[0][prediction])
        else:
            if not TRANSFORMER_MODEL_DIR.exists() or not any(TRANSFORMER_MODEL_DIR.iterdir()):
                st.warning("Le Transformer n'est pas encore entraine. Lance `python src/train_transformer.py`.")
                return

            from transformers import BertTokenizer, pipeline

            tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
            classifier = pipeline(
                "text-classification",
                model=str(TRANSFORMER_MODEL_DIR),
                tokenizer=tokenizer,
            )
            output = classifier(cleaned, truncation=True)[0]
            prediction = int(output["label"].replace("LABEL_", ""))
            score = float(output["score"])

        label = LABEL_NAMES[prediction]
        card_class = "sentiment-positive" if prediction == 1 else "sentiment-negative"
        st.markdown(
            f"""
            <div class="prediction-card {card_class}">
                <div class="metric-label">Sentiment predit</div>
                <div class="metric-value">{label.capitalize()}</div>
                <div style="margin-top: 8px; color: var(--muted);">Confiance : {score:.2%}</div>
                <div class="score-track"><div class="score-fill" style="width: {score * 100:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_conclusion() -> None:
    st.markdown('<div class="section-title">5. Conclusion</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            La baseline classique obtient les meilleurs resultats sur cette experience,
            tandis que le Transformer illustre l'utilisation d'un modele pre-entraine
            dans un cadre reproductible. Le projet couvre donc toute la chaine :
            donnees, nettoyage, entrainement, evaluation et demonstration visuelle.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    render_header()
    try:
        render_dataset_section(load_preview_data())
    except Exception as exc:
        st.error(f"Impossible de charger le dataset Hugging Face : {exc}")

    render_methodology()
    render_results()
    render_prediction()
    render_conclusion()


if __name__ == "__main__":
    main()
