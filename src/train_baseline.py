"""Train and evaluate a TF-IDF + Logistic Regression sentiment baseline."""

import json

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from config import (
    BASELINE_MODEL_PATH,
    LABEL_NAMES,
    OUTPUTS_DIR,
    TEST_SAMPLE_SIZE,
    TRAIN_SAMPLE_SIZE,
)
from data_utils import (
    dataset_to_dataframe,
    get_texts_and_labels,
    load_sentiment_dataset,
    sample_split,
)


def save_confusion_matrix(y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        yticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    plt.xlabel("Prediction")
    plt.ylabel("Vrai label")
    plt.title("Matrice de confusion - baseline")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "baseline_confusion_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    BASELINE_MODEL_PATH.parent.mkdir(exist_ok=True)

    dataset = load_sentiment_dataset()
    train_df = dataset_to_dataframe(sample_split(dataset["train"], TRAIN_SAMPLE_SIZE))
    test_df = dataset_to_dataframe(sample_split(dataset["test"], TEST_SAMPLE_SIZE))

    x_train, y_train = get_texts_and_labels(train_df)
    x_test, y_test = get_texts_and_labels(test_df)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )

    print("Entrainement de la baseline...")
    model.fit(x_train, y_train)

    print("Evaluation...")
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    print("\nRapport de classification:")
    print(classification_report(y_test, predictions, target_names=list(LABEL_NAMES.values())))
    print("\nMetriques:")
    print(json.dumps(metrics, indent=2))

    joblib.dump(model, BASELINE_MODEL_PATH)
    save_confusion_matrix(y_test, predictions)

    with open(OUTPUTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nModele sauvegarde dans: {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()

