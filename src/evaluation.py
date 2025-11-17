import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from .config import PROCESSED_DIR, MODELS_DIR


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "chapters.csv")
    with open(MODELS_DIR / "lda_doc_topic.pkl", "rb") as f:
        lda_doc_topic = pickle.load(f)
    with open(MODELS_DIR / "nmf_doc_topic.pkl", "rb") as f:
        nmf_doc_topic = pickle.load(f)
    return df, lda_doc_topic, nmf_doc_topic


def infer_topic_to_book(df, doc_topic):
    dominant_topic = doc_topic.argmax(axis=1)
    df_tmp = df.copy()
    df_tmp["topic"] = dominant_topic
    mapping = {}
    for t in sorted(df_tmp["topic"].unique()):
        subset = df_tmp[df_tmp["topic"] == t]
        majority = subset["book_title"].value_counts().idxmax()
        mapping[t] = majority
    return mapping, dominant_topic


def build_predictions(df, doc_topic, mapping, model_name):
    dominant_topic = doc_topic.argmax(axis=1)
    pred_books = np.array([mapping[t] for t in dominant_topic])
    out = df.copy()
    out[f"{model_name}_dominant_topic"] = dominant_topic
    out[f"{model_name}_pred_book"] = pred_books
    return out


def evaluate_predictions(true_labels, pred_labels, label_order):
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)
    report = classification_report(true_labels, pred_labels, labels=label_order)
    return cm, report


def main():
    df, lda_doc_topic, nmf_doc_topic = load_data()
    labels = sorted(df["book_title"].unique())

    lda_mapping, lda_dom = infer_topic_to_book(df, lda_doc_topic)
    lda_pred_books = np.array([lda_mapping[t] for t in lda_dom])
    lda_cm, lda_report = evaluate_predictions(df["book_title"].values, lda_pred_books, labels)

    nmf_mapping, nmf_dom = infer_topic_to_book(df, nmf_doc_topic)
    nmf_pred_books = np.array([nmf_mapping[t] for t in nmf_dom])
    nmf_cm, nmf_report = evaluate_predictions(df["book_title"].values, nmf_pred_books, labels)

    combined = df.copy()
    combined["lda_dominant_topic"] = lda_dom
    combined["lda_pred_book"] = lda_pred_books
    combined["nmf_dominant_topic"] = nmf_dom
    combined["nmf_pred_book"] = nmf_pred_books

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / "chapter_topics.csv"
    combined.to_csv(outpath, index=False)

    print("Labels (book titles):")
    for i, lab in enumerate(labels):
        print(f"{i}: {lab}")
    print()
    print("LDA confusion matrix:")
    print(lda_cm)
    print()
    print("LDA classification report:")
    print(lda_report)
    print()
    print("NMF confusion matrix:")
    print(nmf_cm)
    print()
    print("NMF classification report:")
    print(nmf_report)
    print()
    print(f"Saved chapter-level topics to {outpath}")


if __name__ == "__main__":
    main()

