import pickle
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer

from .config import PROCESSED_DIR, MODELS_DIR


def load_chapters():
    path = PROCESSED_DIR / "chapters.csv"
    df = pd.read_csv(path)
    return df


def build_dtm(df):
    texts = df["text"].tolist()
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.9,
        min_df=5,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "dtm.pkl", "wb") as f:
        pickle.dump(X, f)
    return X, vectorizer


def fit_models(X, n_topics):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=42,
    )
    nmf = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=500,
    )
    lda_doc_topic = lda.fit_transform(X)
    nmf_doc_topic = nmf.fit_transform(X)
    with open(MODELS_DIR / "lda_model.pkl", "wb") as f:
        pickle.dump(lda, f)
    with open(MODELS_DIR / "nmf_model.pkl", "wb") as f:
        pickle.dump(nmf, f)
    with open(MODELS_DIR / "lda_doc_topic.pkl", "wb") as f:
        pickle.dump(lda_doc_topic, f)
    with open(MODELS_DIR / "nmf_doc_topic.pkl", "wb") as f:
        pickle.dump(nmf_doc_topic, f)
    return lda, nmf, lda_doc_topic, nmf_doc_topic


def main():
    df = load_chapters()
    n_topics = 6
    X, vectorizer = build_dtm(df)
    lda, nmf, lda_doc_topic, nmf_doc_topic = fit_models(X, n_topics)
    print("Finished training LDA and NMF.")


if __name__ == "__main__":
    main()

