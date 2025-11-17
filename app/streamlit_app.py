import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import PROCESSED_DIR, MODELS_DIR


@st.cache_resource
def load_artifacts():
    df = pd.read_csv(PROCESSED_DIR / "chapter_topics.csv")
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        import pickle

        vectorizer = pickle.load(f)
    with open(MODELS_DIR / "lda_model.pkl", "rb") as f:
        import pickle

        lda = pickle.load(f)
    with open(MODELS_DIR / "nmf_model.pkl", "rb") as f:
        import pickle

        nmf = pickle.load(f)
    with open(MODELS_DIR / "lda_doc_topic.pkl", "rb") as f:
        import pickle

        lda_doc_topic = pickle.load(f)
    with open(MODELS_DIR / "nmf_doc_topic.pkl", "rb") as f:
        import pickle

        nmf_doc_topic = pickle.load(f)
    return df, vectorizer, lda, nmf, lda_doc_topic, nmf_doc_topic


def get_top_words_for_topic(vectorizer, model, topic_idx, top_n):
    terms = np.array(vectorizer.get_feature_names_out())
    topic_weights = model.components_[topic_idx]
    top_indices = topic_weights.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[top_indices], "weight": topic_weights[top_indices]})


@st.cache_resource
def compute_embeddings(doc_topic):
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(doc_topic)
    return embedding


def main():
    st.set_page_config(page_title="Gutenberg Topic Explorer", layout="wide")
    st.title("Topic Modeling Classic Gothic and Sci-Fi Novels")

    df, vectorizer, lda, nmf, lda_doc_topic, nmf_doc_topic = load_artifacts()

    model_choice = st.sidebar.selectbox("Model", ["LDA", "NMF"])
    top_n = st.sidebar.slider("Top words per topic", 5, 20, 10)

    if model_choice == "LDA":
        model = lda
        doc_topic = lda_doc_topic
    else:
        model = nmf
        doc_topic = nmf_doc_topic

    n_topics = doc_topic.shape[1]
    topics = list(range(n_topics))

    tab_topics, tab_chapters, tab_map = st.tabs(
        ["Topic Explorer", "Chapter Explorer", "Chapter Map"]
    )

    with tab_topics:
        st.subheader("Topic Explorer")
        topic_idx = st.selectbox("Topic index", topics)
        top_words = get_top_words_for_topic(vectorizer, model, topic_idx, top_n)
        fig = px.bar(
            top_words.sort_values("weight"),
            x="weight",
            y="term",
            orientation="h",
            title=f"Top {top_n} terms for topic {topic_idx}",
        )
        st.plotly_chart(fig, use_container_width=True)

        df_topic = df.copy()
        df_topic["topic_weight"] = doc_topic[:, topic_idx]
        agg = (
            df_topic.groupby("book_title")["topic_weight"]
            .mean()
            .reset_index()
            .sort_values("topic_weight", ascending=False)
        )
        fig2 = px.bar(
            agg,
            x="book_title",
            y="topic_weight",
            title=f"Average weight of topic {topic_idx} by book",
        )
        fig2.update_layout(xaxis_title="", xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_chapters:
        st.subheader("Chapter Explorer")
        chapter_id = st.selectbox("Chapter", df["chapter_id"].tolist())
        row = df[df["chapter_id"] == chapter_id].iloc[0]
        st.markdown(
            f"**Book:** {row['book_title']}  \n"
            f"**Chapter index:** {row['chapter_idx']}  \n"
            f"**LDA predicted book:** {row['lda_pred_book']}  \n"
            f"**NMF predicted book:** {row['nmf_pred_book']}"
        )
        idx = df.index[df["chapter_id"] == chapter_id][0]
        weights = doc_topic[idx]
        topic_df = pd.DataFrame({"topic": topics, "weight": weights})
        fig3 = px.bar(
            topic_df,
            x="topic",
            y="weight",
            title=f"Topic mixture for {chapter_id} ({model_choice})",
        )
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Show chapter text"):
            st.write(row["text"][:5000])

    with tab_map:
        st.subheader("Chapter Map")
        embedding = compute_embeddings(doc_topic)
        embed_df = pd.DataFrame(
            {
                "x": embedding[:, 0],
                "y": embedding[:, 1],
                "book_title": df["book_title"],
                "chapter_id": df["chapter_id"],
            }
        )
        fig4 = px.scatter(
            embed_df,
            x="x",
            y="y",
            color="book_title",
            hover_name="chapter_id",
            title=f"UMAP projection of chapters ({model_choice})",
        )
        st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()

