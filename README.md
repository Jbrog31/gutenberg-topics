***Gutenberg Topic Modeling Project***

This small project performs unsupervised topic modeling on six public-domain novels (some of my favorites) from Project Gutenberg. I will be using a Python-based pipeline that includes text acquisition, preprocessing, document–term matrix construction, topic modeling (LDA and NMF), evaluation, and an interactive dashboard for exploration.

The central question is whether topic models can reconstruct which chapters belong to which book if all chapters are mixed together. The project demonstrates that topic modeling captures meaningful thematic and stylistic structure across classic literary texts.

***Books Included***

The following novels from Project Gutenberg are used as the corpus:

- Frankenstein (ID 84)
- Dracula (ID 345)
- Dr Jekyll and Mr Hyde (ID 43)
- The Picture of Dorian Gray (ID 174)
- The War of the Worlds (ID 36)
- The Time Machine (ID 35)

All texts are downloaded directly from Gutenberg and are in the public domain.

***Project Structure***
```
gutenberg-topics/
│
├── data/
│   ├── raw/                # Raw Gutenberg texts
│   ├── processed/          # Chapter-level CSVs and topic assignments
│   └── models/             # Saved vectorizer and topic models
│
├── src/
│   ├── config.py           # Project paths and book definitions
│   ├── gutenberg_utils.py  # Text download and cleaning
│   ├── preprocessing.py    # Chapter segmentation and formatting
│   ├── topic_models.py     # Document–term matrix and LDA/NMF models
│   ├── evaluation.py       # Confusion matrices and chapter-level outputs
│   └── __init__.py
│
├── app/
│   └── streamlit_app.py    # Interactive topic exploration dashboard
│
├── requirements.txt
└── README.md
```
This structure supports a reproducible workflow suitable for research and version control.

***Installation***

Clone repository:
```
git clone https://github.com/Jbrog31/gutenberg-topics.git
cd gutenberg-topics
```
Create and activate virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
***Workflow***

1. Download and clean the texts
```
python -m src.gutenberg_utils
```
2. Segment books into chapters
```
python -m src.preprocessing
```
This produces data/processed/chapters.csv.

3. Train topic models (LDA and NMF)
```
python -m src.topic_models
```
This step generates the document–term matrix and saves trained model artifacts.

4. Evaluate reconstruction performance
```
python -m src.evaluation
```
This outputs:

- A confusion matrix for both LDA and NMF
- Chapter-level topic assignments
- chapter_topics.csv used by the dashboard

***Interactive Dashboard***

After generating the topic assignments, launch the Streamlit app:
```
streamlit run app/streamlit_app.py
```
The dashboard provides:

- Topic Explorer: Top terms per topic and topic–book associations
- Chapter Explorer: Topic mixture visualization for individual chapters
- UMAP Projection: A two-dimensional embedding of chapters colored by book

This enables qualitative inspection of topic structure and chapter clustering.

***Results Summary***

The reconstruction experiment evaluates how well each model assigns chapters to their correct book based solely on topic composition. Using six topics:

- LDA achieves approximately 81% accuracy
- NMF achieves approximately 85% accuracy
- Books with stronger thematic consistency across chapters (e.g., Frankenstein, Dracula) cluster more distinctly, while books with fewer chapters or more uniform language exhibit weaker separation.

***Methodology***

The analysis uses the following techniques:

- Text acquisition and cleaning via HTTP requests
- Simple chapter segmentation using heuristic regular expressions
- Document–term matrix using unigrams and bigrams
- Unsupervised topic modeling with:
- Latent Dirichlet Allocation (scikit-learn)
- Non-negative Matrix Factorization
- UMAP dimensionality reduction for visualization
- Evaluation with confusion matrices and standard classification metrics

All code is modular and designed for extension, including the incorporation of additional texts, preprocessing steps, or modeling techniques.

***Future Extensions***

Possible improvements include:

- Incorporating lemmatization using spaCy
- Adding coherence score calculations
- Trying alternative topic modeling frameworks (e.g., BERTopic)
- Adding similarity search or chapter retrieval features
- Deploying the dashboard using Streamlit Cloud

***License***

All novels used in this project are in the public domain. All code in this repository is provided under the MIT License.
