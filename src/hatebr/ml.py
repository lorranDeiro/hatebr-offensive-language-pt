"""Milestone 2: classical ML. Every transformer lives inside a Pipeline, so TF-IDF
vocabulary and IDF weights are fitted on training folds only."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC, SVC

from .rules import clean_tokens


def cleaned_text(texts) -> list[str]:
    """Same preprocessing as the original milestones (NLTK tokens, PT stopwords removed)."""
    return [" ".join(clean_tokens(t)) for t in texts]


def original_models() -> dict[str, Pipeline]:
    """The three Milestone 2 models, unchanged except for living in a Pipeline."""
    def tfidf():
        return TfidfVectorizer(max_features=5000)
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", SVC(kernel="linear", random_state=42))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(solver="liblinear", random_state=42))]),
    }


def word_char_svm(C: float = 0.5) -> Pipeline:
    """Added baseline: word 1-2 grams + char 2-5 grams (robust to spelling like 'vagabu*da', 'LIXOOO')."""
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("feats", feats), ("clf", LinearSVC(C=C))])
