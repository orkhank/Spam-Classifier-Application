from abc import ABC, abstractmethod
from typing import Union
from numpy import ndarray
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline


class Classifier(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_algorithm_settings(self) -> dict[str, any]:
        pass

    def fit(self, X, y, **fit_params):
        # st.write(X, y, self.clf)
        assert hasattr(self, "clf"), f"{self} does not have the `clf` attribute."
        assert self.clf is not None, "Pipeline has not been initialized."
        self.clf.fit(X, y, **fit_params)
        return self.clf

    def predict(self, X):
        assert hasattr(self, "clf"), f"{self} does not have the `clf` attribute."
        assert self.clf is not None and isinstance(
            self.clf, Pipeline
        ), "Pipeline has not been initialized."
        assert self.clf.__sklearn_is_fitted__(), "The Pipeline is not fitted."
        return self.clf.predict(X)

    def get_feature_extractor_settings(self, parameter_settnigs_enabled=True):
        feature_extractor_name = st.selectbox(
            "Feature Extractor",
            ["Count Vectorizer", "TF-IDF Vectorizer"],
            help="The feature extractor to use for extracting features from the text data.",
        )
        selectbox2vectorizer = {
            "Count Vectorizer": CountVectorizer,
            "TF-IDF Vectorizer": TfidfVectorizer,
        }

        # get feature extractor settings
        settings = dict()
        if feature_extractor_name == "Count Vectorizer":
            with st.expander("Count Vectorizer Settings"):
                settings["ngram_range"] = st.selectbox(
                    "N-Gram Range",
                    [(1, 1), (1, 2), (1, 3), (1, 4)],
                    help="The lower and upper boundary of the range of n-values for different n-grams to be extracted.",
                )
                settings["max_df"] = st.slider(
                    "Max DF",
                    0.0,
                    1.0,
                    1.0,
                    0.01,
                    help="When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).",
                )
                settings["min_df"] = st.slider(
                    "Min DF",
                    0.0,
                    1.0,
                    0.0,
                    0.01,
                    help="When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.",
                )
        else:
            with st.expander("TF-IDF Vectorizer Settings"):
                settings["ngram_range"] = st.selectbox(
                    "N-Gram Range",
                    [(1, 1), (1, 2), (1, 3), (1, 4)],
                    help="The lower and upper boundary of the range of n-values for different n-grams to be extracted.",
                )
                settings["max_df"] = st.slider(
                    "Max DF",
                    0.0,
                    1.0,
                    1.0,
                    0.01,
                    help="When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).",
                )
                settings["min_df"] = st.slider(
                    "Min DF",
                    0.0,
                    1.0,
                    0.0,
                    0.01,
                    help="When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.",
                )
                settings["use_idf"] = st.toggle(
                    "Use IDF",
                    True,
                    help="Enable inverse-document-frequency reweighting.",
                )
                settings["smooth_idf"] = st.toggle(
                    "Smooth IDF",
                    True,
                    help="Smooth IDF by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.",
                )
                settings["sublinear_tf"] = st.toggle(
                    "Sublinear TF",
                    True,
                    help="Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).",
                )

        self.feature_extractor = selectbox2vectorizer[feature_extractor_name](
            **settings
        )

    def get_parameters(self):
        self.optimize_hyperparameters = st.toggle(
            "Optimize Hyperparameters", False, disabled=True
        )
        self.get_feature_extractor_settings()
        self.algo_settings = self.get_algorithm_settings()

        self.algorithm = self.algorithm_class(
            **self.algo_settings,
        )
        self.clf = Pipeline(
            [("vectorizer", self.feature_extractor), ("clf", self.algorithm)]
        )

    # TODO: add method for hyperparameter optimization
