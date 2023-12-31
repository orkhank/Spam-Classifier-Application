from abc import ABC, abstractmethod
from typing import Optional, Union
from numpy import ndarray
import streamlit as st
from classifiers.vectorizers import Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

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
        assert hasattr(
            self, "pipeline"
        ), f"{self} does not have the `pipeline` attribute."
        assert self.pipeline is not None, "Pipeline has not been initialized."
        self.pipeline.fit(X, y, **fit_params)
        return self.pipeline

    def predict(self, X):
        assert hasattr(
            self, "pipeline"
        ), f"{self} does not have the `pipeline` attribute."
        assert self.pipeline is not None and isinstance(
            self.pipeline, Pipeline
        ), "Pipeline has not been initialized."
        assert self.pipeline.__sklearn_is_fitted__(), "The Pipeline is not fitted."
        return self.pipeline.predict(X)

    def get_feature_extractor_settings(self, parameter_settnigs_enabled=True):
        self.vectorizer = Vectorizer.vectorizer_selection().get_vectorizer()

    def get_clf(self):
        return self.clf

    def get_parameters(self, vectorizer: Optional[CountVectorizer] = None):
        # TODO: add method for hyperparameter optimization
        # self.optimize_hyperparameters = st.toggle(
        #     "Optimize Hyperparameters", False, disabled=True,
        #     key=f"optimize_hyperparameters_key_{self.__class__.__name__}",
        # )
        if not vectorizer:
            self.get_feature_extractor_settings()
        else:
            self.vectorizer = vectorizer
        self.algo_settings = self.get_algorithm_settings()

        self.clf = self.algorithm_class(
            **self.algo_settings,
        )
        self.pipeline = Pipeline([("vectorizer", self.vectorizer), ("clf", self.clf)])

    
