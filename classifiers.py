from abc import ABC, abstractmethod
from typing import Optional
from vectorizers import Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.pipeline import Pipeline


class Classifier(ABC):
    """Abstract base class for classifiers."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_algorithm_settings(self) -> dict[str, any]:
        pass

    def fit(self, X, y, **fit_params):
        """Fit the classifier to the training data.

        Args:
            X: The input features.
            y: The target labels.
            **fit_params: Additional fitting parameters.

        Returns:
            The fitted pipeline.
        """
        assert hasattr(
            self, "pipeline"
        ), f"{self} does not have the `pipeline` attribute."
        assert self.pipeline is not None, "Pipeline has not been initialized."
        self.pipeline.fit(X, y, **fit_params)
        return self.pipeline

    def predict(self, X):
        """Make predictions on new data.

        Args:
            X: The input features.

        Returns:
            The predicted labels.
        """
        assert hasattr(
            self, "pipeline"
        ), f"{self} does not have the `pipeline` attribute."
        assert self.pipeline is not None and isinstance(
            self.pipeline, Pipeline
        ), "Pipeline has not been initialized."
        assert self.pipeline.__sklearn_is_fitted__(), "The Pipeline is not fitted."
        return self.pipeline.predict(X)

    def get_feature_extractor_settings(self, parameter_settnigs_enabled=True):
        """Get the settings for the feature extractor (vectorizer).

        Args:
            parameter_settnigs_enabled: Whether to enable parameter settings for the feature extractor.

        Returns:
            None
        """
        self.vectorizer = Vectorizer.vectorizer_selection().get_vectorizer()

    def get_clf(self):
        """Get the classifier.

        Returns:
            The classifier.
        """
        return self.clf

    def get_parameters(self, vectorizer: Optional[CountVectorizer] = None):
        """Get the parameters for the classifier.

        Args:
            vectorizer: Optional pre-defined vectorizer.

        Returns:
            None
        """
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


class NaiveBayes(Classifier):
    """Naive Bayes classifier."""

    def __init__(self):
        self.algorithm_class = MultinomialNB

    def get_algorithm_settings(self):
        """Get the algorithm settings for Naive Bayes.

        Returns:
            The algorithm settings.
        """
        settings = dict()
        with st.expander("Naive Bayes Hyperparameters"):
            settings["alpha"] = st.slider(
                "Alpha",
                0.0,
                1.0,
                1.0,
                0.25,
                help="Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).",
            )
            settings["fit_prior"] = st.toggle(
                "Fit Prior",
                True,
                help="Whether to learn class prior probabilities or not. If false, a uniform prior will be used.",
            )

        return settings


class RandomForest(Classifier):
    """Random Forest classifier."""

    def __init__(self):
        self.feature_extractor = None
        self.algorithm_class = RandomForestClassifier
        self.clf = None

    def get_algorithm_settings(self):
        """Get the algorithm settings for Random Forest.

        Returns:
            The algorithm settings.
        """
        # get hyperparameters
        settings = dict()
        with st.expander("Random Forest Hyperparameters"):
            settings["n_estimators"] = st.slider(
                "N Estimators",
                1,
                100,
                100,
                help="The number of trees in the forest.",
            )
            settings["criterion"] = st.selectbox(
                "Criterion",
                ["gini", "entropy"],
                index=0,
                help="The function to measure the quality of a split.",
            )
            settings["min_samples_split"] = st.slider(
                "Min Samples Split",
                2,
                10,
                2,
                help="The minimum number of samples required to split an internal node.",
            )
            settings["min_samples_leaf"] = st.slider(
                "Min Samples Leaf",
                1,
                10,
                1,
                help="The minimum number of samples required to be at a leaf node.",
            )

        return settings


class SVM(Classifier):
    """Support Vector Machine (SVM) classifier."""

    def __init__(self):
        self.algorithm_class = svm.SVC

    def get_algorithm_settings(self):
        """Get the algorithm settings for SVM.

        Returns:
            The algorithm settings.
        """
        settings = dict()
        with st.expander("SVM Hyperparameters"):
            settings["C"] = st.slider(
                "C",
                0.0,
                1.0,
                1.0,
                0.25,
                help="Penalty parameter C of the error term.",
            )
            settings["kernel"] = st.selectbox(
                "Kernel",
                ["linear", "poly", "rbf", "sigmoid"],
                index=0,
                help="Specifies the kernel type to be used in the algorithm.",
            )
            settings["degree"] = st.slider(
                "Degree",
                1,
                100,
                3,
                format="%01d",
                help="Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.",
            )
            settings["gamma"] = st.selectbox(
                "Gamma",
                ["scale", "auto"],
                index=0,
                help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, if 'auto', uses 1 / n_features.",
            )

        return settings
