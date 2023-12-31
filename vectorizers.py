from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st


class Vectorizer(ABC):
    """
    Abstract base class for vectorizers.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_settings(self) -> dict[str, any]:
        pass

    def get_vectorizer(self):
        """
        Returns the vectorizer object.
        """
        return self.vectorizer

    def get_settings(self):
        """
        Returns the settings dictionary for the vectorizer.
        """
        return self.settings

    @classmethod
    def vectorizer_selection(cls):
        """
        Allows the user to select a vectorizer and its settings.
        Returns the selected vectorizer object.
        """
        vectorizer_names = [
            f"{subclass.__name__} Vectorizer" for subclass in cls.__subclasses__()
        ]
        feature_extractor_name = st.selectbox(
            "Feature Extractor",
            vectorizer_names,
            help="The feature extractor to use for extracting features from the text data.",
            key=f"feature_extractor_key_{cls.__name__}",
        )
        selectbox2vectorizer = {
            vectorizer_name: vectorizer()
            for vectorizer_name, vectorizer in zip(
                vectorizer_names, cls.__subclasses__()
            )
        }

        feature_extractor = selectbox2vectorizer[feature_extractor_name]
        # get feature extractor settings
        with st.expander(f"{feature_extractor_name} Settings"):
            settings = feature_extractor.get_settings()

        return feature_extractor


class TFIDF(Vectorizer):
    """
    TF-IDF vectorizer class.
    """

    def __init__(self):
        self.vectorizer_class = TfidfVectorizer

    def get_settings(self):
        """
        Allows the user to select settings for the TF-IDF vectorizer.
        Returns the settings dictionary.
        """
        settings = dict()
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

        self.vectorizer = self.vectorizer_class(**settings)
        self.settings = settings
        return settings


class Count(Vectorizer):
    """
    Count vectorizer class.
    """

    def __init__(self):
        self.vectorizer_class = CountVectorizer

    def get_settings(self):
        """
        Allows the user to select settings for the Count vectorizer.
        Returns the settings dictionary.
        """
        settings = dict()
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

        self.vectorizer = self.vectorizer_class(**settings)
        return settings
