# TODO: implement a better way to preprocess text. the current version is way too inefficient

import re
import string
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class HyperlinkRemover(BaseEstimator, TransformerMixin):
    """
    Remove hyperlinks from a list of texts.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Remove hyperlinks from a list of texts.

        Args:
        - X (list): List of texts containing hyperlinks.

        Returns:
        - list: List of texts with hyperlinks removed.
        """
        return np.array([re.sub(r"http\S+", "", text) for text in X])


class ToLower(BaseEstimator, TransformerMixin):
    """
    Converts a list of text to lowercase.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Converts a list of text to lowercase.

        Parameters:
        - X (list): List of strings to be converted.

        Returns:
        - list: List of strings in lowercase.
        """
        return np.array([text.lower() for text in X])


class NumberRemover(BaseEstimator, TransformerMixin):
    """
    Remove elements from a list of text that contain numbers using regex.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Remove elements from a list of text that contain numbers using regex.

        Parameters:
        - X (list): List of text elements.

        Returns:
        - list: New list with elements that do not contain numbers.
        """
        return np.array([re.sub(r"\d+", "", text) for text in X])


class PunctuationRemover(BaseEstimator, TransformerMixin):
    """
    Remove punctuation from a list of text using string.translate.
    """

    def __init__(self):
        self.translator = str.maketrans("", "", string.punctuation)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Remove punctuation from a list of text using string.translate.

        Parameters:
        - X (list): List of text elements.

        Returns:
        - list: New list with punctuation removed from each text element.
        """
        return np.array([text.translate(self.translator) for text in X])


class WhitespaceRemover(BaseEstimator, TransformerMixin):
    """
    Remove leading and trailing whitespaces from a list of text.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Remove leading and trailing whitespaces from a list of text.

        Parameters:
        - X (list): List of text elements.

        Returns:
        - list: New list with leading and trailing whitespaces removed.
        """
        return np.array([" ".join(text.split()) for text in X])


class NewlineReplacer(BaseEstimator, TransformerMixin):
    """
    Replace newline characters with a space in a list of text.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Replace newline characters with a space in a list of text.

        Parameters:
        - X (list): List of text elements.

        Returns:
        - list: New list with newline characters replaced by spaces.
        """
        return np.array([text.replace("\n", " ") for text in X])


class WordStemmer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible text transformer for stemming words using the Porter Stemmer algorithm.
    """

    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms the input data by stemming each word in the text.

        Parameters:
        - X: array-like or pd.Series, shape (n_samples,): Input data.

        Returns:
        - results: list, shape (n_samples,): Transformed data with stemmed words.
        """
        results = []
        for text in X:
            stem_words = [self.stemmer.stem(o) for o in word_tokenize(text)]
            results.append(" ".join(stem_words))
        return np.array(results)


class WordLemmatizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for lemmatizing words in a list of text using NLTK's WordNetLemmatizer.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform method to lemmatize words in a list of text.

        Parameters:
        - X (list): List of text elements.

        Returns:
        - list: New list with words lemmatized.
        """
        results = []
        for text in X:
            lemma_words = [self.lemmatizer.lemmatize(o) for o in word_tokenize(text)]
            results.append(" ".join(lemma_words))
        return np.array(results)


class StopWordRemover(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for removing stop words from a list of text.
    """

    def __init__(self):
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Removes stop words from each text element in the input list.

        Parameters:
        - X (list): List of text elements.

        Returns:
        - list: List of text elements with stop words removed.
        """
        results = []
        for text in X:
            filtered_sentence = [
                w for w in word_tokenize(text) if not w in self.stop_words
            ]
            results.append(" ".join(filtered_sentence))
        return np.array(results)


preprocess_dict = {
    "to_lower": ToLower(),
    "punctuation_remover": PunctuationRemover(),
    "number_remover": NumberRemover(),
    "hyperlink_remover": HyperlinkRemover(),
    "newline_replacer": NewlineReplacer(),
    "whitespace_remover": WhitespaceRemover(),
    "stopword_remover": StopWordRemover(),
    "word_stemmer": WordStemmer(),
    "word_lemmatizer": WordLemmatizer(),
}


class Preprocess:
    def __init__(self):
        self.steps = preprocess_dict

    def get_steps(self, prompt: str = "Configure Preprocess Steps"):
        import streamlit as st

        if prompt:
            st.write(prompt)

        # get preprocess steps from user
        for preprocess_name, transformer in preprocess_dict.items():
            st.toggle(
                preprocess_name,
                True,
                f"{preprocess_name}_key",
                help=transformer.__doc__,
            )

        # save chosen steps
        self.steps = {
            preprocess_name: transformer
            for preprocess_name, transformer in preprocess_dict.items()
            if st.session_state.get(f"{preprocess_name}_key", False)
        }
        # st.write(self.steps)

    @property
    def list_of_steps(self):
        return list(self.steps.items())

    def transform(self, X):
        results = X.copy()
        for step in self.steps.values():
            results = step.transform(results)

        # import streamlit as st
        # st.write(results)

        return results
