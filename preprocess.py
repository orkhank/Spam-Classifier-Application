import re
import string
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def clean_data(df: pd.DataFrame):
    data = df.copy()
    data.drop(
        data.columns[data.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )

    data.drop_duplicates()
    data.drop(data[data["Body"] == "empty"].index, inplace=True)
    data.dropna(inplace=True)

    return data


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
        return [re.sub(r"http\S+", "", text) for text in X]


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
        return [text.lower() for text in X]


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
        return [re.sub(r"\d+", "", text) for text in X]


class PunctuationRemover(BaseEstimator, TransformerMixin):
    """
    Remove punctuation from a list of text using string.translate.
    """

    def __init__(self):
        pass

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
        translator = str.maketrans("", "", string.punctuation)
        return [text.translate(translator) for text in X]


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
        return [text.strip() for text in X]


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
        return [text.replace("\n", " ") for text in X]


default_preprocess_steps = [
    ("hyperlink_remover", HyperlinkRemover()),
    ("to_lower", ToLower()),
    ("number_remover", NumberRemover()),
    ("punctuation_remover", PunctuationRemover()),
    ("whitespace_remover", WhitespaceRemover()),
    ("newline_replacer", NewlineReplacer()),
]
