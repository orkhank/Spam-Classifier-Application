# TODO: implement RandomForest and its settings


from sklearn.pipeline import Pipeline
from classifiers.base import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV

from preprocess import Preprocess
import streamlit as st


class RandomForest(Classifier):
    def __init__(self):
        self.preprocess = Preprocess()
        self.feature_extractor = None
        self.algorithm = RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1
        )  # TODO: optimize hyperparameters
        self.clf = None

    def get_parameters(self):
        feature_extractor_name = st.selectbox(
            "Select Feature Extractor", ["Count Vectorizer", "TF-IDF Vectorizer"]
        )
        selectbox2vectorizer = {  # TODO: optimize hyperparameters
            "Count Vectorizer": CountVectorizer(),
            "TF-IDF Vectorizer": TfidfVectorizer(),
        }
        self.feature_extractor = selectbox2vectorizer[feature_extractor_name]
        self.clf = Pipeline(
            [("vectorizer", self.feature_extractor), ("nb", self.algorithm)]
        )
