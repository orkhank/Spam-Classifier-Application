# TODO: implement SVM and its settings


from sklearn.pipeline import Pipeline
from classifiers.base import Classifier
from preprocess import Preprocess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import streamlit as st


class SVM(Classifier):
    def __init__(self):
        self.preprocess = Preprocess()
        self.feature_extractor = None
        self.algorithm = svm.SVC(kernel="linear")  # TODO: optimize hyperparameters
        self.clf = None

    def get_parameters(self):
        feature_extractor = st.selectbox(
            "Select Feature Extractor", ["Count Vectorizer", "TF-IDF Vectorizer"]
        )
        selectbox2vectorizer = {  # TODO: optimize hyperparameters
            "Count Vectorizer": CountVectorizer(),
            "TF-IDF Vectorizer": TfidfVectorizer(),
        }
        self.feature_extractor = selectbox2vectorizer[feature_extractor]
        self.clf = Pipeline(
            [("vectorizer", self.feature_extractor), ("svm", self.algorithm)]
        )

        self.optimize_hyperparameters = st.toggle(
            "Optimize Hyperparameters", False, disabled=True
        )
