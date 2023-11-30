import numpy as np
import pandas as pd
from sklearn.naive_bayes import (
    MultinomialNB,
    GaussianNB,
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from preprocess import Preprocess

import streamlit as st


class NaiveBayes:
    def __init__(self):
        self.preprocess = Preprocess()
        self.feature_extractor = TfidfVectorizer()
        self.algorithm = MultinomialNB()
        self.clf = None

    def get_parameters(self):
        feature_extractor = st.selectbox(
            "Select Feature Extractor", ["Count Vectorizer", "TF-IDF Vectorizer"]
        )
        selectbox2vectorizer = {
            "Count Vectorizer": CountVectorizer(),
            "TF-IDF Vectorizer": TfidfVectorizer(),
        }
        self.feature_extractor = selectbox2vectorizer[feature_extractor]
        self.clf = Pipeline(
            [("vectorizer", self.feature_extractor), ("nb", self.algorithm)]
        )

    def fit(self, X, y, **fit_params):
        # st.write(X, y, self.clf)
        assert self.clf is not None, "Pipeline has not been initialized."
        self.clf.fit(X, y, **fit_params)
        return self.clf

    def predict(self, X):
        return self.clf.predict(X)

