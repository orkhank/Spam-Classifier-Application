# TODO: for some reason TF-IDF Vectorizer performs worse than count vectorizer on SOME occasions but not all... Find out why and fix the issue
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
from classifiers.base import Classifier
from preprocess import Preprocess

import streamlit as st


class NaiveBayes(Classifier):
    def __init__(self):
        self.preprocess = Preprocess()
        self.feature_extractor = None
        self.algorithm = MultinomialNB()  # TODO: optimize hyperparameters
        self.clf = None

    def get_parameters(self):
        feature_extractor = st.selectbox(
            "Select Feature Extractor", ["Count Vectorizer", "TF-IDF Vectorizer"]
        )
        selectbox2vectorizer = { #TODO: optimize hyperparameters
            "Count Vectorizer": CountVectorizer(),
            "TF-IDF Vectorizer": TfidfVectorizer(),
        }
        self.feature_extractor = selectbox2vectorizer[feature_extractor]
        self.clf = Pipeline(
            [("vectorizer", self.feature_extractor), ("nb", self.algorithm)]
        )

        self.optimize_hyperparameters = st.toggle(
            "Optimize Hyperparameters", False, disabled=True
        )
