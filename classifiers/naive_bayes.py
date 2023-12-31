import numpy as np
import pandas as pd
from sklearn.naive_bayes import (
    MultinomialNB,
)
from sklearn.pipeline import Pipeline
from classifiers.base import Classifier
from preprocess import Preprocess

import streamlit as st


class NaiveBayes(Classifier):
    def __init__(self):
        self.algorithm_class = MultinomialNB

    def get_algorithm_settings(self):
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
