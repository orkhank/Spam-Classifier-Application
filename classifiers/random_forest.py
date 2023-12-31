from classifiers.base import Classifier
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


class RandomForest(Classifier):
    def __init__(self):
        self.feature_extractor = None
        self.algorithm_class = RandomForestClassifier
        self.clf = None

    def get_algorithm_settings(self):
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
