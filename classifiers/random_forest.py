# TODO: implement RandomForest and its settings


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
                10,
                help="The number of trees in the forest.",
            )
            settings["criterion"] = st.selectbox(
                "Criterion",
                ["gini", "entropy"],
                index=0,
                help="The function to measure the quality of a split.",
            )
            settings["max_depth"] = st.slider(
                "Max Depth",
                1,
                100,
                10,
                format="%01d",
                help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
            )
            settings["min_samples_split"] = st.slider(
                "Min Samples Split",
                1,
                100,
                10,
                help="The minimum number of samples required to split an internal node.",
            )
            settings["min_samples_leaf"] = st.slider(
                "Min Samples Leaf",
                1,
                100,
                10,
                help="The minimum number of samples required to be at a leaf node.",
            )
            settings["max_features"] = st.slider(
                "Max Features",
                1,
                100,
                10,
                help="The number of features to consider when looking for the best split.",
            )

        return settings
