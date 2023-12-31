# TODO: implement SVM and its settings


from classifiers.base import Classifier
from sklearn import svm
import streamlit as st


class SVM(Classifier):
    def __init__(self):
        self.algorithm_class = svm.SVC

    def get_algorithm_settings(self):
        settings = dict()
        with st.expander("SVM Hyperparameters"):
            settings["C"] = st.slider(
                "C",
                0.0,
                1.0,
                1.0,
                0.25,
                help="Penalty parameter C of the error term.",
            )
            settings["kernel"] = st.selectbox(
                "Kernel",
                ["linear", "poly", "rbf", "sigmoid"],
                index=0,
                help="Specifies the kernel type to be used in the algorithm.",
            )
            settings["degree"] = st.slider(
                "Degree",
                1,
                100,
                3,
                format="%01d",
                help="Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.",
            )
            settings["gamma"] = st.selectbox(
                "Gamma",
                ["scale", "auto"],
                index=0,
                help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, if 'auto', uses 1 / n_features.",
            )

        return settings
