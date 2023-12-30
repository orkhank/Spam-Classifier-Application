import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from st_pages import show_pages_from_config
from sklearn.model_selection import train_test_split
from classifiers.base import Classifier
from dataset import Datasets
from preprocess import Preprocess
from classifiers.naive_bayes import NaiveBayes
from classifiers.svm import SVM
from classifiers.random_forest import RandomForest
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


class SpamClassifierApp:
    def __init__(self):
        self.data = None
        self.classifier_name_dict: dict[str, type[Classifier]] = {
            "Naive Bayes": NaiveBayes,
            "SVM": SVM,
            "Random Forest": RandomForest,
        }
        self.classifier = None
        self.preprocess = Preprocess()

    def get_settings(self):
        classifier_selectbox = st.selectbox(
            "# Classifier", self.classifier_name_dict.keys()
        )
        assert classifier_selectbox is not None
        self.classifier = self.classifier_name_dict[classifier_selectbox]()
        self.classifier.get_parameters()
        self.data = Datasets.get_single()
        with st.expander("Configure Preprocess Steps"):
            self.preprocess.get_steps(None)

    def train_clf(self, X, y):
        # tfidf_matrix = self.classifier.feature_extractor.fit_transform(X)
        # print(tfidf_matrix.toarray())
        # print(self.classifier.feature_extractor.vocabulary_)
        # TODO: record train time(completed)
        start_time = time.time()  # Record the start time

        # Your training code here
        self.classifier.fit(X, y)

        end_time = time.time()  # Record the end time
        training_time = end_time - start_time  # Calculate the training time in seconds
        

        st.write(f"Training time: {training_time} seconds")


    def evaluate_clf(self, X_test, y_test):
        # TODO: show evaluation/training time complexity
        # TODO: make the evaluation chart and drawing more appealing and informative
        start_time = time.time()
        y_predict = [1 if o > 0.5 else 0 for o in self.classifier.predict(X_test)]
        end_time = time.time()
        testing_time = end_time - start_time
        st.success(f"Testing finished successfully in `{testing_time:0.2f}` secs.")
        recall_score_ = recall_score(y_test, y_predict)
        precision_score_ = precision_score(y_test, y_predict)
        st.write(
            f"Avarage Prediction Time: `{testing_time*1000/X_test.shape[0]:0.2f}` milliseconds."
        )
        st.write(f"Accuracy: `{100 * accuracy_score(y_test, y_predict):.2f}%`")
        st.write(f"Precision: `{100 * precision_score_:.2f}%`")
        st.write(f"Recall: `{100 * recall_score_:.2f}%`")
        st.write(f"F1 Score: `{100 * f1_score(y_test, y_predict):.2f}%`")

        # Confusion Matrix
        st.header("Confusion Matrix")
        cf_matrix = confusion_matrix(y_test, y_predict)
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(
            cf_matrix, annot=True, ax=ax, cmap="Blues", fmt=""
        )  # annot=True to annotate cells
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(["Not Spam", "Spam"])
        ax.yaxis.set_ticklabels(["Not Spam", "Spam"])
        st.pyplot(ax.figure)  # type: ignore

        # ROC Curve
        st.header("ROC Curve")
        y_score = app.classifier.clf.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_display.plot()
        st.pyplot(plt.gcf())  # type: ignore


if __name__ == "__main__":
    show_pages_from_config()
    app = SpamClassifierApp()
    with st.sidebar:
        app.get_settings()

    st.write(app.classifier.clf)

    with st.spinner("Preparing Data..."):
        X_train, X_test, y_train, y_test = Datasets.split_transform_data(
            app.data, app.preprocess
        )
    with st.spinner("Training The Classifier..."):
        naive_bayes_clf = app.train_clf(X_train, y_train)
    with st.spinner("Evaluating The Classifier..."):
        app.evaluate_clf(X_test, y_test)
