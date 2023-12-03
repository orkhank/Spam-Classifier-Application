import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from st_pages import show_pages_from_config
from sklearn.model_selection import train_test_split
from dataset import Datasets
from preprocess import Preprocess
from sklearn.preprocessing import LabelEncoder
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
        self.classifier_name_dict = {
            "Naive Bayes": NaiveBayes,
            # "SVM": SVM,
            # "Random Forest": RandomForest,
        }
        self.classifier = None
        self.preprocess = Preprocess()

    def get_settings(self):
        classifier_selectbox = st.selectbox(
            "# Classifier", self.classifier_name_dict.keys()
        )
        self.classifier = self.classifier_name_dict[classifier_selectbox]()
        self.classifier.get_parameters()
        self.data = Datasets.get_single()
        with st.expander("Configure Preprocess Steps"):
            self.preprocess.get_steps(None)

    def split_data(self):
        emails_train, emails_test, target_train, target_test = train_test_split(
            self.data["Body"], self.data["Label"], test_size=0.2, random_state=42
        )

        @st.cache_data(show_spinner=False)
        def preprocess(data):
            return self.preprocess.transform(data)

        # TODO: find some way to decrease the decrease preprocessing time (current version TAKES AGES)
        # ? Maybe preprocess all datasets before hand and let the user choose if it is worth to wait for the customs preprocess steps to finish or just use fast and the default preset
        preprocessed_emails_train = preprocess(emails_train)
        preprocessed_emails_test = preprocess(emails_test)

        le = LabelEncoder()
        y_train = np.array(le.fit_transform(target_train.values))
        y_test = np.array(le.transform(target_test.values))

        return preprocessed_emails_train, preprocessed_emails_test, y_train, y_test

    def train_clf(self, X, y):
        # tfidf_matrix = self.classifier.feature_extractor.fit_transform(X)
        # print(tfidf_matrix.toarray())
        # print(self.classifier.feature_extractor.vocabulary_)
        # TODO: record train time
        self.classifier.fit(X, y)

    def evaluate_clf(self, X_test, y_test):
        # TODO: show evaluation/training time complexity
        # TODO: make the evaluation chart and drawing more appealing and informative
        y_predict = [1 if o > 0.5 else 0 for o in self.classifier.predict(X_test)]

        st.write("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_predict)))
        st.write("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predict)))
        st.write("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predict)))
        st.write("F1 Score: {:.2f}%".format(100 * f1_score(y_test, y_predict)))
        # Confusion Matrix
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
        y = np.array([0, 0, 1, 1])
        pred = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, thresholds = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator"
        )
        st.pyplot(display.plot().figure_)


if __name__ == "__main__":
    show_pages_from_config()
    app = SpamClassifierApp()
    with st.sidebar:
        app.get_settings()

    st.write(app.classifier.clf)

    with st.spinner("Preparing Data..."):
        X_train, X_test, y_train, y_test = app.split_data()
    with st.spinner("Training The Classifier..."):
        naive_bayes_clf = app.train_clf(X_train, y_train)
    with st.spinner("Evaluating The Classifier..."):
        app.evaluate_clf(X_test, y_test)
