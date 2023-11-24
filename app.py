import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from st_pages import show_pages_from_config
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dict, clean_data, Preprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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
        self.dataset_folder = "datasets/archive"
        self.dataset_dict = {
            "Spam Assassin": f"{self.dataset_folder}/completeSpamAssassin.csv",
            "EnronSpam": f"{self.dataset_folder}/enronSpamSubset.csv",
            "LingSpam": f"{self.dataset_folder}/lingSpam.csv",
        }
        self.algorithmList = ["Naive Bayes", "SVM", "Random Forest"]
        self.cleaning_utils = preprocess_dict.keys()
        self.preprocess = Preprocess()

    def get_settings(self):
        with st.sidebar:
            dataset_selectbox = st.selectbox("# Dataset", self.dataset_dict.keys())
            algorithm_selectbox = st.selectbox("# Algorithm", self.algorithmList)
            with st.expander("Preprocess Steps"):
                self.preprocess.get_steps()
        if (
            (not dataset_selectbox)
            or (not algorithm_selectbox)
            or (not self.preprocess.steps)
        ):
            st.stop()

        self.data = clean_data(pd.read_csv(self.dataset_dict[dataset_selectbox]))

    def preprocess_steps(self):
        return list(self.preprocess.steps.items())

    def split_data(self):
        emails_train, emails_test, target_train, target_test = train_test_split(
            self.data["Body"], self.data["Label"], test_size=0.2, random_state=42
        )

        le = LabelEncoder()

        y_train = np.array(le.fit_transform(target_train.values))
        y_test = np.array(le.transform(target_test.values))

        return emails_train, emails_test, y_train, y_test

    def train_naive_bayes(self, X_train, y_train):
        naive_bayes_clf = Pipeline(
            self.preprocess_steps()
            + [("vectorizer", CountVectorizer()), ("nb", MultinomialNB())]
        )
        naive_bayes_clf.fit(X_train, y_train)
        return naive_bayes_clf

    def evaluate_naive_bayes(self, clf, X_test, y_test):
        y_predict = [1 if o > 0.5 else 0 for o in clf.predict(X_test)]

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
    app.get_settings()
    X_train, X_test, y_train, y_test = app.split_data()
    naive_bayes_clf = app.train_naive_bayes(X_train, y_train)
    app.evaluate_naive_bayes(naive_bayes_clf, X_test, y_test)
