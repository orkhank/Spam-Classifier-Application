import matplotlib.pyplot as plt
import time
import seaborn as sns
import streamlit as st
from st_pages import show_pages_from_config
from dataset import Datasets
from preprocess import Preprocess
from classifiers import NaiveBayes, SVM, RandomForest, Classifier
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
    """
    This class represents a Streamlit application for spam classification.
    It allows users to select a classifier, configure preprocessing steps,
    train the classifier, evaluate its performance, and make predictions on
    custom text inputs.
    """

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
        """
        Displays the settings sidebar where users can select a classifier
        and configure preprocessing steps.
        """
        classifier_selectbox = st.selectbox(
            "# Classifier",
            self.classifier_name_dict.keys(),
            help="The classifier to use for training and evaluation.",
        )
        assert classifier_selectbox is not None
        self.classifier = self.classifier_name_dict[classifier_selectbox]()
        self.classifier.get_parameters()
        self.data = Datasets.get_multi()
        with st.expander("Configure Preprocess Steps"):
            self.preprocess.get_steps(None)

    def train_clf(self, X, y):
        """
        Trains the classifier using the provided training data.

        Args:
            X (array-like): The training data features.
            y (array-like): The training data labels.

        Returns:
            float: The training time in seconds.
        """
        start_time = time.time()  # Record the start time

        # Your training code here
        self.classifier.fit(X, y)

        end_time = time.time()  # Record the end time
        training_time = end_time - start_time  # Calculate the training time in seconds

        return training_time

    def evaluate_clf(self, X_test, y_test):
        """
        Evaluates the classifier using the provided test data and displays
        evaluation metrics, confusion matrix, and ROC curve.

        Args:
            X_test (array-like): The test data features.
            y_test (array-like): The test data labels.
        """
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
        y_score = app.classifier.pipeline.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_display.plot()
        st.pyplot(plt.gcf())  # type: ignore


if __name__ == "__main__":
    show_pages_from_config()

    # Title and Description
    st.title("Spam Classifier")
    st.write(
        """ This is a Streamlit app for spam classification.
        It allows users to select a classifier, configure preprocessing steps, train the classifier, evaluate its performance, and make predictions on custom text inputs."""
    )
    st.divider()

    app = SpamClassifierApp()
    with st.sidebar:
        app.get_settings()

    # st.write(app.classifier.clf)

    with st.spinner("Preparing Data..."):
        X_train, X_test, y_train, y_test = Datasets.split_transform_data(
            app.data, app.preprocess
        )

    with st.spinner("Training The Classifier..."):
        training_time = app.train_clf(X_train, y_train)

    st.success(
        f"Classifier Trained Successfully! Training Time: `{training_time:0.2f}` secs"
    )

    eval_tab, custom_input_tab = st.tabs(
        ["Evaluate the classifier", "Test with a text input"]
    )

    with custom_input_tab:
        text_input = st.text_area("Enter a text to test the classifier")
        if text_input:
            with st.spinner("Preprocessing The Text..."):
                preprocessed_text = app.preprocess.transform([text_input])
            with st.spinner("Predicting..."):
                prediction = app.classifier.predict(preprocessed_text)
            st.markdown(f"Prediction: **`{'Spam' if prediction[0] == 1 else 'HAM'}`**")

    with eval_tab:
        with st.spinner("Evaluating The Classifier..."):
            app.evaluate_clf(X_test, y_test)
