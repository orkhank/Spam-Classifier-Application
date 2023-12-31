"""
This module is used to compare the different classifiers and their settings.

It provides functions to plot the accuracy and runtime of classifiers as a function of different parameters, such as the number of training examples and runtime. The module also includes a main function that compares the performance of various classifiers on a given dataset using different preprocessing steps and vectorizers.

Functions:
- plot_accuracy: Plots the accuracy as a function of a given parameter.
- plot_runtime: Plots the runtime as a function of a given parameter.
- plot: Plots the accuracy, runtime, and training times of classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from vectorizers import Vectorizer
from dataset import Datasets
from preprocess import Preprocess
from classifiers import NaiveBayes, SVM, RandomForest
from sklearn.metrics import accuracy_score


def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x.

    Args:
        x (list): The values of x.
        y (list): The corresponding accuracy values.
        x_legend (str): The label for the x-axis.

    Returns:
        None
    """
    x = np.array(x)
    y = np.array(y)
    plt.title("Classification accuracy as a function of %s" % x_legend)
    plt.xlabel("%s" % x_legend)
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.plot(x, y)


def plot_runtime(x, y, x_legend):
    """Plot runtime as a function of x.

    Args:
        x (list): The values of x.
        y (list): The corresponding runtime values.
        x_legend (str): The label for the x-axis.

    Returns:
        None
    """
    x = np.array(x)
    y = np.array(y)
    plt.title("Classification runtime as a function of %s" % x_legend)
    plt.xlabel("%s" % x_legend)
    plt.ylabel("runtime (s)")
    plt.grid(True)
    plt.plot(x, y)


def plot(cls_stats, n_test_documents):
    """Plot accuracy, runtime, and training times of classifiers.

    Args:
        cls_stats (dict): A dictionary containing the statistics of each classifier.
        n_test_documents (int): The number of test documents.

    Returns:
        None
    """
    cls_names = list(sorted(cls_stats.keys()))

    # Plot accuracy evolution
    fig = plt.figure()
    for _, stats in sorted(cls_stats.items()):
        # Plot accuracy evolution with #examples
        plot_accuracy(stats["n_samples"], stats["accuracies"], "training examples (#)")
        ax = plt.gca()
        ax.set_ylim((0.8, 1))
    plt.legend(cls_names, loc="best")
    st.pyplot(fig)

    fig = plt.figure()
    for _, stats in sorted(cls_stats.items()):
        # Plot accuracy evolution with runtime
        run_times = stats["fit_times"]
        accuracies = stats["accuracies"]

        # sort the values by runtime
        idx = np.argsort(run_times)
        run_times = np.array(run_times)[idx]
        accuracies = np.array(accuracies)[idx]

        plot_accuracy(run_times, accuracies, "runtime (s)")
        ax = plt.gca()
        ax.set_ylim((0.8, 1))
    plt.legend(cls_names, loc="best")
    st.pyplot(fig)

    # Plot fitting times
    plt.figure()
    fig = plt.gcf()
    cls_runtime = [
        stats["fit_times"][-1] for cls_name, stats in sorted(cls_stats.items())
    ]

    bar_colors = ["b", "g", "r", "c", "m", "y"]

    ax = plt.subplot(111)
    rectangles = plt.bar(
        range(len(cls_names)), cls_runtime, width=0.5, color=bar_colors
    )

    ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
    ax.set_xticklabels(cls_names, fontsize=10)
    ymax = max(cls_runtime) * 1.2
    ax.set_ylim((0, ymax))
    ax.set_ylabel("runtime (s)")
    ax.set_title("Training Times")

    def autolabel(rectangles):
        """Attach some text via autolabel on rectangles.

        Args:
            rectangles (list): List of rectangles.

        Returns:
            None
        """
        for rect in rectangles:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                1.05 * height,
                "%.4f" % height,
                ha="center",
                va="bottom",
            )
            plt.setp(plt.xticks()[1], rotation=30)

    autolabel(rectangles)
    fig.tight_layout()
    st.pyplot(fig)

    # Plot fitting times as a function of the number of training documents
    fig = plt.figure()
    for _, stats in sorted(cls_stats.items()):
        # Plot accuracy evolution with runtime
        plot_runtime(stats["n_samples"], stats["fit_times"], "training examples (#)")
        ax = plt.gca()
        # ax.set_ylim((0.8, 1))
    plt.legend(cls_names, loc="best")
    st.pyplot(fig)

    # Plot prediction times
    fig = plt.figure()
    cls_runtime = []
    cls_names = list(sorted(cls_stats.keys()))
    for cls_name, stats in sorted(cls_stats.items()):
        cls_runtime.append(stats["prediction_times"][-1])

    ax = plt.subplot(111)
    rectangles = plt.bar(
        range(len(cls_names)), cls_runtime, width=0.5, color=bar_colors
    )

    ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
    ax.set_xticklabels(cls_names, fontsize=8)
    plt.setp(plt.xticks()[1], rotation=30)
    ymax = max(cls_runtime) * 1.2
    ax.set_ylim((0, ymax))
    ax.set_ylabel("runtime (s)")
    ax.set_title("Prediction Times (%d instances)" % n_test_documents)
    autolabel(rectangles)
    fig.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    st.title("Spam Classifier App")
    st.markdown(
        """
        This app allows you to compare different classifiers and their settings. It provides a user-friendly interface to analyze and evaluate the performance of various classifiers on a given dataset. You can explore different preprocessing steps, select a vectorizer, and choose from a list of classifiers including Naive Bayes, SVM, and Random Forest. The app provides visualizations of accuracy, runtime, and other performance metrics to help you make informed decisions about classifier selection. 
        """
    )
    # Initialize the results dictionary
    results = {}

    vectorizer = Vectorizer.vectorizer_selection().get_vectorizer()
    # List of classifiers
    classifiers = [NaiveBayes(), SVM(), RandomForest()]

    for classifier in classifiers:
        classifier.get_parameters(vectorizer)
    data = Datasets.get_multi()
    preprocess = Preprocess()
    with st.expander("Configure Preprocess Steps"):
        preprocess.get_steps(None)

    with st.spinner("Preprocessing data..."):
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = Datasets.split_transform_data(data, preprocess)

    # For each classifier
    for clf in classifiers:
        # Initialize lists
        n_samples = []
        fit_times = []
        accuracies = []
        prediction_times = []

        # For each subset of the training data
        for i in range(500, len(X_train), 100):
            # Record the fit start time
            fit_start = time.time()

            # Train the classifier
            clf.fit(X_train[:i], y_train[:i])

            # Record the fit end time
            fit_end = time.time()

            # Record the prediction start time
            prediction_start = time.time()

            # Predict the labels for the test data
            y_pred = clf.predict(X_test)

            # Record the prediction end time
            prediction_end = time.time()

            # Calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Append the number of training samples, runtime, and accuracy
            n_samples.append(i)
            fit_times.append(fit_end - fit_start)
            accuracies.append(accuracy)
            prediction_times.append(prediction_end - prediction_start)

        # Store the lists in the results dictionary
        results[clf.__class__.__name__] = {
            "n_samples": n_samples,
            "fit_times": fit_times,
            "accuracies": accuracies,
            "prediction_times": prediction_times,
        }

    plot(results, len(X_test))
