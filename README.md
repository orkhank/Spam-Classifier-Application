# Spam Classifier Application

This project is a spam classifier application built using Python and the Streamlit library. The application allows users to train and evaluate various machine learning models on a spam classification task.

## Features

- **Classifier Selection**: Users can select from three different classifiers - Naive Bayes, SVM, and Random Forest.
- **Preprocessing Configuration**: Users can configure various preprocessing steps to clean and prepare the data for training.
- **Model Training**: Users can train the selected classifier on the provided data.
- **Model Evaluation**: Users can evaluate the performance of the trained classifier, with metrics such as accuracy, precision, recall, and F1 score displayed. A confusion matrix and ROC curve are also provided.
- **Custom Text Prediction**: Users can input custom text to test the classifier's prediction.

## Usage

To get started, select a classifier from the sidebar and click on the train button. After the classifier is trained, you can evaluate it using the test button. You can also configure the classifier's parameters and the preprocessing steps from the sidebar. The dataset can be configured from the sidebar as well. The dataset can be explored from the data exploration page.

## Installation

This project requires Python and the following Python libraries installed:

- Streamlit
- Matplotlib
- Seaborn
- Scikit-learn

To run this application, download the project and navigate to the project directory. Then run the following command:

```bash
streamlit run app.py
```

