import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from preprocess import preprocess_pipeline


def clean_data(df: pd.DataFrame):
    data = df.copy()
    data.drop(
        data.columns[data.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )

    data.drop_duplicates()
    data.drop(data[data["Body"] == "empty"].index, inplace=True)
    data.dropna(inplace=True)

    return data


# datasets
dataset_folder = "datasets/archive"
dataset_dict = {
    "Spam Assassin": f"{dataset_folder}/completeSpamAssassin.csv",
    "EnronSpam": f"{dataset_folder}/enronSpamSubset.csv",
    "LingSpam": f"{dataset_folder}/lingSpam.csv",
}


with st.sidebar:
    dataset_selectbox = st.selectbox("# Dataset", dataset_dict.keys())

data = clean_data(pd.read_csv(dataset_dict[dataset_selectbox]))

# Plot pie chart of ham/spam distribution
fig = plt.figure()
ax = fig.add_subplot(111)
ax.pie(data["Label"].value_counts(), labels=[" not spam", "spam"], autopct="%0.2f")

st.pyplot(fig)

with st.expander("See dataset"):
    st.table(data)


emails_train, emails_test, target_train, target_test = train_test_split(
    data["Body"], data["Label"], test_size=0.2, random_state=42
)

X_train = [preprocess_pipeline(x) for x in emails_train]
X_test = [preprocess_pipeline(x) for x in emails_test]

# print 5 training samples
st.write(X_train[:5])


# Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(target_train.values)
y_test = le.transform(target_test.values)


# CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

naive_bayes_clf = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())])
naive_bayes_clf.fit(X_train, y_train)


# ------- Metrics and Performance -------
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

# Scores
y_predict = [1 if o > 0.5 else 0 for o in naive_bayes_clf.predict(X_test)]

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

st.pyplot(ax.figure)

# ROC Curve
y = np.array([0, 0, 1, 1])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(
    fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator"
)
st.pyplot(display.plot().figure_)

# ----------------------------
