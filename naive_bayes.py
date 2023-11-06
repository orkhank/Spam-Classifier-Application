import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from preprocess import preprocess_pipeline, clean_data


# datasets
dataset_folder = "datasets/archive"
dataset_dict = {
    "Spam Assassin": f"{dataset_folder}/completeSpamAssassin.csv",
    "EnronSpam": f"{dataset_folder}/enronSpamSubset.csv",
    "LingSpam": f"{dataset_folder}/lingSpam.csv",
}


data = clean_data(pd.read_csv(dataset_dict["Spam Assassin"]))


emails_train, emails_test, target_train, target_test = train_test_split(
    data["Body"], data["Label"], test_size=0.2, random_state=42
)

X_train = [preprocess_pipeline(x) for x in emails_train]
X_test = [preprocess_pipeline(x) for x in emails_test]


# Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = np.array(le.fit_transform(target_train.values))
y_test = np.array(le.transform(target_test.values))


# CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

naive_bayes_clf = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())])
naive_bayes_clf.fit(X_train, y_train)
