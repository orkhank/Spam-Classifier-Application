from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

from preprocess import Preprocess

dataset_folder = "datasets/archive"
dataset_names = ["Spam Assassin", "EnronSpam", "LingSpam"]
dataset_paths = [
    f"{dataset_folder}/completeSpamAssassin.csv",
    f"{dataset_folder}/enronSpamSubset.csv",
    f"{dataset_folder}/lingSpam.csv",
]
dataset_name_2_path_dict = {
    name: path for name, path in zip(dataset_names, dataset_paths)
}


class Datasets:
    @staticmethod
    def get_single(raw: bool = False):
        dataset_selectbox = st.selectbox("# Dataset", dataset_names)
        with st.spinner("Loading Data..."):
            raw_data = pd.read_csv(dataset_name_2_path_dict[dataset_selectbox])
        if raw:
            data = raw_data
        else:
            with st.spinner("Cleaning Raw Data..."):
                data = Datasets.clean_data(raw_data)

        previous_data_size = st.session_state.setdefault("data_size", 1000)
        total_data_size = data.shape[0]
        data_size = st.slider(
            "Configure Data Size",
            10,
            total_data_size,
            min(previous_data_size, total_data_size),
            10,
            "%d",
        )

        return data.sample(data_size, random_state=42)

    @staticmethod
    def get_multi(raw: bool = False):
        dataset_name_multibox = st.multiselect(
            "# Dataset", dataset_names, dataset_names[0]
        )

        if not dataset_name_multibox:
            st.warning("Please select a dataset or a combination of datasets.")
            st.stop()

        # Concatanate given datasets
        raw_data = pd.concat(
            [
                pd.read_csv(dataset_name_2_path_dict[dataset_name])
                for dataset_name in dataset_name_multibox
            ]
        )
        if raw:
            data = raw_data
        else:
            with st.spinner("Cleaning Raw Data..."):
                data = Datasets.clean_data(raw_data)

        previous_data_size = st.session_state.setdefault("data_size", 1000)
        total_data_size = data.shape[0]
        data_size = st.slider(
            "Configure Data Size",
            10,
            total_data_size,
            min(previous_data_size, total_data_size),
            10,
            "%d",
        )

        return data.sample(data_size, random_state=42)

    @staticmethod
    def clean_data(df: pd.DataFrame, sample: Optional[int] = None):
        data = df.copy()
        data.drop(
            data.columns[data.columns.str.contains("unnamed", case=False)],
            axis=1,
            inplace=True,
        )

        data.drop_duplicates(inplace=True)
        data.drop(data[data["Body"] == "empty"].index, inplace=True)
        data.dropna(inplace=True)

        if not sample:
            return data

        assert isinstance(sample, int), "type of sample must be int"
        return data.sample(sample)

    @staticmethod
    def split_transform_data(data, preprocess: Preprocess):
        emails_train, emails_test, target_train, target_test = train_test_split(
            data["Body"], data["Label"], test_size=0.2, random_state=42
        )

        @st.cache_data(show_spinner=False)
        def cached_preprocess(data):
            return preprocess.transform(data)

        # TODO: find some way to decrease the decrease preprocessing time (current version TAKES AGES)
        # ? Maybe preprocess all datasets before hand and let the user choose if it is worth to wait for the customs preprocess steps to finish or just use fast and the default preset
        preprocessed_emails_train = cached_preprocess(emails_train)
        preprocessed_emails_test = cached_preprocess(emails_test)

        le = LabelEncoder()
        y_train = np.array(le.fit_transform(target_train.values))
        y_test = np.array(le.transform(target_test.values))

        return preprocessed_emails_train, preprocessed_emails_test, y_train, y_test
