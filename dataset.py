from typing import Optional
import pandas as pd
import streamlit as st

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
            "Configure Data Size", 10, total_data_size, min(previous_data_size, total_data_size), 10, "%d"
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

        data_size = st.slider(
            "Configure Data Size", 10, data.shape[0], data.shape[0] // 2, 10, "%d"
        )

        return data.sample(data_size)

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
