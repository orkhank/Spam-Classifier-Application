import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from st_pages import add_indentation

from preprocess import Preprocess, clean_data


def explore_data(data: pd.DataFrame):
    # Print samples
    with st.expander("Samples from the dataset(s)"):
        st.write(data.sample(5))

    # Plot pie chart of ham/spam distribution
    with st.expander("Ham/Spam Distribution"):
        label_counts = data["Label"].value_counts()
        label_counts.rename({0: "Not Spam", 1: "Spam"}, inplace=True)
        st.write("## Label Counts")
        col1, col2 = st.columns(2)
        with col1:
            st.write(label_counts)
        with col2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.pie(label_counts, labels=[" not spam", "spam"], autopct="%0.2f")

            st.pyplot(fig)

    # Information about the data
    import io

    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # TODO: Show word clouds

add_indentation()

raw_data_tab, preprocessed_data_tab = st.tabs(["Raw Data", "Preprocessed Data"])

# datasets
dataset_folder = "datasets/archive"
dataset_names = ["Spam Assassin", "EnronSpam", "LingSpam"]
dataset_paths = [
    f"{dataset_folder}/completeSpamAssassin.csv",
    f"{dataset_folder}/enronSpamSubset.csv",
    f"{dataset_folder}/lingSpam.csv",
]
dataset_dict = {name: path for name, path in zip(dataset_names, dataset_paths)}


with st.sidebar:
    dataset_name_multibox = st.multiselect("# Dataset", dataset_names, dataset_names[0])

if not dataset_name_multibox:
    st.warning("Please select a dataset or a combination of datasets from the sidebar.")
    st.stop()

# Concatanate given datasets
raw_data = pd.concat(
    [pd.read_csv(dataset_dict[dataset_name]) for dataset_name in dataset_name_multibox]
)

with raw_data_tab:
    explore_data(raw_data)

with preprocessed_data_tab:
    data = clean_data(raw_data)
    limited_data = data.sample(500)
    preprocess = Preprocess()
    with st.expander("Preprocess Steps"):
        preprocess.get_steps()

    from sklearn.preprocessing import LabelEncoder

    st.write(limited_data)
    st.write(limited_data["Label"])
    st.write(pd.DataFrame(preprocess.transform(limited_data["Body"])))
    preprocessed_data = pd.concat(
        [
            pd.DataFrame(preprocess.transform(limited_data["Body"])),
            limited_data["Label"],
        ],
        axis=1,
    )
    # le = LabelEncoder()
    # limited_data["Label"] = le.fit_transform(limited_data["Label"])
    explore_data(preprocessed_data)
