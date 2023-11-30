import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from st_pages import add_indentation

from preprocess import Preprocess
from dataset import Datasets


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

with st.sidebar:
    raw_data = Datasets.get_multi(raw=True)

with raw_data_tab:
    explore_data(raw_data)

with preprocessed_data_tab:
    data = Datasets.clean_data(raw_data)
    limited_data = data.sample(500)
    preprocess = Preprocess()
    with st.expander("Preprocess Steps"):
        preprocess.get_steps()

    from sklearn.preprocessing import LabelEncoder

    # TODO: correctly combine the preprocessed text (X) with labels (Y)
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
