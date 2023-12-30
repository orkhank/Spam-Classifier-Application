import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from st_pages import add_indentation
from wordcloud import WordCloud
from preprocess import Preprocess
from dataset import Datasets


def explore_data(data: pd.DataFrame):
    # TODO: add character_count, word_count, sentence_count information for each email and compare them based on ham/spam
    # Print samples

    data['Character_Count'] = data['Body'].apply(len)
    data['Word_Count'] = data['Body'].apply(lambda x: len(x.split()))
    data['Sentence_Count'] = data['Body'].apply(lambda x: len(x.split('.')))


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
    with st.expander("Word Clouds"):
        spam_wordcloud = WordCloud(width=800, height=400).generate(' '.join(data[data['Label'] == 1]['Body']))
        ham_wordcloud = WordCloud(width=800, height=400).generate(' '.join(data[data['Label'] == 0]['Body']))

        st.write("## Spam Word Cloud")
        st.image(spam_wordcloud.to_image(), caption="Spam Word Cloud", use_column_width=True)

        st.write("## Ham Word Cloud")
        st.image(ham_wordcloud.to_image(), caption="Ham Word Cloud", use_column_width=True)
    # TODO: Show word clouds


add_indentation()

raw_data_tab, preprocessed_data_tab = st.tabs(["Raw Data", "Preprocessed Data"])

with st.sidebar:
    raw_data = Datasets.get_multi(raw=True)

with raw_data_tab:
    explore_data(raw_data)

with preprocessed_data_tab:
    data = Datasets.clean_data(raw_data)
    preprocess = Preprocess()
    with st.expander("Preprocess Steps"):
        preprocess.get_steps()

    # from sklearn.preprocessing import LabelEncoder

    st.write(data)
    preprocessed_body = pd.DataFrame(
        preprocess.transform(data["Body"]),
        columns=["Body"],
        index=data.index,
    )
    st.write(preprocessed_body)
    preprocessed_data = pd.concat(
        [
            preprocessed_body,
            data["Label"],
        ],
        axis=1,
    )
    # le = LabelEncoder()
    # limited_data["Label"] = le.fit_transform(limited_data["Label"])
    explore_data(preprocessed_data)
