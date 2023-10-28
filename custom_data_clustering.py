# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:38:53 2023

@author: SK075631
"""

import streamlit as st


import hdbscan
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from bertopic import BERTopic
import genieclust
from top2vec import Top2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import re
import unicodedata
import spacy
import nltk
import nltkmodule
from nltk.corpus import stopwords


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.contractions_dict = {
        "ain't": "are not",
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "isn't": "is not",
        "shan't": "shall not",
        "shouldn't": "should not",
        "wasn't": "was not",      
        "weren't": "were not",    
        "won't": "will not",
        "wouldn't": "would not",
        "'cause": 'because',
         "could've": 'could have',
         "he'd": 'he would',
         "he'll": 'he will',
         "he's": 'he is',
         "i'd": 'I would',
         "i'll": 'I will',
         "i'm": 'I am',
         "i've": 'I have',
         "it'd": 'it would',
         "it'll": 'it will',
         "it's": 'it is',
         "let's": 'let us',
         "might've": 'might have',
         "must've": 'must have',
         "she'd": 'she would',
         "she'll": 'she will',
         "she's": 'she is',
         "should've": 'should have',
         "that's": 'that is',
         "there's": 'there is',
         "they'd": 'they would',
         "they'll": 'they will',
         "they're": 'they are',
         "they've": 'they have',
         "we'd": 'we would',
         "we'll": 'we will',
         "we're": 'we are',
         "we've": 'we have',
         "what'll": 'what will',
         "what're": 'what are',
         "what's": 'what is',
         "what've": 'what have',
         "where's": 'where is',
         "who'd": 'who would',
         "who'll": 'who will',
         "who're": 'who are',
         "who's": 'who is',
         "who've": 'who have',
         "would've": 'would have',
         "you'd": 'you would',
         "you'll": 'you will',
         "you're": 'you are',
         "you've": 'you have'

    }
        
    def expand_contractions(self, text):
        for key, value in self.contractions_dict.items():
            text = text.replace(key, value)
        return text

    def clean_special_characters(self, text):
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        negative_words = ["no", "not", "don't", "doesn't", "aren't", "isn't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "couldn't",
                        "never", "neither", "nor", "nothing", "nowhere", "noone", "none", "nobody", "hardly", "scarcely", "barely", "rarely"]
        stop_words = [word for word in stop_words if word.lower() not in negative_words]

        text = self.expand_contractions(text)
        words = nltk.word_tokenize(text)
        words = [word for word in words if (
            (len(word) >= 3 and word.lower() not in stop_words) or
            (len(word) >= 3 and any(neg_word in word.lower() for neg_word in negative_words)) or
            (word.lower() == "no")
        )]
        cleaned_text = ' '.join(words)
        return cleaned_text

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc]
        return ' '.join(lemmas)

    def text_preprocessing(self, text):
        text = text.lower()
        text = self.expand_contractions(text)
        text = self.clean_special_characters(text)
        text = re.sub(r'\brt\b', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s\']+', ' ', text)
        text = re.sub(r'\b\w*\d\w*\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = self.remove_stop_words(text)
        text = self.lemmatize_text(text)
        return text


class TextClustering:
    def __init__(self, min_subcluster_size=20):

        self.min_subcluster_size = min_subcluster_size

    def calculate_umap_args(self, doc_size):
        umap_defaults = {'metric': 'cosine', 'random_state': 42}

        if doc_size > 5000:
            return {'n_neighbors': 25, 'n_components': 20, **umap_defaults}
        elif doc_size > 2500:
            return {'n_neighbors': 20, 'n_components': 15, **umap_defaults}
        elif doc_size > 1000:
            return {'n_neighbors': 15, 'n_components': 10, **umap_defaults}
        elif doc_size > 500:
            return {'n_neighbors': 10, 'n_components': 5, **umap_defaults}
        else:
            return {'n_neighbors': 5, 'n_components': 5, **umap_defaults}

    def create_top2vec_model(self, dataframe, column_name):
        # Extract the specified column from the DataFrame
        documents = dataframe[column_name].tolist()

        # Calculate UMAP arguments based on the document size
        doc_size = len(documents)
        umap_args = self.calculate_umap_args(doc_size)

        hdbscan_args = {'min_cluster_size': 2, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}
      
        model = Top2Vec(
            documents=documents,
            speed='deep-learn',
            workers=8,
            min_count=5,
            embedding_model='all-MiniLM-L6-v2',
            umap_args=umap_args,
            hdbscan_args=hdbscan_args
        )

        # Get cluster labels and add them to the DataFrame
        cluster_labels = model.doc_top
        dataframe['INITAL_CLUSTERS'] = cluster_labels

        return dataframe

    def subcluster_data(self, dataframe, cleaned_summary_column):
        
        
        subcluster_dfs = []  # List to store sub-cluster DataFrames
        grouped = dataframe.groupby('INITAL_CLUSTERS')

        for group_name, group_data in tqdm(grouped, desc="Subclustering Progress"):
            if len(group_data) >= self.min_subcluster_size:
                group_data[cleaned_summary_column].fillna('NaN', inplace=True)
                docs_summary = group_data[cleaned_summary_column].tolist()

                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=5, min_samples=2, metric='euclidean', cluster_selection_method='eom',
                    prediction_data=True
                )
                summary_topic_model = BERTopic(hdbscan_model=hdbscan_model)
                summary_topics, summary_probs = summary_topic_model.fit_transform(docs_summary)

                if len(set(summary_topics)) < 4:
                    n_cluster = int(np.sqrt(len(set(docs_summary))))
                    genie_model = genieclust.Genie(
                        n_clusters=n_cluster, M=2, affinity='cosine', postprocess="all", gini_threshold=0.3
                    )
                    summary_topic_model = BERTopic(hdbscan_model=genie_model)
                    summary_topics, summary_probs = summary_topic_model.fit_transform(docs_summary)

                try:
                    if -1 in summary_topics:
                        new_summary_topics = summary_topic_model.reduce_outliers(docs_summary, summary_topics)
                    else:
                        new_summary_topics = summary_topics

                    group_data['SUB_CLUSTERS'] = new_summary_topics
                except Exception as e:
                    print(f"Error in sub-clustering: {e}")
                    pass

                subcluster_dfs.append(group_data)
            else:
                group_data['SUB_CLUSTERS'] = 0
                subcluster_dfs.append(group_data)

        # Concatenate sub-cluster DataFrames to get the final result
        final_df = pd.concat(subcluster_dfs, ignore_index=True)

        return final_df

    def process_data(self, dataframe, input_column_name,cleaned_column_name):
        
        preprocessor = TextPreprocessor()
        dataframe = dataframe[~dataframe[input_column_name].isnull()].reset_index(drop=True)
        dataframe[cleaned_column_name] = dataframe[input_column_name].progress_apply(lambda x: preprocessor.text_preprocessing(x))
        
        
        # Create the Top2Vec model and return the DataFrame with cluster labels
        labeled_dataframe = self.create_top2vec_model(dataframe, input_column_name)

        # Perform subclustering
        result_df = self.subcluster_data(labeled_dataframe, cleaned_column_name)

        # Add cluster descriptions and labels
        result_df = self.add_cluster_labels(result_df, 'INITAL_CLUSTERS', 'SUB_CLUSTERS', cleaned_column_name)

        return result_df

    @staticmethod
    def extract_keywords(text, num_keywords):
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            keyword_scores = dict(zip(feature_names, tfidf_scores))
            sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            keywords = sorted_keywords[:num_keywords]
            return ' '.join([k[0] for k in keywords])
        except:
            return "undefined"

    @staticmethod
    def add_cluster_labels(df, final_cluster_column, summary_cluster_column, text_column, num_keywords=5):
        unique_final_clusters = df[final_cluster_column].unique()

        # Create a dictionary to store cluster descriptions
        cluster_descriptions = {}

        for final_cluster in unique_final_clusters:
            final_cluster_indices = df[df[final_cluster_column] == final_cluster].index.tolist()
            samples_to_choose = min(len(final_cluster_indices), 20)

            if samples_to_choose == 0:
                final_cluster_label = "undefined_" + str(final_cluster)
            else:
                sampled_texts = df.loc[random.sample(final_cluster_indices, samples_to_choose), text_column]
                text = ' '.join(sampled_texts)
                keywords = TextClustering.extract_keywords(text, num_keywords)

                if keywords == "undefined":
                    final_cluster_label = "undefined_" + str(final_cluster)
                else:
                    final_cluster_label = ' '.join(list(set(keywords.split(" "))))

            # Add the cluster description to the dictionary
            cluster_descriptions[final_cluster] = final_cluster_label

        sub_cluster_descriptions = {}
        # Now, generate descriptions for SUMMARY_SUB_CLUSTERS within each FINAL_CLUSTER
        for final_cluster in unique_final_clusters:
            sub_cluster_labels = df[df[final_cluster_column] == final_cluster][summary_cluster_column].unique()

            for sub_cluster in sub_cluster_labels:
                sub_cluster_indices = df[(df[final_cluster_column] == final_cluster) & (df[summary_cluster_column] == sub_cluster)].index.tolist()
                samples_to_choose = min(len(sub_cluster_indices), 20)

                if samples_to_choose == 0:
                    sub_cluster_label = "undefined_" + str(sub_cluster)
                else:
                    sampled_texts = df.loc[random.sample(sub_cluster_indices, samples_to_choose), text_column]
                    text = ' '.join(sampled_texts)
                    keywords = TextClustering.extract_keywords(text, num_keywords)

                    if keywords == "undefined":
                        sub_cluster_label = "undefined_" + str(sub_cluster)
                    else:
                        sub_cluster_label = ' '.join(list(set(keywords.split(" "))))

                # Add the sub-cluster description to the dictionary
                combined_cluster = f"{final_cluster}_{sub_cluster}"
                sub_cluster_descriptions[combined_cluster] = sub_cluster_label

        # Map cluster descriptions to labels for FINAL_CLUSTER
        df['INITIAL_CLUSTERS_LABEL'] = df[final_cluster_column].map(cluster_descriptions)

        # Map cluster descriptions to labels for SUMMARY_SUB_CLUSTERS
        df['SUB_CLUSTERS_LABEL'] = (df[final_cluster_column].map(str) + '_' + df[summary_cluster_column].map(str)).map(sub_cluster_descriptions)

        return df

# Define a function to run clustering and display results
def run_clustering(dataframe, text_column):
    # Initialize the TextClustering class
    clustering = TextClustering()

    # Process the data
    result_df = clustering.process_data(dataframe, text_column, 'cleaned_text')

    # Display the clustering results
    st.header("Clustering Results")
    st.dataframe(result_df, height=500)

    # Add a download button for the CSV file
    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering_results.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Text Clustering App",
        page_icon="📚",
        layout="wide",  # Enable a wide layout for more content
        initial_sidebar_state="expanded"  # Expand the sidebar by default
    )

    # Apply a colorful background
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f8ff; /* Light blue background color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Text Clustering App")

    # Upload an Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Read the Excel file into a DataFrame
            dataframe = pd.read_excel(uploaded_file)

            # Display the DataFrame
            st.header("Uploaded DataFrame")
            st.dataframe(dataframe, height=300)

            # Select the text column for clustering
            text_columns = dataframe.columns.tolist()
            selected_text_column = st.selectbox("Select the text column for clustering", text_columns)

            if st.button("Run Clustering"):
                st.info("Clustering in progress...")
                # Run clustering on the selected text column
                run_clustering(dataframe, selected_text_column)
                st.success("Clustering completed!")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
