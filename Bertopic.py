# # %%
# import numpy as np
# import pandas as pd
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_punctuation
# from gensim import corpora
# import re
# from gensim import models
# import gensim.downloader as api
# from nltk.stem.wordnet import WordNetLemmatizer
# from bertopic import BERTopic
# # from umap import UMAP
# # from hdbscan import HDBSCAN
# from cuml.cluster import HDBSCAN
# from cuml.manifold import UMAP
# from bertopic.vectorizers import ClassTfidfTransformer
# from bertopic.representation import MaximalMarginalRelevance
# from sentence_transformers import SentenceTransformer
# # %%
# def remove_specific_words(s):
#     words_to_remove = ["Lyrics", "\[.+\]", "Contributors", "translators",
#                        "Embed", "You might also like",'Chorus',"Verse",
#                        "Intro","Outro","Hook","Bridge","Pre-Chorus",
#                        "Post-Chorus","Refrain","Interlude","Instrumenta"]
#     for word in words_to_remove:
#         s = re.sub(word, " ", s, flags=re.IGNORECASE)
#     return s

# words_to_delete = pd.read_csv("data/delete_word_list.txt",header=None)

# def delete_words(s):
#     for word in words_to_delete[1]:
#         s = re.sub(r'\b{}\b'.format(word), ' ', s, flags=re.IGNORECASE)
#     return s

# df = pd.read_csv("data/billboard_lyrics_genres.csv")
# df["lyrics"] = df["lyrics"].map(remove_specific_words)
# df["lyrics"] = df["lyrics"].map(delete_words)
# df_tmp = df.reset_index(drop=True)
# df_tmp.to_csv("../lyrics-data/lyrics4Bert.csv",index=False)

# # %%
# # =============================================================================
# # strip changerow and remove \n and \r
# # =============================================================================

# def strip_changerow(l):
#     l = re.sub(r"\r"," ",l)
#     l = re.sub(r"\n"," ",l)
#     return l

# corpus = []
# year = []
# df_tmp = pd.read_csv("data/lyrics4Bert.csv",index_col=0)

# for i, row in df_tmp.iterrows():
#     ltmp = list(map(strip_changerow, row["lyrics"].split("\n")))
#     ltmp = list(map(strip_multiple_whitespaces, ltmp))
#     ltmp = [x.strip(' ') for x in ltmp]
#     ltmp = [x for x in ltmp if x != ""]
#     corpus = corpus + ltmp
#     year = year + list(np.ones(len(ltmp), dtype=np.int32) * row["year"])
# # %%
# # =============================================================================
# # fit BERTopic with different parameters
# # =============================================================================
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = sentence_model.encode(corpus)
# umap_model = UMAP(n_neighbors=10,n_components=10,metric='cosine',low_memory=False)
# hdbscan_model = HDBSCAN(min_cluster_size=10,metric="euclidean",prediction_data=True)
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# representation_model = MaximalMarginalRelevance(diversity=.4)
# topic_model = BERTopic(
#     verbose=True,
#     n_gram_range=(1,2),
#     embedding_model=sentence_model,
#     umap_model=umap_model,
#     hdbscan_model=hdbscan_model,
#     ctfidf_model=ctfidf_model,
#     representation_model=representation_model
#     )
# topics, probs = topic_model.fit_transform(corpus,embeddings)

# # %%
# topic_model.save("models/BERTmodel_3")
# # topic_model = BERTopic.load("models/BERTmodel_1")
# # %%
# topics_over_time = topic_model.topics_over_time(corpus,year,nr_bins=20)
# topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
# # %%
# fig_topics_over_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
# fig_topics_over_time.write_html("./Figures/html/topics_20_v3.html")
# %%
# =============================================================================
# Multiprocessing for BERTopic
# =============================================================================

import os
import json
from itertools import product
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from nltk.stem.wordnet import WordNetLemmatizer
from bertopic import BERTopic
# from cuml.cluster import HDBSCAN
# from cuml.manifold import UMAP
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer


def remove_specific_words(s):
    words_to_remove = ["Lyrics", "\[.+\]", "Contributors", "translators",
                       "Embed", "You might also like", 'Chorus', "Verse",
                       "Intro", "Outro", "Hook", "Bridge", "Pre-Chorus",
                       "Post-Chorus", "Refrain", "Interlude", "Instrumenta"]

    for word in words_to_remove:
        s = re.sub(word, " ", s, flags=re.IGNORECASE)

    return s


def delete_words(s):
    delete_words_list = pd.read_csv("data/delete_word_list.txt", header=None)[1].values.tolist()
    for word in delete_words_list:
        s = re.sub(r'\b{}\b'.format(word), ' ', s, flags=re.IGNORECASE)
    return s


def strip_changerow(l):
    l = re.sub(r"\r", " ", l)
    l = re.sub(r"\n", " ", l)
    return l


def process_combination(*args):

    parameters, corpus, year, figures_folder, data_folder = args

    print(f"Starting combination{parameters}...")

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(corpus)

    umap_model = UMAP(n_neighbors=parameters["n_neighbors"], n_components=parameters["n_components"],
                      metric=parameters["metric"], low_memory=False)
    hdbscan_model = HDBSCAN(min_cluster_size=parameters["min_cluster_size"], metric="euclidean",
                            prediction_data=True)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = MaximalMarginalRelevance(diversity=parameters["diversity"])

    topic_model = BERTopic(
        embedding_model=sentence_model,
        verbose=True,
        n_gram_range=parameters["n_gram_range"],
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model
    )

    topics, probs = topic_model.fit_transform(corpus, embeddings)

    print("Model fitted.\nSaving figures...")

    # Save figures with unique names based on hyperparameters
    figures_subfolder = f"{figures_folder}/{parameters['n_neighbors']}_{parameters['n_components']}_" \
                        f"{parameters['metric']}_{parameters['min_cluster_size']}_{parameters['diversity']}"
    data_subfolder = f"{data_folder}/{parameters['n_neighbors']}_{parameters['n_components']}_" \
                     f"{parameters['metric']}_{parameters['min_cluster_size']}_{parameters['diversity']}"
    os.makedirs(figures_subfolder, exist_ok=True)
    os.makedirs(data_subfolder, exist_ok=True)

    topics_over_time = topic_model.topics_over_time(corpus,year,nr_bins=20)
    fig_topics_over_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
    fig_topics_over_time.write_html(f"{figures_subfolder}/topics_over_time.html")

    print("Figures saved!\nSaving model...")
    topic_model.save(f"{data_subfolder}/BERTmodel")
    # print(f"Model saved!\nFinished combination{args}!")

# =============================================================================
# main funtion
# =============================================================================

if not os.path.exists("data/corpus.csv"):
    print("Loading data...")
    df= pd.read_csv("data/billboard_lyrics_genres.csv")
    df = df[['lyrics','year']]
    print("Data loaded!\nCleaning data...")

    # Clean lyrics
    df["lyrics"] = df["lyrics"].map(remove_specific_words)
    df["lyrics"] = df["lyrics"].map(delete_words)
    df["lyrics"] = df["lyrics"].map(strip_changerow)

    # Lemmatize lyrics
    lemmatizer = WordNetLemmatizer()
    df["lyrics"] = df["lyrics"].map(lemmatizer.lemmatize)

    print("Data cleaned!\nCreating corpus...")
    # Create corpus and year lists
    corpus = []
    year = []

    for _, row in df.iterrows():
        lyrics = row["lyrics"]
        lyrics_paragraphs = lyrics.split("\n\n")  # split by paragraphs
        lyrics_sentences = [sentence.strip() for paragraph in lyrics_paragraphs for sentence in paragraph.split(".")]
        lyrics_sentences = [sentence for sentence in lyrics_sentences if sentence]
        
        # split into lyrics_sentences
        # corpus.extend(lyrics_sentences)
        # year.extend([row["year"]] * len(lyrics_sentences))

        # split into lyrics_paragraphs
        corpus.extend(lyrics_paragraphs)
        year.extend([row["year"]] * len(lyrics_paragraphs))

    print("Corpus created!")

    # save corpus and year information
    df_corpus = pd.DataFrame({"corpus":corpus,"year":year})
    df_corpus.to_csv("data/corpus.csv",index=False)
    print("Corpus saved!")

# if corpus has been created
# Load corpus and year lists
print("Loading corpus...")
corpus = pd.read_csv("data/corpus.csv")["corpus"].values.tolist()
year = pd.read_csv("data/corpus.csv")["year"].values.tolist()
print("Corpus loaded!")

# Define hyperparameters
hyperparameters = {
    "n_neighbors": [5, 10, 15],
    "n_components": [5, 10, 15],
    "metric": ["cosine", "euclidean"],
    "min_cluster_size": [5, 10, 15],
    "diversity": [0.3, 0.4, 0.5],
    "n_gram_range": [(1, 1), (1, 2), (2, 2)]
}

# Define folders
figures_folder = "Figures"
data_folder = "data"

# Generate combinations of hyperparameters
combinations = list(product(*hyperparameters.values()))
num_combinations = len(combinations)

# Create arguments for parallel processing
arguments = [(dict(zip(hyperparameters.keys(), combination)), corpus, year, figures_folder, data_folder) for  combination in combinations]
print(f"Total combinations to process: {num_combinations}")

# Run parallel processes
print("Creating pool...")
results = []
with Pool(10) as pool:
    async_results = [pool.apply_async(process_combination, args) for args in arguments]
    print("Number of processes running:", pool._processes)

    # Retrieve the results
    for async_result in async_results:
        result = async_result.get()
        results.append(result)

# Save results
with open(f"{data_folder}/results.json", "w") as f:
    json.dump(results, f)

print("Done!")