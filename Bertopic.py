# %%
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_punctuation
from gensim import corpora
import re
from gensim import models
import gensim.downloader as api
from nltk.stem.wordnet import WordNetLemmatizer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
# %%
def remove_specific_words(s):
    words_to_remove = ["Lyrics", "\[.+\]", "Contributors", "translators",
                       "Embed", "You might also like",'Chorus',"Verse",
                       "Intro","Outro","Hook","Bridge","Pre-Chorus",
                       "Post-Chorus","Refrain","Interlude","Instrumenta"]
    
    for word in words_to_remove:
        s = re.sub(word, " ", s, flags=re.IGNORECASE)
    
    return s

words_to_delete = pd.read_csv("data/delete_word_list.txt",header=None)[1].values.tolist()

def delete_words(s):
    for word in words_to_delete:
        s = re.sub(r'\b{}\b'.format(word), ' ', s, flags=re.IGNORECASE)
    return s

df = pd.read_csv("data/billboard_lyrics_genres.csv")
df["lyrics"] = df["lyrics"].map(remove_specific_words)
df_tmp = df.reset_index(drop=True)
df_tmp["lyrics"] = df_tmp["lyrics"].map(delete_words)
df_tmp.to_csv("data/lyrics4Bert.csv",index=False)

# %%
def strip_changerow(l):
    l = re.sub(r"\r"," ",l)
    l = re.sub(r"\n"," ",l)
    return l

corpus = []
year = []
df_tmp = pd.read_csv("data/lyrics4Bert.csv",index_col=0)

for i, row in df_tmp.iterrows():
    ltmp = list(map(strip_changerow, row["lyrics"].split("\n\n")))
    ltmp = list(map(strip_multiple_whitespaces, ltmp))
    ltmp = [x.strip(' ') for x in ltmp]
    ltmp = [x for x in ltmp if x != ""]
    corpus = corpus + ltmp
    year = year + list(np.ones(len(ltmp), dtype=np.int32) * row["year"])

# %%
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(corpus)
umap_model = UMAP(n_neighbors=10,n_components=10,metric='cosine',low_memory=False)
hdbscan_model = HDBSCAN(min_cluster_size=10,metric="euclidean",prediction_data=True)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = MaximalMarginalRelevance(diversity=.4)
topic_model = BERTopic(embedding_model=sentence_model,verbose=True,n_gram_range=(1,2),
                       umap_model=umap_model,hdbscan_model=hdbscan_model,ctfidf_model=ctfidf_model,
                       representation_model=representation_model)
topics, probs = topic_model.fit_transform(corpus,embeddings)

# %%
topic_model.save("models/BERTmodel")
topic_model = BERTopic.load("models/BERTmodel")
# %%
topics_over_time = topic_model.topics_over_time(corpus,year,nr_bins=20)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
# %%
fig_topics_over_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
fig_topics_over_time.write_html("./Figures/html/topics_50_v1.html")

# %%
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
fig_docs = topic_model.visualize_documents(corpus, reduced_embeddings=reduced_embeddings)
fig_docs.write_html("./Figures/html/docs_original_embedding_v1.html")
