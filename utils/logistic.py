# word embedding + svm/logistic regression
# %%
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import sklearn.svm as svm
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import re
# %%
# load data
test_lyrics = pd.read_csv("../data/df_train.csv")
train_lyrics = pd.read_csv("../data/df_train.csv")


def str2list(s):
    return re.findall(r"'([^']*)'", s)


test_lyrics['lyrics'] = test_lyrics['lyrics'].apply(str2list)
train_lyrics['lyrics'] = train_lyrics['lyrics'].apply(str2list)
# %%
# word2vec embedding
all_lyrics = test_lyrics['lyrics'].tolist() + train_lyrics['lyrics'].tolist()
model = Word2Vec(all_lyrics, min_count=1)
# %%
# add a new column to the data matrix
test_lyrics['embedded_lyrics'] = test_lyrics['lyrics'].apply(
    lambda x: [model.wv[word] for word in x])
train_lyrics['embedded_lyrics'] = train_lyrics['lyrics'].apply(
    lambda x: [model.wv[word] for word in x])
# %%

# %%
# Prepare the data
X_train = [np.mean(embeddings, axis=0)
           for embeddings in train_lyrics['embedded_lyrics']]
y_train = train_lyrics['year'].astype(str)

X_test = [np.mean(embeddings, axis=0)
          for embeddings in test_lyrics['embedded_lyrics']]
y_test = test_lyrics['year'].astype(str)

# Map years to decade categories
y_train_decades = np.where(train_lyrics['year'] < 1960, '1960s',
                           np.where(train_lyrics['year'] > 2020, '2010s',
                                    (train_lyrics['year'] // 10 * 10).astype(str) + 's'))

y_test_decades = np.where(test_lyrics['year'] < 1960, '1960s',
                          np.where(test_lyrics['year'] > 2020, '2010s',
                                   (test_lyrics['year'] // 10 * 10).astype(str) + 's'))

# Train the model
model = SVC(kernel='rbf')
model.fit(X_train, y_train_decades)

# Predict on the test set
y_pred_decades = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test_decades, y_pred_decades)
print("Accuracy:", accuracy)
# %%
