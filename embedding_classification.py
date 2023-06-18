# word embedding + svm/logistic regression -> classification
# i.e. predict year from lyrics
# %%
# =============================================================================
# Section 1: Import packages
# =============================================================================
# general packages
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn packages
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# scipy packages
from scipy.interpolate import griddata
from scipy.cluster.hierarchy import dendrogram, linkage

# gensim packages
from gensim.models import Word2Vec, FastText

# nltk packages
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# pickle: save trained models
import pickle

# =============================================================================
# Section 2: Load data & word embedding
# =============================================================================
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pickle

# load data
test_lyrics = pd.read_csv("./data/df_test.csv",index_col=0)
train_lyrics = pd.read_csv("./data/df_train.csv",index_col=0)


def str2list(s):
    return re.findall(r"'([^']*)'", s)


test_lyrics['lyrics'] = test_lyrics['lyrics'].apply(str2list)
train_lyrics['lyrics'] = train_lyrics['lyrics'].apply(str2list)


# FastText embedding
all_lyrics = test_lyrics['lyrics'].tolist() + train_lyrics['lyrics'].tolist()
size = 200
if not os.path.exists(f"./models/fasttext_{size}.model"):
    print(f"Training FastText model_{size}...")
    model = FastText(all_lyrics, min_count=1, vector_size=size, workers=4, window=10, sg=1)
    model.save(f"./models/fasttext_{size}.model")
else:
    print(f"Loading FastText model_{size}...")
    model = FastText.load(f"./models/fasttext_{size}.model")

# add a new column to the data matrix
test_lyrics['embedded_lyrics'] = test_lyrics['lyrics'].apply(
    lambda x: [model.wv[word] for word in x])
train_lyrics['embedded_lyrics'] = train_lyrics['lyrics'].apply(
    lambda x: [model.wv[word] for word in x])


# TF-IDF
vectorizer = TfidfVectorizer()
vectorizer.fit([' '.join(lyric) for lyric in all_lyrics])
print(vectorizer)

tfidf_matrix_train = vectorizer.transform([' '.join(lyric) for lyric in train_lyrics['lyrics']])
tfidf_matrix_test = vectorizer.transform([' '.join(lyric) for lyric in test_lyrics['lyrics']])

# Map years to 6 decade categories 
y_train_decades = np.where(train_lyrics['year'] < 1960, '1960s',
                           np.where(train_lyrics['year'] > 2020, '2010s',
                                    (train_lyrics['year'] // 10 * 10).astype(str) + 's'))
y_test_decades = np.where(test_lyrics['year'] < 1960, '1960s',
                          np.where(test_lyrics['year'] > 2020, '2010s',
                                   (test_lyrics['year'] // 10 * 10).astype(str) + 's'))

# Map years to binary classes: "before 2000" and "after 2000"
y_train_binary = np.where(train_lyrics['year'] < 2000, 0, 1)
y_test_binary = np.where(test_lyrics['year'] < 2000, 0, 1)

# Map years to 4 classes: "60s, 70s, and 80s", "90s", "00s", "10s"
y_train_4classes = np.where(train_lyrics['year'] < 1990, 0,
                            np.where(train_lyrics['year'] < 2000, 1,
                                     np.where(train_lyrics['year'] < 2010, 2, 3)))
y_test_4classes = np.where(test_lyrics['year'] < 1990, 0,
                           np.where(test_lyrics['year'] < 2000, 1,
                                    np.where(test_lyrics['year'] < 2010, 2,3)))

X_train = [np.max(embeddings, axis=0)
           for embeddings in train_lyrics['embedded_lyrics']]

X_test = [np.max(embeddings, axis=0)
          for embeddings in test_lyrics['embedded_lyrics']]

# %%
# Average Word Embeddings using TF-IDF weights
tfidf_feature_names = vectorizer.get_feature_names_out()

tfidf_weights_train = np.asarray(tfidf_matrix_train.mean(axis=0)).ravel().tolist()
tfidf_weights_test = np.asarray(tfidf_matrix_test.mean(axis=0)).ravel().tolist()

weights_df_train = pd.DataFrame({'weight': tfidf_weights_train}, index=tfidf_feature_names)
weights_df_test = pd.DataFrame({'weight': tfidf_weights_test}, index=tfidf_feature_names)

X_train_tfidf = [np.average(embeddings, axis=0, weights=[weights_df_train.loc[word, 'weight'] if word in weights_df_train.index else 0 for word in lyric]) 
                 for lyric, embeddings in zip(train_lyrics['lyrics'], train_lyrics['embedded_lyrics'])]
X_test_tfidf = [np.average(embeddings, axis=0, weights=[weights_df_test.loc[word, 'weight'] if word in weights_df_test.index else 0 for word in lyric])
                for lyric, embeddings in zip(test_lyrics['lyrics'], test_lyrics['embedded_lyrics'])]

# max pooling
X_train = [np.max(embeddings, axis=0)
           for embeddings in train_lyrics['embedded_lyrics']]

X_test = [np.max(embeddings, axis=0)
          for embeddings in test_lyrics['embedded_lyrics']]

# %% Grid search
# =============================================================================
# Section 3: SVM for Classification
# =============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
    0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Initialize the SVM model
svc_model = SVC()

# Perform grid search
grid_search = GridSearchCV(svc_model, param_grid, cv=5)
grid_search.fit(X_train, y_train_decades)

# Retrieve the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Predict on the test set using the best model
best_model = grid_search.best_estimator_
y_pred_decades = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test_decades, y_pred_decades)
print("SVC Accuracy:", accuracy)

# %% without grid search
# decades: 6 classes
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train_decades)

# Predict on the test set
y_pred_decades = svc_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test_decades, y_pred_decades)
print("6 classes accuracy:", accuracy)

# 4 classes
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train_4classes)

# Predict on the test set
y_pred_4classes = svc_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test_4classes, y_pred_4classes)
print("4 classes accuracy:", accuracy)

# 2 classes
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train_binary)

# Predict on the test set
y_pred_binary = svc_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print("2 classes accuracy:", accuracy)

# %% Logistic Regression with L2 Penalty
# =============================================================================
# Section 4: Logistic Regression for Classification
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
# ridge regression
ridge_model = LogisticRegression(penalty='l2', solver='liblinear')
ridge_model.fit(X_train, y_train_decades)
y_pred_ridge = ridge_model.predict(X_test)
accuracy_ridge = accuracy_score(y_test_decades, y_pred_ridge)
print("Ridge Regression Accuracy:", accuracy_ridge)

# %% Ridge Regression with L1 Penalty
# lasso
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train, y_train_decades)
y_pred_lasso = lasso_model.predict(X_test)
accuracy_lasso = accuracy_score(y_test_decades, y_pred_lasso)
print("Lasso 6 classes accuracy:", accuracy_lasso)

lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train, y_train_binary)
y_pred_lasso_binary = lasso_model.predict(X_test)
accuracy_lasso = accuracy_score(y_test_binary, y_pred_lasso_binary)
print("Lasso 2 classes Accuracy:", accuracy_lasso)

lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train, y_train_4classes)
y_pred_lasso_4classes = lasso_model.predict(X_test)
accuracy_lasso = accuracy_score(y_test_4classes, y_pred_lasso_4classes)
print("Lasso 4 classes accuracy:", accuracy_lasso)

# %% Kernel Logistic Regression
# Initialize the RBFSampler
rbf_sampler = RBFSampler(n_components=1000, random_state=42)  # todo: adjust the n_components

# Apply the RBF kernel approximation to the data
X_train_rbf = rbf_sampler.fit_transform(X_train)
X_test_rbf = rbf_sampler.transform(X_test)

# Initialize the Logistic Regression model with a kernel
logistic_model = LogisticRegression(solver='liblinear',penalty='l2')
logistic_model.fit(X_train_rbf, y_train_decades)

# Make predictions on the test set
y_pred_kernel_logistic = logistic_model.predict(X_test_rbf)
accuracy_kernel_logistic = accuracy_score(
    y_test_decades, y_pred_kernel_logistic)
print("Kernel Logistic Regression Accuracy:", accuracy_kernel_logistic)

# =============================================================================
# Section 5: Plotting the Embeddings of Lyrics
# =============================================================================
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

n_songs = len(X_train)
n_dims = len(X_train[0])
songs, dims = np.meshgrid(np.arange(n_songs), np.arange(n_dims))

Z = np.vstack(X_train).T

x1 = songs.flatten()
x2 = dims.flatten()
z = Z.flatten()

new_songs = np.linspace(songs.min(), songs.max(), 100)
new_dims = np.linspace(dims.min(), dims.max(), 100)
new_songs, new_dims = np.meshgrid(new_songs, new_dims)
new_Z = griddata((x1, x2), z, (new_songs, new_dims), method='cubic')

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(new_songs, new_dims, new_Z, cmap='coolwarm')

ax.view_init(30, 10)

fig.colorbar(surface, shrink=0.5, aspect=5)

# Add contour lines to highlight specific value ranges
cset = ax.contourf(new_songs, new_dims, new_Z, zdir='z', offset=Z.min(), cmap='coolwarm', alpha=0)
cset = ax.contourf(new_songs, new_dims, new_Z, zdir='x', offset=songs.min(), cmap='coolwarm', alpha=0)
cset = ax.contourf(new_songs, new_dims, new_Z, zdir='y', offset=dims.max(), cmap='coolwarm', alpha=0)

ax.set_xlabel('Songs')
ax.set_ylabel('Dimensions')
ax.set_zlabel('Embedding Values')
ax.set_title('3D plot of Lyrics Embeddings')

plt.show()

# %%
# =============================================================================
# Section 6: Plotting the Clustering Results of Words
# =============================================================================
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pickle
import matplotlib.pyplot as plt

# Get the word embeddings
word_vectors = model.wv.vectors

if not os.path.exists('models/cluster_model.pkl'):
    # Define the clustering model
    cluster_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    # Fit the model to the data
    cluster_model = cluster_model.fit(word_vectors)

    # Create a linkage matrix
    linkage_matrix = linkage(word_vectors, 'ward')

    with open('models/cluster_model.pkl', 'wb') as file:
        pickle.dump(cluster_model, file)

    with open('models/linkage_matrix.pkl', 'wb') as file:
        pickle.dump(linkage_matrix, file)
                      
else:
    with open('models/cluster_model.pkl', 'rb') as file:
        cluster_model = pickle.load(file)

    with open('models/linkage_matrix.pkl', 'rb') as file:
        linkage_matrix = pickle.load(file)

plt.figure(figsize=(15, 15))
dendrogram(linkage_matrix)
plt.show()
# %%