# word embedding + svm/logistic regression
# %%
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem

import pandas as pd

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

# word2vec embedding
all_lyrics = test_lyrics['lyrics'].tolist() + train_lyrics['lyrics'].tolist()
model = Word2Vec(all_lyrics, min_count=1)

# add a new column to the data matrix
test_lyrics['embedded_lyrics'] = test_lyrics['lyrics'].apply(
    lambda x: [model.wv[word] for word in x])
train_lyrics['embedded_lyrics'] = train_lyrics['lyrics'].apply(
    lambda x: [model.wv[word] for word in x])

# Map years to decade categories
y_train_decades = np.where(train_lyrics['year'] < 1960, '1960s',
                           np.where(train_lyrics['year'] > 2020, '2010s',
                                    (train_lyrics['year'] // 10 * 10).astype(str) + 's'))

y_test_decades = np.where(test_lyrics['year'] < 1960, '1960s',
                          np.where(test_lyrics['year'] > 2020, '2010s',
                                   (test_lyrics['year'] // 10 * 10).astype(str) + 's'))


# %%
# taking average of all the word vectors in a song
X_train = [np.mean(embeddings, axis=0)
           for embeddings in train_lyrics['embedded_lyrics']]

X_test = [np.mean(embeddings, axis=0)
          for embeddings in test_lyrics['embedded_lyrics']]

# %%
# another method: max pooling
X_train = [np.max(embeddings, axis=0)
           for embeddings in train_lyrics['embedded_lyrics']]

X_test = [np.max(embeddings, axis=0)
          for embeddings in test_lyrics['embedded_lyrics']]

# %%
# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
    0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Initialize the SVM model
model = SVC()

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5)
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


# %%
# elastic net
elasticnet_model = LogisticRegression(
    penalty='elasticnet', solver='saga', l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train_decades)
y_pred_elasticnet = elasticnet_model.predict(X_test)
accuracy_elasticnet = accuracy_score(y_test_decades, y_pred_elasticnet)
print("Elastic Net Accuracy:", accuracy_elasticnet)

# %%
# ridge regression
ridge_model = LogisticRegression(penalty='l2', solver='liblinear')
ridge_model.fit(X_train, y_train_decades)
y_pred_ridge = ridge_model.predict(X_test)
accuracy_ridge = accuracy_score(y_test_decades, y_pred_ridge)
print("Ridge Regression Accuracy:", accuracy_ridge)

# %%
# lasso
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train, y_train_decades)
y_pred_lasso = lasso_model.predict(X_test)
accuracy_lasso = accuracy_score(y_test_decades, y_pred_lasso)
print("Lasso Accuracy:", accuracy_lasso)

# %%
# Prepare the data
X_train = [np.max(embeddings, axis=0)
           for embeddings in train_lyrics['embedded_lyrics']]
y_train = train_lyrics['year'].astype(str)

X_test = [np.max(embeddings, axis=0)
          for embeddings in test_lyrics['embedded_lyrics']]
y_test = test_lyrics['year'].astype(str)

# Map years to decade categories
y_train_decades = np.where(train_lyrics['year'] < 1960, '1960s',
                           np.where(train_lyrics['year'] > 2020, '2010s',
                                    (train_lyrics['year'] // 10 * 10).astype(str) + 's'))

y_test_decades = np.where(test_lyrics['year'] < 1960, '1960s',
                          np.where(test_lyrics['year'] > 2020, '2010s',
                                   (test_lyrics['year'] // 10 * 10).astype(str) + 's'))

# Initialize the RBFSampler
rbf_sampler = RBFSampler(n_components=100, random_state=42)

# Apply the RBF kernel approximation to the data
X_train_rbf = rbf_sampler.fit_transform(X_train)
X_test_rbf = rbf_sampler.transform(X_test)

# Initialize the Logistic Regression model with a kernel
logistic_model = LogisticRegression(solver='saga', random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.1, 1, 10],
    'l1_ratio': [0, 0.5, 1]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
grid_search.fit(X_train_rbf, y_train_decades)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Fit the final model with the best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_rbf, y_train_decades)

# Make predictions on the test set
y_pred_kernel_logistic = final_model.predict(X_test_rbf)
accuracy_kernel_logistic = accuracy_score(
    y_test_decades, y_pred_kernel_logistic)
print("Kernel Logistic Regression Accuracy:", accuracy_kernel_logistic)

# %%
