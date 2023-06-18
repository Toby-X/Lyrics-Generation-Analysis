# Lyrics-Generation-Analysis

## Project Structure

`preprocess.ipynb` is the code for preprocessing.

`lr_svm_xgb.R` is the code for Cross Validation and Boosting L1 penalty logistic regression, svm and xgboost.

`ETM.R` is the code for Embedded Topic Models.

In `embedded_classification.py` we first trained word embedding models then use SVM and logistic regression to perform 2,4 and 6 classes classification.

In `L1logit.py` we generated word cloud via logistic regression based on a word count embedding.

`deep learning` directory contains all code for deep learning methods.

In `deep learning/Bertopic.py` we used a integrated framework called BERTopic to analysis the trends of topics hidden in lyrics.

In `deep learning/BERT4classification.py` we used Bert to directly perform classification.

In `deep learning/neural_classifier.py` we trained a neural network to perform lyrics classification and use the outputs of the hidden layer to perform clustering analysis.

In `deep learning/VAE.py` we experimented with a Variational Auto-encoder to learn the features of lyrics in a low-dimensional latent space.
