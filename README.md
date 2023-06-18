# Lyrics-Generation-Analysis

## Project Structure

In `preprocess.ipynb` we preprocess the data and transform original data into word frequency table.

In `lr_svm_xgb.R` we try out Cross Validation and Boosting $L_1$ penalty logistic regression, svm and xgboost.

In `ETM.R` we perform embedded topic models over the lyrics.

In `embedded_classification.py` we first trained word embedding models then use SVM and logistic regression to perform 2,4 and 6 classes classification.

In `L1logit.py` we generated word cloud via logistic regression based on a word count embedding.

`deep learning` directory contains all code for deep learning methods.

In `deep learning/Bertopic.py` we used a integrated framework called BERTopic to analysis the trends of topics hidden in lyrics.

In `deep learning/BERT4classification.py` we used Bert to directly perform classification.

In `deep learning/neural_classifier.py` we trained a neural network to perform lyrics classification and use the outputs of the hidden layer to perform clustering analysis.

In `deep learning/VAE.py` we experimented with a Variational Auto-encoder to learn the features of lyrics in a low-dimensional latent space.

## Original Data

`data/billboard_lyrics_genres.csv` is the original data scraped with genere.

`data/active_years.csv` is the data of active years for each song's artist.

`data/train_data.csv` and `data/test_data.csv` are the data of MaxAbsScaling after transformed to word frequency table, respectively for training set and test set.

`data/train_data.csv` and `data/test_data.csv` are the data of TF-IDF and transformed to word frequency table, respectively for training set and test set.

`data/delete_word_list.txt` is the list of words that are deleted by artificial screening.