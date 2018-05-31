## Quora Question Pairs Kaggle Challenge
We've tried different machine learning approaches to take a stab at this Kaggle challenge. The idea was to compare their effectiveness and also understand why a certain approach works (or doesn't). 
 - Feed forward neural network trained on features derived from sentence vectors for both questions.
 - Siamese convolutional neural network trained on the word vectors of both questions.
 - Using TF-IDF scores as weights
 - Using different sources for word vectors (Wikipedia, google news & training data set) to see if one representation is better than the other ones.

#### Datasets
 - Training datasets can be downloaded from https://www.kaggle.com/c/quora-question-pairs/data
 - Wikipedia word vectors can be downloaded from https://github.com/idio/wiki2vec/
 - Google News word vectors can be downloaded from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

#### Installation & setup
We used the following third party libraries - gensim, nltk, scipy, sklearn. We also used editdistance package (to compute levenshtein distance), fuzzywuzzy package(to compute partial ratio).
They can be installed using :
```
sudo pip install gensim
sudo pip install scipy
sudo pip install sklearn
sudo pip install editdistance
sudo pip install fuzzywuzzy
```

#### How to run the code
```
python main.py classify neural_network data/train_40k_qn_pairs_features.csv data/test_qn_pairs_features.csv latest_predictions-2.csv
```
