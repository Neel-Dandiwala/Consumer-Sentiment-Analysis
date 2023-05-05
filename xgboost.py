import pandas as pd
from pandas import MultiIndex, Int16Dtype

from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# get the training datatset
training_data = pd.read_csv('data/train_file.csv')
training_data

# check dataset info
print(training_data.info())

# check for missing values
print(training_data.isna().sum())

training_data['feature'] = training_data['Headline'] + training_data['Topic']

# display dataset
print(training_data.head())

# get the stopwords and punctuation
import string
stop = stopwords.words('english')
punc = list(string.punctuation)

# remove stop words
text_clean = []
for i in range(len(training_data.feature)):
    char_clean = []
    for char in str(training_data['feature'][i]).split():
        char = char.lower()
        if char not in stop:
            char_clean.append(char)
        else:
            continue
    char_clean = ' '.join(char_clean)
    text_clean.append(char_clean)
training_data['feature'] = text_clean

# remove punctuations
text_clean = []
for i in range(len(training_data.feature)):
    char_clean = []
    for char in training_data['feature'][i]:
        char = char.lower()
        if char not in punc:
            char_clean.append(char)
        else:
            continue
    char_clean = ''.join(char_clean)
    text_clean.append(char_clean)
training_data['feature'] = text_clean

labels = list()
sentiment_headlines = training_data['SentimentHeadline']
for i in range(len(sentiment_headlines)):
    if(sentiment_headlines[i] < 0):
        labels.append(0)
    elif(sentiment_headlines[i] > 0):
        labels.append(2)
    else:
        labels.append(1)
training_data['SentimentHeadline'] = pd.DataFrame(labels)

# split training and testing data
from sklearn.model_selection import train_test_split
X = training_data.feature
y = training_data.SentimentHeadline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

import daal4py as d4p
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

model = xgb.XGBClassifier()

# fit the data
xgb_model = model.fit(X_train, y_train)

# get predictons on test data
# predictions = model.predict(X_test)

# XGBoost prediction (for accuracy comparison)
import daal4py as d4p
daal_model = d4p.get_gbt_model_from_xgboost(xgb_model.get_booster())
# Make a faster prediction with oneDAL
daal_prediction = d4p.gbt_classification_prediction(nClasses = n_classes).compute(X_test, daal_model).prediction

probabilities = daal_prediction.probabilities
print(probabilities)
