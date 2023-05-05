import pandas as pd
from sklearn.metrics import accuracy_score

# Importing oneAPI libraries
from daal4py.sklearn.feature_extraction.text import CountVectorizer
from daal4py.sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Read the train and test datasets
train_data = pd.read_csv("train_file.csv")
test_data = pd.read_csv("test_file.csv")

import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

train_data['processed_title'] = train_data['Title'].apply(preprocess_text)
test_data['processed_title'] = test_data['Title'].apply(preprocess_text)

train_data['processed_headline'] = train_data['Headline'].apply(preprocess_text)
test_data['processed_headline'] = test_data['Headline'].apply(preprocess_text)

# Concatenate the processed title and headline into a single feature
train_data['text'] = train_data['processed_title'] + ' ' + train_data['processed_headline']
test_data['text'] = test_data['processed_title'] + ' ' + test_data['processed_headline']

# Using oneAPI CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

train_data['sentiment_title_label'] = train_data['SentimentTitle'].apply(lambda x: 1 if x >= 0 else 0)
train_data['sentiment_headline_label'] = train_data['SentimentHeadline'].apply(lambda x: 1 if x >= 0 else 0)

# Combined the sentiment labels for title and headline into a single label
train_data['sentiment_label'] = train_data.apply(lambda x: x['sentiment_title_label'] if x['sentiment_title_label'] != 0 else x['sentiment_headline_label'], axis=1)

# Using oneAPI MultinomialNB
model = MultinomialNB()
model.fit(X_train, train_data['sentiment_label'])

test_data['predicted_sentiment_label'] = model.predict(X_test)

# Splited the training data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, train_data['sentiment_label'], test_size=0.2, random_state=42)

# Train the model on the training set
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the sentiment labels for the validation set
y_pred = model.predict(X_val)

# The accuracy score of the model on the validation set
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)
Accuracy: 0.7454187896665773

test_data['predicted_sentiment_score'] = test_data['predicted_sentiment_label'].apply(lambda x: 1 if x == 1 else -1)
submission_data = test_data[['IDLink', 'predicted_sentiment_score']]
submission_data.to_csv('combined_submission1.csv', index=False)

