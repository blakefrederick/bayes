# This file is for learning purposes. It is a simple example of how to use the Naive Bayes classifier to detect spam messages.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import pickle

# Get spam/ham example data
df = pd.read_csv('https://raw.githubusercontent.com/paulwababu/datasets/main/spam.csv', encoding = 'latin-1')
print(df.head())

# Drop unwanted columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Convert spam/ham to numerical values
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

# Separate feature and target columns
X = df['v2'] 
y = df['label']

# Build the Naive Bayes classifier model #

# Initialize the CountVectorizer, which converts a collection of text documents to a matrix of token counts
cv = CountVectorizer()
# Transform text data X into a numeric format that can be used by the Naive Bayes classifier
X = cv.fit_transform(X) 
# Split the dataset into training and testing sets. 33% of the data is used for testing, and the random state is set for reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Initialize the Multinomial Naive Bayes classifier
model = MultinomialNB()
# Train the Naive Bayes classifier on the training data
model.fit(X_train,y_train)
#model.score(X_test,y_test)
# Predict the labels for the test set
y_pred = model.predict(X_test)
# Print a report showing the main classification metrics (like precision, recall, and F1-score) for the model's performance on the test set
print(classification_report(y_test, y_pred))

