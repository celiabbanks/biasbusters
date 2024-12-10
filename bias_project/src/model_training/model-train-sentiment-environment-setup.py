# Set the training environment

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack  # for combining sparse matrices

# Load saved datasets - use original ones before submitting for postgresql 
twitter2_df = pd.read_csv('complete2_twitter_data-sentiment.csv')
reddit2_df = pd.read_csv('complete2_reddit_data-sentiment.csv')

# Combine both datasets to train on both
data = pd.concat([twitter2_df, reddit2_df], ignore_index=True)

# Drop rows with NaN in 'processed_tokens' or 'true_bias' columns
data = data.dropna(subset=['processed_tokens', 'true_bias'])  

# Define features (text) and labels
X_text = data['processed_tokens']  # Text data to be transformed by TF-IDF
X_bias = data['true_bias']  # Multibinary bias feature (1=bias, 0=not bias, 2=false detection)
y = data['has_flag']  # The label indicating bias presence

# Split data into train, validation, and test sets
X_text_train, X_text_temp, X_bias_train, X_bias_temp, y_train, y_temp = train_test_split(
    X_text, X_bias, y, test_size=0.3, random_state=42)
X_text_val, X_text_test, X_bias_val, X_bias_test, y_val, y_test = train_test_split(
    X_text_temp, X_bias_temp, y_temp, test_size=0.5, random_state=42)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

# Fill NaN values in text columns with empty strings before TF-IDF transformation
X_text_train = X_text_train.fillna("")
X_text_val = X_text_val.fillna("")
X_text_test = X_text_test.fillna("")

# Convert text to TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_val_tfidf = vectorizer.transform(X_text_val)
X_test_tfidf = vectorizer.transform(X_text_test)

# Reshape X_bias columns to make them compatible for hstack
X_bias_train = X_bias_train.values.reshape(-1, 1)
X_bias_val = X_bias_val.values.reshape(-1, 1)
X_bias_test = X_bias_test.values.reshape(-1, 1)

# Combine TF-IDF text features with the `true_bias` feature
X_train_combined = hstack([X_train_tfidf, X_bias_train])
X_val_combined = hstack([X_val_tfidf, X_bias_val])
X_test_combined = hstack([X_test_tfidf, X_bias_test])

# Next: X_train_combined, X_val_combined, and X_test_combined will be used to train model
