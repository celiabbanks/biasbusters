### has_bias
# Comparing with the corrected true_bias

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load the datasets
twitter2_df = pd.read_csv('complete2_twitter_data-sentiment.csv')
reddit2_df = pd.read_csv('complete2_reddit_data-sentiment.csv')

# Combine datasets
data = pd.concat([twitter2_df, reddit2_df], ignore_index=True)

# Drop rows with NaN in the relevant columns
data = data.dropna(subset=['processed_tokens', 'true_bias'])  

# Define features and labels
X = data['processed_tokens']  # Text data
X_bias = data['true_bias']  # Multibinary bias feature
y = data['has_flag']  # Target variable

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X_vec = vectorizer.fit_transform(X)  # Convert text to TF-IDF matrix

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)


# Define and train the XGBoost model
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": 2.76,  
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)

# Predict probabilities for the positive class (bias class)
y_prob = model.predict_proba(X_test)[:, 1]  # Class 1 (bias)

# Calculate the calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

# Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # 45-degree line
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Without X_bias Feature)')
plt.legend()
plt.grid()
plt.show()
