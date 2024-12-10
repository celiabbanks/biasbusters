# Conduct a Learning Curve
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Split data into features and labels
X = data['processed_tokens']
y = data['true_bias']  # Use multibinary true_bias as the label instead of 'has_flag'

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Create an XGBoost model
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
best_model = grid_search.best_estimator_
print(f"Best parameters from grid search: {grid_search.best_params_}")

# Plot learning curve with log loss (alternative to accuracy)
train_sizes, train_scores, valid_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='neg_log_loss', n_jobs=-1)

# Plotting learning curve with log loss
train_scores_mean = -train_scores.mean(axis=1)
valid_scores_mean = -valid_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training log loss', color='blue')
plt.plot(train_sizes, valid_scores_mean, label='Validation log loss', color='red')
plt.title('Learning Curve (Log Loss)')
plt.xlabel('Training Size')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()

