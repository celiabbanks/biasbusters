import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Convert the combined feature set and labels into a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_combined, label=y_train)

# Set up parameters for XGBoost
params = {
    'objective': 'binary:logistic',  # Binary classification objective
    'eval_metric': 'logloss',        # Logarithmic loss for binary classification
    'max_depth': 6,                  # Depth of trees (tune as needed)
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Fraction of samples used per tree
    'colsample_bytree': 0.8,         # Fraction of features used per tree
    'seed': 42                       # Seed for reproducibility
}

# Perform 5-fold cross-validation
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=5,                         # 5-fold cross-validation
    early_stopping_rounds=10,        # Set to stop if no improvement for 10 rounds
    metrics="logloss",
    as_pandas=True,
    seed=42
)

# Display cross-validation results
print("Cross-Validation Results:")
print(cv_results)

# Get the best number of boosting rounds based on cross-validation
best_num_boost_round = cv_results['test-logloss-mean'].idxmin() + 1
print(f"Best number of boosting rounds: {best_num_boost_round}")

# Train final model using the optimal number of boosting rounds
final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

# Make predictions on the test set and evaluate
dtest = xgb.DMatrix(X_test_combined)
y_pred_prob = final_model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))