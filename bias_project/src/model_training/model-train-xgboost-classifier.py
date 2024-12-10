# Run the xgboost classifier

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Convert data to DMatrix format, which is optimized for XGBoost
dtrain = xgb.DMatrix(X_train_combined, label=y_train)
dval = xgb.DMatrix(X_val_combined, label=y_val)
dtest = xgb.DMatrix(X_test_combined, label=y_test)

# Set up parameters for XGBoost
params = {
    'objective': 'binary:logistic',  # for binary classification
    'eval_metric': 'logloss',        # Evaluation metric for binary classification
    'max_depth': 6,                  # Maximum depth of trees 
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Subsample ratio of the training instance
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'seed': 42                       # Set random seed for reproducibility
}

# Train the model with early stopping on validation set
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10, evals=evals, verbose_eval=True)

# Make predictions on the test set
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
