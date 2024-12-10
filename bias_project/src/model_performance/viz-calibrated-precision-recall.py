# RUN A CALIBRATED PRECISION-RECALL 
# (PRECISION-RECALL GAVE AP SCORE OF .30)

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Step 1: Train an XGBoost model using XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_tfidf, y_train)  # Ensure the model is fitted first

# Step 2: Calibrate the trained model using CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train_tfidf, y_train)  # Fit the calibrated model on the training data

# Step 3: Predict probabilities on the test set using the calibrated model
y_pred_prob_calibrated = calibrated_model.predict_proba(X_test_tfidf)[:, 1]

# Step 4: Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob_calibrated)

# Step 5: Compute the Average Precision (AP) score
ap_score_calibrated = average_precision_score(y_test, y_pred_prob_calibrated)

# Step 6: Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'AP = {ap_score_calibrated:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Calibrated Precision-Recall Curve')
plt.legend()
plt.show()