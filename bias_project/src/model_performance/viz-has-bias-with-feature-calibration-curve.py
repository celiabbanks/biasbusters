# Append X_bias as a feature 

from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Split data 
X_train, X_test, y_train, y_test, X_bias_train, X_bias_test = train_test_split(
    X_vec, y, X_bias, test_size=0.2, random_state=42
)

# Convert the pandas Series to numpy arrays and reshape
X_bias_train = X_bias_train.values.reshape(-1, 1)  
X_bias_test = X_bias_test.values.reshape(-1, 1)

# Combine the bias feature with the text data for both training and testing
X_train_with_bias = hstack([X_train, X_bias_train])  # Reshape bias feature to add
X_test_with_bias = hstack([X_test, X_bias_test])

# Re-train the model with X_bias as a feature
model_with_bias = XGBClassifier(**params)
model_with_bias.fit(X_train_with_bias, y_train)

# Predict probabilities
y_prob_with_bias = model_with_bias.predict_proba(X_test_with_bias)[:, 1]

# Generate the calibration curve
prob_true_with_bias, prob_pred_with_bias = calibration_curve(y_test, y_prob_with_bias, n_bins=10)

# Plot the calibration curve with bias feature
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_with_bias, prob_true_with_bias, marker='o', label='Calibration Curve with X_bias')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (With X_bias Feature)')
plt.legend()
plt.grid()
plt.show()

