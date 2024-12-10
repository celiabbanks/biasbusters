# Try side-by-side plotting

# Generate the calibration curve without bias feature
y_prob_without_bias = model.predict_proba(X_test)[:, 1]
prob_true_without_bias, prob_pred_without_bias = calibration_curve(y_test, y_prob_without_bias, n_bins=10)

# Plot both calibration curves
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_without_bias, prob_true_without_bias, marker='o', label='Calibration Curve (has_bias)')
plt.plot(prob_pred_with_bias, prob_true_with_bias, marker='o', label='Calibration Curve (With true_bias)')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # 45-degree line
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Comparison of Calibration Curves')
plt.legend()
plt.grid()
plt.show()