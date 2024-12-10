# Confusion matrix of true and false bias

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Predict the class labels (not probabilities) on the test set
y_pred = calibrated_model.predict(X_test_tfidf)

# Step 2: Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 3: Display confusion matrix as a heatmap
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot(cmap='Blues', values_format='d', xticks_rotation='horizontal')
plt.title('Confusion Matrix')
plt.show()