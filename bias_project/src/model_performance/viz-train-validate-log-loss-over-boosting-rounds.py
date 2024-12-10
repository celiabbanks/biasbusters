import matplotlib.pyplot as plt

# Plot training and validation log loss
plt.figure(figsize=(10, 6))
plt.plot(cv_results['train-logloss-mean'], label='Training Log Loss')
plt.plot(cv_results['test-logloss-mean'], label='Validation Log Loss')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss Over Boosting Rounds')
plt.legend()
plt.grid(True)
plt.show()