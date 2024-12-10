# Plot the top 20 features from XGBoost run

import matplotlib.pyplot as plt
import xgboost as xgb

# Prepare data for the custom plot
sorted_features = list(feature_importance_dict_sorted.items())[:20]
features, importance_scores = zip(*sorted_features)

# Create the figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left subplot: XGBoost plot (F score) - default plot_importance
xgb.plot_importance(final_model, importance_type='gain', max_num_features=20, ax=axes[0])
axes[0].set_title('XGBoost Top 20 Feature Importance (F Score)')

# Right subplot: Custom bar plot showing feature names and gain scores
axes[1].barh(features, importance_scores, color='teal')
axes[1].set_xlabel('Gain')
axes[1].set_title('Top 20 Named Features by Gain')
axes[1].invert_yaxis()  # Dsplay the most important features on top

# Adjust the layout to avoid overlap
plt.tight_layout()
plt.show()
