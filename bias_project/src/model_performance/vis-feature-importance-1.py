# Get feature importance from the final model as a dictionary
importance_dict = final_model.get_score(importance_type='gain')

# Retrieve feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Map feature indices in importance_dict to actual tokens, but only if they are within bounds
feature_importance_dict = {}
for key, importance in importance_dict.items():
    # Extract the index from the feature key, e.g., "f0" -> 0
    index = int(key[1:])
    # Only add the feature if the index is within bounds of the feature_names array
    if index < len(feature_names):
        feature_importance_dict[feature_names[index]] = importance

# Sort the dictionary by importance score in descending order for easier reading
feature_importance_dict_sorted = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

# Print the feature importance with actual tokens
# print("Feature Importance with Tokens:")
for token, importance in feature_importance_dict_sorted.items():
    print(f"{token}: {importance}")