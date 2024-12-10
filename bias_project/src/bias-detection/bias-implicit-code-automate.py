# Import libraries and include progress bar

import pandas as pd
from tqdm import tqdm

# Add progress bar functionality to pandas
tqdm.pandas()

# Load saved datasets
twitter_df = pd.read_csv('complete2_twitter_data-sentiment.csv')
reddit_df = pd.read_csv('complete2_reddit_data-sentiment.csv')

# Add source columns to each DataFrame
twitter_df['source'], reddit_df['source'] = 'Twitter', 'Reddit'

# Combine both datasets to train on both
combined_data = pd.concat([twitter_df, reddit_df], ignore_index=True)

# Add empty columns to datasets
combined_data['implicit_explicit'] = ''

# Define function to determine implicit_explicit based on conditions
def determine_implicit_explicit(row):
    bias_terms = row['bias_terms'].lower()
    true_bias = row['true_bias']
    
    # Apply conditions based on flagged_bias and true_bias
    if any(term in bias_terms for term in ['lazy', 'grandma', 'grandpa', 'foreigner']):
        return 0
    elif any(term in bias_terms for term in ['poor', 'low-life', 'blue collar', "you're ugly", "you're a moron", "you're stupid", 'overqualified', 'patriarchy', 'skinhead', 'Holocaust denier', 'freeloader', 'elitist', 'privileged', 'Nazism', 'feminazi', 'bonehead', 'criminal', 'semite', 'immigrant', 'migrant', 'hoe', 'towelhead', 'slut']):
        return 2
    elif any(term in bias_terms for term in ['republican', 'democrat', 'far right', 'far left', 'party loyalist']):
        return 3
    elif any(term in bias_terms for term in ['my brand', 'brand']):
        return 4
    elif any(term in bias_terms for term in ['nigger', 'darkie', 'whore', 'bitch', 'wetback', 'white trash','thug', 'hoodlum', 'terrorist', 'ghetto', 'coon', 'spic']):
        return 1
    elif true_bias == 0 or true_bias == 2:
        return 0
    else:
        return None  

# Add progress bar to the apply function
tqdm.pandas(desc="Processing bias evaluation")

# Apply function to DataFrame with progress bar
print("Updating implicit_explicit...")
combined_data['implicit_explicit'] = combined_data.progress_apply(determine_implicit_explicit, axis=1)

# Ensure type and no NaNs
combined_data['implicit_explicit'] = combined_data['implicit_explicit'].fillna(0).astype(int)

# Next, add a id column to dataset for unique identifier
# Add a comment_id column
# Inserting the column at the
# beginning in the DataFrame
combined_data.insert(loc = 0,
          column = 'comment_id',
          value = range(1, len(combined_data) + 1))

# Save the updated data to CSV
combined_data.to_csv('updated_bias_data.csv', index=False)

# Subset to new dataframe and save to CSV
new_data = combined_data[['comment_id', 'username', 'body', 'bias_terms', 'processed_tokens', 'flagged_bias', 'has_flag', 'true_bias', 'bias_type', 'source', 'Predicted_sentiment', 'subreddit', 'subreddit_id', 'implicit_explicit']].copy()
new_data.to_csv('updated_bias_data1.csv', index=False)

print("Automated implicit_explicit update completed.")