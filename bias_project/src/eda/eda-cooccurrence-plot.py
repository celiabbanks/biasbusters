# CO-OCCURRENCE OF BIAS TERMS PLOT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Function to count occurrences of co-occurring bias categories
def count_bias_cooccurrence(df, bias_columns):
    cooccurrence_counts = {}
    # Generate all possible pairs of bias categories, plus an 'all' combination
    for combo in combinations(bias_columns, 2):
        cooccurrence_counts[f"{combo[0]}_and_{combo[1]}"] = (df[combo[0]] & df[combo[1]]).sum()
    # Add count for all categories co-occurring
    cooccurrence_counts['all_six'] = df[bias_columns].all(axis=1).sum()
    return cooccurrence_counts

# Specify the columns related to bias categories
bias_columns = [
    'racial_present', 'antisemitic_present', 'sexist_present', 
    'classist_present', 'ageism_present', 'bullying_present',
    'political_present', 'brand_present'
]

# Count co-occurrences in both datasets
twitter_cooccurrence = count_bias_cooccurrence(stratified_twitter_df, bias_columns)
reddit_cooccurrence = count_bias_cooccurrence(stratified_reddit_df, bias_columns)

# Convert co-occurrence counts to a combined DataFrame for both datasets
def create_combined_cooccurrence_df(twitter_data, reddit_data):
    df_twitter = pd.DataFrame(list(twitter_data.items()), columns=['Bias Combination', 'Count']).assign(Source='Twitter')
    df_reddit = pd.DataFrame(list(reddit_data.items()), columns=['Bias Combination', 'Count']).assign(Source='Reddit')
    return pd.concat([df_twitter, df_reddit])

combined_cooccurrence_df = create_combined_cooccurrence_df(twitter_cooccurrence, reddit_cooccurrence)
print(combined_cooccurrence_df)  # Optional: Check the structure

# Set the plot style globally for readability
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(12, 8))

# Visualize the co-occurrence counts
sns.barplot(
    x='Bias Combination',
    y='Count',
    hue='Source',
    data=combined_cooccurrence_df,
    palette='Set2'
)

# Set axis properties and show the plot
plt.xticks(rotation=45, ha='right')
plt.title('Bias Co-occurrence Counts on Twitter and Reddit')
plt.xlabel('Bias Combinations')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
