# CORRELATION HEATMAP

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the bias summary
plt.figure(figsize=(10,8))
bias_summary.plot(kind='bar', x='Platform', stacked=True, color=['#FF9999', '#013220','#66B2FF','#99FF99', '#AF4FA2', '#FFCC00'], alpha=0.85) 
plt.title('Bias Term Occurrences by Platform')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=0)
plt.legend(title='Bias Type')
plt.show()


# Function to extract bias terms by category
def extract_terms_by_category(text, category_terms):
    words = text.split()
    category_words = [word for word in words if word in category_terms]
    return ' '.join(category_words)

# note: use original uncleaned datasets

# Apply category filtering
stratified_twitter_df['racial_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['racial']))
stratified_twitter_df['antisemitic_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['antisemitic']))
stratified_twitter_df['sexist_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['sexist']))
stratified_twitter_df['classist_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['classist']))
stratified_twitter_df['ageism_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['ageism']))
stratified_twitter_df['bullying_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['bullying']))
stratified_twitter_df['political_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['political']))
stratified_twitter_df['brand_terms'] = stratified_twitter_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['brand']))


stratified_reddit_df['racial_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['racial']))
stratified_reddit_df['antisemitic_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['antisemitic']))
stratified_reddit_df['sexist_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['sexist']))
stratified_reddit_df['classist_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['classist']))
stratified_reddit_df['ageism_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['ageism']))
stratified_reddit_df['bullying_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['bullying']))
stratified_reddit_df['political_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['political']))
stratified_reddit_df['brand_terms'] = stratified_reddit_df['processed_tokens'].apply(lambda x: extract_terms_by_category(x, bias_terms['brand']))


# Create a combined column with all bias terms (racial, sexist, classist)
def combine_bias_terms(row):
    combined_terms = []
    if row['racial_terms']:
        combined_terms.append(row['racial_terms'])
    if row['antisemitic_terms']:
        combined_terms.append(row['antisemitic_terms'])
    if row['sexist_terms']:
        combined_terms.append(row['sexist_terms'])
    if row['classist_terms']:
        combined_terms.append(row['classist_terms'])
    if row['ageism_terms']:
        combined_terms.append(row['ageism_terms'])
    if row['bullying_terms']:
        combined_terms.append(row['bullying_terms'])
    if row['political_terms']:
        combined_terms.append(row['political_terms'])
    if row['brand_terms']:
        combined_terms.append(row['brand_terms'])
    return ' '.join(combined_terms)

# Apply this function to create a combined terms column
stratified_twitter_df['combined_terms'] = stratified_twitter_df.apply(combine_bias_terms, axis=1)
stratified_reddit_df['combined_terms'] = stratified_reddit_df.apply(combine_bias_terms, axis=1)


# Function to detect presence of bias terms in each category, handling non-string values
def detect_bias_terms(text, bias_terms):
    if isinstance(text, str):  # Ensure the input is a string
        text_lower = text.lower()
        return any(term in text_lower for term in bias_terms)
    return False  # Return False if the input is not a valid string


# Function to add binary columns for racial, sexist, bullying, classist, political, and brand term presence, handling NaNs
def add_bias_category_columns(df):
    df['racial_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['racial']))
    df['antisemitic_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['antisemitic']))
    df['sexist_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['sexist']))
    df['classist_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['classist']))
    df['ageism_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['ageism']))
    df['bullying_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['bullying']))
    df['political_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['political']))
    df['brand_present'] = df['processed_tokens'].apply(lambda x: detect_bias_terms(x, bias_terms['brand']))
    
    return df

# Apply to both datasets
stratified_twitter_df = add_bias_category_columns(stratified_twitter_df)
stratified_reddit_df = add_bias_category_columns(stratified_reddit_df)

# Save processed data
stratified_twitter_df.to_csv('complete2_twitter_data-sentiment.csv', index=False)
stratified_reddit_df.to_csv('complete2_reddit_data-sentiment.csv', index=False)

# Function to calculate correlation matrix for bias categories
def calculate_bias_correlation(df):
    # Select only the binary columns representing bias categories
    bias_columns = df[['racial_present', 'antisemitic_present', 'sexist_present', 'classist_present', 'ageism_present', 'bullying_present', 'political_present', 'brand_present']]
    
    # Calculate the correlation matrix
    correlation_matrix = bias_columns.corr()
    return correlation_matrix

# Generate correlation matrices for Twitter and Reddit datasets
twitter_correlation_matrix = calculate_bias_correlation(stratified_twitter_df)
reddit_correlation_matrix = calculate_bias_correlation(stratified_reddit_df)

# Function to plot the bias correlation heatmap
def plot_bias_correlation_heatmap(corr_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
    plt.title(title)
    plt.show()

# Plot for both Twitter and Reddit
plot_bias_correlation_heatmap(twitter_correlation_matrix, 'Twitter Bias Categories Correlation Heatmap')
plot_bias_correlation_heatmap(reddit_correlation_matrix, 'Reddit Bias Categories Correlation Heatmap')
