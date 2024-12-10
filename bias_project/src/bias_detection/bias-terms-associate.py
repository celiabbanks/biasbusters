from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

# Initialize tqdm for pandas
tqdm.pandas()

# Repeat the bias terms list for this code block
# Allows for separate code run later after preprocessing
# List of bias terms and phrases

bias_terms = {
    'racial': [
        "nigger", "thug", "hoodlum", "spic", "chinc", "coon", "wetback", "black women are hostile", "they take our jobs",
        "Haitian immigrants eat pets", "affirmative action is reverse discrimination", "darkie", "migrant", 
        "you’re here because of DEI", "I won’t get in an elevator alone with a black person", "angry black woman", 
        "blacks have to be twice as good", "illegal alien", "terrorist", "towelhead", "they don't belong here", 
        "they are too lazy to work", "they are good at sports, bad at school", "immigrants are rapists and criminals"
    ],
    "antisemitic": [ "jews own the media", "jews run America", "jew bitch", "stingy as a jew", "Holocaust denier", 
                    "Nazism", "skinhead", "Semite"
    ],
    'sexist': [
        "female dike", "female butch", "bitch", "bitch slut", "whore",  "bitches and hoes", "women do not belong in the workplace", 
        "women should not be sports casters", "they are only good for bearing children", "feminazi", "slut", "gold digger", "women are weak", 
        "a woman's role is in the home", "patriarchy", "women are too emotional", "men are stronger than women", "women should not have careers" 
        "I will not vote for a woman for US president", "men don't cry", "only men should become a US president", "she is a weak executive and grins alot"
    ],
    'classist': [
        "lazy", "freeloader", "criminal", "hoe", "ghetto", "white trash", "you did not go to an ivy league school", 
        "welfare queen", "elitist", "blue-collar", "low-life", "privileged", "self-made vs. born with a silver spoon",
        "poor people don't work hard enough", "rich people are greedy"

    ],
    'ageism':[
        "old fart", "overqualified", "past her prime", "past your prime", "go play with your grandchildren", "grandma", "grandpa",
        "sleepy joe", "cannot keep up", "old people are a drain on society", "you are no longer relevant", "she's ancient", 
        "you're ancient", "over the hill"
    ],
    'political':[
        "republican", "conservative party", "democrat", "liberal party", "green party", "the squad", "far right", "far left", "project 2025",
        "president trump", "president biden", "vice president harris", "vice president jd vance", "obama", "jill biden", "melania trump",  
        "antifa", "klu klux klan", "NRA", "guns and abortion"
    ],
    'brand':[
        "love this brand", "my brand", "prefer this brand", "birkin bag", "expensive taste", "favorite team", "favorite movie", "favorite apparel", "favorite food"
    ],
    'bullying':[
        "scaredy cat", "can't cut it", "dare you", "spit on you", "you should kill yourself", "go hide in a corner",  
        "not popular enough", "you don't fit in", "shame on you", "you're ugly", "too fat", "too skinny", "put a bag over your head",
        "hide your face", "cover your titties", "hide yourself", "body not meant for those clothes", "you can't do anything right", "bonehead",
        "you can't handle it", "you're stupid", "you're a moron", "she's big as an apartment complex", "she can afford to miss a meal"

    ]
}

# Define predict_labels function
def predict_labels(data, feature_col):
    """
    Predict labels for the given feature column in the data.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        feature_col (str): The column in the DataFrame for which to predict labels.

    Returns:
        pd.Series: A series of predicted labels for the given feature column.
    """
    le = LabelEncoder()
    labels = le.fit_transform(data[feature_col].fillna("unknown"))
    return pd.Series(labels, index=data.index, name=f"Predicted_{feature_col}")

# Load saved datasets
stratified_twitter_df = pd.read_csv('complete_twitter_data-sentiment.csv')
stratified_reddit_df = pd.read_csv('complete_reddit_data-sentiment.csv')

# Step 1: Define 'true_bias' column based on conditions for 'has_flag' and 'has_bias'
stratified_twitter_df['has_flag'] = 0  # Default
stratified_reddit_df['has_flag'] = 0  # Default

# Step 2: Define 'has_flag' column by checking if 'flagged_bias' contains any terms.
stratified_twitter_df['has_flag'] = stratified_twitter_df['flagged_bias'].apply(
    lambda x: 1 if isinstance(ast.literal_eval(x), list) and len(ast.literal_eval(x)) > 0 else 0)
stratified_reddit_df['has_flag'] = stratified_reddit_df['flagged_bias'].apply(
    lambda x: 1 if isinstance(ast.literal_eval(x), list) and len(ast.literal_eval(x)) > 0 else 0)

# Step 3: Define 'true_bias' column based on conditions for 'has_flag' and 'has_bias'
stratified_twitter_df['true_bias'] = 0  # Default
stratified_reddit_df['true_bias'] = 0  # Default

stratified_twitter_df.loc[
    (stratified_twitter_df['has_flag'] == 1) & (stratified_twitter_df['has_bias'] == 1), 'true_bias'] = 1
stratified_reddit_df.loc[
    (stratified_reddit_df['has_flag'] == 1) & (stratified_reddit_df['has_bias'] == 1), 'true_bias'] = 1

stratified_twitter_df.loc[
    (stratified_twitter_df['has_flag'] == 0) & (stratified_twitter_df['has_bias'] == 1), 'true_bias'] = 2
stratified_reddit_df.loc[
    (stratified_reddit_df['has_flag'] == 0) & (stratified_reddit_df['has_bias'] == 1), 'true_bias'] = 2

# Function to assign bias type(s) based on the text
def assign_bias_type(text, bias_terms):
    if not isinstance(text, str):
        return None
    found_biases = []
    for bias_type, terms in bias_terms.items():
        for term in terms:
            if term.lower() in text.lower():
                found_biases.append(bias_type)
                break
    return ','.join(found_biases) if found_biases else None

# Apply the function to the 'processed_tokens' column with a progress bar
stratified_twitter_df['bias_type'] = stratified_twitter_df['processed_tokens'].progress_apply(
    lambda x: assign_bias_type(x, bias_terms))
stratified_reddit_df['bias_type'] = stratified_reddit_df['processed_tokens'].progress_apply(
    lambda x: assign_bias_type(x, bias_terms))

# Apply predict_labels on processed_tokens for sentiment prediction
stratified_twitter_df['predicted_sentiment'] = predict_labels(stratified_twitter_df, 'processed_tokens')
stratified_reddit_df['predicted_sentiment'] = predict_labels(stratified_reddit_df, 'processed_tokens')

# Save processed data
stratified_twitter_df.to_csv('complete2_twitter_data-sentiment.csv', index=False)
stratified_reddit_df.to_csv('complete2_reddit_data-sentiment.csv', index=False)

# Function to find bias term occurrences
def count_bias_terms(text, bias_dict):
    if not isinstance(text, str):
        return {key: 0 for key in bias_dict}
    term_counts = {key: 0 for key in bias_dict}
    for category, terms in bias_dict.items():
        for term in terms:
            if term.lower() in text.lower():
                term_counts[category] += 1
    return term_counts

# Apply the function to Twitter and Reddit data with a progress bar
stratified_twitter_df['bias_term_counts'] = stratified_twitter_df['processed_tokens'].progress_apply(
    lambda x: count_bias_terms(x, bias_terms))
stratified_reddit_df['bias_term_counts'] = stratified_reddit_df['processed_tokens'].progress_apply(
    lambda x: count_bias_terms(x, bias_terms))

# Extract counts for each category
def extract_bias_counts(df):
    bias_categories = ['racial', 'antisemitic', 'sexist', 'classist', 'ageism', 'bullying', 'political', 'brand']
    bias_counts = {}
    for category in bias_categories:
        count = df['bias_term_counts'].apply(lambda x: x[category]).sum()
        if count > 0:
            bias_counts[category] = count
    return bias_counts

# Extract bias counts for Twitter and Reddit
twitter_bias_counts = extract_bias_counts(stratified_twitter_df)
reddit_bias_counts = extract_bias_counts(stratified_reddit_df)

# Create a dictionary for summary
summary_data = {'Platform': ['Twitter', 'Reddit']}
for category in twitter_bias_counts.keys():
    twitter_count = twitter_bias_counts.get(category, 0)
    reddit_count = reddit_bias_counts.get(category, 0)
    if twitter_count > 0 or reddit_count > 0:
        summary_data[category.capitalize() + ' Bias'] = [twitter_count, reddit_count]

# Create a DataFrame for visualizing
bias_summary = pd.DataFrame(summary_data)

print(bias_summary)
