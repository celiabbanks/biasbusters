#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MILESTONE II TEAM 6 C. BANKS - USING ML ALGORITHMS TO IDENTIFY BIAS IN SOCIAL NETWORKS


# CAPSTONE TEAM 21 C. BANKS - MITIGATING IMPLICIT AND EXPLICIT BIAS ON SOCIAL NETWORK ENVIRONMENTS


# ############################################
# # DATA PREPROCESSING 
# ############################################

# ########### THESE LIBRARIES MAY REQUIRE INSTALLATION ################

# In[1]:


# !pip install --force-reinstall numpy nltk spacy
# !pip install nltk
# !pip install spacy==3.5.3
# Install spaCy model within Jupyter Notebook
# !python -m spacy download en_core_web_sm
# !pip install numpy==1.24.3
# !pip uninstall -y numpy pandas scipy
# !pip install numpy==1.24.3 pandas==1.5.3 scipy==1.10.1
# spacy.load("en_core_web_sm")
# !pip install textblob
# import nltk
# nltk.download('vader_lexicon')


# ########### PASSWORD INFO ################

# In[ ]:


from dotenv import load_dotenv
load_dotenv()


# ########### SENTIMENT IN PHRASES OF BIAS TERMS IDENTIFICATION ################

# In[ ]:


# YIELDS BETTER FALSE POSITIVES IDENTIFICATION
# YIELDS BETTER PERFORMANCE SCORE
# VERSUS SIMILARITY OF WORDS 

import nltk
nltk.download('vader_lexicon')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

import spacy
from nltk.stem import WordNetLemmatizer
from spacy.matcher import PhraseMatcher
import pandas as pd
from tqdm import tqdm


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Extra language support for wordnet
nltk.download('all')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download necessary nltk data
nltk.download('vader_lexicon')

# Initialize lemmatizer, stopwords, and sentiment analyzer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

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

# list of safe phrases
safe_phrases = [
    "love my grandparents", "my adorable grandparents", "grandma's cooking is delicious",
    "My grandma makes the best apple pie", "Grandma shared stories from her youth",
    "Grandma's advice is always so helpful", "My grandma loves gardening",
    "My grandpa used to be a teacher", "Spending time with my grandparents is the best",
    "I am a foreigner to learning technology", "let's pig out and enjoy this ice cream",
    "you're so skinny oh how I want to be too", "we are better than that is not a classist statement",
    "having wealth can make you more wealth", "your behavior is ugly and needs improvement",
    "The older generation has a lot to teach us", "Young people today have so many new opportunities",  
    "he is a successful foreigner who has made America his home", 
    "I am too lazy to run for the U.S. Congress", "I feel lazy today", 
    "Lazy Saturday mornings are the best", 
    "George Washington Carver was a prominent Black scientist of the early 20th century with his sustainable farming techniques", 
    "African American hip hop artists are some of the wealthiest people in the U.S."
]

# Function to check safe context
def is_safe_context(text, flagged_phrase):
    for safe_phrase in safe_phrases:
        if flagged_phrase in safe_phrase and safe_phrase in text:
            return True  # Safe context found
    return False  # No safe context

# Flatten bias terms into a single list of strings, ensure all elements are strings
flat_bias_terms = [str(term) for sublist in bias_terms.values() for term in sublist]

# Simple detection of bias in the data for purposes of stratifying data
def has_bias_terms(text, bias_terms):
    """
    Check if the text contains any bias terms.

    Args:
        text (str): The text to analyze.
        bias_terms (dict): A dictionary of bias categories and their associated terms.

    Returns:
        list: A list of bias terms found in the text.
    """
    if not isinstance(text, str):
        return []
    
    found_bias_terms = []
    for category, terms in bias_terms.items():
        for term in terms:
            # Check if term is in the text as a whole word
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                found_bias_terms.append(term)
    
    return found_bias_terms

# Function to replace phrases with underscore-joined tokens
def replace_phrases(text, bias_terms):
    # Ensure text is a string
    if isinstance(text, list):
        text = ' '.join(map(str, text))  # Join list items into a single string

    # Process each term in bias_terms, which should now only contain strings
    for phrase in bias_terms:
        # Ensure phrase is a string (in case there was any unexpected non-string)
        if not isinstance(phrase, str):
            print(f"Skipping non-string bias term: {phrase}")
            continue

        # Replace spaces with underscores in the phrase
        joined_phrase = phrase.replace(" ", "_")
        
        # Substitute the phrase in the text
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', joined_phrase, text)
    
    return text

# Initialize the PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)

# Convert bias terms into spaCy patterns
patterns = [nlp.make_doc(term) for term in flat_bias_terms]
phrase_matcher.add("BIAS_TERMS", patterns)

# Preprocessing function with sentiment scoring
def preprocess_text(text):
    # Handle missing or non-string values
    if not isinstance(text, str):
        return "", []

    # Convert text to lowercase and replace phrases with underscores
    text = text.lower()
    text = replace_phrases(text, flat_bias_terms)

    # Use spaCy to parse the standardized text
    doc = nlp(text)
    
    # Tokenize and lemmatize with nltk
    tokens = [lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words]
    
    # Find flagged phrases using PhraseMatcher
    flagged_tokens = []
    matches = phrase_matcher(doc)  # Use the phrase_matcher
    for match_id, start, end in matches:
        matched_span = doc[start:end].text  # Get matched phrase

        # Check if the flagged phrase is in a safe context or has positive sentiment
        if not is_safe_context(text, matched_span):  # Check safe context
            
            # Perform sentiment analysis on the sentence containing the flagged term
            sentence = doc[start:end].sent  # Get the sentence containing the flagged term
            blob = TextBlob(sentence.text)
            sentiment_score = blob.sentiment.polarity  # Polarity ranges from -1 to 1
            
            # Only flag if sentiment is negative or neutral (indicating potential bias)
            if sentiment_score <= 0:  # Negative or neutral sentiment
                flagged_tokens.append(matched_span)
                print(f"Flagged for potential bias (negative/neutral sentiment): '{matched_span}' in text: {sentence.text}")
    
    return tokens, flagged_tokens  # Return tokenized text and flagged tokens

# Preprocessing function with sentiment scoring
def preprocess_text(text):
    
    # Handle missing or non-string values
    if not isinstance(text, str):
        return "", []

    # Convert text to lowercase and replace phrases with underscores
    text = text.lower()
    text = replace_phrases(text, flat_bias_terms)  # Use flat_bias_terms as list

    # Use spaCy to parse the standardized text
    doc = nlp(text)
    
    # Tokenize and lemmatize with nltk
    tokens = [lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words]
    
    # Find flagged phrases using PhraseMatcher
    flagged_tokens = []
    matches = phrase_matcher(doc)
    for match_id, start, end in matches:
        matched_span = doc[start:end].text  # Get matched phrase

        # Check if the flagged phrase is in a safe context or has positive sentiment
        if not is_safe_context(text, matched_span):  # Check safe context
            
            # Perform sentiment analysis on the sentence containing the flagged term
            sentence = doc[start:end].sent  # Get the sentence containing the flagged term
            blob = TextBlob(sentence.text)
            sentiment_score = blob.sentiment.polarity  # Polarity ranges from -1 to 1
            
            # Only flag if sentiment is negative or neutral (indicating potential bias)
            if sentiment_score <= 0:  # Negative or neutral sentiment
                flagged_tokens.append(matched_span)
                print(f"Flagged for potential bias (negative/neutral sentiment): '{matched_span}' in text: {sentence.text}")
    
    return tokens, flagged_tokens  # Return tokenized text and flagged tokens


# Function to stratify dataset based on bias presence
def stratify_dataset(df, bias_terms, ratio=1.5):
    df['bias_terms'] = df['body'].apply(lambda x: has_bias_terms(x, bias_terms))
    df['has_bias'] = df['bias_terms'].apply(lambda terms: 1 if terms else 0)

    bias_df = df[df['has_bias'] == 1]
    non_bias_df = df[df['has_bias'] == 0]
    non_bias_sample = non_bias_df.sample(n=int(len(bias_df) * ratio), random_state=42)
    
    return pd.concat([bias_df, non_bias_sample]).reset_index(drop=True)

# Load datasets
twitter_df = pd.read_csv('twitter1Mtweets_2009.csv', encoding='utf-8', encoding_errors='ignore')
reddit_df = pd.read_csv('reddit_combined_file.csv', encoding='utf-8', encoding_errors='ignore')

# Fill NaN values with empty strings in the 'body' column
twitter_df['body'] = twitter_df['body'].fillna("")
reddit_df['body'] = reddit_df['body'].fillna("")

# Stratify the datasets
stratified_twitter_df = stratify_dataset(twitter_df, bias_terms)
stratified_reddit_df = stratify_dataset(reddit_df, bias_terms)

# Preprocess and flag bias in batches for Twitter data
batch_size = 50
twitter_results = []
for doc in nlp.pipe(stratified_twitter_df['body'], batch_size=batch_size):
    tokens, flagged_tokens = preprocess_text(doc.text)
    twitter_results.append((tokens, flagged_tokens))

# Add results to the Twitter DataFrame
stratified_twitter_df['processed_tokens'] = [result[0] for result in twitter_results]
stratified_twitter_df['flagged_bias'] = [result[1] for result in twitter_results]

# Preprocess and flag bias in batches for Reddit data
reddit_results = []
for doc in nlp.pipe(stratified_reddit_df['body'], batch_size=batch_size):
    tokens, flagged_tokens = preprocess_text(doc.text)
    reddit_results.append((tokens, flagged_tokens))

# Add results to the Reddit DataFrame
stratified_reddit_df['processed_tokens'] = [result[0] for result in reddit_results]
stratified_reddit_df['flagged_bias'] = [result[1] for result in reddit_results]

# Save processed data
stratified_twitter_df.to_csv('complete_twitter_data-sentiment.csv', index=False)
stratified_reddit_df.to_csv('complete_reddit_data-sentiment.csv', index=False)

# Check the output
print("Twitter DataFrame Sample:\n", stratified_twitter_df[['body', 'processed_tokens', 'flagged_bias']].head())
print("Reddit DataFrame Sample:\n", stratified_reddit_df[['body', 'processed_tokens', 'flagged_bias']].head())

###################


# ################### ASSOCIATE BIAS TERMS AND PHRASES ###################

# ############## ADD PREDICTED LABELS FOR MITIGATION RESPONSE ##############

# In[ ]:


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
    return pd.Series(labels, index=data.index, name=f"predicted_{feature_col}")

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


# ################### AUTOMATED PROCESS FOR ADDING BIAS_CODE AND IMPLICIT_EXPLICIT CODING ###################

# In[ ]:


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
new_data = combined_data[['comment_id', 'username', 'body', 'bias_terms', 'processed_tokens', 'flagged_bias', 'has_flag', 'true_bias', 'bias_type', 'source', 'predicted_sentiment', 'subreddit', 'subreddit_id', 'implicit_explicit']].copy()
new_data.to_csv('updated_bias_data1.csv', index=False)

print("Automated implicit_explicit update completed.")


# ########################################
# # EDA - VISUALIZATIONS
# ########################################

# In[ ]:


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


# In[ ]:


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


# In[ ]:


# WORD CLOUD OF BIAS TERMS 

import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Repeat the bias terms list for this code block

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

# Simplify the term extraction by using set operations
def extract_bias_terms(text, bias_dict):
    if not isinstance(text, str):
        return ''
    words = set(text.lower().split())
    bias_words = set()
    for terms in bias_dict.values():
        bias_words.update(words.intersection(map(str.lower, terms)))
    return ' '.join(bias_words)

# Add source columns to each DataFrame
stratified_twitter_df['source'], stratified_reddit_df['source'] = 'Twitter', 'Reddit'

########################################################################################
# This section for vectorizing...

stratified_twitter_df['cleaned_body'] = stratified_twitter_df['processed_tokens'].fillna('')
stratified_reddit_df['cleaned_body'] = stratified_reddit_df['processed_tokens'].fillna('')

# Reconvert tokenized text into a single string for each entry (to allow for bias term search later on)
stratified_twitter_df['cleaned_body_str'] = stratified_twitter_df['cleaned_body'].apply(lambda x: ' '.join(x))
stratified_reddit_df['cleaned_body_str'] = stratified_reddit_df['cleaned_body'].apply(lambda x: ' '.join(x))

# Save updates to file for other visualizations ...
stratified_twitter_df.to_csv('complete_twitter_data-sentiment.csv', index=False)
stratified_reddit_df.to_csv('complete_reddit_data-sentiment.csv', index=False)


########################################################################################

# Combine non-empty bias terms from both dataframes
combined_bias_terms = pd.concat([stratified_twitter_df['bias_terms'], stratified_reddit_df['bias_terms']]).loc[lambda x: x != '']

print(combined_bias_terms.head())
print(f'Number of combined bias terms: {len(combined_bias_terms)}')

# Vectorize bias terms for clustering
vectorizer = TfidfVectorizer(max_features=1000)
X_bias = vectorizer.fit_transform(combined_bias_terms)

# Apply KMeans clustering
kmeans_bias = KMeans(n_clusters=5, random_state=42)
kmeans_bias.fit(X_bias)

# Print top bias terms per cluster
def print_top_bias_terms_per_cluster(kmeans, vectorizer, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    for i, order in enumerate(kmeans.cluster_centers_.argsort()[:, ::-1]):
        top_terms = ", ".join(terms[ind] for ind in order[:n_terms])
        print(f"Cluster {i}: {top_terms}\n")

print_top_bias_terms_per_cluster(kmeans_bias, vectorizer)

# Function to generate and save word clouds for clusters
def plot_wordcloud_for_clusters(kmeans, vectorizer, n_clusters=5, n_top_terms=15, save_dir='wordclouds'):
    os.makedirs(save_dir, exist_ok=True)
    terms = vectorizer.get_feature_names_out()
    for i, order in enumerate(kmeans.cluster_centers_.argsort()[:, ::-1]):
        wordcloud_text = ' '.join(terms[ind] for ind in order[:n_top_terms])
        wordcloud = WordCloud(background_color='white', colormap='Set2').generate(wordcloud_text)
        
        plt.figure(figsize=(3, 3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Cluster {i} - Top Bias Terms")
        
        # Save and display the word cloud
        plt.savefig(os.path.join(save_dir, f'wordcloud_cluster_{i}.png'))
        plt.show()

# Generate word clouds for each cluster
plot_wordcloud_for_clusters(kmeans_bias, vectorizer)


# ##################################################
# # TRAIN SENTIMENT MODEL 
# 
# ##################################################

# #################### WITH HAS_FLAG LABEL, TRUE_BIAS FEATURE ####################

# In[ ]:


# Set the training environment

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack  # for combining sparse matrices

# Load saved datasets - use original ones before submitting for postgresql 
twitter2_df = pd.read_csv('complete2_twitter_data-sentiment.csv')
reddit2_df = pd.read_csv('complete2_reddit_data-sentiment.csv')

# Combine both datasets to train on both
data = pd.concat([twitter2_df, reddit2_df], ignore_index=True)

# Drop rows with NaN in 'processed_tokens' or 'true_bias' columns
data = data.dropna(subset=['processed_tokens', 'true_bias'])  

# Define features (text) and labels
X_text = data['processed_tokens']  # Text data to be transformed by TF-IDF
X_bias = data['true_bias']  # Multibinary bias feature (1=bias, 0=not bias, 2=false detection)
y = data['has_flag']  # The label indicating bias presence

# Split data into train, validation, and test sets
X_text_train, X_text_temp, X_bias_train, X_bias_temp, y_train, y_temp = train_test_split(
    X_text, X_bias, y, test_size=0.3, random_state=42)
X_text_val, X_text_test, X_bias_val, X_bias_test, y_val, y_test = train_test_split(
    X_text_temp, X_bias_temp, y_temp, test_size=0.5, random_state=42)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

# Fill NaN values in text columns with empty strings before TF-IDF transformation
X_text_train = X_text_train.fillna("")
X_text_val = X_text_val.fillna("")
X_text_test = X_text_test.fillna("")

# Convert text to TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_val_tfidf = vectorizer.transform(X_text_val)
X_test_tfidf = vectorizer.transform(X_text_test)

# Reshape X_bias columns to make them compatible for hstack
X_bias_train = X_bias_train.values.reshape(-1, 1)
X_bias_val = X_bias_val.values.reshape(-1, 1)
X_bias_test = X_bias_test.values.reshape(-1, 1)

# Combine TF-IDF text features with the `true_bias` feature
X_train_combined = hstack([X_train_tfidf, X_bias_train])
X_val_combined = hstack([X_val_tfidf, X_bias_val])
X_test_combined = hstack([X_test_tfidf, X_bias_test])

# Next: X_train_combined, X_val_combined, and X_test_combined will be used to train model


# In[ ]:


# Run the xgboost classifier

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Convert data to DMatrix format, which is optimized for XGBoost
dtrain = xgb.DMatrix(X_train_combined, label=y_train)
dval = xgb.DMatrix(X_val_combined, label=y_val)
dtest = xgb.DMatrix(X_test_combined, label=y_test)

# Set up parameters for XGBoost
params = {
    'objective': 'binary:logistic',  # for binary classification
    'eval_metric': 'logloss',        # Evaluation metric for binary classification
    'max_depth': 6,                  # Maximum depth of trees 
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Subsample ratio of the training instance
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'seed': 42                       # Set random seed for reproducibility
}

# Train the model with early stopping on validation set
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10, evals=evals, verbose_eval=True)

# Make predictions on the test set
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# ###################################################################################################
# 
# # VALIDATE XGBOOST MODEL WITH CROSS-VALIDATION
# 
# ###################################################################################################
# 

# In[ ]:


import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Convert the combined feature set and labels into a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_combined, label=y_train)

# Set up parameters for XGBoost
params = {
    'objective': 'binary:logistic',  # Binary classification objective
    'eval_metric': 'logloss',        # Logarithmic loss for binary classification
    'max_depth': 6,                  # Depth of trees (tune as needed)
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Fraction of samples used per tree
    'colsample_bytree': 0.8,         # Fraction of features used per tree
    'seed': 42                       # Seed for reproducibility
}

# Perform 5-fold cross-validation
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=5,                         # 5-fold cross-validation
    early_stopping_rounds=10,        # Set to stop if no improvement for 10 rounds
    metrics="logloss",
    as_pandas=True,
    seed=42
)

# Display cross-validation results
print("Cross-Validation Results:")
print(cv_results)

# Get the best number of boosting rounds based on cross-validation
best_num_boost_round = cv_results['test-logloss-mean'].idxmin() + 1
print(f"Best number of boosting rounds: {best_num_boost_round}")

# Train final model using the optimal number of boosting rounds
final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

# Make predictions on the test set and evaluate
dtest = xgb.DMatrix(X_test_combined)
y_pred_prob = final_model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# ######################################################################
# # MODEL PERFORMANCE VISUALIZATIONS
# ######################################################################

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


## LDA for top 10 biased phrases

# Visualization of LDA Topic Top 10 Biased Words Distributions

import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to display the LDA topic distributions with vertical labels
def plot_lda_topic_distribution_with_labels(lda_model, topic_distributions, feature_names, title):
    num_topics = topic_distributions.shape[1]
    avg_topic_distribution = topic_distributions.mean(axis=0)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the bars
    bars = plt.bar(range(num_topics), avg_topic_distribution, color='blue', alpha=0.6)
    
    # Add vertical labels with topic numbers and top words
    for i, bar in enumerate(bars):
        height = bar.get_height()
        top_words = " ".join([feature_names[j] for j in lda_model.components_[i].argsort()[:-6 - 1:-1]])
        
        # Annotate the bar with the topic number and top words, rotated vertically
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'Topic {i+1}\n{top_words}', 
                 ha='center', va='bottom', rotation=90, fontsize=10, color='black')
    
    # Remove the x-axis labels and ticks
    plt.xticks([])
    
    # Add y-axis label
    plt.ylabel('Average Topic Proportion')
    
    # Move the title to the bottom of the chart
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin to make more space for the title
    plt.suptitle(title, fontsize=16, y=0.02)  # Title positioned even lower

    plt.tight_layout()
    plt.show()

# Updated LDA Topic Modeling Function with Visualization
def lda_topic_modeling_with_vertical_labels(twitter2_df, reddit2_df):
    # Vectorize Twitter and Reddit text data
    tfidf_twitter = TfidfVectorizer(max_features=1000)
    tfidf_reddit = TfidfVectorizer(max_features=1000)

    twitter_tfidf = tfidf_twitter.fit_transform(twitter2_df['processed_tokens'])
    reddit_tfidf = tfidf_reddit.fit_transform(reddit2_df['processed_tokens'])

    # LDA on Twitter data
    lda_twitter = LatentDirichletAllocation(n_components=10, random_state=42)
    twitter_topics = lda_twitter.fit_transform(twitter_tfidf)

    # LDA on Reddit data
    lda_reddit = LatentDirichletAllocation(n_components=10, random_state=42)
    reddit_topics = lda_reddit.fit_transform(reddit_tfidf)

    # Plot Twitter Topic Distribution with Vertical Labels
    plot_lda_topic_distribution_with_labels(lda_twitter, twitter_topics, 
                                            tfidf_twitter.get_feature_names_out(), 
                                            'LDA Topic Top 10 Biases Distribution for Twitter')

    # Plot Reddit Topic Distribution with Vertical Labels
    plot_lda_topic_distribution_with_labels(lda_reddit, reddit_topics, 
                                            tfidf_reddit.get_feature_names_out(), 
                                            'LDA Topic Top 10 Biases Distribution for Reddit')

lda_topic_modeling_with_vertical_labels(twitter2_df, reddit2_df)


# In[ ]:


# RUN A CALIBRATED PRECISION-RECALL 
# (PRECISION-RECALL GAVE AP SCORE OF .30)

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Step 1: Train an XGBoost model using XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_tfidf, y_train)  # Ensure the model is fitted first

# Step 2: Calibrate the trained model using CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train_tfidf, y_train)  # Fit the calibrated model on the training data

# Step 3: Predict probabilities on the test set using the calibrated model
y_pred_prob_calibrated = calibrated_model.predict_proba(X_test_tfidf)[:, 1]

# Step 4: Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob_calibrated)

# Step 5: Compute the Average Precision (AP) score
ap_score_calibrated = average_precision_score(y_test, y_pred_prob_calibrated)

# Step 6: Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'AP = {ap_score_calibrated:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Calibrated Precision-Recall Curve')
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Predict probabilities on the test set using the calibrated model
y_pred_prob_calibrated = calibrated_model.predict_proba(X_test_tfidf)[:, 1]  # Probabilities for class 1

# Step 2: Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_calibrated)
roc_auc = roc_auc_score(y_test, y_pred_prob_calibrated)

# Step 3: Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line (no discrimination)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:


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


# In[ ]:


# Conduct a Learning Curve
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Split data into features and labels
X = data['processed_tokens']
y = data['true_bias']  # Use multibinary true_bias as the label instead of 'has_flag'

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Create an XGBoost model
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
best_model = grid_search.best_estimator_
print(f"Best parameters from grid search: {grid_search.best_params_}")

# Plot learning curve with log loss (alternative to accuracy)
train_sizes, train_scores, valid_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='neg_log_loss', n_jobs=-1)

# Plotting learning curve with log loss
train_scores_mean = -train_scores.mean(axis=1)
valid_scores_mean = -valid_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training log loss', color='blue')
plt.plot(train_sizes, valid_scores_mean, label='Validation log loss', color='red')
plt.title('Learning Curve (Log Loss)')
plt.xlabel('Training Size')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()



# ################### HAS_BIAS VS TRUE_BIAS CALIBRATION CURVES ###################

# In[ ]:


### has_bias
# Comparing with the corrected true_bias

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load the datasets
twitter2_df = pd.read_csv('complete2_twitter_data-sentiment.csv')
reddit2_df = pd.read_csv('complete2_reddit_data-sentiment.csv')

# Combine datasets
data = pd.concat([twitter2_df, reddit2_df], ignore_index=True)

# Drop rows with NaN in the relevant columns
data = data.dropna(subset=['processed_tokens', 'true_bias'])  

# Define features and labels
X = data['processed_tokens']  # Text data
X_bias = data['true_bias']  # Multibinary bias feature
y = data['has_flag']  # Target variable

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X_vec = vectorizer.fit_transform(X)  # Convert text to TF-IDF matrix

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)


# Define and train the XGBoost model
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": 2.76,  
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)

# Predict probabilities for the positive class (bias class)
y_prob = model.predict_proba(X_test)[:, 1]  # Class 1 (bias)

# Calculate the calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

# Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # 45-degree line
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Without X_bias Feature)')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


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


# In[ ]:


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


# ####################################################################################################
# 
# # CAPSTONE PORTION -- MITIGATING THE DETECTED BIAS
# 
# ####################################################################################################
# 

# ##############################
# DATABASE SECTION
# ##############################

# ############### SET UP POSTGRESQL DATABASE FOR TRACKING BIAS INSTANCES ###############

# #################### INTEGRATE CHATGPT TO GENERATE RESPONSES ####################

# In[ ]:


# TEMPLATES SETUP
# RUN BEFORE DATABASE CREATION AND API INTEGRATION

# Templates for implicit bias (educational)
implicit_templates = {
    "bullying": [
        "Let's aim for a positive conversation. Could you rephrase this without hurtful language?",
        "It's important to avoid language that could be interpreted as bullying. Please consider rephrasing."
    ],
    "racial": [
        "This comment might be interpreted as racially biased. How about framing it more inclusively?",
        "Let's try to avoid racially charged language. Could you express your point differently?"
    ],
    "sexist": [
        "Your comment seems to carry gendered bias. Consider using neutral language.",
        "Language free of gender bias helps foster inclusivity. Please rephrase this."
    ],
    "ageism": [
        "Your comment seems to carry ageism bias. Consider using neutral language.",
        "Language free of ageism bias helps foster inclusivity. Please rephrase this."
    ],
    "classist": [
        "Your comment seems to carry classist bias. Consider using neutral language.",
        "Language free of classist bias helps foster inclusivity. Please rephrase this."
    ],
}


# Templates for explicit bias (warnings and guidelines)
explicit_templates = {
    "general": [
        "Your comment violates our community guidelines. Please refrain from using such language.",
        "This comment has been flagged for explicit bias. Continuing this behavior may lead to action."
    ]
}


# Selection Function

import random

def select_template(bias_type, category=None):
    if bias_type == 0:  # Implicit Bias
        if category and category in implicit_templates:
            return random.choice(implicit_templates[category])
        else:
            return "This comment appears biased. Please consider rephrasing."
    elif bias_type == 1:  # Explicit Bias
        return random.choice(explicit_templates["general"])
    else:
        return "This comment could not be processed. Please contact support."


# ChatGPT Prompt Generation

def generate_prompt(body, bias_type, category):
    """
    Generate a prompt for ChatGPT based on bias type and category.
    """
    if bias_type == 0:  # Implicit bias
        template = implicit_templates.get(category, "JenAI says this comment may be biased: '{}'")
    elif bias_type == 1:  # Explicit bias
        template = explicit_templates.get(category, "JenAI says this comment is biased: '{}'")
    else:
        template = "Evaluate this comment for bias: '{}'"

    # Fill the template with the flagged comment
    prompt = template.format(body)
    return prompt


# ############ INTERACT WITH POSTGRESQL ON SINGULARITY-DOCKER CONTAINER #########

# In[ ]:


# Create the comments_data table

import os
import psycopg2
from psycopg2 import sql

# load API key
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
conn = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")


csv_file_path = 'updated_bias_data1.csv'

try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Create table
    create_table_sql = """

    CREATE TABLE IF NOT EXISTS public.comments_data (
        comments_id BIGSERIAL PRIMARY KEY,
        username VARCHAR(150),
        body VARCHAR(10000),
        bias_terms VARCHAR(4000),
        processed_tokens VARCHAR(20000),
        flagged_bias VARCHAR(4000),
        has_flag BIGINT,
        true_bias BIGINT,
        bias_type VARCHAR(4000),
        source1 VARCHAR(25),
        predicted_sentiment integer,
        subreddit VARCHAR(150),
        subreddit_id VARCHAR(25),
        implicit_explicit BIGINT
    );

    ALTER TABLE IF EXISTS public.comments_data
    OWNER to postgres;

    """
    cursor.execute(create_table_sql)
    print("Table created successfully.")

except Exception as e:
    print("An error occurred:", e)

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()


# In[ ]:


# Create the Responses Table

import psycopg2
from psycopg2 import sql
from psycopg2 import sql, OperationalError, Error

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")


# Create the Responses table

create_table_sql = """

DROP TABLE IF EXISTS public.Responses;

CREATE TABLE IF NOT EXISTS public.Responses(
    response_id SERIAL PRIMARY KEY,
    comments_id INT NOT NULL,
    JenAI_responds TEXT NOT NULL,
    bias_type VARCHAR(50),  -- 'implicit' or 'explicit'
    template_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (comments_id) REFERENCES comments_data (comments_id)
);


ALTER TABLE IF EXISTS public.Responses
    OWNER to postgres;
"""

# Connect to the database and execute the SQL commands
try:
    # Establish connection
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True  # Enable autocommit for DDL statements
    cursor = conn.cursor()

    # Execute CREATE TABLE command
    cursor.execute(create_table_sql)
    print("Table created successfully.")

except Exception as e:
    print("An error occurred:", e)

finally:
    # Close the connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()


# In[ ]:


# Add data from csv into comments_data table

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import psycopg2
from psycopg2 import sql, OperationalError, Error

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

# Path to your CSV file
csv_file_path = 'updated_bias_data1.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Connect to the database
try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Prepare the INSERT statement
    insert_query = """
    INSERT INTO public.comments_data (
        comments_id, username, body, bias_terms, processed_tokens, flagged_bias,
        has_flag, true_bias, bias_type, 
        source1, predicted_sentiment, subreddit,
        subreddit_id, implicit_explicit 
    ) VALUES %s
    """

    # Convert the DataFrame to a list of tuples
    records = df.values.tolist()

    # Insert data into the table using execute_values
    execute_values(cursor, insert_query, records)
    
    # Commit the transaction
    conn.commit()
    print("Data uploaded successfully.")
except Exception as e:
    print("Error:", e)
finally:
    if conn:
        cursor.close()
        conn.close()


# ############### CUSTOMIZE CHATGPT'S RESPONSE #######################

# ############### RESPONSE SETUP AND EVALUATION #######################

# In[ ]:


# TO ADDRESS TIMEOUT ISSUES WITH API INTEGRATION, ADD PROCESSED COLUMN TO IDENTIFY RECORDS PROCESSED
# SO THAT YOU CAN PICK UP WITH NON-PROCESSED ONES IF SYSTEM TIMEOUTS, FREEZES, ETC.

# ADD PROCESSED COLUMN TO COMMENTS_DATA TABLE
import psycopg2
from psycopg2 import sql, OperationalError, Error
from openai import OpenAI

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
conn = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Rename the column
    cur.execute("""
        ALTER TABLE public.comments_data
        ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE;
    """)
    conn.commit()  # Commit the change
    print("Column 'processed' successfully added to 'comments_data'.")

except psycopg2.Error as e:
    print(f"Error occurred: {e}")

finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()


# In[ ]:


# Modify process_comments function to process only records where processed = FALSE
import psycopg2
from psycopg2 import sql, OperationalError, Error
from openai import OpenAI

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Ensure the API key is loaded properly
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")


# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
conn = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

# OpenAI client initialization
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  

# Safe phrases
safe_phrases = [
    "love my grandparents", "my adorable grandparents", "grandma's cooking is delicious",
    "My grandma makes the best apple pie", "Grandma shared stories from her youth",
    "Grandma's advice is always so helpful", "My grandma loves gardening",
    "My grandpa used to be a teacher", "Spending time with my grandparents is the best",
    "I am a foreigner to learning technology", "let's pig out and enjoy this ice cream",
    "you're so skinny oh how I want to be too", "we are better than that is not a classist statement",
    "having wealth can make you more wealth", "your behavior is ugly and needs improvement",
    "The older generation has a lot to teach us", "Young people today have so many new opportunities",
    "he is a successful foreigner who has made America his home",
    "I am too lazy to run for the U.S. Congress", "I feel lazy today",
    "Lazy Saturday mornings are the best",
    "George Washington Carver was a prominent Black scientist of the early 20th century with his sustainable farming techniques",
    "African American hip hop artists are some of the wealthiest people in the U.S."
]

# Response templates
templates = {
    "implicit_bias": {
        "general": "The comment may have implicit bias: \"{comment}\". A more neutral approach could be: {suggestion}.",
        "racial": "The comment contains potential implicit racial bias: \"{comment}\". Consider rephrasing to be more inclusive, e.g., {suggestion}.",
        "sexist": "The comment contains potential implicit gender bias: \"{comment}\". A more equitable phrasing might be: {suggestion}.",
    },
    "explicit_bias": {
        "general": "The comment: \"{comment}\" shows explicit bias. To clarify intent, consider this alternative: {suggestion}.",
        "bullying": "The comment: \"{comment}\" is an example of explicit bullying. To foster a positive dialogue, you might say: {suggestion}.",
    },
    "political_bias": "The comment may contain political bias: \"{comment}\". A more neutral approach could be: {suggestion}.",
    "brand_bias": "The comment may contain brand bias: \"{comment}\". A more neutral approach could be: {suggestion}."
}


# Function to generate suggestion from safe phrases
def generate_suggestion(comment, sentiment):
    for phrase in safe_phrases:
        if phrase.lower() in comment.lower():
            return f"Consider focusing on a positive aspect, such as: '{phrase}'"
    
    # Incorporate sentiment into suggestions
    if sentiment == "negative":
        return "The tone of the comment seems negative. Consider rephrasing to express positivity or neutrality."
    elif sentiment == "neutral":
        return "The tone is neutral. Ensure clarity and avoid any unintended bias."
    elif sentiment == "positive":
        return "The tone is positive. Maintain this perspective while ensuring inclusivity."
    
    return "Rephrase to remove any potential bias."

# Function to generate a response using templates and ChatGPT
def generate_response(comment_data):
    comment = comment_data['body']
    implicit_explicit = comment_data['implicit_explicit']
    bias_type = comment_data['bias_type']
    sentiment = comment_data.get('predicted_sentiment', 'neutral')  # Default to neutral if not present

    # Select the appropriate template
    if implicit_explicit == 1:  # Explicit bias
        template = templates['explicit_bias'].get(bias_type, templates['explicit_bias']['general'])
    elif implicit_explicit == 2:  # Implicit bias
        template = templates['implicit_bias'].get(bias_type, templates['implicit_bias']['general'])
    else:
        template = templates.get(bias_type, templates['implicit_bias']['general'])

    # Generate suggestion
    suggestion = generate_suggestion(comment, sentiment)

    # Populate the template
    JenAI_responds = template.format(comment=comment, suggestion=suggestion)

    # Use ChatGPT to enhance the response
    try:
        system_message = (
            "You are JenAI, a friendly and approachable assistant who helps with identifying bias. "
            "Respond in a friendly, professional, and empathetic tone, ensuring clarity and warmth."
        )

        user_message = f"JenAI advises: {JenAI_responds}"

        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return gpt_response.choices[0].message.content
    except Exception as e:
        print(f"Error generating GPT response for comment: {comment}")
        print(f"OpenAI error: {e}")
        return JenAI_responds  # Fallback to template response

# Process flagged comments and insert responses in batches
def process_comments(batch_size=1000):
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Fetch unprocessed flagged comments
        cur.execute("""
            SELECT comments_id, body, implicit_explicit, bias_type, predicted_sentiment
            FROM comments_data
            WHERE implicit_explicit > 0 AND processed = FALSE
        """)
        rows = cur.fetchall()
        
        total_records = len(rows)
        print(f"Total records to process: {total_records}")

        # Process records in batches
        for start in range(0, total_records, batch_size):
            batch = rows[start:start + batch_size]
            print(f"Processing batch {start // batch_size + 1}: Records {start + 1} to {min(start + batch_size, total_records)}")

            for row in batch:
                comment_data = {
                    "comments_id": row[0],
                    "body": row[1],
                    "implicit_explicit": row[2],
                    "bias_type": row[3],
                    "predicted_sentiment": row[4]
                }

                # Generate response
                response = generate_response(comment_data)

                # Insert response into the responses table
                cur.execute(
                    "INSERT INTO responses (comments_id, JenAI_responds) VALUES (%s, %s)",
                    (comment_data['comments_id'], response)
                )

                # Update the processed flag
                cur.execute(
                    "UPDATE comments_data SET processed = TRUE WHERE comments_id = %s",
                    (comment_data['comments_id'],)
                )

                conn.commit()
                print(f"Response inserted and processed flag updated for comment_id: {comment_data['comments_id']}")

            print(f"Batch {start // batch_size + 1} completed.")

    except (OperationalError, Error) as e:
        print(f"Database error: {e}")
        raise

    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    try:
        process_comments(batch_size=1000)  # Set batch size as needed
    except Exception as e:
        print(f"Execution stopped due to error: {e}")


# #################### VIEW CHATGPT MITIGATION RESPONSE ####################

# In[ ]:


# JUST SOURCE, USER COMMENT, AND JENAI RESPONDS COLUMNS...

import psycopg2
from psycopg2 import sql, OperationalError, Error
from openai import OpenAI

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
conn = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

# Create a cursor object
cur = conn.cursor()

# Execute the SELECT query
cur.execute("""
    SELECT 
        cd.source1 AS source, 
        CASE 
            WHEN cd.source1 = 'Twitter' THEN cd.username 
            WHEN cd.source1 = 'Reddit' THEN cd.subreddit_id 
            ELSE NULL 
        END AS identifier,
        cd.comments_id,
        cd.bias_type,
        cd.body,
        r.JenAI_responds 
    FROM 
        comments_data cd 
    JOIN 
        responses r 
    ON 
        cd.comments_id = r.comments_id 
        
    WHERE cd.source1 = 'Reddit' AND cd.bias_type = 'racial'

    LIMIT 2;
""")



# Fetch the results
rows = cur.fetchall()

# Print the results
for row in rows:
   print(row)

cur.close()
conn.close()


# In[ ]:


# Optional routine to save combined data table as pandas df and csv file
import pandas as pd
from psycopg2 import sql, OperationalError, Error
from openai import OpenAI

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Ensure the API key is loaded properly
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

def fetch_bias_data():
    try:
        # Connect to the database
        conn = psycopg2.connect(**db_config)
        query = """
            SELECT 
                cd.comments_id,
                cd.bias_type, 
                cd.bias_terms,
                cd.body AS user_comment,
                cd.source1 AS source, 
                cd.username, 
                cd.subreddit_id, 
                cd.predicted_sentiment,
                cd.processed,
                r.JenAI_responds 
            FROM 
                comments_data cd 
            JOIN 
                responses r 
            ON 
                cd.comments_id = r.comments_id 
            LIMIT 25;
        """
        # Execute query and load into DataFrame
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# Fetch and display the data
bias_data = fetch_bias_data()
# print(bias_data.head())


# Save DataFrame to a CSV file
csv_filename = "bias_mitigation_results.csv"
bias_data.to_csv(csv_filename, index=False)

# Display the DataFrame and confirm save
print(f"DataFrame saved to {csv_filename}")
print(bias_data.head())


# In[4]:


# Close database session
cur.close()
conn.close()


# ############################## MITIGATION ANALYSIS SECTION ##############################

# ############### SET UP MITIGATION EVALUATION VISUALIZATIONS ###############

# In[ ]:


# Join data for analysis
# Fetch data and process into pandas dataframe 

import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
conn = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")


# Create a cursor object
cur = conn.cursor()

try:

    # SQL Query to join tables
    sql_query = """
        SELECT
            cd.comments_id,
            cd.processed,
            cd.predicted_sentiment,
            r.JenAI_responds
        FROM
            comments_data cd
        LEFT JOIN
            responses r
        ON
            cd.comments_id = r.comments_id;
    """
    cur.execute(sql_query)

    # Fetch all results
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=columns)
    print("Data fetched successfully.")

except Exception as e:
    print(f"Error fetching data: {e}")

finally:
    if 'conn' in locals() and conn:
        cur.close()
        conn.close()


# In[ ]:


# Add a column to evaluate if mitigation response was
# successfully associated with comment

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
conn = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")


# Create a cursor object
cur = conn.cursor()

# Batch process 

def update_mitigation_status(batch_size=1000):
    try:
        # Connect to the database
        # conn = psycopg2.connect(**db_config)
        # cur = conn.cursor()

        # Step 1: Check if 'mitigation_successful' column exists
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'comments_data' AND column_name = 'mitigation_successful';
        """)
        if not cur.fetchone():
            print("Adding 'mitigation_successful' column...")
            cur.execute("ALTER TABLE comments_data ADD COLUMN mitigation_successful BOOLEAN;")
            conn.commit()
            print("'mitigation_successful' column added successfully.")

        # Step 2: Fetch all rows to process
        cur.execute("""
            SELECT comments_id, processed, predicted_sentiment
            FROM comments_data
        """)
        rows = cur.fetchall()

        total_records = len(rows)
        print(f"Total records to process: {total_records}")

        # Step 3: Batch processing with progress bar
        with tqdm(total=total_records, desc="Processing comments") as pbar:
            for start in range(0, total_records, batch_size):
                batch = rows[start:start + batch_size]

                for row in batch:
                    comments_id, processed, predicted_sentiment = row

                    # Mitigation success logic
                    mitigation_successful = processed and predicted_sentiment >= 0

                    # Update the table
                    cur.execute("""
                        UPDATE comments_data
                        SET mitigation_successful = %s
                        WHERE comments_id = %s
                    """, (mitigation_successful, comments_id))

                conn.commit()  # Commit after each batch
                pbar.update(len(batch))  # Update progress bar

        print("Batch processing completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()  # Roll back in case of error

    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

# Run the update process
update_mitigation_status()


# In[ ]:


# Conduct mitigation bias analysis

import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

def analyze_mitigation():
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # 1. Summary of Processed Records
        cur.execute("""
            SELECT processed, COUNT(*) AS count
            FROM comments_data
            GROUP BY processed;
        """)
        processed_summary = cur.fetchall()
        processed_df = pd.DataFrame(processed_summary, columns=['processed', 'count'])
        print("Processed Summary:")
        print(processed_df)

        # 2. Bias and Sentiment Analysis
        cur.execute("""
            SELECT 
                predicted_sentiment,
                COUNT(*) AS count,
                AVG(CASE WHEN processed THEN predicted_sentiment ELSE NULL END) AS avg_processed_sentiment,
                AVG(CASE WHEN NOT processed THEN predicted_sentiment ELSE NULL END) AS avg_unprocessed_sentiment
            FROM comments_data
            GROUP BY predicted_sentiment;
        """)
        sentiment_analysis = cur.fetchall()
        sentiment_df = pd.DataFrame(sentiment_analysis, columns=[
            'predicted_sentiment', 'count', 'avg_processed_sentiment', 'avg_unprocessed_sentiment'
        ])
        print("Sentiment Analysis:")
        print(sentiment_df)

        # 3. Successful Mitigations (if a success column exists or success criteria is calculated)
        cur.execute("""
            SELECT 
                COUNT(*) AS successful_count, 
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS success_rate 
            FROM comments_data
            WHERE mitigation_successful = TRUE;
        """)
        success_rate = cur.fetchall()
        success_df = pd.DataFrame(success_rate, columns=['successful_count', 'success_rate'])
        print("Mitigation Success Rates:")
        print(success_df)

        # Close the cursor and connection
        cur.close()
        conn.close()

        # Visualization
        # visualize_data(processed_df, sentiment_df, success_df)

        # Return the DataFrames to access outside the function
        return processed_df, sentiment_df, success_df
    
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as ex:
        print(f"Error: {ex}")
        return None, None, None  # Return None if there's an error

processed_df, sentiment_df, success_df = analyze_mitigation()

def visualize_data(processed_df):
    # 1. Bar Chart: Processed Records
    processed_df['processed'] = processed_df['processed'].map({True: 'Processed', False: 'Unprocessed'})
    plt.figure(figsize=(8, 6))
    processed_df.plot(kind='bar', x='processed', y='count', color=['blue', 'orange'], legend=False)
    plt.title('Processed vs Unprocessed Records')
    plt.xlabel('Processing Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

visualize_data(processed_df)
    
if __name__ == "__main__":
    analyze_mitigation()


# In[ ]:


# Various plots
import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

def fetch_data_from_db():
    try:
        # Establishing the connection to the database
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # SQL query to join comments_data and responses, considering only valid mitigations
        cur.execute("""
            SELECT 
                c.bias_type, 
                c.true_bias, 
                c.username, 
                c.source1, 
                c.subreddit_id,
                c.predicted_sentiment,
                c.implicit_explicit,
                r.JenAI_responds
            FROM comments_data c
            JOIN responses r ON c.comments_id = r.comments_id
            WHERE c.true_bias = 1;
        """)
        
        # Fetching the data from the query result
        data = cur.fetchall()

        # Closing the cursor and connection
        cur.close()
        conn.close()

        # Converting to DataFrame
        df = pd.DataFrame(data, columns=['bias_type', 'true_bias', 'username', 'source1', 'subreddit_id', 'predicted_sentiment', 'implicit_explicit','JenAI_responds'])

        return df
    
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None

def visualize_bias_type_distribution(df):
    # Count the number of mitigation responses for each bias type
    # Step 1: Filter for valid mitigation responses
    valid_responses_df = df[df['true_bias'] == 1]
    
    # Step 2: Filter for the specific bias types and order them
    desired_bias_types = ['sexist', 'classist', 'political', 'racial', 'bullying', 'ageism']
    filtered_df = valid_responses_df[valid_responses_df['bias_type'].isin(desired_bias_types)]
    
    # Step 3: Count the occurrences of each bias type in the specified order
    bias_type_counts = filtered_df['bias_type'].value_counts()
    
    # Step 4: Plot the distribution
    plt.figure(figsize=(10, 6))
    bias_type_counts.plot(kind='bar', color='skyblue')
    
    # Step 5: Customize the plot
    plt.title('Bias Type Distribution of Valid Mitigation Responses')
    plt.xlabel('Bias Type')
    plt.ylabel('Count of Mitigation Responses')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Step 5: Pie of Implicit Explicit Mitigation Responses
    implicit_explicit_counts = df['implicit_explicit'].value_counts()
    
    # Define the labels for the pie chart
    labels = {1: 'Explicit', 2: 'Implicit', 3: 'False Bias'}
    
    # Plot pie chart with descriptive labels
    implicit_explicit_counts.plot(kind='pie', 
                                  autopct='%1.1f%%', 
                                  figsize=(8, 6), 
                                  labels=[labels.get(x, x) for x in implicit_explicit_counts.index])  # Map to descriptive labels
    
    plt.title('Distribution of Implicit vs Explicit Bias in Mitigation Responses')
    plt.ylabel('')
    plt.show()


    ####################

    # Define bins and labels for sentiment categories
    bins = [-1, 10000, 20000, 30000, 40000, 50000]
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    
    # Create a new column for aggregated sentiment categories
    df['sentiment_category'] = pd.cut(df['predicted_sentiment'], bins=bins, labels=labels, right=False)
    
    # Check the distribution of sentiment categories
    print(df['sentiment_category'].value_counts())
    
    # Step 1: Filter the DataFrame to include only the specified bias_types
    selected_bias_types = ["sexist", "classist", "political", "racial", "bullying", "ageism"]
    
    # Filter the dataset to only include the specified bias_types
    filtered_df = df[df['bias_type'].isin(selected_bias_types)]
    
    # Step 2: Create the boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=filtered_df,  # Use the filtered data
        x='bias_type', 
        y='predicted_sentiment', 
        hue='sentiment_category',  # Sentiment categories as hue
        palette="Set2",  # Color palette
    )
    
    plt.title("Bias Type Distribution with Predicted Sentiment Categories")
    plt.xlabel("Bias Type")
    plt.ylabel("Predicted Sentiment")
    plt.xticks(rotation=45)
    
    # Adjusting the legend placement and font size
    plt.legend(title='Sentiment Category', loc='upper right', fontsize=10)
    
    plt.show()

    
    ####################
    # Define bins and labels for sentiment categories
    bins = [-1, 10000, 20000, 30000, 40000, 50000]
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    
    # Create a new column for aggregated sentiment categories
    df['sentiment_category'] = pd.cut(df['predicted_sentiment'], bins=bins, labels=labels, right=False)
    
    # Check the distribution of sentiment categories
    print(df['sentiment_category'].value_counts())
    
    # Visualize the distribution using a bar plot
    plt.figure(figsize=(10, 6))
#    sns.countplot(data=df, x='sentiment_category', palette="pastel")
    sns.countplot(data=df, x='sentiment_category', color="grey")
    plt.title("Distribution of Aggregated Sentiment Categories")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

############################

# Fetch data from the database
df = fetch_data_from_db()

# Check if the DataFrame is not empty before plotting
if df is not None and not df.empty:

    # Visualize the Bias Type Distribution
    visualize_bias_type_distribution(df)
    
    # Visualize the Mitigations by Source
    # visualize_mitigations_by_source(df)
    
    # Visualize the Mitigations by Subreddit
    # visualize_mitigations_by_subreddit(df)
else:
    print("No data found or failed to fetch data.")


# In[ ]:


# Mitigation Response Frequency Viz
import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

try:
    # Connect to database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Fetch Twitter data
    twitter_query = """
    SELECT
        c.username AS user_id,
        COUNT(r.jenai_responds) AS response_count
    FROM
        comments_data c
    JOIN
        responses r ON c.comments_id = r.comments_id
    WHERE
        c.source1 = 'Twitter' AND r.jenai_responds IS NOT NULL
    GROUP BY
        c.username
    ORDER BY
        response_count DESC
    LIMIT 10;
    """
    cur.execute(twitter_query)
    twitter_data = cur.fetchall()
    twitter_df = pd.DataFrame(twitter_data, columns=['user_id', 'response_count'])

    # Fetch Reddit data
    reddit_query = """
    SELECT
        c.subreddit_id AS user_id,
        COUNT(r.jenai_responds) AS response_count
    FROM
        comments_data c
    JOIN
        responses r ON c.comments_id = r.comments_id
    WHERE
        c.source1 = 'Reddit' AND r.jenai_responds IS NOT NULL
    GROUP BY
        c.subreddit_id
    ORDER BY
        response_count DESC
    LIMIT 10;
    """
    cur.execute(reddit_query)
    reddit_data = cur.fetchall()
    reddit_df = pd.DataFrame(reddit_data, columns=['user_id', 'response_count'])

    # Close database connection
    cur.close()
    conn.close()

    # Plot side-by-side bar charts
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    # Twitter graph
    axes[0].barh(twitter_df['user_id'], twitter_df['response_count'], color='#1f77b4')
    axes[0].set_title('Mitigation Response Frequency of Top 10 Twitter Users')
    axes[0].set_xlabel('Response Count')
    axes[0].set_ylabel('User ID')

    # Reddit graph
    axes[1].barh(reddit_df['user_id'], reddit_df['response_count'], color='#add8e6')
    axes[1].set_title('Mitigation Response Frequency of Top 10 Reddit Users')
    axes[1].set_xlabel('Response Count')

    plt.tight_layout()
    plt.show()

except psycopg2.Error as e:
    print(f"Database error: {e}")
except Exception as ex:
    print(f"Error: {ex}")


# In[ ]:


# Distribution of Biased Comments to Mitigation Responses by Source

import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Retrieve PostgreSQL access info
# Establish the connection
db_config = psycopg2.connect(
    POSTGRES_HOST = os.getenv('POSTGRES_HOST'),
    POSTGRES_PORT = os.getenv('POSTGRES_PORT'),
    POSTGRES_USER = os.getenv('POSTGRES_USER'),
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD'),
    POSTGRES_DB = os.getenv('POSTGRES_DB')
)

# Verifying database access info
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
    raise ValueError("Missing one or more required PostgreSQL environment variables.")

try:
    # Connect to the database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # SQL query to get mitigation responses and biased comments
    query = """
    SELECT
        c.source1,
        COUNT(DISTINCT CASE WHEN r.jenai_responds IS NOT NULL THEN c.comments_id END) AS mitigation_responses,
        COUNT(DISTINCT CASE WHEN c.true_bias = 1 THEN c.comments_id END) AS biased_comments
    FROM
        comments_data c
    LEFT JOIN
        responses r ON c.comments_id = r.comments_id
    GROUP BY
        c.source1;
    """
    cur.execute(query)
    data = cur.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['source', 'mitigation_responses', 'biased_comments'])
    print(df)

    # Close the connection
    cur.close()
    conn.close()

except psycopg2.Error as e:
    print(f"Database error: {e}")
except Exception as ex:
    print(f"Error: {ex}")


# In[ ]:


import matplotlib.pyplot as plt

# Bar plot
df.plot(x='source', kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Mitigation Responses vs Biased Comments by Source')
plt.ylabel('Count')
plt.xlabel('Source')
plt.legend(['Mitigation Responses', 'Biased Comments'])
plt.tight_layout()
plt.show()


# In[4]:


get_ipython().system('pip install -r requirements.txt')


# ## END OF CODE RUN
