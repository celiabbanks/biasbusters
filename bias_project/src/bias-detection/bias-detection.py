########### SENTIMENT IN PHRASES OF BIAS TERMS IDENTIFICATION ################

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
