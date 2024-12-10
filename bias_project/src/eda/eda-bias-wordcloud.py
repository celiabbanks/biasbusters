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
