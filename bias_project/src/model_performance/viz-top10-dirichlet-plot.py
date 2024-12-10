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