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
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
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

