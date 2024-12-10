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
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
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
