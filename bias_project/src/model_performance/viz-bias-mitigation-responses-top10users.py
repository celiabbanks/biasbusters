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
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
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
