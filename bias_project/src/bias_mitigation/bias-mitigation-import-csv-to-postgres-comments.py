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
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
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