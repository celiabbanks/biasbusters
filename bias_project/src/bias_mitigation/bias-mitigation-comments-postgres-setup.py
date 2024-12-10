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
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
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

