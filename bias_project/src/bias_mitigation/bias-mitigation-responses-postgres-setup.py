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
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
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
