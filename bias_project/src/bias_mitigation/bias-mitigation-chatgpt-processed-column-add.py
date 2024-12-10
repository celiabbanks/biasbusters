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

