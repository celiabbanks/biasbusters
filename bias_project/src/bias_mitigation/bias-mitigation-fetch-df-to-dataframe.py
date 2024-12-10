# Join data for analysis
# Fetch data and process into pandas dataframe 

import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2

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


# Create a cursor object
cur = conn.cursor()

try:

    # SQL Query to join tables
    sql_query = """
        SELECT
            cd.comments_id,
            cd.processed,
            cd.predicted_sentiment,
            r.JenAI_responds
        FROM
            comments_data cd
        LEFT JOIN
            responses r
        ON
            cd.comments_id = r.comments_id;
    """
    cur.execute(sql_query)

    # Fetch all results
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=columns)
    print("Data fetched successfully.")

except Exception as e:
    print(f"Error fetching data: {e}")

finally:
    if 'conn' in locals() and conn:
        cur.close()
        conn.close()
