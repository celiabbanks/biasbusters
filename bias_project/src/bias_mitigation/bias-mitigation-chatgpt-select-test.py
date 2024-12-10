# VIEW CHATGPT RESPONSE -- JUST SOURCE, USER COMMENT, AND JENAI RESPONDS...

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

# Create a cursor object
cur = conn.cursor()

# Execute the SELECT query
cur.execute("""
    SELECT 
        cd.source1 AS source, 
        CASE 
            WHEN cd.source1 = 'Twitter' THEN cd.username 
            WHEN cd.source1 = 'Reddit' THEN cd.subreddit_id 
            ELSE NULL 
        END AS identifier,
        cd.comments_id,
        cd.bias_type,
        cd.body,
        r.JenAI_responds 
    FROM 
        comments_data cd 
    JOIN 
        responses r 
    ON 
        cd.comments_id = r.comments_id 
        
    WHERE cd.source1 = 'Reddit' AND cd.bias_type = 'racial'

    LIMIT 2;
""")



# Fetch the results
rows = cur.fetchall()

# Print the results
for row in rows:
   print(row)

cur.close()
conn.close()