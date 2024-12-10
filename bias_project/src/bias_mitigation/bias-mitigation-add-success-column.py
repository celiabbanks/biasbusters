# Add a column to evaluate if mitigation response was
# successfully associated with comment

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from tqdm import tqdm

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

# Batch process 

def update_mitigation_status(batch_size=1000):
    try:
        # Connect to the database
        # conn = psycopg2.connect(**db_config)
        # cur = conn.cursor()

        # Step 1: Check if 'mitigation_successful' column exists
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'comments_data' AND column_name = 'mitigation_successful';
        """)
        if not cur.fetchone():
            print("Adding 'mitigation_successful' column...")
            cur.execute("ALTER TABLE comments_data ADD COLUMN mitigation_successful BOOLEAN;")
            conn.commit()
            print("'mitigation_successful' column added successfully.")

        # Step 2: Fetch all rows to process
        cur.execute("""
            SELECT comments_id, processed, predicted_sentiment
            FROM comments_data
        """)
        rows = cur.fetchall()

        total_records = len(rows)
        print(f"Total records to process: {total_records}")

        # Step 3: Batch processing with progress bar
        with tqdm(total=total_records, desc="Processing comments") as pbar:
            for start in range(0, total_records, batch_size):
                batch = rows[start:start + batch_size]

                for row in batch:
                    comments_id, processed, predicted_sentiment = row

                    # Mitigation success logic
                    mitigation_successful = processed and predicted_sentiment >= 0

                    # Update the table
                    cur.execute("""
                        UPDATE comments_data
                        SET mitigation_successful = %s
                        WHERE comments_id = %s
                    """, (mitigation_successful, comments_id))

                conn.commit()  # Commit after each batch
                pbar.update(len(batch))  # Update progress bar

        print("Batch processing completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()  # Roll back in case of error

    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

# Run the update process
update_mitigation_status()
