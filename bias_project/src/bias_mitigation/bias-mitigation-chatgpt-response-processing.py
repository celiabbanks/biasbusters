# Modify process_comments function to process only records where processed = FALSE
import psycopg2
from psycopg2 import sql, OperationalError, Error
from openai import OpenAI

# load API key
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Ensure the API key is loaded properly
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")


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

# OpenAI client initialization
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  

# Safe phrases
safe_phrases = [
    "love my grandparents", "my adorable grandparents", "grandma's cooking is delicious",
    "My grandma makes the best apple pie", "Grandma shared stories from her youth",
    "Grandma's advice is always so helpful", "My grandma loves gardening",
    "My grandpa used to be a teacher", "Spending time with my grandparents is the best",
    "I am a foreigner to learning technology", "let's pig out and enjoy this ice cream",
    "you're so skinny oh how I want to be too", "we are better than that is not a classist statement",
    "having wealth can make you more wealth", "your behavior is ugly and needs improvement",
    "The older generation has a lot to teach us", "Young people today have so many new opportunities",
    "he is a successful foreigner who has made America his home",
    "I am too lazy to run for the U.S. Congress", "I feel lazy today",
    "Lazy Saturday mornings are the best",
    "George Washington Carver was a prominent Black scientist of the early 20th century with his sustainable farming techniques",
    "African American hip hop artists are some of the wealthiest people in the U.S."
]

# Response templates
templates = {
    "implicit_bias": {
        "general": "The comment may have implicit bias: \"{comment}\". A more neutral approach could be: {suggestion}.",
        "racial": "The comment contains potential implicit racial bias: \"{comment}\". Consider rephrasing to be more inclusive, e.g., {suggestion}.",
        "sexist": "The comment contains potential implicit gender bias: \"{comment}\". A more equitable phrasing might be: {suggestion}.",
    },
    "explicit_bias": {
        "general": "The comment: \"{comment}\" shows explicit bias. To clarify intent, consider this alternative: {suggestion}.",
        "bullying": "The comment: \"{comment}\" is an example of explicit bullying. To foster a positive dialogue, you might say: {suggestion}.",
    },
    "political_bias": "The comment may contain political bias: \"{comment}\". A more neutral approach could be: {suggestion}.",
    "brand_bias": "The comment may contain brand bias: \"{comment}\". A more neutral approach could be: {suggestion}."
}


# Function to generate suggestion from safe phrases
def generate_suggestion(comment, sentiment):
    for phrase in safe_phrases:
        if phrase.lower() in comment.lower():
            return f"Consider focusing on a positive aspect, such as: '{phrase}'"
    
    # Incorporate sentiment into suggestions
    if sentiment == "negative":
        return "The tone of the comment seems negative. Consider rephrasing to express positivity or neutrality."
    elif sentiment == "neutral":
        return "The tone is neutral. Ensure clarity and avoid any unintended bias."
    elif sentiment == "positive":
        return "The tone is positive. Maintain this perspective while ensuring inclusivity."
    
    return "Rephrase to remove any potential bias."

# Function to generate a response using templates and ChatGPT
def generate_response(comment_data):
    comment = comment_data['body']
    implicit_explicit = comment_data['implicit_explicit']
    bias_type = comment_data['bias_type']
    sentiment = comment_data.get('predicted_sentiment', 'neutral')  # Default to neutral if not present

    # Select the appropriate template
    if implicit_explicit == 1:  # Explicit bias
        template = templates['explicit_bias'].get(bias_type, templates['explicit_bias']['general'])
    elif implicit_explicit == 2:  # Implicit bias
        template = templates['implicit_bias'].get(bias_type, templates['implicit_bias']['general'])
    else:
        template = templates.get(bias_type, templates['implicit_bias']['general'])

    # Generate suggestion
    suggestion = generate_suggestion(comment, sentiment)

    # Populate the template
    JenAI_responds = template.format(comment=comment, suggestion=suggestion)

    # Use ChatGPT to enhance the response
    try:
        system_message = (
            "You are JenAI, a friendly and approachable assistant who helps with identifying bias. "
            "Respond in a friendly, professional, and empathetic tone, ensuring clarity and warmth."
        )

        user_message = f"JenAI advises: {JenAI_responds}"

        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return gpt_response.choices[0].message.content
    except Exception as e:
        print(f"Error generating GPT response for comment: {comment}")
        print(f"OpenAI error: {e}")
        return JenAI_responds  # Fallback to template response

# Process flagged comments and insert responses in batches
def process_comments(batch_size=1000):
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Fetch unprocessed flagged comments
        cur.execute("""
            SELECT comments_id, body, implicit_explicit, bias_type, predicted_sentiment
            FROM comments_data
            WHERE implicit_explicit > 0 AND processed = FALSE
        """)
        rows = cur.fetchall()
        
        total_records = len(rows)
        print(f"Total records to process: {total_records}")

        # Process records in batches
        for start in range(0, total_records, batch_size):
            batch = rows[start:start + batch_size]
            print(f"Processing batch {start // batch_size + 1}: Records {start + 1} to {min(start + batch_size, total_records)}")

            for row in batch:
                comment_data = {
                    "comments_id": row[0],
                    "body": row[1],
                    "implicit_explicit": row[2],
                    "bias_type": row[3],
                    "Predicted_sentiment": row[4]
                }

                # Generate response
                response = generate_response(comment_data)

                # Insert response into the responses table
                cur.execute(
                    "INSERT INTO responses (comments_id, JenAI_responds) VALUES (%s, %s)",
                    (comment_data['comments_id'], response)
                )

                # Update the processed flag
                cur.execute(
                    "UPDATE comments_data SET processed = TRUE WHERE comments_id = %s",
                    (comment_data['comments_id'],)
                )

                conn.commit()
                print(f"Response inserted and processed flag updated for comment_id: {comment_data['comments_id']}")

            print(f"Batch {start // batch_size + 1} completed.")

    except (OperationalError, Error) as e:
        print(f"Database error: {e}")
        raise

    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    try:
        process_comments(batch_size=1000)  # Set batch size as needed
    except Exception as e:
        print(f"Execution stopped due to error: {e}")
