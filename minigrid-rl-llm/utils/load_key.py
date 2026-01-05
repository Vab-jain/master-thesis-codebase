from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Now you can access your API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')