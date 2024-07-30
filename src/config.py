import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

DB_PATH = os.getenv('DB_PATH')
EMBEDDING_MODEL = "all-MiniLM-L6-v2"