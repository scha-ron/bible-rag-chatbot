from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH_ENV = os.getenv('DB_PATH')
DB_PATH = DB_PATH_ENV
EMBEDDING_MODEL = "all-MiniLM-L6-v2"