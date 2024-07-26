from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DB_PATH, EMBEDDING_MODEL
import requests
from tqdm import tqdm
#import json

BASE_URL = "https://api.biblesupersearch.com/api"
BIBLE_VERSION = "kjv"
LANGUAGE = "en"  # English

def get_bible_versions():
    try:
        response = requests.get(f"{BASE_URL}/bibles")
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and isinstance(data['results'], dict):
            return data['results']
        else:
            print("Unexpected data structure in API response for Bible versions")
            return {}
    except requests.RequestException as e:
        print(f"Error fetching Bible versions: {e}")
        return {}

def get_books():
    try:
        params = {
            'language': LANGUAGE,
        }
        response = requests.get(f"{BASE_URL}/books", params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and isinstance(data['results'], list):
            return data['results']
        else:
            print("Unexpected data structure in API response for books")
            return []
    except requests.RequestException as e:
        print(f"Error fetching books: {e}")
        return []

def get_book_content(bible_module, book_name):
    try:
        params = {
            'bible': bible_module,
            'reference': book_name,
            'data_format': 'passage',
            'formatting': 'plain'
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
            return data['results'][0].get('verses', {}).get(bible_module, {})
        else:
            print(f"No verses found for {book_name}")
            return {}
    except requests.RequestException as e:
        print(f"Error fetching book content: {e}")
        return {}

def extract_bible_content(bible_module):
    bible_content = []
    books = get_books()
    
    if not books:
        print("No books fetched. Check your internet connection or API access.")
        return bible_content

    for book in tqdm(books, desc="Processing books"):
        book_content = get_book_content(bible_module, book['name'])
        for chapter, verses in book_content.items():
            for verse_num, verse_data in verses.items():
                bible_content.append({
                    'book': book['name'],
                    'chapter': chapter,
                    'verse': verse_num,
                    'text': verse_data['text']
                })
    
    return bible_content

def preprocess_bible_content(bible_module):
    bible_content = extract_bible_content(bible_module)
    
    if not bible_content:
        print("No Bible content extracted. Aborting preprocessing.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    bible_splits = []
    for entry in bible_content:
        splits = text_splitter.split_text(entry['text'])
        for i, split in enumerate(splits):
            bible_splits.append({
                'book': entry['book'],
                'chapter': entry['chapter'],
                'verse': entry['verse'],
                'chunk': i + 1,
                'text': split,
            })
    
    return bible_splits

def create_vectorstore(bible_splits):
    if not bible_splits:
        print("No Bible splits to process. Aborting vector store creation.")
        return

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    texts = [split['text'] for split in bible_splits]
    metadatas = [{k: v for k, v in split.items() if k != 'text'} for split in bible_splits]
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"Vector store created and saved to {DB_PATH}")

if __name__ == "__main__":
    bible_versions = get_bible_versions()
    if BIBLE_VERSION not in bible_versions:
        print(f"Bible version {BIBLE_VERSION} not found. Available versions:")
        for module, info in bible_versions.items():
            print(f"- {module}: {info['name']}")
        exit(1)
    
    bible_module = bible_versions[BIBLE_VERSION]['module']
    bible_splits = preprocess_bible_content(bible_module)
    if bible_splits:
        create_vectorstore(bible_splits)
    else:
        print("Preprocessing failed. Unable to create vector store.")