from src.database import VectorDB
import logging
import os

# Suppress log noise
logging.basicConfig(level=logging.ERROR)

db = VectorDB()
# Get a few items
try:
    results = db.collection.get(limit=10, include=['metadatas'])
    print("--- Stored Paths ---")
    for meta in results['metadatas']:
        path = meta.get('path', 'MISSING')
        print(f"Path: {path}")
        print(f"Exists: {os.path.exists(path)}")
except Exception as e:
    print(e)
