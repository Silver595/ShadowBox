from src.database import VectorDB
import logging
import os

logging.basicConfig(level=logging.ERROR)

db = VectorDB()
print(f"--- Database Diagnostics ---")
try:
    count = db.collection.count()
    print(f"Total Images in Index: {count}")
    
    if count > 0:
        # IDs are always returned, don't ask for them in 'include'
        results = db.collection.get(limit=20, include=['metadatas'])
        print(f"\nChecking first 20 entries:")
        valid_count = 0
        ids = results['ids']
        metas = results['metadatas']
        
        for i, meta in enumerate(metas):
            path = meta.get('path', 'MISSING')
            exists = os.path.exists(path)
            status = "EXISTS" if exists else "MISSING/INVALID"
            print(f"ID: {ids[i][:10]}... | Path: {path} -> {status}")
            if exists: valid_count += 1
        
        print(f"\nValid paths in sample: {valid_count}/20")
    else:
        print("Database is empty.")

except Exception as e:
    print(f"Error: {e}")
