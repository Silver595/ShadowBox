import chromadb
from chromadb.config import Settings

class VectorDB:
    def __init__(self, persist_directory="d:/image_python/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="image_embeddings")

    def add_images(self, ids, embeddings, metadatas):
        if not ids:
            return
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_images(self, query_embedding, n_results=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
