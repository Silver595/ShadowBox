import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, persist_directory="d:/image_python/chroma_db"):
        logger.info(f"Connecting to ChromaDB at {persist_directory}...")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="image_embeddings")

    def add_images(self, ids, embeddings, metadatas):
        if not ids:
            return
        logger.info(f"Adding batch of {len(ids)} images to database.")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_images(self, query_embedding, n_results=5):
        logger.debug(f"Querying database for top {n_results} results.")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def get_existing_ids(self):
        """
        Returns a set of all IDs currently in the database.
        """
        try:
            # ChromaDB get() returns all if no args provided, but check limit
            # For large datasets, this might need pagination, but for local use <100k it's fine
            existing = self.collection.get(include=[])
            return set(existing['ids'])
        except Exception as e:
            logger.error(f"Failed to fetch existing IDs: {e}")
            return set()
