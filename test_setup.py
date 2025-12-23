import torch
import chromadb
from transformers import CLIPProcessor, CLIPModel
import PIL

def test_setup():
    print("Testing setup...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Running on CPU.")

    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path="d:/image_python/chroma_db_test")
        print("ChromaDB initialized successfully.")
    except Exception as e:
        print(f"ChromaDB initialization failed: {e}")

    # Check Transformers/CLIP
    try:
        model_id = "openai/clip-vit-base-patch32"
        print(f"Loading {model_id}...")
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        print("CLIP model loaded successfully.")
    except Exception as e:
        print(f"CLIP model loading failed: {e}")

    print("Setup test complete.")

if __name__ == "__main__":
    test_setup()
