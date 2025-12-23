import gradio as gr
import os
import glob
import logging
from src.processor import ImageProcessor
from src.metadata import get_exif_data
from src.database import VectorDB
from src.tags import COMMON_TAGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global instances for lazy loading
_processor = None
_db = None

def get_processor():
    global _processor
    if _processor is None:
        logger.info("Initializing ImageProcessor...")
        try:
            _processor = ImageProcessor()
            logger.info("ImageProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ImageProcessor: {e}")
            raise
    return _processor

def get_db():
    global _db
    if _db is None:
        logger.info("Initializing VectorDB...")
        try:
            _db = VectorDB()
            logger.info("VectorDB initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize VectorDB: {e}")
            raise
    return _db

def index_images(directory_path, progress=gr.Progress()):
    if not directory_path:
        return "Please provide a directory path."
    
    # Security check: Ensure path is absolute and exists
    safe_path = os.path.abspath(directory_path)
    if not os.path.exists(safe_path):
        return "Directory not found."
    
    if not os.path.isdir(safe_path):
        return "Path is not a directory."

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    try:
        progress(0, desc="Scanning directory...")
        for ext in image_extensions:
            # Case-insensitive search simulation for Windows/Linux compatibility if strict case needed
            # But standard glob is usually sufficient. 
            image_files.extend(glob.glob(os.path.join(safe_path, ext)))
            # Also try uppercase extensions
            image_files.extend(glob.glob(os.path.join(safe_path, ext.upper())))
            
        # Deduplicate
        image_files = list(set(image_files))
        
        if not image_files:
            return "No images found in directory."

        processor = get_processor()
        db = get_db()

        ids = []
        embeddings = []
        metadatas = []
        count = 0
        total_images = len(image_files)
        
        logger.info(f"Found {total_images} images to process in {safe_path}")

        for i, img_path in enumerate(image_files):
            # Update progress
            progress((i / total_images), desc=f"Processing {os.path.basename(img_path)}...")
            
            try:
                emb = processor.get_image_embedding(img_path)
                if emb is None:
                    continue
                    
                meta = get_exif_data(img_path)
                
                # Sanitize metadata for ChromaDB
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                
                # Auto-Tagging
                try:
                    probs = processor.get_probs(img_path, COMMON_TAGS)
                    top_tags = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    found_tags = [tag for tag, score in top_tags if score > 0.05]
                    clean_meta['tags'] = ", ".join(found_tags)
                except Exception as tag_err:
                    logger.warning(f"Tagging failed for {img_path}: {tag_err}")
                    clean_meta['tags'] = ""
                
                clean_meta['path'] = img_path
                
                # Use filename as ID, ensure uniqueness could be handled better in production
                ids.append(os.path.basename(img_path))
                embeddings.append(emb)
                metadatas.append(clean_meta)
                count += 1
                
                # Batch add - increased to 50 for performance
                if len(ids) >= 50:
                    db.add_images(ids, embeddings, metadatas)
                    ids, embeddings, metadatas = [], [], []
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Add remaining
        if ids:
            db.add_images(ids, embeddings, metadatas)

        logger.info(f"Indexing complete. Indexed {count} images.")
        return f"Indexing Complete! Indexed {count} images from {safe_path}"

    except Exception as e:
        logger.error(f"Indexing process failed: {e}")
        return f"Error during indexing: {str(e)}"

def search_images(query_text):
    if not query_text:
        return []
    
    try:
        processor = get_processor()
        db = get_db()
        
        text_emb = processor.get_text_embedding(query_text)
        if text_emb is None:
            raise ValueError("Embedding generation failed.")
            
        results = db.query_images(text_emb, n_results=9)
        
        images = []
        if results and results.get('metadatas') and len(results['metadatas']) > 0:
            for meta in results['metadatas'][0]:
                if 'path' in meta:
                    caption = f"{meta.get('model', 'Unknown')} - {meta.get('date', 'No Date')}\nTags: {meta.get('tags','')}"
                    images.append((meta['path'], caption))
        
        return images
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise gr.Error(f"Search failed: {str(e)}")


def search_similar_images(image_path):
    if not image_path:
        return []
        
    try:
        processor = get_processor()
        db = get_db()
        
        # Determine strictness? For now just raw cosine similarity search
        image_emb = processor.get_image_embedding(image_path)
        if image_emb is None:
            raise ValueError("Embedding generation failed.")
            
        results = db.query_images(image_emb, n_results=9)
        
        images = []
        if results and results.get('metadatas') and len(results['metadatas']) > 0:
            for meta in results['metadatas'][0]:
                if 'path' in meta:
                    caption = f"{meta.get('model', 'Unknown')} - {meta.get('date', 'No Date')}\nTags: {meta.get('tags','')}"
                    images.append((meta['path'], caption))
        
        return images
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise gr.Error(f"Search failed: {str(e)}")

# UI Layout
with gr.Blocks(title="Local Semantic Image Search") as demo:
    gr.Markdown("# Local Semantic Image Search & Privacy Tool")
    
    with gr.Tab("Text Search"):
        search_input = gr.Textbox(label="Search Query", placeholder="e.g., 'A dog running in the park'")
        search_btn = gr.Button("Search")
        gallery = gr.Gallery(label="Results", columns=3, height="auto")
        search_btn.click(fn=search_images, inputs=search_input, outputs=gallery)
        search_input.submit(fn=search_images, inputs=search_input, outputs=gallery)

    with gr.Tab("Image Search"):
        gr.Markdown("Upload an image to find similar ones in your index.")
        with gr.Row():
            img_input = gr.Image(label="Upload Image", type="filepath", height=300)
            img_gallery = gr.Gallery(label="Similar Images", columns=3, height="auto")
        
        img_find_btn = gr.Button("Find Similar Images")
        img_find_btn.click(fn=search_similar_images, inputs=img_input, outputs=img_gallery)

    with gr.Tab("Index"):
        dir_input = gr.Textbox(label="Image Directory Path", value=os.path.abspath("./images"))
        index_btn = gr.Button("Start Indexing")
        index_output = gr.Textbox(label="Status")
        index_btn.click(fn=index_images, inputs=dir_input, outputs=index_output)

if __name__ == "__main__":
    # Allow access to drives where images might be stored
    # This is necessary for a local tool that accesses user files across the system
    allowed_paths = ["C:\\", "D:\\"]
    demo.queue() # Enable queuing for progress bars
    demo.launch(allowed_paths=allowed_paths)

