import gradio as gr
import os
import glob
import logging
import hashlib
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
            # Recursive search with **
            image_files.extend(glob.glob(os.path.join(safe_path, "**", ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(safe_path, "**", ext.upper()), recursive=True))
            
        # Deduplicate
        image_files = list(set(image_files))
        
        if not image_files:
            return "No images found in directory."

        processor = get_processor()
        db = get_db()

        # Incremental Indexing: Filter out already indexed images
        # IDs are now hashes of the absolute path
        existing_ids = db.get_existing_ids()
        
        # Helper to compute hash
        def get_id(path):
            return hashlib.sha256(path.encode()).hexdigest()

        # Filter candidates
        new_files = [f for f in image_files if get_id(f) not in existing_ids]
        
        if not new_files:
            return f"No new images to index. Checked {len(image_files)} files."

        logger.info(f"Found {len(new_files)} new images to process out of {len(image_files)} total.")

        ids = []
        embeddings = []
        metadatas = []
        count = 0
        total_new = len(new_files)
        batch_size = 32
        
        # Batch Processing Loop
        for i in range(0, total_new, batch_size):
            batch_paths = new_files[i : i + batch_size]
            current_batch_size = len(batch_paths)
            
            progress((i / total_new), desc=f"Processing batch {i}/{total_new}...")
            
            try:
                # 1. Batch Inference
                batch_embeddings = processor.get_image_embeddings_batch(batch_paths)
                
                # 2. Process Metadata & Tags for the batch
                for idx, img_path in enumerate(batch_paths):
                    emb = batch_embeddings[idx]
                    if emb is None:
                        continue
                        
                    meta = get_exif_data(img_path)
                    
                    # Sanitize metadata
                    clean_meta = {}
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool)):
                            clean_meta[k] = v
                        else:
                            clean_meta[k] = str(v)
                    
                    try:
                        probs = processor.get_probs(img_path, COMMON_TAGS)
                        top_tags = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                        found_tags = [tag for tag, score in top_tags if score > 0.05]
                        clean_meta['tags'] = ", ".join(found_tags)
                    except Exception:
                        clean_meta['tags'] = ""
                    
                    clean_meta['path'] = img_path
                    
                    # Generate Unique ID
                    file_id = get_id(img_path)
                    
                    ids.append(file_id)
                    embeddings.append(emb)
                    metadatas.append(clean_meta)
                    count += 1
            
                # 3. Add to DB (flush every batch to keep memory low)
                if ids:
                    db.add_images(ids, embeddings, metadatas)
                    ids, embeddings, metadatas = [], [], []
                    
            except Exception as e:
                logger.error(f"Error processing batch starting at {i}: {e}")

        logger.info(f"Indexing complete. Added {count} new images.")
        return f"Indexing Complete! Added {count} new images from {safe_path}"

    except Exception as e:
        logger.error(f"Indexing process failed: {e}")
        return f"Error during indexing: {str(e)}"

def search_images(query_text, n_results=9):
    if not query_text:
        return []
    
    try:
        processor = get_processor()
        db = get_db()
        
        text_emb = processor.get_text_embedding(query_text)
        if text_emb is None:
            raise ValueError("Embedding generation failed.")
            
        # Oversample to handle potential ghost/missing files
        # We ask for 5x the results, filter valid ones, then slice to n_results
        results = db.query_images(text_emb, n_results=n_results * 5)
        
        images = []
        if results and results.get('metadatas') and len(results['metadatas']) > 0:
            for meta in results['metadatas'][0]:
                if 'path' in meta:
                    raw_path = meta['path']
                    # Normalize for OS compatibility (fixes mixed slashes)
                    norm_path = os.path.normpath(raw_path)
                    
                    if os.path.exists(norm_path):
                        caption = f"{meta.get('model', 'Unknown')} - {meta.get('date', 'No Date')}\nTags: {meta.get('tags','')}"
                        images.append((norm_path, caption))
                        
                        if len(images) >= n_results:
                            break
        
        return images
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise gr.Error(f"Search failed: {str(e)}")


def search_similar_images(image_path, n_results=9):
    if not image_path:
        return []
        
    try:
        processor = get_processor()
        db = get_db()
        
        # Determine strictness? For now just raw cosine similarity search
        image_emb = processor.get_image_embedding(image_path)
        if image_emb is None:
            raise ValueError("Embedding generation failed.")
            
        # Oversample
        results = db.query_images(image_emb, n_results=n_results * 5)
        
        images = []
        if results and results.get('metadatas') and len(results['metadatas']) > 0:
            for meta in results['metadatas'][0]:
                if 'path' in meta:
                    raw_path = meta['path']
                    norm_path = os.path.normpath(raw_path)
                    
                    if os.path.exists(norm_path):
                        caption = f"{meta.get('model', 'Unknown')} - {meta.get('date', 'No Date')}\nTags: {meta.get('tags','')}"
                        images.append((norm_path, caption))
                        
                        if len(images) >= n_results:
                            break
        
        return images
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise gr.Error(f"Search failed: {str(e)}")

# UI Layout
# Minimalist / Architectural Theme
theme = gr.themes.Monochrome(
    primary_hue="neutral",
    secondary_hue="neutral",
    neutral_hue="neutral",
    radius_size=gr.themes.sizes.radius_none,
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Local Semantic Image Search", theme=theme) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 2rem auto; font-family: 'Space Grotesk', sans-serif;">
            <h1 style="font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;">Local Semantic Image Search</h1>
            <p style="font-size: 0.9rem; letter-spacing: 1px; color: #666; text-transform: uppercase;">Private / Offline / AI-Powered</p>
        </div>
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("TEXT SEARCH"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    search_input = gr.Textbox(
                        label="SEARCH QUERY", 
                        placeholder="E.g. A golden retriever playing in the snow...",
                        lines=2,
                        show_label=True
                    )
                    with gr.Accordion("SEARCH SETTINGS", open=False):
                        n_results_text = gr.Slider(minimum=1, maximum=50, value=9, step=1, label="NUMBER OF RESULTS")
                    
                    search_btn = gr.Button("SEARCH", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="RESULTS", columns=[3], height="auto", object_fit="contain")
            
            # Event triggers
            search_btn.click(fn=search_images, inputs=[search_input, n_results_text], outputs=gallery)
            search_input.submit(fn=search_images, inputs=[search_input, n_results_text], outputs=gallery)

        with gr.TabItem("IMAGE SEARCH"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    img_input = gr.Image(label="UPLOAD IMAGE", type="filepath", height=300)
                    with gr.Accordion("SEARCH SETTINGS", open=False):
                        n_results_img = gr.Slider(minimum=1, maximum=50, value=9, step=1, label="NUMBER OF RESULTS")
                    img_find_btn = gr.Button("FIND SIMILAR", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    img_gallery = gr.Gallery(label="SIMILAR IMAGES", columns=[3], height="auto", object_fit="contain")
            
            img_find_btn.click(fn=search_similar_images, inputs=[img_input, n_results_img], outputs=img_gallery)

        with gr.TabItem("INDEX MANAGEMENT"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### INDEX NEW IMAGES")
                    dir_input = gr.Textbox(
                        label="DIRECTORY PATH", 
                        value=os.path.abspath("./images"),
                        placeholder="C:\\Users\\Photos..."
                    )
                    gr.Markdown("*Note: The indexer scans all subfolders recursively.*")
                    index_btn = gr.Button("START INDEXING", variant="secondary")
                
                with gr.Column():
                     index_output = gr.Textbox(label="STATUS LOG", lines=10, interactive=False)
            
            index_btn.click(fn=index_images, inputs=dir_input, outputs=index_output)

if __name__ == "__main__":
    # Allow access to drives where images might be stored
    # This is necessary for a local tool that accesses user files across the system
    allowed_paths = ["C:\\", "D:\\"]
    demo.queue() # Enable queuing for progress bars
    demo.launch(allowed_paths=allowed_paths)

