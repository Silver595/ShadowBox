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
            # Recursive search with **
            image_files.extend(glob.glob(os.path.join(safe_path, "**", ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(safe_path, "**", ext.upper()), recursive=True))
            
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

def search_images(query_text, n_results=9):
    if not query_text:
        return []
    
    try:
        processor = get_processor()
        db = get_db()
        
        text_emb = processor.get_text_embedding(query_text)
        if text_emb is None:
            raise ValueError("Embedding generation failed.")
            
        results = db.query_images(text_emb, n_results=n_results)
        
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
            
        results = db.query_images(image_emb, n_results=n_results)
        
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
# UI Layout
# Custom CSS for Animations and Font
custom_css = """
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

.gradio-container {
    animation: fadeIn 0.8s ease-out;
}

/* Button Hover Effects */
button {
    transition: all 0.3s ease !important;
}

button:hover {
    transform: scale(1.02);
    opacity: 0.9;
}
"""

# Minimalist / Architectural Theme
theme = gr.themes.Monochrome(
    primary_hue="neutral",
    secondary_hue="neutral",
    neutral_hue="neutral",
    radius_size=gr.themes.sizes.radius_none,
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Local Semantic Image Search", theme=theme, css=custom_css) as demo:
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

