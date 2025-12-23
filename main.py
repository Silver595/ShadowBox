import gradio as gr
import os
import glob
from src.processor import ImageProcessor
from src.metadata import get_exif_data
from src.database import VectorDB
from src.tags import COMMON_TAGS

# Initialize components
# Note: In a real app, might want to lazy load or load on startup with progress
processor = ImageProcessor()
db = VectorDB()

def index_images(directory_path):
    if not os.path.exists(directory_path):
        return "Directory not found."
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        # Also check case-insensitive or upper case extensions on Windows if needed, 
        # but glob is case-insensitive on Windows usually.
    
    if not image_files:
        return "No images found in directory."

    ids = []
    embeddings = []
    metadatas = []
    
    count = 0
    for img_path in image_files:
        try:
            # Check if already indexed? For now, just re-index or add. 
            # ChromaDB handles duplicates by ID if we use path as ID.
            
            emb = processor.get_image_embedding(img_path)
            if emb is None:
                continue
                
            meta = get_exif_data(img_path)
            # ChromaDB metadata must be str, int, float, bool. 
            # Convert complex types if any.
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            
            # Auto-Tagging
            probs = processor.get_probs(img_path, COMMON_TAGS)
            # filter tags > 0.1 probability, take top 5
            top_tags = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            found_tags = [tag for tag, score in top_tags if score > 0.05]
            
            clean_meta['tags'] = ", ".join(found_tags)
            
            # Add path to metadata for retrieval
            clean_meta['path'] = img_path
            
            ids.append(os.path.basename(img_path)) # ID is filename, might want full path if duplicates possible across folders
            embeddings.append(emb)
            metadatas.append(clean_meta)
            count += 1
            
            # Batch add to avoid memory issues if many images
            if len(ids) >= 10:
                db.add_images(ids, embeddings, metadatas)
                ids, embeddings, metadatas = [], [], []
                
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # Add remaining
    if ids:
        db.add_images(ids, embeddings, metadatas)

    return f"Indexed {count} images from {directory_path}"

def search_images(query_text):
    if not query_text:
        return []
    
    try:
        text_emb = processor.get_text_embedding(query_text)
        if text_emb is None:
            raise gr.Error("Embedding generation failed. See logs.")
    except Exception as e:
        raise gr.Error(f"Search failed: {str(e)}")
    
    results = db.query_images(text_emb, n_results=9)
    
    # Results structure: {'ids': [[]], 'metadatas': [[]], ...}
    images = []
    if results['metadatas'] and len(results['metadatas']) > 0:
        for meta in results['metadatas'][0]:
            if 'path' in meta:
                # Gradio Gallery expects list of (image, caption) tuples or just images
                # We can add caption with metadata
                caption = f"{meta.get('model', 'Unknown')} - {meta.get('date', 'No Date')}"
                images.append((meta['path'], caption))
    
    return images

# UI Layout
with gr.Blocks(title="Local Semantic Image Search") as demo:
    gr.Markdown("# Local Semantic Image Search & Privacy Tool")
    
    with gr.Tab("Search"):
        search_input = gr.Textbox(label="Search Query", placeholder="e.g., 'A dog running in the park'")
        search_btn = gr.Button("Search")
        gallery = gr.Gallery(label="Results", columns=3, height="auto")
        search_btn.click(fn=search_images, inputs=search_input, outputs=gallery)
        search_input.submit(fn=search_images, inputs=search_input, outputs=gallery)

    with gr.Tab("Index"):
        dir_input = gr.Textbox(label="Image Directory Path", value="d:/image_python/images")
        index_btn = gr.Button("Start Indexing")
        index_output = gr.Textbox(label="Status")
        index_btn.click(fn=index_images, inputs=dir_input, outputs=index_output)

if __name__ == "__main__":
    demo.launch(allowed_paths=["C:\\", "D:\\"])
