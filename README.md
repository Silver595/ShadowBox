# Local Semantic Image Search 🖼️

A private, offline tool to search your photos using AI. Just type what you're looking for (e.g., "A dog playing in the snow") or upload a photo to find similar ones.

## ✨ What it does

-   **Search with words**: Just describe the photo you want to find.
-   **Find lookalikes**: Upload a photo to find others just like it.
-   **100% Private**: Everything happens right on your computer. Your photos never go to the cloud.
-   **Smart Indexing**: It scans your folders (and subfolders) automatically. It also remembers what it has already seen, so it's super fast when you add new photos.

## 🚀 How to use it

1.  **Get the code**:
    ```bash
    git clone https://github.com/YourUsername/LocalImageSearch.git
    cd LocalImageSearch
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    python main.py
    ```
    Click the local link it shows (usually `http://127.0.0.1:7860`).

4.  **Start Searching**:
    -   Go to the **INDEX MANAGEMENT** tab.
    -   Paste the path to your photo folder (e.g., `D:\MyPhotos`).
    -   Click **START INDEXING**.
    -   Once it's done, head over to the Search tab and find your memories!

## ⚙️ Under the Hood
Built with Python, Gradio, ChromaDB, and OpenAI CLIP (running locally).

---
[MIT License](https://choosealicense.com/licenses/mit/)
