import os
import shutil
from src.processor import ImageProcessor
from src.metadata import get_exif_data
from src.tags import COMMON_TAGS
import PIL.Image

# Ensure dummy image exists
if not os.path.exists("dummy_test.jpg"):
    img = PIL.Image.new('RGB', (100, 100), color = 'red')
    img.save("dummy_test.jpg")

processor = ImageProcessor()

print("--- Testing Auto-Tagging ---")
probs = processor.get_probs("dummy_test.jpg", COMMON_TAGS)
top_tags = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
print("Top Tags:", top_tags)

print("\n--- Testing Metadata ---")
# Create a dummy image with no real exif, but we check structurally
meta = get_exif_data("dummy_test.jpg")
print("Metadata extracted:", meta)

print("\nVerification Complete.")
