import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ImageProcessor:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_image_embedding(self, image_path):
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def get_text_embedding(self, text):
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return None

    def get_probs(self, image_path, text_list):
        """
        Returns a dictionary of {text_label: probability} for the given image and text candidates.
        """
        try:
            image = Image.open(image_path)
            inputs = self.processor(text=text_list, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # probs is a tensor of shape (1, n_classes)
            # we take softmax to get probabilities
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            result = {text: float(prob) for text, prob in zip(text_list, probs)}
            return result
        except Exception as e:
            print(f"Error calculating probabilities for {image_path}: {e}")
            return {}
