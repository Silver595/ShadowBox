import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model on {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _prepare_image(self, image_path):
        """Helper to open and optimization image size."""
        image = Image.open(image_path)
        # Resize if too large to save memory/compute, CLIP expects 224x224 anyway so huge inputs are wasteful
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
        return image

    def get_image_embedding(self, image_path):
        try:
            image = self._prepare_image(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.warning(f"Error processing image {image_path} on {self.device}: {e}")
            if self.device == "cuda":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                self.model = self.model.to("cpu")
                return self.get_image_embedding(image_path)
            return None

    def get_image_embeddings_batch(self, image_paths):
        """
        Process a batch of images and return a list of embeddings.
        Returns None for failed images in the list (maintaining order is hard if some fail mid-batch, 
        so we process carefully or return list of (emb|None)).
        """
        valid_images = []
        valid_indices = []
        embeddings = [None] * len(image_paths)

        # 1. Load Images
        for idx, path in enumerate(image_paths):
            try:
                img = self._prepare_image(path)
                valid_images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Failed to load image for batch {path}: {e}")

        if not valid_images:
            return embeddings

        # 2. Batch Inference
        try:
            inputs = self.processor(images=valid_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            features_np = features.cpu().numpy()

            # 3. Map back to original order
            for i, real_idx in enumerate(valid_indices):
                embeddings[real_idx] = features_np[i].tolist()
        
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Fallback to single processing if batch fails (e.g. OOM)
            for idx in valid_indices:
                embeddings[idx] = self.get_image_embedding(image_paths[idx])

        return embeddings

    def get_text_embedding(self, text):
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.warning(f"Error processing text '{text}' on {self.device}: {e}")
            if self.device == "cuda":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                self.model = self.model.to("cpu")
                return self.get_text_embedding(text)
            return None

    def get_probs(self, image_path, text_list):
        """
        Returns a dictionary of {text_label: probability} for the given image and text candidates.
        """
        try:
            image = self._prepare_image(image_path)
            inputs = self.processor(text=text_list, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # probs is a tensor of shape (1, n_classes)
            # we take softmax to get probabilities
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            result = {text: float(prob) for text, prob in zip(text_list, probs)}
            return result
        except Exception as e:
            logger.warning(f"Error calculating probabilities for {image_path}: {e}")
            # Simple fallback for probs too?
            if self.device == "cuda":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                self.model = self.model.to("cpu")
                return self.get_probs(image_path, text_list)
            return {}
