import torch
import open_clip
from PIL import Image
import numpy as np
from typing import Union, Tuple
import os
from app.config import settings


class CLIPService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
    def load_model(self):
        """Cargar modelo CLIP"""
        print(f"Loading CLIP model: {settings.clip_model} ({settings.clip_pretrained})")
        print(f"Using device: {self.device}")
        
        # Crear directorio de caché si no existe
        os.makedirs(settings.cache_dir, exist_ok=True)
        
        # Cargar modelo con caché
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
            device=self.device,
            cache_dir=settings.cache_dir
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        self.model.eval()
        
        print("CLIP model loaded successfully")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Extraer embedding de texto"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()[0]
    
    def encode_image(self, image: Union[Image.Image, str, bytes]) -> np.ndarray:
        """Extraer embedding de imagen"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Cargar imagen según el tipo
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            import io
            pil_image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("image must be a PIL Image, file path, or bytes")
        
        # Preprocesar y extraer features
        with torch.no_grad():
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()[0]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcular similitud coseno entre dos embeddings
        Retorna un valor entre 0 y 1 (1 = idénticos, 0 = completamente diferentes)
        """
        # Normalizar embeddings si no están normalizados
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Producto punto (cosine similarity)
        similarity = np.dot(embedding1, embedding2)
        
        # Convertir de [-1, 1] a [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def validate_image_text_similarity(
        self, 
        image: Union[Image.Image, str, bytes], 
        text: str,
        threshold: float = None
    ) -> Tuple[bool, float]:
        """
        Validar que una imagen y un texto sean consistentes
        
        Args:
            image: Imagen a validar
            text: Texto descriptivo de la imagen
            threshold: Umbral de similitud (usa config por defecto si None)
        
        Returns:
            (es_valido, score_similitud)
        """
        if threshold is None:
            threshold = settings.similarity_threshold
        
        # Extraer embeddings
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text)
        
        # Calcular similitud
        similarity = self.compute_similarity(image_embedding, text_embedding)
        
        # Validar contra umbral
        is_valid = similarity >= threshold
        
        return is_valid, similarity
    
    def get_embedding_dimension(self) -> int:
        """Obtener dimensión del embedding"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.visual.output_dim


# Instancia singleton
clip_service = CLIPService()