import faiss
import numpy as np
import os
import pickle
import asyncio
from typing import List, Tuple
from app.config import settings


class FAISSService:
    def __init__(self):
        self.index = None
        self.image_ids = []
        self.dimension = None
        self.lock = asyncio.Lock()
        
    def initialize_index(self, dimension: int):
        """Inicializar índice FAISS"""
        self.dimension = dimension
        # Usar IndexFlatIP para búsqueda por producto interno (cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        print(f"FAISS index initialized with dimension {dimension}")
    
    async def add_embedding(self, image_id: str, embedding: np.ndarray):
        """Agregar embedding al índice (thread-safe)"""
        async with self.lock:
            if self.index is None:
                raise RuntimeError("Index not initialized")
            
            # Asegurar que el embedding sea float32 y 2D
            embedding = embedding.astype('float32').reshape(1, -1)
            
            # Agregar al índice
            self.index.add(embedding)
            self.image_ids.append(image_id)
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 20, 
        threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Buscar embeddings más similares (thread-safe)"""
        async with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Asegurar que el embedding sea float32 y 2D
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            
            # Buscar los k más cercanos
            k_search = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k_search)
            
            # Convertir distancias (producto interno) a similitud [0, 1]
            # Como los embeddings están normalizados, el producto interno es cosine similarity
            # Rango: [-1, 1] → convertir a [0, 1]
            similarities = (distances[0] + 1) / 2
            
            # Filtrar por threshold y retornar resultados
            results = []
            for sim, idx in zip(similarities, indices[0]):
                if idx != -1 and sim >= threshold:
                    results.append((self.image_ids[idx], float(sim)))
            
            return results
    
    async def save_index(self):
        """Guardar índice FAISS y mapeo de IDs (thread-safe)"""
        async with self.lock:
            if self.index is None:
                return
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
            
            # Guardar índice FAISS
            faiss.write_index(self.index, settings.faiss_index_path)
            
            # Guardar mapeo de IDs
            metadata_path = settings.faiss_index_path + ".metadata"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'image_ids': self.image_ids,
                    'dimension': self.dimension
                }, f)
            
            print(f"FAISS index saved with {self.index.ntotal} vectors")
    
    def load_index(self) -> bool:
        """Cargar índice FAISS y mapeo de IDs"""
        if not os.path.exists(settings.faiss_index_path):
            return False
        
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(settings.faiss_index_path)
            
            # Cargar mapeo de IDs
            metadata_path = settings.faiss_index_path + ".metadata"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.image_ids = metadata['image_ids']
                self.dimension = metadata['dimension']
            
            print(f"FAISS index loaded with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False
    
    def get_count(self) -> int:
        """Obtener número de vectores en el índice"""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    async def rebuild_index(self, embeddings_data: List[Tuple[str, np.ndarray]]):
        """Reconstruir índice desde cero (thread-safe)"""
        async with self.lock:
            if not embeddings_data:
                return
            
            # Obtener dimensión del primer embedding
            dimension = embeddings_data[0][1].shape[0]
            self.initialize_index(dimension)
            
            # Preparar todos los embeddings para inserción batch
            all_embeddings = []
            all_ids = []
            
            for image_id, embedding in embeddings_data:
                all_embeddings.append(embedding.astype('float32'))
                all_ids.append(image_id)
            
            # Inserción batch
            embeddings_array = np.vstack(all_embeddings)
            self.index.add(embeddings_array)
            self.image_ids = all_ids
            
            print(f"Index rebuilt with {len(all_ids)} vectors")
    
    async def remove_embedding(self, image_id: str) -> bool:
        """
        Eliminar un embedding del índice
        Nota: FAISS no soporta eliminación directa, se debe reconstruir el índice
        """
        async with self.lock:
            if image_id not in self.image_ids:
                return False
            
            # Encontrar índice
            idx = self.image_ids.index(image_id)
            
            # Remover de lista de IDs
            self.image_ids.pop(idx)
            
            # Para FAISS, necesitamos reconstruir el índice sin ese vector
            # Esto es costoso, considerar hacer batch removals
            print(f"Warning: FAISS doesn't support direct removal. Index should be rebuilt.")
            return True


# Instancia singleton
faiss_service = FAISSService()