from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CLIP Model
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    cache_dir: str = "./model_cache"
    
    # PostgreSQL (Cloud - Railway)
    postgres_url: str
    
    # MinIO / S3
    minio_endpoint: str = "visiontext-minio:9000"
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False
    minio_bucket_images: str = "visiontext-images"
    minio_bucket_thumbnails: str = "visiontext-thumbnails"
    
    # FAISS
    faiss_index_path: str = "./faiss_indices/faiss_index.bin"
    
    # Search & Upload
    max_upload_size: int = 10485760  # 10MB
    thumbnail_size: int = 256
    default_search_limit: int = 20
    default_search_limit: int = 20
    text_search_threshold: float = 0.18  # Default para texto
    image_search_threshold: float = 0.65  # Default para imagen
    
    # Similarity validation for upload
    similarity_threshold: float = 0.20  # 20% de similitud mínima imagen-texto
    
    # CORS
    cors_origins: str 
    
    # Base URL
    base_url: str = "http://localhost:8000" 
    
    # Admin
    admin_username: str
    admin_password: str
    jwt_secret_key: str 
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()