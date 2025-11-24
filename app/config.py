from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CLIP Model
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    cache_dir: str = "./model_cache"
    
    # PostgreSQL (Cloud - Railway)
    postgres_url: str
    
    # Redis (Cloud - Railway)
    redis_url: str
    
    # MinIO / S3
    minio_endpoint: str = "minio:9000"
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
    default_threshold: float = 0.1
    
    # Similarity validation for upload
    similarity_threshold: float = 0.25  # Umbral mínimo de similitud imagen-texto
    
    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000"
    
    # Worker
    worker: bool = False
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def celery_broker(self) -> str:
        return self.celery_broker_url or self.redis_url
    
    @property
    def celery_backend(self) -> str:
        return self.celery_result_backend or self.redis_url
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()