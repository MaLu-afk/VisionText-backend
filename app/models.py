from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ImageMetadata(BaseModel):
    width: int
    height: int
    size: int
    format: str


class SearchResult(BaseModel):
    id: str
    filename: str
    url: str
    thumbnail_url: str
    similarity_score: float
    description: Optional[str] = None
    metadata: Optional[ImageMetadata] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    query_time: float


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # None = usa default de .env


class UploadStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # Por baja similitud


class UploadResponse(BaseModel):
    success: bool
    message: str
    image_id: Optional[str] = None
    filename: Optional[str] = None
    status: Optional[UploadStatus] = None
    similarity_score: Optional[float] = None  
    required_similarity: Optional[float] = None
    banned_words: Optional[List[str]] = None 
    rejection_reason: Optional[str] = None


class SimilarityValidationResult(BaseModel):
    is_valid: bool
    similarity_score: float
    threshold: float
    message: str


class ImageRecord(BaseModel):
    id: str
    filename: str
    description: Optional[str] = None
    image_path: str  # Path en MinIO
    thumbnail_path: str  # Path en MinIO
    width: int
    height: int
    size: int
    format: str
    status: UploadStatus = UploadStatus.COMPLETED
    similarity_score: Optional[float] = None  
    created_at: datetime
    updated_at: Optional[datetime] = None


class ImageUploadRequest(BaseModel):
    """Modelo para request de upload con descripción opcional"""
    description: Optional[str] = Field(None, max_length=500)