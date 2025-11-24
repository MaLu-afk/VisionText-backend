import uuid
from PIL import Image
import io
from typing import Tuple
from datetime import datetime
from app.config import settings
from app.models import ImageRecord, UploadStatus


class ImageProcessingService:
    def __init__(self):
        self.thumbnail_size = settings.thumbnail_size
    
    def process_image(
        self, 
        file_content: bytes, 
        filename: str,
        description: str = None,
        similarity_score: float = None
    ) -> Tuple[ImageRecord, Image.Image, bytes, bytes]:
        """
        Procesar imagen: validar, redimensionar, crear thumbnail
        
        Returns:
            (image_record, pil_image, original_bytes, thumbnail_bytes)
        """
        
        # Generar ID único
        image_id = str(uuid.uuid4())
        
        # Abrir imagen
        image = Image.open(io.BytesIO(file_content))
        image = image.convert('RGB')  # Asegurar formato RGB
        
        # Obtener metadatos
        width, height = image.size
        file_format = image.format or 'JPEG'
        file_size = len(file_content)
        
        # Guardar imagen original en memoria
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='JPEG', quality=95)
        original_bytes = original_buffer.getvalue()
        
        # Crear y guardar thumbnail
        thumbnail = image.copy()
        thumbnail.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)
        thumbnail_buffer = io.BytesIO()
        thumbnail.save(thumbnail_buffer, format='JPEG', quality=85)
        thumbnail_bytes = thumbnail_buffer.getvalue()
        
        # Paths en MinIO
        image_path = f"{image_id}.jpg"
        thumbnail_path = f"{image_id}_thumb.jpg"
        
        # Crear registro
        image_record = ImageRecord(
            id=image_id,
            filename=filename,
            description=description,
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            width=width,
            height=height,
            size=file_size,
            format=file_format,
            status=UploadStatus.COMPLETED,
            similarity_score=similarity_score,
            created_at=datetime.utcnow()
        )
        
        return image_record, image, original_bytes, thumbnail_bytes
    
    def validate_image_size(self, file_content: bytes) -> bool:
        """Validar tamaño de imagen"""
        return len(file_content) <= settings.max_upload_size
    
    def validate_image_format(self, file_content: bytes) -> bool:
        """Validar formato de imagen"""
        try:
            image = Image.open(io.BytesIO(file_content))
            # Formatos soportados
            return image.format in ['JPEG', 'JPG', 'PNG', 'GIF', 'BMP', 'WEBP']
        except Exception:
            return False


# Instancia singleton
image_processing_service = ImageProcessingService()