from minio import Minio
from minio.error import S3Error
import io
from typing import Optional
from app.config import settings


class MinIOService:
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self.bucket_images = settings.minio_bucket_images
        self.bucket_thumbnails = settings.minio_bucket_thumbnails
    
    def init_buckets(self):
        """Inicializar buckets de MinIO"""
        try:
            # Crear bucket de imágenes si no existe
            if not self.client.bucket_exists(self.bucket_images):
                self.client.make_bucket(self.bucket_images)
                print(f"Bucket '{self.bucket_images}' created")
            
            # Crear bucket de thumbnails si no existe
            if not self.client.bucket_exists(self.bucket_thumbnails):
                self.client.make_bucket(self.bucket_thumbnails)
                print(f"Bucket '{self.bucket_thumbnails}' created")
            
            print("MinIO buckets initialized")
        except S3Error as e:
            print(f"Error initializing MinIO buckets: {e}")
            raise
    
    def upload_file(self, bucket: str, object_name: str, data: bytes, 
                   content_type: str = "image/jpeg") -> bool:
        """Subir archivo a MinIO"""
        try:
            self.client.put_object(
                bucket,
                object_name,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type
            )
            return True
        except S3Error as e:
            print(f"Error uploading file to MinIO: {e}")
            return False
    
    def upload_image(self, image_id: str, data: bytes, 
                    content_type: str = "image/jpeg") -> Optional[str]:
        """Subir imagen original"""
        object_name = f"{image_id}.jpg"
        if self.upload_file(self.bucket_images, object_name, data, content_type):
            return object_name
        return None
    
    def upload_thumbnail(self, image_id: str, data: bytes) -> Optional[str]:
        """Subir thumbnail"""
        object_name = f"{image_id}_thumb.jpg"
        if self.upload_file(self.bucket_thumbnails, object_name, data, "image/jpeg"):
            return object_name
        return None
    
    def get_file(self, bucket: str, object_name: str) -> Optional[bytes]:
        """Obtener archivo de MinIO"""
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            print(f"Error getting file from MinIO: {e}")
            return None
    
    def get_image(self, object_name: str) -> Optional[bytes]:
        """Obtener imagen original"""
        return self.get_file(self.bucket_images, object_name)
    
    def get_thumbnail(self, object_name: str) -> Optional[bytes]:
        """Obtener thumbnail"""
        return self.get_file(self.bucket_thumbnails, object_name)
    
    def delete_file(self, bucket: str, object_name: str) -> bool:
        """Eliminar archivo de MinIO"""
        try:
            self.client.remove_object(bucket, object_name)
            return True
        except S3Error as e:
            print(f"Error deleting file from MinIO: {e}")
            return False
    
    def delete_image(self, object_name: str) -> bool:
        """Eliminar imagen original"""
        return self.delete_file(self.bucket_images, object_name)
    
    def delete_thumbnail(self, object_name: str) -> bool:
        """Eliminar thumbnail"""
        return self.delete_file(self.bucket_thumbnails, object_name)
    
    def get_presigned_url(self, bucket: str, object_name: str) -> Optional[str]:
        """Obtener URL prefirmada para acceso temporal"""
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket,
                object_name,
                expires=timedelta(hours=1)
            )
            return url
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            return None


# Instancia singleton
minio_service = MinIOService()