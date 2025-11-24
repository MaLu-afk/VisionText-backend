from celery import Celery
from app.config import settings
from app.services.clip import clip_service
from app.services.faiss import faiss_service
from app.database import db_service
from app.services.minio import minio_service
import asyncio

# Crear aplicación Celery
celery_app = Celery(
    'visiontext',
    broker=settings.celery_broker,
    backend=settings.celery_backend
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


# Inicializar servicios al inicio del worker
@celery_app.on_after_configure.connect
def setup_services(sender, **kwargs):
    """Inicializar servicios cuando el worker arranca"""
    print("Initializing Celery worker services...")
    
    # Inicializar base de datos
    loop = asyncio.get_event_loop()
    loop.run_until_complete(db_service.init_db())
    
    # Inicializar MinIO
    minio_service.init_buckets()
    
    # Cargar modelo CLIP
    clip_service.load_model()
    
    # Cargar índice FAISS
    if not faiss_service.load_index():
        dimension = clip_service.get_embedding_dimension()
        faiss_service.initialize_index(dimension)
    
    print("Celery worker ready!")


@celery_app.task(name='process_image_upload')
def process_image_upload(image_id: str):
    """
    Procesar imagen subida de forma asíncrona
    
    Este task puede ser usado para procesamiento pesado como:
    - Extracción de embeddings
    - Indexación en FAISS
    - Post-procesamiento de imágenes
    - Generación de variantes
    """
    try:
        print(f"Processing image upload: {image_id}")
        
        # Obtener imagen de la base de datos
        loop = asyncio.get_event_loop()
        image_record = loop.run_until_complete(db_service.get_image(image_id))
        
        if not image_record:
            print(f"Image {image_id} not found in database")
            return {'success': False, 'error': 'Image not found'}
        
        # Obtener imagen de MinIO
        image_bytes = minio_service.get_image(image_record.image_path)
        if not image_bytes:
            print(f"Image file {image_id} not found in MinIO")
            return {'success': False, 'error': 'Image file not found'}
        
        # Extraer embedding
        embedding = clip_service.encode_image(image_bytes)
        
        # Agregar al índice FAISS
        loop.run_until_complete(faiss_service.add_embedding(image_id, embedding))
        loop.run_until_complete(faiss_service.save_index())
        
        print(f"Image {image_id} processed successfully")
        return {'success': True, 'image_id': image_id}
        
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return {'success': False, 'error': str(e)}


@celery_app.task(name='rebuild_faiss_index')
def rebuild_faiss_index():
    """
    Reconstruir índice FAISS desde cero
    
    Útil para:
    - Mantenimiento periódico
    - Después de eliminaciones masivas
    - Migración de índices
    """
    try:
        print("Rebuilding FAISS index...")
        
        loop = asyncio.get_event_loop()
        
        # Obtener todas las imágenes
        images = loop.run_until_complete(db_service.get_all_images())
        
        if not images:
            print("No images found to rebuild index")
            return {'success': True, 'count': 0}
        
        print(f"Found {len(images)} images to index")
        
        # Extraer embeddings
        embeddings_data = []
        for img in images:
            try:
                image_bytes = minio_service.get_image(img.image_path)
                if image_bytes:
                    embedding = clip_service.encode_image(image_bytes)
                    embeddings_data.append((img.id, embedding))
            except Exception as e:
                print(f"Error processing {img.filename}: {e}")
        
        # Reconstruir índice
        if embeddings_data:
            loop.run_until_complete(faiss_service.rebuild_index(embeddings_data))
            loop.run_until_complete(faiss_service.save_index())
        
        print(f"Index rebuilt with {len(embeddings_data)} vectors")
        return {'success': True, 'count': len(embeddings_data)}
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        return {'success': False, 'error': str(e)}


@celery_app.task(name='cleanup_old_images')
def cleanup_old_images(days: int = 30):
    """
    Limpiar imágenes antiguas
    
    Útil para:
    - Liberar espacio
    - Cumplir políticas de retención
    """
    try:
        from datetime import datetime, timedelta
        
        print(f"Cleaning up images older than {days} days...")
        
        # Esta es una tarea de ejemplo
        # Implementar lógica de limpieza según necesidades
        
        return {'success': True, 'message': 'Cleanup completed'}
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return {'success': False, 'error': str(e)}