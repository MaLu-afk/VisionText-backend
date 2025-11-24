from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import io
from typing import Optional

from app.config import settings
from app.models import (
    TextSearchRequest, SearchResponse, SearchResult, 
    UploadResponse, ImageMetadata, UploadStatus,
    SimilarityValidationResult
)
from app.database import db_service
from app.services.clip import clip_service
from app.services.faiss import faiss_service
from app.services.image import image_processing_service
from app.services.minio import minio_service
from app.services.redis import redis_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializar servicios al inicio de la aplicación"""
    print("Starting VisionText Backend...")
    
    # Conectar a Redis
    await redis_service.connect()
    
    # Inicializar base de datos PostgreSQL
    await db_service.init_db()
    
    # Inicializar buckets de MinIO
    minio_service.init_buckets()
    
    # Cargar modelo CLIP
    clip_service.load_model()
    
    # Inicializar o cargar índice FAISS
    if not faiss_service.load_index():
        print("No existing FAISS index found, creating new one...")
        dimension = clip_service.get_embedding_dimension()
        faiss_service.initialize_index(dimension)
        
        # Cargar imágenes existentes si las hay
        images = await db_service.get_all_images()
        if images:
            print(f"Rebuilding index with {len(images)} existing images...")
            embeddings_data = []
            
            for img in images:
                try:
                    # Obtener imagen de MinIO
                    image_bytes = minio_service.get_image(img.image_path)
                    if image_bytes:
                        embedding = clip_service.encode_image(image_bytes)
                        embeddings_data.append((img.id, embedding))
                except Exception as e:
                    print(f"Error processing {img.filename}: {e}")
            
            if embeddings_data:
                await faiss_service.rebuild_index(embeddings_data)
                await faiss_service.save_index()
    
    print("VisionText Backend ready!")
    yield
    
    # Cleanup
    print("Shutting down VisionText Backend...")
    await faiss_service.save_index()
    await redis_service.disconnect()


# Crear aplicación FastAPI
app = FastAPI(
    title="VisionText Backend",
    description="API para búsqueda de imágenes con CLIP y validación de similitud",
    version="2.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "VisionText Backend API v2",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Text-to-Image search",
            "Image-to-Image search",
            "Image-Text similarity validation"
        ]
    }


@app.get("/api/health")
async def health_check():
    """Verificar estado del sistema"""
    image_count = await db_service.get_image_count()
    faiss_count = faiss_service.get_count()
    
    return {
        "status": "healthy",
        "images_in_db": image_count,
        "vectors_in_index": faiss_count,
        "model": f"{settings.clip_model} ({settings.clip_pretrained})",
        "similarity_threshold": settings.similarity_threshold,
        "services": {
            "postgresql": "connected",
            "redis": "connected",
            "minio": "connected",
            "clip": "loaded",
            "faiss": "ready"
        }
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(
    image: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Subir nueva imagen al sistema con validación de similitud imagen-texto
    
    Si se proporciona una descripción, se valida que la imagen y el texto
    sean consistentes (similitud >= threshold). Si no pasan la validación,
    se rechaza la subida.
    """
    try:
        # Validar tipo de archivo
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer contenido
        content = await image.read()
        
        # Validar tamaño
        if not image_processing_service.validate_image_size(content):
            raise HTTPException(
                status_code=400, 
                detail=f"Imagen demasiado grande. Máximo {settings.max_upload_size / 1024 / 1024}MB"
            )
        
        # Validar formato
        if not image_processing_service.validate_image_format(content):
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado")
        
        # VALIDACIÓN DE SIMILITUD IMAGEN-TEXTO
        similarity_score = None
        if description and description.strip():
            print(f"Validating image-text similarity for: {image.filename}")
            
            is_valid, similarity_score = clip_service.validate_image_text_similarity(
                content,
                description,
                threshold=settings.similarity_threshold
            )
            
            print(f"Similarity score: {similarity_score:.4f} (threshold: {settings.similarity_threshold})")
            
            if not is_valid:
                return UploadResponse(
                    success=False,
                    message=(
                        f"La imagen y la descripción no son consistentes. "
                        f"Similitud: {similarity_score:.2%} "
                        f"(mínimo requerido: {settings.similarity_threshold:.2%}). "
                        f"Por favor proporciona una descripción más precisa o una imagen diferente."
                    ),
                    status=UploadStatus.REJECTED,
                    similarity_score=similarity_score,
                    required_similarity=settings.similarity_threshold
                )
        
        # Procesar imagen
        image_record, pil_image, original_bytes, thumbnail_bytes = \
            image_processing_service.process_image(
                content, 
                image.filename,
                description=description.strip() if description else None,
                similarity_score=similarity_score
            )
        
        # Subir a MinIO
        minio_service.upload_image(image_record.id, original_bytes)
        minio_service.upload_thumbnail(image_record.id, thumbnail_bytes)
        
        # Extraer embedding
        embedding = clip_service.encode_image(pil_image)
        
        # Guardar en base de datos
        success = await db_service.insert_image(image_record)
        if not success:
            raise HTTPException(status_code=500, detail="Error al guardar en base de datos")
        
        # Agregar al índice FAISS
        await faiss_service.add_embedding(image_record.id, embedding)
        await faiss_service.save_index()
        
        message = f"Imagen '{image.filename}' subida exitosamente"
        if similarity_score:
            message += f" (similitud con descripción: {similarity_score:.2%})"
        
        return UploadResponse(
            success=True,
            message=message,
            image_id=image_record.id,
            filename=image.filename,
            status=UploadStatus.COMPLETED,
            similarity_score=similarity_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {str(e)}")


@app.post("/api/validate-similarity", response_model=SimilarityValidationResult)
async def validate_similarity(
    image: UploadFile = File(...),
    description: str = Form(...)
):
    """
    Endpoint para validar la similitud entre una imagen y su descripción
    sin subirla al sistema. Útil para pre-validación en el frontend.
    """
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        content = await image.read()
        
        is_valid, similarity_score = clip_service.validate_image_text_similarity(
            content,
            description,
            threshold=settings.similarity_threshold
        )
        
        message = "Válido" if is_valid else "Similitud insuficiente"
        
        return SimilarityValidationResult(
            is_valid=is_valid,
            similarity_score=similarity_score,
            threshold=settings.similarity_threshold,
            message=message
        )
        
    except Exception as e:
        print(f"Error validating similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """Buscar imágenes por texto"""
    try:
        # Verificar caché
        cached_result = await redis_service.get_search_cache(request.query, "text")
        if cached_result:
            print("Returning cached results")
            return SearchResponse(**cached_result)
        
        start_time = time.time()
        
        # Extraer embedding del texto
        text_embedding = clip_service.encode_text(request.query)
        
        # Buscar en FAISS
        search_results = await faiss_service.search(
            text_embedding, 
            k=request.limit,
            threshold=request.threshold
        )
        
        # Obtener información de las imágenes
        results = []
        for image_id, similarity in search_results:
            image_record = await db_service.get_image(image_id)
            if image_record:
                results.append(SearchResult(
                    id=image_record.id,
                    filename=image_record.filename,
                    url=f"/api/images/{image_record.id}",
                    thumbnail_url=f"/api/thumbnails/{image_record.id}",
                    similarity_score=similarity,
                    description=image_record.description,
                    metadata=ImageMetadata(
                        width=image_record.width,
                        height=image_record.height,
                        size=image_record.size,
                        format=image_record.format
                    )
                ))
        
        query_time = time.time() - start_time
        
        response = SearchResponse(
            results=results,
            total_count=len(results),
            query_time=query_time
        )
        
        # Guardar en caché
        await redis_service.set_search_cache(
            request.query, 
            response.model_dump(), 
            "text"
        )
        
        return response
        
    except Exception as e:
        print(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@app.post("/api/search/image", response_model=SearchResponse)
async def search_by_image(
    image: UploadFile = File(...),
    limit: int = Form(default=20),
    threshold: float = Form(default=0.1)
):
    """Buscar imágenes similares a una imagen"""
    try:
        start_time = time.time()
        
        # Validar tipo de archivo
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer imagen
        content = await image.read()
        
        # Extraer embedding
        image_embedding = clip_service.encode_image(content)
        
        # Buscar en FAISS
        search_results = await faiss_service.search(
            image_embedding,
            k=limit,
            threshold=threshold
        )
        
        # Obtener información de las imágenes
        results = []
        for image_id, similarity in search_results:
            image_record = await db_service.get_image(image_id)
            if image_record:
                results.append(SearchResult(
                    id=image_record.id,
                    filename=image_record.filename,
                    url=f"/api/images/{image_record.id}",
                    thumbnail_url=f"/api/thumbnails/{image_record.id}",
                    similarity_score=similarity,
                    description=image_record.description,
                    metadata=ImageMetadata(
                        width=image_record.width,
                        height=image_record.height,
                        size=image_record.size,
                        format=image_record.format
                    )
                ))
        
        query_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_time=query_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Obtener imagen original de MinIO"""
    image_record = await db_service.get_image(image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    image_bytes = minio_service.get_image(image_record.image_path)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Archivo de imagen no encontrado")
    
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/thumbnails/{image_id}")
async def get_thumbnail(image_id: str):
    """Obtener thumbnail de imagen de MinIO"""
    image_record = await db_service.get_image(image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    thumbnail_bytes = minio_service.get_thumbnail(image_record.thumbnail_path)
    if not thumbnail_bytes:
        raise HTTPException(status_code=404, detail="Thumbnail no encontrado")
    
    return Response(content=thumbnail_bytes, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )