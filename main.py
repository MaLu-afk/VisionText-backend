from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Response, Depends, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import time
from typing import List, Optional, Tuple

from app.config import settings
from app.models import (
    TextSearchRequest, SearchResponse, SearchResult,
    UploadResponse, ImageMetadata, UploadStatus,
)
from app.database import db_service
from app.services.clip import clip_service
from app.services.faiss import faiss_service
from app.services.image import image_processing_service
from app.services.minio import minio_service
from app.services.validation import content_validator
from app.services.auth import verify_admin_credentials, create_access_token, verify_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializar servicios al inicio de la aplicación"""
    print("Starting VisionText Backend...")

    # Inicializar base de datos
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
    print("Shutting down...")
    await faiss_service.save_index()


# Crear aplicación FastAPI
app = FastAPI(
    title="VisionText Backend",
    description="API para búsqueda de imágenes con CLIP y validación de similitud",
    version="2.1.0",
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

# Security (auto_error=False para responder 401, no 403, cuando falta el token)
security = HTTPBearer(auto_error=False)


# Dependency para verificar autenticación de admin
async def verify_admin_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verificar que el token sea válido"""
    payload = verify_token(credentials.credentials) if credentials else None

    if payload is None or payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autorizado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def read_validated_image(image: UploadFile) -> bytes:
    """Leer un archivo subido validando tipo, tamaño y formato"""
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    content = await image.read()

    if not image_processing_service.validate_image_size(content):
        raise HTTPException(
            status_code=413,
            detail=f"Imagen demasiado grande. Máximo {settings.max_upload_size / 1024 / 1024:.0f}MB"
        )

    if not image_processing_service.validate_image_format(content):
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

    return content


async def build_search_results(search_results: List[Tuple[str, float]]) -> List[SearchResult]:
    """Convertir resultados de FAISS en respuestas con la información de cada imagen"""
    results = []
    for image_id, similarity in search_results:
        image_record = await db_service.get_image(image_id)
        if image_record:
            results.append(SearchResult(
                id=image_record.id,
                filename=image_record.filename,
                url=f"{settings.base_url}/api/images/{image_record.id}",
                thumbnail_url=f"{settings.base_url}/api/thumbnails/{image_record.id}",
                similarity_score=similarity,
                description=image_record.description,
                metadata=ImageMetadata(
                    width=image_record.width,
                    height=image_record.height,
                    size=image_record.size,
                    format=image_record.format
                )
            ))
    return results


def rejected_upload(response: UploadResponse) -> JSONResponse:
    """Subida rechazada por validación de contenido: 422 con el detalle del rechazo"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response.model_dump(mode="json"),
    )


@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "VisionText Backend API",
        "status": "running",
        "features": [
            "Text-to-Image search",
            "Image-to-Image search",
            "Image-Text similarity validation"
        ]
    }


@app.get("/api/health")
async def health_check(response: Response):
    """Verificar estado del sistema (503 si alguna dependencia no está disponible)"""
    database_ok = await db_service.ping()
    minio_ok = await run_in_threadpool(minio_service.ping)
    clip_ok = clip_service.model is not None
    faiss_ok = faiss_service.index is not None

    healthy = database_ok and minio_ok and clip_ok and faiss_ok
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if healthy else "unhealthy",
        "images_in_db": await db_service.get_image_count() if database_ok else None,
        "vectors_in_index": faiss_service.get_count(),
        "model": f"{settings.clip_model} ({settings.clip_pretrained})",
        "similarity_threshold": settings.similarity_threshold,
        "services": {
            "postgresql": "connected" if database_ok else "unavailable",
            "minio": "connected" if minio_ok else "unavailable",
            "clip": "loaded" if clip_ok else "not loaded",
            "faiss": "ready" if faiss_ok else "not ready"
        }
    }


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    responses={422: {"model": UploadResponse, "description": "Imagen rechazada por validación de contenido"}},
)
async def upload_image(
    image: UploadFile = File(...),
    description: Optional[str] = Form(None),
    admin: dict = Depends(verify_admin_token)
):
    """
    Subir nueva imagen al sistema con validación de similitud imagen-texto (requiere admin)

    Si se proporciona una descripción, se valida que la imagen y el texto
    sean consistentes (similitud >= threshold). Si no pasan la validación,
    se rechaza la subida con 422.
    """
    image_record = None
    try:
        # VALIDACIÓN INICIAL (tipo de archivo, tamaño, formato)
        content = await read_validated_image(image)
        description = description.strip() if description and description.strip() else None

        # CAPA 1: VALIDACIÓN DE CONTENIDO TEXTUAL (LISTA NEGRA)
        is_text_valid, banned_words = content_validator.validate_image_metadata(
            image.filename, description
        )

        if not is_text_valid:
            return rejected_upload(UploadResponse(
                success=False,
                message=f"Contenido no permitido detectado: {', '.join(banned_words)}",
                status=UploadStatus.REJECTED,
                banned_words=banned_words,
                rejection_reason="banned_words"
            ))

        # CAPA 2: VALIDACIÓN DE SIMILITUD IMAGEN-TEXTO (CLIP)
        similarity_score = None
        if description:
            print(f"Validating image-text similarity for: {image.filename}")

            is_valid, similarity_score = await run_in_threadpool(
                clip_service.validate_image_text_similarity,
                content,
                description,
                settings.similarity_threshold
            )

            print(f"Similarity score: {similarity_score:.4f} (threshold: {settings.similarity_threshold})")

            if not is_valid:
                return rejected_upload(UploadResponse(
                    success=False,
                    message=(
                        f"La imagen y la descripción no son consistentes. "
                        f"Similitud: {similarity_score:.2%} "
                        f"(mínimo requerido: {settings.similarity_threshold:.2%}). "
                        f"Por favor proporciona una descripción más precisa o una imagen diferente."
                    ),
                    status=UploadStatus.REJECTED,
                    similarity_score=similarity_score,
                    required_similarity=settings.similarity_threshold,
                    rejection_reason="low_similarity"
                ))

        # SI PASÓ TODAS LAS VALIDACIONES, PROCESAR IMAGEN
        image_record, pil_image, original_bytes, thumbnail_bytes = await run_in_threadpool(
            image_processing_service.process_image,
            content,
            image.filename,
            description,
            similarity_score
        )

        # Extraer embedding antes de persistir nada
        embedding = await run_in_threadpool(clip_service.encode_image, pil_image)

        # Subir a MinIO
        image_saved = await run_in_threadpool(minio_service.upload_image, image_record.id, original_bytes)
        thumbnail_saved = await run_in_threadpool(minio_service.upload_thumbnail, image_record.id, thumbnail_bytes)
        if not image_saved or not thumbnail_saved:
            raise RuntimeError("MinIO upload failed")

        # Guardar en base de datos
        if not await db_service.insert_image(image_record):
            raise RuntimeError("Database insert failed")

        # Agregar al índice FAISS
        await faiss_service.add_embedding(image_record.id, embedding)
        await faiss_service.save_index()

        message = f"Imagen '{image.filename}' subida exitosamente"
        if similarity_score is not None:
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
        # No dejar archivos huérfanos en MinIO si falló un paso posterior
        if image_record is not None and await db_service.get_image(image_record.id) is None:
            await run_in_threadpool(minio_service.delete_image, image_record.image_path)
            await run_in_threadpool(minio_service.delete_thumbnail, image_record.thumbnail_path)
        raise HTTPException(status_code=500, detail="Error interno al procesar la imagen")


@app.post("/api/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """Buscar imágenes por texto"""
    try:
        start_time = time.time()

        # Usar threshold del request o default de settings (.env)
        threshold = request.threshold if request.threshold is not None else settings.text_search_threshold

        # Extraer embedding del texto
        text_embedding = await run_in_threadpool(clip_service.encode_text, request.query)

        # Buscar en FAISS
        search_results = await faiss_service.search(
            text_embedding,
            k=request.limit,
            threshold=threshold
        )

        results = await build_search_results(search_results)

        return SearchResponse(
            results=results,
            total_count=len(results),
            query_time=time.time() - start_time
        )

    except Exception as e:
        print(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la búsqueda")


@app.post("/api/search/image", response_model=SearchResponse)
async def search_by_image(
    image: UploadFile = File(...),
    limit: int = Form(default=20, ge=1, le=100),
    threshold: Optional[float] = Form(default=None, ge=0.0, le=1.0)
):
    """Buscar imágenes similares a una imagen"""
    content = await read_validated_image(image)

    try:
        start_time = time.time()

        # Usar threshold del request o default de settings (.env)
        search_threshold = threshold if threshold is not None else settings.image_search_threshold

        # Extraer embedding
        image_embedding = await run_in_threadpool(clip_service.encode_image, content)

        # Buscar en FAISS
        search_results = await faiss_service.search(
            image_embedding,
            k=limit,
            threshold=search_threshold
        )

        results = await build_search_results(search_results)

        return SearchResponse(
            results=results,
            total_count=len(results),
            query_time=time.time() - start_time
        )

    except Exception as e:
        print(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la búsqueda")


# Las imágenes nunca cambian para un mismo ID: se pueden cachear en el navegador
IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400"}


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Obtener imagen original de MinIO"""
    image_record = await db_service.get_image(image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    image_bytes = await run_in_threadpool(minio_service.get_image, image_record.image_path)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Archivo de imagen no encontrado")

    return Response(content=image_bytes, media_type="image/jpeg", headers=IMAGE_CACHE_HEADERS)


@app.get("/api/thumbnails/{image_id}")
async def get_thumbnail(image_id: str):
    """Obtener thumbnail de imagen de MinIO"""
    image_record = await db_service.get_image(image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    thumbnail_bytes = await run_in_threadpool(minio_service.get_thumbnail, image_record.thumbnail_path)
    if not thumbnail_bytes:
        raise HTTPException(status_code=404, detail="Thumbnail no encontrado")

    return Response(content=thumbnail_bytes, media_type="image/jpeg", headers=IMAGE_CACHE_HEADERS)


# ==================== ADMIN ENDPOINTS ====================

@app.post("/api/admin/login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    """Login de administrador"""
    if not verify_admin_credentials(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas"
        )

    # Crear token
    access_token = create_access_token(
        data={"sub": username, "role": "admin"}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.get("/api/admin/images")
async def get_all_images_admin(
    admin: dict = Depends(verify_admin_token),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
):
    """Obtener todas las imágenes del sistema (requiere autenticación de admin)"""
    images = await db_service.get_all_images_paginated(limit, offset)
    total = await db_service.get_image_count()

    return {
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "description": img.description,
                "thumbnail_url": f"{settings.base_url}/api/thumbnails/{img.id}",
                "metadata": {
                    "width": img.width,
                    "height": img.height,
                    "size": img.size,
                    "format": img.format,
                    "created_at": img.created_at.isoformat() if img.created_at else None,
                    "similarity_score": img.similarity_score
                },
                "vector_id": img.id  # El ID de la imagen es el mismo que el ID del vector
            }
            for img in images
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.delete("/api/admin/images/{image_id}")
async def delete_image_admin(
    image_id: str,
    admin: dict = Depends(verify_admin_token)
):
    """Eliminar imagen completamente del sistema (requiere autenticación de admin)"""
    image_record = await db_service.get_image(image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    try:
        # 1. Eliminar de PostgreSQL PRIMERO
        # Esto asegura que get_all_images() no incluya esta imagen al reconstruir FAISS
        if not await db_service.delete_image(image_id):
            raise RuntimeError("Database delete failed")

        # 2. Eliminar de FAISS (vector e ID, manteniendo el índice alineado)
        await faiss_service.remove_embedding(image_id)
        await faiss_service.save_index()

        # 3. Eliminar de MinIO AL FINAL
        # Los archivos se eliminan después de actualizar el índice
        await run_in_threadpool(minio_service.delete_image, image_record.image_path)
        await run_in_threadpool(minio_service.delete_thumbnail, image_record.thumbnail_path)

        return {
            "success": True,
            "message": f"Imagen '{image_record.filename}' eliminada completamente",
            "deleted_from": ["postgresql", "faiss", "minio"]
        }

    except Exception as e:
        print(f"Error deleting image {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al eliminar la imagen"
        )


@app.get("/api/admin/storage")
async def storage_status_admin(admin: dict = Depends(verify_admin_token)):
    """Verificar conexión y contenido de MinIO (requiere autenticación de admin)"""
    def count_objects(bucket: str) -> int:
        return sum(1 for _ in minio_service.client.list_objects(bucket, recursive=True))

    try:
        buckets = await run_in_threadpool(minio_service.client.list_buckets)
        return {
            "status": "connected",
            "buckets": [bucket.name for bucket in buckets],
            "images_bucket_count": await run_in_threadpool(count_objects, settings.minio_bucket_images),
            "thumbnails_bucket_count": await run_in_threadpool(count_objects, settings.minio_bucket_thumbnails)
        }
    except Exception as e:
        print(f"Error checking MinIO storage: {e}")
        raise HTTPException(status_code=503, detail="MinIO no disponible")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
