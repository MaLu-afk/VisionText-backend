#!/usr/bin/env python3
"""
Carga dataset_final/ escribiendo DIRECTAMENTE en MinIO y PostgreSQL, sin pasar por la API.

Pensado para cuando MinIO corre en tu máquina (expuesto a Railway con ngrok): las
imágenes van a tu MinIO local sin cruzar el túnel, y los embeddings CLIP se calculan en
tu máquina y se guardan en PostgreSQL. El backend en Railway reconstruye su índice FAISS
con esos embeddings al reiniciarse, sin descargar ninguna imagen.

Uso (desde la raíz del proyecto, con docker compose):
    docker compose run --rm visiontext-backend python load_dataset_local.py

Usa la configuración normal del backend: POSTGRES_URL, MINIO_ENDPOINT, MINIO_ACCESS_KEY,
MINIO_SECRET_KEY, CLIP_MODEL... Es seguro volver a ejecutarlo: se salta los archivos ya cargados.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from app.config import settings
from app.database import db_service
from app.services.clip import clip_service
from app.services.image import image_processing_service
from app.services.minio import minio_service
from app.services.validation import content_validator

DATASET_DIR = os.getenv(
    "LOAD_DATASET_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_final")
)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


async def load_image(image_file: Path) -> str:
    """Procesar y guardar una imagen. Devuelve None si se cargó, o el motivo si se saltó."""
    content = image_file.read_bytes()

    # Mismas validaciones que /api/upload (sin descripción, no hay validación de similitud)
    if not image_processing_service.validate_image_size(content):
        return "imagen demasiado grande"
    if not image_processing_service.validate_image_format(content):
        return "formato no soportado"
    is_valid, banned_words = content_validator.validate_image_metadata(image_file.name)
    if not is_valid:
        return f"contenido no permitido ({', '.join(banned_words)})"

    image_record, pil_image, original_bytes, thumbnail_bytes = \
        image_processing_service.process_image(content, image_file.name)
    embedding = clip_service.encode_image(pil_image)

    if not minio_service.upload_image(image_record.id, original_bytes) or \
       not minio_service.upload_thumbnail(image_record.id, thumbnail_bytes):
        raise RuntimeError("no se pudo subir a MinIO")

    if not await db_service.insert_image(image_record, embedding):
        # No dejar archivos huérfanos en MinIO
        minio_service.delete_image(image_record.image_path)
        minio_service.delete_thumbnail(image_record.thumbnail_path)
        raise RuntimeError("no se pudo guardar en PostgreSQL")

    return None


async def main():
    print("=" * 60)
    print("VisionText - Carga directa del dataset (sin API)")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"MinIO:   {settings.minio_endpoint}")
    print(f"Modelo:  {settings.clip_model} ({settings.clip_pretrained})")

    dataset_path = Path(DATASET_DIR)
    if not dataset_path.exists():
        print(f"❌ El directorio {DATASET_DIR} no existe")
        sys.exit(1)

    await db_service.init_db()
    minio_service.init_buckets()
    clip_service.load_model()

    existing = await db_service.get_all_filenames()
    image_files = sorted(
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.name not in existing
    )
    print(f"✓ Ya cargadas: {len(existing)}. Pendientes: {len(image_files)}")
    if not image_files:
        return

    successful = skipped = failed = 0
    start_time = time.time()

    with tqdm(total=len(image_files), unit="img") as pbar:
        for image_file in image_files:
            try:
                reason = await load_image(image_file)
                if reason:
                    pbar.write(f"⚠ Saltada {image_file.name}: {reason}")
                    skipped += 1
                else:
                    successful += 1
            except Exception as e:
                pbar.write(f"❌ Error {image_file.name}: {e}")
                failed += 1
            pbar.update(1)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"✓ Cargadas:  {successful}")
    print(f"⚠ Saltadas:  {skipped}")
    print(f"❌ Fallidas: {failed}")
    print(f"Tiempo: {time.time() - start_time:.0f} s")
    print(f"Imágenes en PostgreSQL: {await db_service.get_image_count()}")
    print("\nSiguiente paso: reinicia el backend en Railway para que reconstruya el índice FAISS.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠ Interrumpido. Puedes volver a ejecutarlo: se salta lo ya cargado.")
