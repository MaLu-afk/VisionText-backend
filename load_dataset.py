#!/usr/bin/env python3
"""
Script para cargar imágenes del dataset_final/ al sistema VisionText usando la API.
Garantiza sincronización en tiempo real del índice FAISS.
"""

import os
import sys
import time
import json
import mimetypes
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm

# Configuración (sobrescribible con variables de entorno)
# Por defecto usa dataset_final/ junto al script: /app/dataset_final dentro del contenedor,
# o ./dataset_final si se ejecuta desde la raíz del proyecto en el host
API_BASE_URL = os.getenv("LOAD_API_URL", "http://localhost:8000").rstrip("/")
DATASET_DIR = os.getenv(
    "LOAD_DATASET_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_final")
)
# /api/upload requiere token de admin; dentro del contenedor ya existen ADMIN_USERNAME/ADMIN_PASSWORD
ADMIN_USERNAME = os.getenv("LOAD_ADMIN_USERNAME", os.getenv("ADMIN_USERNAME"))
ADMIN_PASSWORD = os.getenv("LOAD_ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD"))


def request_json(req):
    """Ejecutar una petición y devolver (status, json)"""
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        try:
            return e.code, json.loads(error_body)
        except ValueError:
            return e.code, {"detail": error_body}
    except Exception as e:
        raise Exception(f"Error de conexión: {str(e)}")


def login():
    """Obtener token de admin"""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise Exception("Define LOAD_ADMIN_USERNAME y LOAD_ADMIN_PASSWORD (o ADMIN_USERNAME/ADMIN_PASSWORD)")
    
    body = urllib.parse.urlencode({"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}).encode()
    req = urllib.request.Request(f"{API_BASE_URL}/api/admin/login", data=body)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    
    code, data = request_json(req)
    if code != 200:
        raise Exception(f"Login fallido (HTTP {code}): {data.get('detail')}")
    return data["access_token"]


def get_existing_filenames(token):
    """Nombres de archivo ya cargados, para no duplicar si se vuelve a ejecutar el script"""
    filenames = set()
    offset = 0
    while True:
        req = urllib.request.Request(f"{API_BASE_URL}/api/admin/images?limit=500&offset={offset}")
        req.add_header('Authorization', f'Bearer {token}')
        code, data = request_json(req)
        if code != 200:
            raise Exception(f"No se pudo listar imágenes (HTTP {code}): {data.get('detail')}")
        
        filenames.update(img["filename"] for img in data["images"])
        offset += len(data["images"])
        if not data["images"] or offset >= data["total"]:
            return filenames


def upload_image(file_path, token):
    """
    Subir una imagen a través de la API usando multipart/form-data
    """
    filename = os.path.basename(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'application/octet-stream'
    
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    
    # Construir el cuerpo del mensaje multipart
    data = []
    
    # Parte del archivo
    data.append(f'--{boundary}'.encode())
    data.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode())
    data.append(f'Content-Type: {mime_type}'.encode())
    data.append(b'')
    
    with open(file_path, 'rb') as f:
        data.append(f.read())
        
    data.append(b'')
    
    # Finalizar body
    data.append(f'--{boundary}--'.encode())
    data.append(b'')
    
    body = b'\r\n'.join(data)
    
    req = urllib.request.Request(f"{API_BASE_URL}/api/upload", data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('Authorization', f'Bearer {token}')
    
    code, response = request_json(req)
    # 422 = rechazada por validación de contenido (el cuerpo trae el motivo)
    if code in (200, 422) and "success" in response:
        return response
    raise Exception(f"HTTP {code}: {response.get('detail')}")

def main():
    print("=" * 60)
    print("VisionText - Carga Masiva vía API")
    print("=" * 60)
    print(f"Target API: {API_BASE_URL}")
    print(f"Dataset: {DATASET_DIR}")
    
    token = login()
    existing = get_existing_filenames(token)
    print(f"✓ Autenticado. Imágenes ya cargadas: {len(existing)}")
    
    dataset_path = Path(DATASET_DIR)
    
    if not dataset_path.exists():
        print(f"❌ Error: El directorio {DATASET_DIR} no existe")
        return
        
    # Buscar imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = sorted(
        f for f in dataset_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions and f.name not in existing
    )
    
    total_images = len(image_files)
    print(f"✓ Encontradas {total_images} imágenes pendientes de cargar")
    
    if total_images == 0:
        return

    successful = 0
    failed = 0
    
    print("\nIniciando carga...")
    
    with tqdm(total=total_images, unit="img") as pbar:
        for image_file in image_files:
            try:
                # Subir imagen (sin descripción para saltar validación)
                response = upload_image(str(image_file), token)
                
                if response.get('success'):
                    successful += 1
                else:
                    pbar.write(f"⚠ Falló {image_file.name}: {response.get('message')}")
                    failed += 1
                    
            except Exception as e:
                pbar.write(f"❌ Error {image_file.name}: {str(e)}")
                failed += 1
            
            pbar.update(1)
            
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total: {total_images}")
    print(f"✓ Exitosas: {successful}")
    print(f"❌ Fallidas: {failed}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Interrumpido por usuario")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
