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
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm

# Configuración
API_URL = "http://localhost:8000/api/upload"
DATASET_DIR = "/app/dataset_final"  # Ruta dentro del contenedor

def upload_image(file_path):
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
    
    req = urllib.request.Request(API_URL, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        raise Exception(f"HTTP {e.code}: {error_body}")
    except Exception as e:
        raise Exception(f"Error de conexión: {str(e)}")

def main():
    print("=" * 60)
    print("VisionText - Carga Masiva vía API")
    print("=" * 60)
    print(f"Target API: {API_URL}")
    print(f"Dataset: {DATASET_DIR}")
    
    dataset_path = Path(DATASET_DIR)
    
    if not dataset_path.exists():
        print(f"❌ Error: El directorio {DATASET_DIR} no existe")
        return
        
    # Buscar imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [
        f for f in dataset_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    total_images = len(image_files)
    print(f"✓ Encontradas {total_images} imágenes")
    
    if total_images == 0:
        return

    successful = 0
    failed = 0
    
    print("\nIniciando carga...")
    
    with tqdm(total=total_images, unit="img") as pbar:
        for image_file in image_files:
            try:
                # Subir imagen (sin descripción para saltar validación)
                response = upload_image(str(image_file))
                
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
