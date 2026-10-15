FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch solo CPU: evita descargar CUDA (varios GB) en un servidor sin GPU.
# requirements.txt fija torch==2.5.1, que esta build (2.5.1+cpu) ya satisface.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install -r requirements.txt

# Copiar código
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/faiss_indices /app/model_cache

# Exponer puerto
EXPOSE 8000

# Railway inyecta PORT; en local se usa 8000. --proxy-headers para respetar
# el esquema https del proxy de Railway.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
