import os
import sys

# Configuración mínima para importar la app sin servicios reales.
# Ningún test abre conexiones: el engine de SQLAlchemy y el cliente de MinIO son perezosos.
os.environ.setdefault("POSTGRES_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("MINIO_ACCESS_KEY", "test")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "test-password")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
