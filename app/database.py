from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Enum as SQLEnum
from datetime import datetime
from typing import List, Optional
import enum

from app.config import settings
from app.models import ImageRecord, UploadStatus


# SQLAlchemy Base
Base = declarative_base()


# Tabla de imágenes
class ImageDB(Base):
    __tablename__ = "images"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    image_path = Column(String, nullable=False)
    thumbnail_path = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    size = Column(Integer, nullable=False)
    format = Column(String, nullable=False)
    status = Column(String, default="completed")
    similarity_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)


# Database Service
class DatabaseService:
    def __init__(self):
        self.engine = create_async_engine(
            settings.postgres_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_db(self):
        """Inicializar base de datos y crear tablas"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_session(self) -> AsyncSession:
        """Obtener sesión de base de datos"""
        async with self.async_session() as session:
            yield session
    
    async def insert_image(self, image_record: ImageRecord) -> bool:
        """Insertar nuevo registro de imagen"""
        try:
            async with self.async_session() as session:
                db_image = ImageDB(
                    id=image_record.id,
                    filename=image_record.filename,
                    description=image_record.description,
                    image_path=image_record.image_path,
                    thumbnail_path=image_record.thumbnail_path,
                    width=image_record.width,
                    height=image_record.height,
                    size=image_record.size,
                    format=image_record.format,
                    status=image_record.status.value if isinstance(image_record.status, UploadStatus) else image_record.status,
                    similarity_score=image_record.similarity_score,
                    created_at=image_record.created_at
                )
                session.add(db_image)
                await session.commit()
                return True
        except Exception as e:
            print(f"Error inserting image: {e}")
            return False
    
    async def get_image(self, image_id: str) -> Optional[ImageRecord]:
        """Obtener registro de imagen por ID"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(ImageDB).where(ImageDB.id == image_id)
                )
                db_image = result.scalar_one_or_none()
                
                if db_image:
                    return ImageRecord(
                        id=db_image.id,
                        filename=db_image.filename,
                        description=db_image.description,
                        image_path=db_image.image_path,
                        thumbnail_path=db_image.thumbnail_path,
                        width=db_image.width,
                        height=db_image.height,
                        size=db_image.size,
                        format=db_image.format,
                        status=UploadStatus(db_image.status),
                        similarity_score=db_image.similarity_score,
                        created_at=db_image.created_at,
                        updated_at=db_image.updated_at
                    )
                return None
        except Exception as e:
            print(f"Error getting image: {e}")
            return None
    
    async def get_all_images(self, limit: Optional[int] = None) -> List[ImageRecord]:
        """Obtener todos los registros de imágenes"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import select
                query = select(ImageDB).where(
                    ImageDB.status == "completed"
                ).order_by(ImageDB.created_at.desc())
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                db_images = result.scalars().all()
                
                return [
                    ImageRecord(
                        id=img.id,
                        filename=img.filename,
                        description=img.description,
                        image_path=img.image_path,
                        thumbnail_path=img.thumbnail_path,
                        width=img.width,
                        height=img.height,
                        size=img.size,
                        format=img.format,
                        status=UploadStatus(img.status),
                        similarity_score=img.similarity_score,
                        created_at=img.created_at,
                        updated_at=img.updated_at
                    )
                    for img in db_images
                ]
        except Exception as e:
            print(f"Error getting all images: {e}")
            return []
    
    async def update_image_status(self, image_id: str, status: UploadStatus) -> bool:
        """Actualizar estado de una imagen"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import select, update
                await session.execute(
                    update(ImageDB)
                    .where(ImageDB.id == image_id)
                    .values(status=status.value, updated_at=datetime.utcnow())
                )
                await session.commit()
                return True
        except Exception as e:
            print(f"Error updating image status: {e}")
            return False
    
    async def delete_image(self, image_id: str) -> bool:
        """Eliminar registro de imagen"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import delete
                await session.execute(
                    delete(ImageDB).where(ImageDB.id == image_id)
                )
                await session.commit()
                return True
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
    
    async def get_image_count(self) -> int:
        """Obtener conteo total de imágenes completadas"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import select, func
                result = await session.execute(
                    select(func.count(ImageDB.id)).where(
                        ImageDB.status == "completed"
                    )
                )
                return result.scalar() or 0
        except Exception as e:
            print(f"Error getting image count: {e}")
            return 0
    
    async def get_all_images_paginated(
        self, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[ImageRecord]:
        """Obtener imágenes con paginación"""
        try:
            async with self.async_session() as session:
                from sqlalchemy import select
                query = select(ImageDB).where(
                    ImageDB.status == "completed"
                ).order_by(ImageDB.created_at.desc()).limit(limit).offset(offset)
                
                result = await session.execute(query)
                db_images = result.scalars().all()
                
                return [
                    ImageRecord(
                        id=img.id,
                        filename=img.filename,
                        description=img.description,
                        image_path=img.image_path,
                        thumbnail_path=img.thumbnail_path,
                        width=img.width,
                        height=img.height,
                        size=img.size,
                        format=img.format,
                        status=UploadStatus(img.status),
                        similarity_score=img.similarity_score,
                        created_at=img.created_at,
                        updated_at=img.updated_at
                    )
                    for img in db_images
                ]
        except Exception as e:
            print(f"Error getting paginated images: {e}")
            return []


# Instancia singleton
db_service = DatabaseService()