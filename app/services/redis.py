import redis.asyncio as aioredis
import json
from typing import Optional, Any
from app.config import settings


class RedisService:
    def __init__(self):
        self.redis = None
    
    async def connect(self):
        """Conectar a Redis"""
        self.redis = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        print("Redis connected")
    
    async def disconnect(self):
        """Desconectar de Redis"""
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor de caché"""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        """Guardar valor en caché con expiración (segundos)"""
        if not self.redis:
            return False
        
        try:
            serialized = json.dumps(value)
            await self.redis.set(key, serialized, ex=expire)
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str):
        """Eliminar clave de caché"""
        if not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Verificar si una clave existe"""
        if not self.redis:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            print(f"Redis exists error: {e}")
            return False
    
    async def increment(self, key: str) -> int:
        """Incrementar contador"""
        if not self.redis:
            return 0
        
        try:
            return await self.redis.incr(key)
        except Exception as e:
            print(f"Redis incr error: {e}")
            return 0
    
    async def get_search_cache(self, query: str, query_type: str = "text") -> Optional[dict]:
        """Obtener resultados de búsqueda en caché"""
        cache_key = f"search:{query_type}:{query}"
        return await self.get(cache_key)
    
    async def set_search_cache(self, query: str, results: dict, 
                              query_type: str = "text", expire: int = 3600):
        """Guardar resultados de búsqueda en caché"""
        cache_key = f"search:{query_type}:{query}"
        return await self.set(cache_key, results, expire)


# Instancia singleton
redis_service = RedisService()