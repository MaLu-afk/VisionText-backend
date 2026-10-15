import math
import time
from collections import deque
from typing import Deque, Dict


class LoginRateLimiter:
    """
    Limitador de intentos fallidos con ventana deslizante, en memoria.

    Suficiente para una sola réplica del backend: si se escalara a varias réplicas,
    el estado debería vivir en un almacén compartido (p. ej. Redis).
    """

    # Por encima de este número de claves se purgan las expiradas, para que
    # alguien que prueba miles de usuarios distintos no haga crecer la memoria sin límite
    MAX_TRACKED_KEYS = 10000

    def __init__(self, max_attempts: int, window_seconds: int):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._failures: Dict[str, Deque[float]] = {}

    def _prune(self, key: str, now: float) -> Deque[float]:
        attempts = self._failures.get(key)
        if attempts is None:
            return deque()
        while attempts and now - attempts[0] >= self.window_seconds:
            attempts.popleft()
        if not attempts:
            del self._failures[key]
        return attempts

    def retry_after(self, key: str) -> int:
        """Segundos que faltan para poder reintentar (0 si está permitido)"""
        now = time.monotonic()
        attempts = self._prune(key, now)
        if len(attempts) < self.max_attempts:
            return 0
        return max(1, math.ceil(self.window_seconds - (now - attempts[0])))

    def register_failure(self, key: str) -> None:
        now = time.monotonic()
        if len(self._failures) >= self.MAX_TRACKED_KEYS:
            for tracked_key in list(self._failures):
                self._prune(tracked_key, now)
        self._failures.setdefault(key, deque()).append(now)

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)

    def clear(self) -> None:
        self._failures.clear()
