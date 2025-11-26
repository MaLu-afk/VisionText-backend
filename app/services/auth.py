from datetime import datetime, timedelta
from jose import JWTError, jwt
from app.config import settings


def verify_admin_credentials(username: str, password: str) -> bool:
    """Verificar credenciales del administrador"""
    return (
        username == settings.admin_username and 
        password == settings.admin_password
    )


def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    """Crear token JWT"""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verificar y decodificar token JWT"""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        return payload
    except JWTError:
        return None
