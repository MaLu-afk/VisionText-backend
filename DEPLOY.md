# Despliegue en Railway

VisionText se despliega como **4 servicios dentro de un mismo proyecto de Railway**:

| Servicio (nombre exacto) | Origen | Público | Volumen |
|---|---|---|---|
| `Postgres` | Plantilla PostgreSQL de Railway | No | Sí (lo crea Railway) |
| `minio` | Imagen Docker `pgsty/minio:RELEASE.2026-08-04T00-00-00Z` | No | `/data` |
| `backend` | Repo `VisionText-backend`, rama `production` | Sí | `/data` |
| `frontend` | Repo `VisionText-frontend`, rama `production` | Sí | No |

```
Navegador ──► frontend (nginx, estático)
         └──► backend (FastAPI + CLIP) ──► minio.railway.internal:9000
                                      └──► postgres.railway.internal:5432
```

Solo `frontend` y `backend` tienen dominio público. El navegador ejecuta la app React y llama
directamente al backend; las imágenes siempre pasan por el backend (`/api/images/{id}`),
así que MinIO y PostgreSQL solo se usan por la red privada.

> Los nombres de servicio importan: las variables de abajo los referencian
> (`${{backend.…}}`, `${{minio.…}}`, `${{Postgres.…}}`) y el nombre define el DNS interno.

---

## Requisitos

- **Plan Hobby o superior.** Hacen falta 3 volúmenes (Postgres, MinIO y backend). El plan Free
  admite solo 1 por proyecto y el Trial, 3.
- **Proyecto nuevo.** Los proyectos creados después del 16-10-2025 tienen red privada IPv4 e IPv6;
  los anteriores solo IPv6, y uvicorn/asyncpg pueden no conectar.
- **Memoria:** CLIP ViT-L-14 en CPU usa varios GB de RAM de forma continua. Revisa el consumo
  en *Metrics* después del primer despliegue.
- Ambos repos en GitHub y Railway conectado a tu cuenta de GitHub.

---

## Flujo de despliegue (CI/CD)

```
push a main ──► nada (solo sube el código)
tag vX.Y.Z  ──► CI (build + tests) ──► mueve rama production ──► Railway construye y despliega
```

- `.github/workflows/deploy.yml`: corre al publicar un tag `vX.Y.Z`. Ejecuta CI, comprueba
  que el commit del tag está en `main` y fuerza la rama `production` a ese commit.
- `.github/workflows/ci.yml`: no tiene disparador propio; lo llama `deploy.yml`. También se
  puede lanzar a mano (*Actions → CI → Run workflow*) para probar `main` sin desplegar.
- Railway observa **solo** la rama `production`, así que los push a `main` nunca despliegan.
- Si CI falla, el despliegue se detiene y producción sigue con la versión anterior.

Cada repo se versiona por separado: un tag en `VisionText-backend` despliega solo el backend.

### Publicar una versión

```bash
git checkout main && git pull
git tag -a v1.1.0 -m "Descripción de los cambios"
git push origin v1.1.0
```

Sigue el progreso en GitHub (*Actions → Deploy*) y luego en Railway (*Deployments*).

### Rollback

- **Rápido:** en Railway, *Deployments →* despliegue anterior *→ Rollback*.
- **Por Git:** crea un tag nuevo sobre el commit bueno (`git tag -a v1.1.1 <sha>`) y publícalo.

### Reglas

- No hagas push manual a `production`: es un puntero que maneja el workflow.
- Usa versiones semánticas: `MAJOR.MINOR.PATCH`.
- Opcional: en GitHub, *Settings → Environments → production*, añade *Required reviewers*
  para aprobar manualmente cada despliegue.

---

## Paso a paso (primera vez)

### 0. Crear la rama `production`

Sube el código a `main` en ambos repos y publica el primer tag en cada uno:

```bash
git tag -a v1.0.0 -m "Primer despliegue" && git push origin v1.0.0
```

Cuando el workflow *Deploy* termine en verde, la rama `production` existirá en cada repo.

### 1. Proyecto y PostgreSQL

1. En Railway: *New Project → Empty Project*.
2. *Create → Database → PostgreSQL*. Deja el nombre `Postgres`.

No hace falta crear tablas: el backend crea la tabla `images` al arrancar.

### 2. MinIO

1. *Create → Docker Image →* `pgsty/minio:RELEASE.2026-08-04T00-00-00Z`.
   Renombra el servicio a `minio`.
2. *Variables:*
   ```
   MINIO_ROOT_USER=<usuario, mín. 3 caracteres>
   MINIO_ROOT_PASSWORD=<clave, mín. 8 caracteres>
   ```
3. *Settings → Deploy → Custom Start Command:*
   ```
   minio server /data --console-address :9001
   ```
   Railway reemplaza el ENTRYPOINT de la imagen con este comando (sin shell), por eso
   empieza con el binario `minio`.
4. Clic derecho sobre el servicio *→ Attach Volume →* mount path `/data`.
5. **No** generes dominio público ni configures healthcheck.

> **Por qué esta imagen:** la imagen oficial `minio/minio` dejó de publicarse en octubre de
> 2025 y ya no se puede descargar. Railway solo acepta imágenes de Docker Hub, GHCR, Quay,
> GitLab y MCR (por eso `cgr.dev/...` da *Invalid Docker image*). `pgsty/minio` es el fork
> comunitario mantenido por Pigsty, publicado en Docker Hub, compatible con la API de MinIO
> y que corre como root, así que no necesita permisos extra para el volumen.

### 3. Backend

1. *Create → GitHub Repo → VisionText-backend*. Renombra el servicio a `backend`.
2. *Settings → Source → Branch:* `production`. Railway detecta el `Dockerfile`.
3. *Attach Volume →* mount path `/data` (índice FAISS y caché del modelo CLIP).
4. *Variables* (usa *Raw Editor*):
   ```
   PORT=8080
   RAILWAY_HEALTHCHECK_TIMEOUT_SEC=900

   POSTGRES_URL=postgresql+asyncpg://${{Postgres.PGUSER}}:${{Postgres.PGPASSWORD}}@${{Postgres.PGHOST}}:${{Postgres.PGPORT}}/${{Postgres.PGDATABASE}}

   MINIO_ENDPOINT=${{minio.RAILWAY_PRIVATE_DOMAIN}}:9000
   MINIO_ACCESS_KEY=${{minio.MINIO_ROOT_USER}}
   MINIO_SECRET_KEY=${{minio.MINIO_ROOT_PASSWORD}}
   MINIO_SECURE=false

   BASE_URL=https://${{backend.RAILWAY_PUBLIC_DOMAIN}}
   CORS_ORIGINS=https://${{frontend.RAILWAY_PUBLIC_DOMAIN}}

   FAISS_INDEX_PATH=/data/faiss_indices/faiss_index.bin
   CACHE_DIR=/data/model_cache

   ADMIN_USERNAME=<usuario admin>
   ADMIN_PASSWORD=<clave larga>
   JWT_SECRET_KEY=<resultado de: openssl rand -hex 32>
   ```
   - `PORT=8080` fija el puerto en el que escucha uvicorn, para que el dominio apunte siempre ahí.
   - `RAILWAY_HEALTHCHECK_TIMEOUT_SEC=900`: el primer arranque descarga ~1.6 GB del modelo CLIP
     al volumen; el valor por defecto (300 s) puede no alcanzar.
   - `POSTGRES_URL` se arma a mano porque el backend necesita el prefijo `postgresql+asyncpg://`.
   - `BASE_URL` es obligatoria: con ella se construyen las URLs de imágenes que recibe el navegador.
   - Los umbrales (`TEXT_SEARCH_THRESHOLD`, etc.) son opcionales: los valores por defecto de
     `app/config.py` son los mismos del `.env` local.
   - La referencia a `frontend` se resuelve cuando ese servicio exista (paso 4).
5. *Settings → Networking → Generate Domain →* puerto **8080**.
6. *Settings → Deploy:*
   - *Healthcheck Path:* `/api/health`
   - *Restart Policy:* `On Failure`

`/api/health` responde **503** si PostgreSQL, MinIO, CLIP o FAISS no están listos. Railway
solo considera sano un 2xx, así que no da por bueno un despliegue que no puede atender peticiones.

### 4. Frontend

1. *Create → GitHub Repo → VisionText-frontend*. Renombra el servicio a `frontend`.
2. *Settings → Source → Branch:* `production`.
3. *Variables:*
   ```
   PORT=8080
   VITE_API_URL=https://${{backend.RAILWAY_PUBLIC_DOMAIN}}
   ```
   Vite incrusta `VITE_API_URL` al **compilar**. Si cambia el dominio del backend, hay que
   redesplegar el frontend. Si falta, el build falla a propósito.
4. *Settings → Networking → Generate Domain →* puerto **8080**.
5. *Settings → Deploy → Healthcheck Path:* `/`

### 5. Primer despliegue

Railway agrupa los cambios: pulsa *Deploy* en el aviso de cambios pendientes. Cuando todo
esté arriba, **redespliega el backend** una vez para que tome `CORS_ORIGINS` con el dominio
ya existente del frontend.

En *Deployments → View logs* del backend deberías ver `VisionText Backend ready!`.

### 6. Verificar

```bash
curl https://<dominio-backend>/api/health
```

La respuesta debe ser `"status": "healthy"`, con `images_in_db: 0` y `vectors_in_index: 0`.

### 7. Cargar el dataset

Desde tu máquina, en la carpeta `VisionText-backend/` (el script usa `dataset_final/` junto a él):

```bash
pip install tqdm
LOAD_API_URL=https://<dominio-backend> \
LOAD_ADMIN_USERNAME=<usuario admin> \
LOAD_ADMIN_PASSWORD=<clave admin> \
python3 load_dataset.py
```

El script se salta las imágenes que ya existen (por nombre de archivo), así que puedes
volver a ejecutarlo si se interrumpe.

---

## Endpoints (para Postman)

Documentación interactiva: `https://<dominio-backend>/docs`

| Método | Ruta | Auth | Respuestas |
|---|---|---|---|
| GET | `/api/health` | — | 200 sano / 503 alguna dependencia caída |
| POST | `/api/search/text` | — | 200. Body JSON: `{"query": "...", "limit": 20, "threshold": 0.18}` |
| POST | `/api/search/image` | — | 200 / 400 no es imagen / 413 >10 MB. Form: `image`, `limit`, `threshold` |
| GET | `/api/images/{id}` | — | 200 JPEG / 404 |
| GET | `/api/thumbnails/{id}` | — | 200 JPEG / 404 |
| POST | `/api/admin/login` | — | 200 `{access_token}` / 401 / 429 demasiados intentos (cabecera `Retry-After`). Form: `username`, `password` |
| POST | `/api/upload` | Bearer | 200 subida / 400 / 401 / 413 / 422 rechazada por contenido o similitud. Form: `image`, `description` (opcional) |
| GET | `/api/admin/images` | Bearer | 200. Query: `limit` (1-500), `offset` |
| DELETE | `/api/admin/images/{id}` | Bearer | 200 / 401 / 404 |
| GET | `/api/admin/storage` | Bearer | 200 estado de MinIO / 503 |

En Postman: llama a `/api/admin/login` y usa el `access_token` en
*Authorization → Bearer Token*. El token dura 24 horas.

**Límite de login:** tras 5 intentos fallidos desde una IP, o 10 contra un mismo usuario,
el login responde 429 durante 15 minutos (incluso con la clave correcta). Un login exitoso
reinicia los contadores. Se configura con `LOGIN_MAX_ATTEMPTS_PER_IP`,
`LOGIN_MAX_ATTEMPTS_PER_USERNAME` y `LOGIN_WINDOW_SECONDS`. El estado vive en memoria:
se reinicia con cada despliegue y asume una sola réplica del backend.

---

## Desarrollo local

El `docker-compose.yml` de la carpeta raíz del proyecto (fuera de los repos) levanta backend,
frontend (con `Dockerfile.dev` y hot reload) y MinIO. La configuración está en `.env.example`.

Tests del backend:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
