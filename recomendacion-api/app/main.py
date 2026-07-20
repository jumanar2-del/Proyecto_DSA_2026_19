from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app.api import api_router
from app.config import settings, setup_app_logging

# Configurar logging lo antes posible
setup_app_logging(config=settings)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Página de inicio con enlace a la documentación."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Recomendacion Productos API</h1>"
        "<div>"
        "Documentación: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Habilitar CORS para los orígenes configurados
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    # Solo para desarrollo/depuración local
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
