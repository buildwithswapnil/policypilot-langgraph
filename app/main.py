from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    application.state.ready = False

    # Later: initialize Chroma, PolicyPilot, and shared resources.
    application.state.ready = True

    try:
        yield
    finally:
        application.state.ready = False


def create_app() -> FastAPI:
    settings = get_settings()

    is_production = (
        settings.environment == "production"
    )

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=(
            None
            if is_production
            else "/openapi.json"
        ),
    )

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return application


app = create_app()