from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    status,
)

from app.core.config import Settings, get_settings
from app.schemas.health import (
    LivenessResponse,
    ReadinessResponse,
)


router = APIRouter(
    prefix="/health",
    tags=["health"],
)


SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Check whether the API process is alive",
)
def liveness(
    settings: SettingsDependency,
) -> LivenessResponse:
    return LivenessResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The application is not ready",
        }
    },
    summary="Check whether the API can receive traffic",
)
def readiness(
    request: Request,
    response: Response,
) -> ReadinessResponse:
    application_ready = bool(
        getattr(
            request.app.state,
            "ready",
            False,
        )
    )

    checks = {
        "application_initialized": application_ready,
    }

    if not all(checks.values()):
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

        return ReadinessResponse(
            status="not_ready",
            checks=checks,
        )

    return ReadinessResponse(
        status="ready",
        checks=checks,
    )