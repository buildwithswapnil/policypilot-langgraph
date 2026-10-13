from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    application = create_app()

    with TestClient(application) as test_client:
        yield test_client


def test_liveness(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/health/live"
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "PolicyPilot API",
        "version": "0.1.0",
    }


def test_readiness_when_initialized(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/health/ready"
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "application_initialized": True,
        },
    }

def test_readiness_when_not_initialized(
    client: TestClient,
) -> None:
    client.app.state.ready = False

    response = client.get(
        "/api/v1/health/ready"
    )

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "application_initialized": False,
        },
    }                  