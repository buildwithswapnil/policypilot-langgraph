from functools import lru_cache
from typing import Literal

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


Environment = Literal[
    "development",
    "testing",
    "production",
]


class Settings(BaseSettings):
    app_name: str = "PolicyPilot API"
    app_version: str = "0.1.0"
    environment: Environment = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="POLICYPILOT_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()