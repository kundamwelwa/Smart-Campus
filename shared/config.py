"""
Shared Configuration Module

Centralized configuration management for all Argos services using Pydantic Settings.
Supports environment variables, .env files, and runtime overrides.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=True, description="Enable debug mode")

    # Database - PostgreSQL
    postgres_user: str = "postgres"
    postgres_password: str = "kelly12345"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "smart-campus"

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # MongoDB - Event Store
    mongodb_user: str = "argos"
    mongodb_password: str = "argos_dev_password"
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_db: str = "argos_events"

    @property
    def mongodb_url(self) -> str:
        """Construct MongoDB connection URL."""
        return f"mongodb://{self.mongodb_user}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_db}?authSource=admin"

    # Redis - Cache & Session Store
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Kafka - Event Streaming
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "argos"

    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production-min-32-chars",
        min_length=32,
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    encryption_key: str = Field(
        default="dev-encryption-key-change-in-production-32-bytes",
        description="Fernet encryption key (base64 encoded 32 bytes)",
    )

    # API Configuration
    api_v1_prefix: str = "/api/v1"
    api_gateway_host: str = "0.0.0.0"
    api_gateway_port: int = 8000

    # Service Ports (HTTP)
    user_service_port: int = 8001
    academic_service_port: int = 8002
    scheduler_service_port: int = 8003
    analytics_service_port: int = 8004
    facility_service_port: int = 8005
    security_service_port: int = 8006
    consensus_service_port: int = 8007

    # Service Ports (gRPC)
    user_service_grpc_port: int = 50051
    academic_service_grpc_port: int = 50052
    scheduler_service_grpc_port: int = 50053
    analytics_service_grpc_port: int = 50054
    facility_service_grpc_port: int = 50055
    security_service_grpc_port: int = 50056
    consensus_service_grpc_port: int = 50057

    # ML Configuration
    ml_model_path: str = "./ml/models/saved"
    ml_training_data_path: str = "./ml/datasets"
    ml_device: Literal["cpu", "cuda", "mps"] = "cpu"
    ml_batch_size: int = 32
    ml_random_seed: int = 42

    # External auto-grading / assignment service
    external_grader_base_url: str = Field(
        default="",
        description="http://localhost:9000 ",
    )
    external_grader_api_key: str | None = Field(
        default=None,
        description="Optional API key for external auto-grading service",
    )
    external_grader_enabled: bool = Field(
        default=False,
        description="Enable external auto-grading integration",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])

    # Performance
    max_workers: int = 4
    request_timeout: int = 30
    max_connections_pool: int = 100

    # Raft Consensus
    raft_node_id: int = 1
    raft_cluster_size: int = 3
    raft_election_timeout_min: int = 150
    raft_election_timeout_max: int = 300
    raft_heartbeat_interval: int = 50


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings singleton
    """
    return Settings()


# Convenience exports
settings = get_settings()

