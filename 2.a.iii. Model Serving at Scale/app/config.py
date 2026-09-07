from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    backend: str = "mock"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    mock_base_latency_ms: float = 8.0
    mock_per_item_latency_ms: float = 6.0
    mock_embed_dim: int = 64

    batch_max_size: int = 16
    batch_max_wait_ms: float = 20.0

    cache_max_size: int = 2000
    cache_ttl_seconds: float = 300.0


settings = Settings()
