from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "openrouter"
    litellm_model: str = "nvidia/nemotron-3.5-lightning:free"
    litellm_api_key: str | None = None
    litellm_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = None
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_reasoning_enabled: bool = False

    semantic_scholar_api_key: str | None = None
    arxiv_api_url: str = "https://export.arxiv.org/api/query"

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/literature_agent"
    )
    embedding_provider: str = "lmstudio"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_api_key: str = "lm-studio"
    embedding_api_base: str = "http://localhost:1234/v1"
    embedding_dimensions: int = 768

    recent_days: int = 730
    enforce_recent_filter: bool = False
    max_papers: int = 5
    max_loop_iterations: int = 3
    min_relevance_score: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()
