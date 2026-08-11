from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "lmstudio"
    litellm_model: str = "google/gemma-3-4b"
    litellm_api_key: str = "lm-studio"
    litellm_api_base: str = "http://localhost:1234/v1"

    semantic_scholar_api_key: str | None = None
    arxiv_api_url: str = "https://export.arxiv.org/api/query"

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/literature_agent"
    )
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 1536

    recent_days: int = 730
    max_papers: int = 5
    max_loop_iterations: int = 3
    min_relevance_score: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()
