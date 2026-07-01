"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embed_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBED_MODEL"
    )
    openai_embed_dim: int = Field(default=1536, alias="OPENAI_EMBED_DIM")

    # --- MongoDB ---
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="docutrust", alias="MONGO_DB")

    # --- Vector store ---
    vector_backend: Literal["faiss", "atlas"] = Field(
        default="faiss", alias="VECTOR_BACKEND"
    )
    atlas_vector_index: str = Field(
        default="vector_index", alias="ATLAS_VECTOR_INDEX"
    )
    atlas_vector_dim: int = Field(default=1536, alias="ATLAS_VECTOR_DIM")
    faiss_index_path: str = Field(default="./data/faiss", alias="FAISS_INDEX_PATH")

    # --- Grader ---
    grader_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="GRADER_MODEL"
    )
    grader_threshold: float = Field(default=0.55, alias="GRADER_THRESHOLD")
    grader_device: str = Field(default="cpu", alias="GRADER_DEVICE")

    # --- Retrieval ---
    retrieval_k: int = Field(default=8, alias="RETRIEVAL_K")

    # --- Web fallback ---
    web_domains_allowlist: str = Field(
        default="sec.gov,europa.eu,nist.gov,iso.org,fas.org,bis.doc.gov",
        alias="WEB_DOMAINS_ALLOWLIST",
    )
    web_max_results: int = Field(default=5, alias="WEB_MAX_RESULTS")

    @property
    def web_allowlist(self) -> list[str]:
        return [d.strip().lower() for d in self.web_domains_allowlist.split(",") if d.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
