from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BGR_", env_file=".env", extra="forbid")

    project_root: Path = Field(default_factory=Path.cwd)
    qdrant_url: HttpUrl = HttpUrl("http://localhost:6789")
    qdrant_api_key: SecretStr | None = None
    ollama_url: HttpUrl = HttpUrl("http://localhost:11434")
    ollama_model: str = Field(default="gemma4:e2b", min_length=1)
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    chunk_tokens: int = Field(default=384, ge=64, le=512)
    context_tokens: int = Field(default=8192, ge=2048)
    answer_tokens: int = Field(default=768, ge=128)
    timeout_seconds: int = Field(default=180, gt=0)
    max_pdf_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    retrieval_limit: int = Field(default=6, ge=1, le=20)
    scan_interval_seconds: int = Field(default=30, ge=5)
    file_settle_seconds: int = Field(default=10, ge=1)
    prefect_url: HttpUrl = HttpUrl("http://127.0.0.1:4201/api")

    @model_validator(mode="after")
    def validate_paths_and_budget(self) -> Settings:
        self.project_root = self.project_root.expanduser().resolve()
        if not self.project_root.is_dir():
            raise ValueError("Project root must be an existing directory.")
        if self.answer_tokens >= self.context_tokens:
            raise ValueError("Answer token budget must be smaller than the context window.")
        return self

    @property
    def catalog_path(self) -> Path:
        return self.project_root / "catalog" / "rulebooks.json"

    @property
    def artifacts_path(self) -> Path:
        return self.project_root / "data" / "parsed"

    @property
    def rulebooks_path(self) -> Path:
        return self.project_root / "data" / "rulebooks"

    @property
    def state_path(self) -> Path:
        return self.project_root / "data" / "ingestion.sqlite"

    @property
    def vector_collection(self) -> str:
        import hashlib

        model_hash = hashlib.sha256(self.embedding_model.encode()).hexdigest()[:12]
        return f"board_game_rules_v1_{model_hash}"
