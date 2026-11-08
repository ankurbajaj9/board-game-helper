from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

type QueryObserver = Callable[[str, dict[str, object]], None]
type PromptStrategy = Literal["plain", "few_shot_citations"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Rulebook(Model):
    document_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    game_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1, max_length=120)
    edition: str = Field(min_length=1)
    language: Literal["en"]
    kind: Literal["base", "expansion", "standalone"]
    base_game_id: str | None
    compatible_base_document_ids: tuple[str, ...] = ()
    publisher: str = Field(min_length=1)
    source_page: HttpUrl | None
    download_url: HttpUrl | None
    local_path: str = Field(min_length=1)
    downloaded_at: date
    pdf_pages: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_expansion(self) -> Rulebook:
        if self.kind == "expansion":
            if not self.base_game_id or not self.compatible_base_document_ids:
                raise ValueError("An expansion needs a base game and compatible base document IDs.")
            if self.base_game_id == self.game_id:
                raise ValueError("An expansion cannot require itself.")
        elif self.base_game_id or self.compatible_base_document_ids:
            raise ValueError("Base and standalone games cannot declare expansion dependencies.")
        return self


class Catalog(Model):
    schema_version: Literal[1]
    documents: tuple[Rulebook, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dependencies(self) -> Catalog:
        by_id = {doc.document_id: doc for doc in self.documents}
        if len(by_id) != len(self.documents):
            raise ValueError("Duplicate document IDs in catalog.")
        for doc in self.documents:
            for base_id in doc.compatible_base_document_ids:
                base = by_id.get(base_id)
                if base is None or base.kind == "expansion" or base.game_id != doc.base_game_id:
                    raise ValueError(f"Invalid compatible base document {base_id!r}.")
        return self

    def document(self, document_id: str) -> Rulebook:
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        raise ValueError(f"Unknown document ID: {document_id}")

class Chunk(Model):
    document_id: str
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    contextualized_text: str = Field(min_length=1)
    pdf_pages: tuple[int, ...] = Field(min_length=1)
    headings: tuple[str, ...] = ()
    token_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pages(self) -> Chunk:
        if any(page < 1 for page in self.pdf_pages):
            raise ValueError("PDF page numbers must be positive and 1-based.")
        return self


class ParseReport(Model):
    document_id: str
    pdf_pages: int
    chunk_count: int
    max_chunk_tokens: int
    pages_without_chunks: tuple[int, ...]
    picture_count: int
    fingerprint: str
    artifact_directory: str


class Evidence(Model):
    citation_id: int = Field(gt=0)
    chunk: Chunk
    score: float


class Answer(Model):
    answer: str = Field(min_length=1, max_length=2000)
    supported: bool
    citations: tuple[int, ...] = Field(max_length=20)
    missing_information: str = Field(max_length=1000)

    @model_validator(mode="after")
    def validate_support(self) -> Answer:
        if self.supported and not self.citations:
            raise ValueError("A supported answer must cite evidence.")
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("Duplicate evidence citations.")
        if not self.supported and not self.missing_information.strip():
            raise ValueError("An unsupported answer must explain the missing information.")
        return self
