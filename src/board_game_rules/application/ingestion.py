from typing import Protocol

from board_game_rules.application.smoke import Embedder
from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk, ParseReport, Rulebook
from board_game_rules.infrastructure.catalog import verify_pdf
from board_game_rules.infrastructure.state import IndexState


class DocumentParser(Protocol):
    def fingerprint(self, doc: Rulebook) -> str: ...
    def parse(self, doc: Rulebook) -> tuple[list[Chunk], ParseReport]: ...


class IndexStore(Protocol):
    def ensure_index(self, collection: str, dimension: int) -> None: ...
    def indexed_count(self, collection: str, document_id: str, fingerprint: str) -> int: ...
    def put_indexed(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]], fingerprint: str
    ) -> None: ...


def is_indexed(
    doc: Rulebook, fingerprint: str, settings: Settings, state: IndexState, store: IndexStore
) -> bool:
    active = state.active(doc.document_id)
    if active is None:
        return False
    _, report, collection = active
    return (
        report.fingerprint == fingerprint
        and collection == settings.vector_collection
        and store.indexed_count(collection, doc.document_id, fingerprint) == report.chunk_count
    )


def publish_vectors(
    doc: Rulebook,
    chunks: list[Chunk],
    vectors: list[list[float]],
    report: ParseReport,
    settings: Settings,
    dimension: int,
    state: IndexState,
    store: IndexStore,
) -> None:
    if (
        not chunks
        or len(chunks) != report.chunk_count
        or any(chunk.document_id != doc.document_id for chunk in chunks)
    ):
        raise ValueError("Parsed chunks do not match the selected document report.")
    store.ensure_index(settings.vector_collection, dimension)
    store.put_indexed(settings.vector_collection, chunks, vectors, report.fingerprint)
    # A writer may still replace the source while conversion is running.
    verify_pdf(settings.project_root, doc, settings.max_pdf_bytes)
    state.publish(doc, report, settings.vector_collection)


def ingest_document(
    doc: Rulebook,
    settings: Settings,
    parser: DocumentParser,
    embeddings: Embedder,
    state: IndexState,
    store: IndexStore,
) -> str:
    fingerprint = parser.fingerprint(doc)
    if is_indexed(doc, fingerprint, settings, state, store):
        return "unchanged"
    chunks, report = parser.parse(doc)
    vectors = embeddings.passages([chunk.contextualized_text for chunk in chunks])
    publish_vectors(doc, chunks, vectors, report, settings, embeddings.dimension, state, store)
    return "indexed"
