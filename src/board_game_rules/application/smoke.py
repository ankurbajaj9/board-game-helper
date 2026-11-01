from typing import Protocol
from uuid import uuid4

from board_game_rules.application.answering import Generator, answer_question
from board_game_rules.config import Settings
from board_game_rules.domain.models import Answer, Chunk, Evidence, Rulebook


class Embedder(Protocol):
    dimension: int

    def passages(self, texts: list[str]) -> list[list[float]]: ...
    def query(self, text: str) -> list[float]: ...


class SmokeStore(Protocol):
    def create(self, collection: str, dimension: int) -> None: ...
    def put(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def search(
        self, collection: str, document_id: str, vector: list[float], limit: int
    ) -> list[Evidence]: ...
    def delete(self, collection: str) -> None: ...


def smoke_test(
    doc: Rulebook,
    chunks: list[Chunk],
    query: str,
    embeddings: Embedder,
    store: SmokeStore,
    generator: Generator,
    settings: Settings,
) -> tuple[Answer, list[Evidence]]:
    if not query.strip():
        raise ValueError("Please enter a question.")
    if not chunks or any(chunk.document_id != doc.document_id for chunk in chunks):
        raise ValueError("Smoke test requires chunks from the selected document only.")
    vectors = embeddings.passages([chunk.contextualized_text for chunk in chunks])
    query_vector = embeddings.query(query)
    collection = f"board_game_rules_smoke_{uuid4().hex}"
    store.create(collection, embeddings.dimension)
    try:
        store.put(collection, chunks, vectors)
        evidence = store.search(collection, doc.document_id, query_vector, settings.retrieval_limit)
        return answer_question(doc, query, evidence, generator, settings)
    finally:
        store.delete(collection)
