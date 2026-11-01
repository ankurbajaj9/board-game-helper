from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from board_game_rules.application.ingestion import ingest_document
from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk, ParseReport, Rulebook
from board_game_rules.infrastructure.qdrant import Vectors
from board_game_rules.infrastructure.state import IndexState


def report(doc: Rulebook, fingerprint: str, count: int = 1) -> ParseReport:
    return ParseReport(
        document_id=doc.document_id,
        pdf_pages=1,
        chunk_count=count,
        max_chunk_tokens=20,
        pages_without_chunks=(),
        picture_count=0,
        fingerprint=fingerprint,
        artifact_directory="data/parsed/test",
    )


@pytest.fixture
def store(settings: Settings, monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    vectors = Vectors(settings)
    yield vectors
    vectors.close()


def test_ingestion_is_persistent_and_unchanged_is_a_noop(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors
) -> None:
    state = IndexState(settings.state_path)
    parser = Mock()
    parser.fingerprint.return_value = "first"
    parser.parse.return_value = ([chunk], report(rulebook, "first"))
    embeddings = Mock(dimension=2)
    embeddings.passages.return_value = [[1.0, 0.0]]
    assert ingest_document(rulebook, settings, parser, embeddings, state, store) == "indexed"
    restarted = IndexState(settings.state_path)
    assert restarted.active(rulebook.document_id)[1].fingerprint == "first"
    assert ingest_document(rulebook, settings, parser, embeddings, restarted, store) == "unchanged"
    parser.parse.assert_called_once()
    embeddings.passages.assert_called_once()
    assert store.indexed_count(settings.vector_collection, rulebook.document_id, "first") == 1


def test_failed_replacement_keeps_old_version_visible(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors
) -> None:
    state = IndexState(settings.state_path)
    parser = Mock()
    parser.fingerprint.return_value = "old"
    parser.parse.return_value = ([chunk], report(rulebook, "old"))
    embeddings = Mock(dimension=2)
    embeddings.passages.return_value = [[1.0, 0.0]]
    ingest_document(rulebook, settings, parser, embeddings, state, store)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))
    embeddings.passages.side_effect = ValueError("Embedding failed")
    with pytest.raises(ValueError, match="Embedding failed"):
        ingest_document(rulebook, settings, parser, embeddings, state, store)
    assert state.active(rulebook.document_id)[1].fingerprint == "old"
    hits = store.search_indexed(
        settings.vector_collection, rulebook.document_id, "old", [1.0, 0.0], 5
    )
    assert len(hits) == 1


def test_replacement_with_fewer_chunks_does_not_leak_old_points(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors
) -> None:
    state = IndexState(settings.state_path)
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "old"
    parser.parse.return_value = (
        [chunk, chunk.model_copy(update={"index": 1})],
        report(rulebook, "old", 2),
    )
    embeddings.passages.return_value = [[1.0, 0.0], [1.0, 0.0]]
    ingest_document(rulebook, settings, parser, embeddings, state, store)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))
    embeddings.passages.return_value = [[1.0, 0.0]]
    ingest_document(rulebook, settings, parser, embeddings, state, store)
    assert state.active(rulebook.document_id)[1].fingerprint == "new"
    assert (
        len(
            store.search_indexed(
                settings.vector_collection, rulebook.document_id, "new", [1.0, 0.0], 5
            )
        )
        == 1
    )
    assert store.indexed_count(settings.vector_collection, rulebook.document_id, "old") == 2


def test_incomplete_upsert_is_not_activated(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, monkeypatch
) -> None:
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))
    embeddings.passages.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(store, "indexed_count", lambda *args: 0)
    state = IndexState(settings.state_path)
    with pytest.raises(ValueError, match="count"):
        ingest_document(rulebook, settings, parser, embeddings, state, store)
    assert state.active(rulebook.document_id) is None


def test_missing_qdrant_data_is_rebuilt(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors
) -> None:
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "same"
    parser.parse.return_value = ([chunk], report(rulebook, "same"))
    embeddings.passages.return_value = [[1.0, 0.0]]
    state = IndexState(settings.state_path)
    ingest_document(rulebook, settings, parser, embeddings, state, store)
    store.delete(settings.vector_collection)
    assert ingest_document(rulebook, settings, parser, embeddings, state, store) == "indexed"
    assert embeddings.passages.call_count == 2


def test_source_changed_during_embedding_is_not_activated(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors
) -> None:
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))

    def embed(texts):
        (settings.project_root / rulebook.local_path).write_bytes(b"changed")
        return [[1.0, 0.0]]

    embeddings.passages.side_effect = embed
    state = IndexState(settings.state_path)
    with pytest.raises(ValueError, match="size"):
        ingest_document(rulebook, settings, parser, embeddings, state, store)
    assert state.active(rulebook.document_id) is None
