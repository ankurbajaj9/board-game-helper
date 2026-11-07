from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from board_game_rules import bootstrap
from board_game_rules.config import Settings
from board_game_rules.domain.models import Catalog, Chunk, ParseReport, Rulebook
from board_game_rules.infrastructure import prefect_pipeline as pipeline
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


@pytest.fixture
def ingest(settings: Settings, rulebook: Rulebook, store: Vectors, monkeypatch):
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    monkeypatch.setattr(pipeline, "inspect_source", lambda *args: rulebook)
    for name in ("parse_source", "embed_chunks", "publish_source"):
        monkeypatch.setattr(pipeline, name, getattr(pipeline, name).fn)
    monkeypatch.setattr(
        pipeline.bootstrap,
        "local_services",
        lambda settings: (Mock(wraps=store, close=Mock()), Mock()),
    )

    def run(parser, embeddings):
        return pipeline.ingest_pdf.fn(
            settings.project_root / rulebook.local_path,
            catalog,
            settings,
            bootstrap.Parsing(embeddings, parser),
        )

    return run


def test_ingestion_is_persistent_and_unchanged_is_a_noop(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, ingest
) -> None:
    state = IndexState(settings.state_path)
    parser = Mock()
    parser.fingerprint.return_value = "first"
    parser.parse.return_value = ([chunk], report(rulebook, "first"))
    embeddings = Mock(dimension=2)
    embeddings.passages.return_value = [[1.0, 0.0]]
    assert ingest(parser, embeddings) == "indexed"
    restarted = IndexState(settings.state_path)
    assert restarted.active(rulebook.document_id)[1].fingerprint == "first"
    assert ingest(parser, embeddings) == "unchanged"
    parser.parse.assert_called_once()
    embeddings.passages.assert_called_once()
    assert store.indexed_count(settings.vector_collection, rulebook.document_id, "first") == 1


def test_failed_replacement_keeps_old_version_visible(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, ingest
) -> None:
    state = IndexState(settings.state_path)
    parser = Mock()
    parser.fingerprint.return_value = "old"
    parser.parse.return_value = ([chunk], report(rulebook, "old"))
    embeddings = Mock(dimension=2)
    embeddings.passages.return_value = [[1.0, 0.0]]
    ingest(parser, embeddings)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))
    embeddings.passages.side_effect = ValueError("Embedding failed")
    with pytest.raises(ValueError, match="Embedding failed"):
        ingest(parser, embeddings)
    assert state.active(rulebook.document_id)[1].fingerprint == "old"
    hits = store.search_indexed(
        settings.vector_collection, rulebook.document_id, "old", [1.0, 0.0], 5
    )
    assert len(hits) == 1


def test_replacement_with_fewer_chunks_does_not_leak_old_points(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, ingest
) -> None:
    state = IndexState(settings.state_path)
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "old"
    parser.parse.return_value = (
        [chunk, chunk.model_copy(update={"index": 1})],
        report(rulebook, "old", 2),
    )
    embeddings.passages.return_value = [[1.0, 0.0], [1.0, 0.0]]
    ingest(parser, embeddings)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))
    embeddings.passages.return_value = [[1.0, 0.0]]
    ingest(parser, embeddings)
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
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, monkeypatch, ingest
) -> None:
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "new"
    parser.parse.return_value = ([chunk], report(rulebook, "new"))
    embeddings.passages.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(store, "indexed_count", lambda *args: 0)
    state = IndexState(settings.state_path)
    with pytest.raises(ValueError, match="count"):
        ingest(parser, embeddings)
    assert state.active(rulebook.document_id) is None


def test_missing_qdrant_data_is_rebuilt(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, ingest
) -> None:
    parser, embeddings = Mock(), Mock(dimension=2)
    parser.fingerprint.return_value = "same"
    parser.parse.return_value = ([chunk], report(rulebook, "same"))
    embeddings.passages.return_value = [[1.0, 0.0]]
    state = IndexState(settings.state_path)
    ingest(parser, embeddings)
    store.delete(settings.vector_collection)
    assert ingest(parser, embeddings) == "indexed"
    assert embeddings.passages.call_count == 2


def test_source_changed_during_embedding_is_not_activated(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, store: Vectors, ingest
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
        ingest(parser, embeddings)
    assert state.active(rulebook.document_id) is None
