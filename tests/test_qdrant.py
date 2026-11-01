from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk
from board_game_rules.infrastructure.qdrant import Vectors


def test_qdrant_round_trip_is_document_scoped(
    settings: Settings, chunk: Chunk, monkeypatch
) -> None:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    store = Vectors(settings)
    store.create("test", 2)
    other = chunk.model_copy(update={"document_id": "other-en", "index": 1})
    store.put("test", [chunk, other], [[1.0, 0.0], [1.0, 0.0]])
    hits = store.search("test", chunk.document_id, [1.0, 0.0], 10)
    assert len(hits) == 1
    assert hits[0].chunk == chunk
    assert hits[0].citation_id == 1
    store.delete("test")
    assert store.collections() == []
    store.close()


def test_qdrant_rejects_mismatched_batches(settings: Settings, chunk: Chunk, monkeypatch) -> None:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    store = Vectors(settings)
    with pytest.raises(ValueError, match="one embedding"):
        store.put("test", [chunk], [])
    store.close()


def test_indexed_trace_matches_query_and_all_returned_chunks(
    settings: Settings, chunk: Chunk, monkeypatch
) -> None:
    client = Mock()
    client.query_points.return_value.points = [
        Mock(payload={**chunk.model_dump(mode="json"), "ingestion_id": "active"}, score=0.9),
        Mock(
            payload={
                **chunk.model_copy(update={"index": 1}).model_dump(mode="json"),
                "ingestion_id": "active",
            },
            score=0.8,
        ),
    ]
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    observer = Mock()
    store = Vectors(settings, observer)
    hits = store.search_indexed("test", chunk.document_id, "active", [1.0, 0.0], 6)
    request = observer.call_args_list[0].args[1]["request"]
    actual = client.query_points.call_args.kwargs
    assert request == {
        "query": actual["query"],
        "filter": actual["query_filter"].model_dump(mode="json", exclude_none=True),
        "limit": actual["limit"],
        "with_payload": actual["with_payload"],
    }
    assert observer.call_args_list[0].args[0] == "Querying Qdrant"
    assert observer.call_args_list[1].args == (
        "Retrieved chunks",
        {"count": 2, "chunks": [entry.model_dump(mode="json") for entry in hits]},
    )
    store.close()


def test_multi_manual_query_excludes_unselected_and_stale_versions(
    settings: Settings, chunk: Chunk, monkeypatch
) -> None:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    store = Vectors(settings)
    store.create("test", 2)
    second = chunk.model_copy(update={"document_id": "second-en"})
    unrelated = chunk.model_copy(update={"document_id": "unrelated-en"})
    store.put_indexed("test", [chunk], [[1.0, 0.0]], "active-first")
    store.put_indexed("test", [chunk], [[1.0, 0.0]], "stale-first")
    store.put_indexed("test", [second], [[1.0, 0.0]], "active-second")
    store.put_indexed("test", [unrelated], [[1.0, 0.0]], "active-third")
    hits = store.search_selected(
        "test",
        ((chunk.document_id, "active-first"), ("second-en", "active-second")),
        [1.0, 0.0],
        6,
    )
    assert {entry.chunk.document_id for entry in hits} == {chunk.document_id, "second-en"}
    assert [entry.citation_id for entry in hits] == [1, 2]
    assert len(hits) == 2
    store.close()
