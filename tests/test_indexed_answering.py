from unittest.mock import Mock

import httpx
import pytest
from test_ingestion import report

from board_game_rules import bootstrap
from board_game_rules.config import Settings
from board_game_rules.domain.models import Answer, Chunk, Evidence, Rulebook
from board_game_rules.infrastructure.ollama import Ollama
from board_game_rules.infrastructure.state import IndexState


def test_ui_service_searches_active_vectors_without_rebuilding(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, monkeypatch
) -> None:
    state = IndexState(settings.state_path)
    state.publish(rulebook, report(rulebook, "ready"), settings.vector_collection)
    services, vectors, ollama = Mock(), Mock(), Mock()
    services.embeddings.query.return_value = [1.0, 0.0]
    vectors.search_indexed.return_value = [Evidence(citation_id=1, chunk=chunk, score=1)]
    ollama.generate.return_value = Answer(
        answer="Two cards.", supported=True, citations=(1,), missing_information=""
    )
    monkeypatch.setattr(bootstrap, "local_services", lambda settings: (vectors, ollama))
    result, _, _ = bootstrap.ask(settings, services, rulebook.document_id, "How many cards?")
    assert result.supported
    vectors.search_indexed.assert_called_once_with(
        settings.vector_collection,
        rulebook.document_id,
        "ready",
        [1.0, 0.0],
        settings.retrieval_limit,
    )
    services.parser.parse.assert_not_called()
    services.embeddings.passages.assert_not_called()
    vectors.close.assert_called_once()


def test_unindexed_rulebook_is_an_explicit_error(settings: Settings) -> None:
    with pytest.raises(ValueError, match="not indexed"):
        bootstrap.ask(settings, Mock(), "unknown", "Question?")


def test_changed_embedding_model_requires_reindex(settings: Settings, rulebook: Rulebook) -> None:
    IndexState(settings.state_path).publish(rulebook, report(rulebook, "old"), "other-model")
    with pytest.raises(ValueError, match="re-indexing"):
        bootstrap.ask(settings, Mock(), rulebook.document_id, "Question?")


def test_trace_records_all_hits_but_only_budgeted_evidence_sent_to_gemma(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, monkeypatch
) -> None:
    IndexState(settings.state_path).publish(
        rulebook, report(rulebook, "ready"), settings.vector_collection
    )
    large_chunk = chunk.model_copy(
        update={"index": 1, "contextualized_text": "overflow " * settings.context_tokens}
    )
    client = Mock()
    client.query_points.return_value.points = [
        Mock(payload={**item.model_dump(mode="json"), "ingestion_id": "ready"}, score=score)
        for item, score in ((chunk, 0.9), (large_chunk, 0.8))
    ]
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    monkeypatch.setattr(Ollama, "check_model", lambda self: {})
    answer = Answer(answer="Two cards.", supported=True, citations=(1,), missing_information="")
    post = Mock(
        return_value=httpx.Response(
            200,
            json={"done_reason": "stop", "message": {"content": answer.model_dump_json()}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )
    )
    monkeypatch.setattr(httpx, "post", post)
    services = Mock()
    services.embeddings.query.return_value = [1.0, 0.0]
    observer = Mock()
    _, used, _ = bootstrap.ask(
        settings, services, rulebook.document_id, "How many cards?", observer
    )
    events = {call.args[0]: call.args[1] for call in observer.call_args_list}
    assert events["Embedding question"]["question"] == "How many cards?"
    assert events["Querying Qdrant"]["request"]["query"] == [1.0, 0.0]
    assert events["Retrieved chunks"]["count"] == 2
    assert len(used) == 1
    sent = events["Asking Gemma"]["request"]
    assert sent == post.call_args.kwargs["json"]
    assert chunk.contextualized_text in sent["messages"][1]["content"]
    assert large_chunk.contextualized_text not in sent["messages"][1]["content"]
    client.close.assert_called_once()


def test_multi_manual_service_embeds_once_and_uses_each_active_version(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, monkeypatch
) -> None:
    second = rulebook.model_copy(
        update={"document_id": "second-en", "game_id": "second", "title": "Second Manual"}
    )
    state = IndexState(settings.state_path)
    state.publish(rulebook, report(rulebook, "first"), settings.vector_collection)
    state.publish(second, report(second, "second"), settings.vector_collection)
    services, vectors, ollama = Mock(), Mock(), Mock()
    services.embeddings.query.return_value = [1.0, 0.0]
    second_chunk = chunk.model_copy(update={"document_id": second.document_id})
    vectors.search_selected.return_value = [
        Evidence(citation_id=1, chunk=chunk, score=0.9),
        Evidence(citation_id=2, chunk=second_chunk, score=0.8),
    ]
    ollama.generate.return_value = Answer(
        answer="Both manuals.", supported=True, citations=(1, 2), missing_information=""
    )
    monkeypatch.setattr(bootstrap, "local_services", lambda settings: (vectors, ollama))
    answer, hits, reports = bootstrap.ask_selected(
        settings, services, (rulebook.document_id, second.document_id), "Question?"
    )
    assert answer.citations == (1, 2)
    assert len(hits) == len(reports) == 2
    services.embeddings.query.assert_called_once_with("Question?")
    vectors.search_selected.assert_called_once_with(
        settings.vector_collection,
        ((rulebook.document_id, "first"), (second.document_id, "second")),
        [1.0, 0.0],
        settings.retrieval_limit,
    )
    assert "Second Manual" in ollama.generate.call_args.args[0]
    services.parser.parse.assert_not_called()


@pytest.mark.parametrize("selection", [(), ("unknown",), ("same", "same")])
def test_invalid_multi_selection_is_explicit(
    settings: Settings, selection: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        bootstrap.ask_selected(settings, Mock(), selection, "Question?")
