from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from board_game_rules.application.answering import answer_question
from board_game_rules.config import Settings
from board_game_rules.domain.models import Answer, Chunk, Evidence, Rulebook
from board_game_rules.infrastructure.qdrant import Vectors


def test_supported_answer_cites_retrieved_evidence(
    settings: Settings, rulebook: Rulebook, chunk: Chunk
) -> None:
    generator = Mock()
    generator.generate.return_value = Answer(
        answer="Two cards.", supported=True, citations=(1,), missing_information=""
    )
    evidence = [Evidence(citation_id=1, chunk=chunk, score=0.8)]
    answer, used = answer_question(rulebook, "How many cards?", evidence, generator, settings)
    assert answer.citations == (1,)
    assert used == evidence
    assert "Test Game" in generator.generate.call_args.args[0]


def test_fabricated_citation_rejected(settings: Settings, rulebook: Rulebook, chunk: Chunk) -> None:
    generator = Mock()
    generator.generate.return_value = Answer(
        answer="Two cards.", supported=True, citations=(99,), missing_information=""
    )
    with pytest.raises(ValueError, match="not supplied"):
        answer_question(
            rulebook,
            "How many?",
            [Evidence(citation_id=1, chunk=chunk, score=1)],
            generator,
            settings,
        )


def test_no_evidence_abstains_without_model_call(settings: Settings, rulebook: Rulebook) -> None:
    generator = Mock()
    answer, evidence = answer_question(rulebook, "How many?", [], generator, settings)
    assert not answer.supported and not evidence
    generator.generate.assert_not_called()


def test_qdrant_retrieval_produces_a_cited_answer(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, monkeypatch
) -> None:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        "board_game_rules.infrastructure.qdrant.QdrantClient", lambda **kwargs: client
    )
    store = Vectors(settings)
    collection = "answering-test"
    store.create(collection, 2)
    try:
        store.put_indexed(collection, [chunk], [[1.0, 0.0]], "active")
        evidence = store.search_indexed(
            collection, rulebook.document_id, "active", [1.0, 0.0], settings.retrieval_limit
        )
        generator = Mock()
        generator.generate.return_value = Answer(
            answer="Two cards.", supported=True, citations=(1,), missing_information=""
        )

        answer, used = answer_question(rulebook, "How many cards?", evidence, generator, settings)

        assert answer.supported
        assert answer.citations == (1,)
        assert used == evidence
    finally:
        store.delete(collection)
        store.close()


def test_evidence_from_another_game_is_rejected(
    settings: Settings, rulebook: Rulebook, chunk: Chunk
) -> None:
    other = chunk.model_copy(update={"document_id": "other-en"})
    with pytest.raises(ValueError, match="selected document"):
        answer_question(
            rulebook, "Test?", [Evidence(citation_id=1, chunk=other, score=1)], Mock(), settings
        )


def test_long_question_is_not_truncated(
    settings: Settings, rulebook: Rulebook, chunk: Chunk
) -> None:
    generator = Mock()
    with pytest.raises(ValueError, match="context window"):
        answer_question(
            rulebook,
            "x" * settings.context_tokens,
            [Evidence(citation_id=1, chunk=chunk, score=1)],
            generator,
            settings,
        )
    generator.generate.assert_not_called()


@pytest.mark.parametrize(
    "data",
    [
        {"supported": True, "citations": (), "missing_information": ""},
        {"supported": False, "citations": (), "missing_information": ""},
        {"supported": True, "citations": (1, 1), "missing_information": ""},
    ],
)
def test_invalid_answer_contract(data: dict) -> None:
    with pytest.raises(ValidationError):
        Answer(answer="Test", **data)


