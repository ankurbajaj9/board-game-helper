from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from board_game_rules.application.answering import answer_question
from board_game_rules.application.smoke import smoke_test
from board_game_rules.config import Settings
from board_game_rules.domain.models import Answer, Chunk, Evidence, Rulebook


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


def test_smoke_cleanup_on_failure(settings: Settings, rulebook: Rulebook, chunk: Chunk) -> None:
    store = Mock()
    store.put.side_effect = ValueError("Upsert failed")
    embeddings = Mock(dimension=2)
    embeddings.passages.return_value = [[1.0, 0.0]]
    with pytest.raises(ValueError, match="Upsert failed"):
        smoke_test(rulebook, [chunk], "Test?", embeddings, store, Mock(), settings)
    collection = store.create.call_args.args[0]
    assert collection.startswith("board_game_rules_smoke_")
    store.delete.assert_called_once_with(collection)


def test_smoke_does_not_delete_existing_collection_on_create_failure(
    settings: Settings, rulebook: Rulebook, chunk: Chunk
) -> None:
    store = Mock()
    store.create.side_effect = ValueError("Create failed")
    with pytest.raises(ValueError, match="Create failed"):
        smoke_test(rulebook, [chunk], "Test?", Mock(dimension=2), store, Mock(), settings)
    store.delete.assert_not_called()
