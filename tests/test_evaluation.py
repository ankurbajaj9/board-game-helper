from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from board_game_rules.config import Settings
from board_game_rules.domain.models import (
    Answer,
    Catalog,
    Chunk,
    Evidence,
    ParseReport,
    Rulebook,
)
from board_game_rules.evaluation import (
    EvaluationCase,
    EvaluationReport,
    evaluate_with_ragas,
    load_dataset,
    run_evaluation,
    summarize_faithfulness,
    summarize_ragas,
)


def test_fixed_dataset_has_twenty_unique_source_backed_cases() -> None:
    dataset, digest = load_dataset(Path("evaluation/questions.json"))
    catalog = Catalog.model_validate_json(Path("catalog/rulebooks.json").read_bytes())

    assert len(dataset.cases) == 20
    assert len({case.case_id for case in dataset.cases}) == 20
    assert len(digest) == 64
    assert all(
        case.evidence_pages
        and case.reference_answer
        and max(case.evidence_pages) <= catalog.document(case.document_id).pdf_pages
        for case in dataset.cases
    )


def test_evaluation_reuses_one_retrieval_for_both_prompt_strategies(
    settings: Settings, rulebook: Rulebook, chunk: Chunk, monkeypatch
) -> None:
    case = EvaluationCase(
        case_id="test_case",
        document_id=rulebook.document_id,
        question="How many cards?",
        reference_answer="Two.",
        evidence_pages=(1,),
    )
    report = ParseReport(
        document_id=rulebook.document_id,
        pdf_pages=1,
        chunk_count=1,
        max_chunk_tokens=12,
        pages_without_chunks=(),
        picture_count=0,
        fingerprint="a" * 64,
        artifact_directory="data/parsed/test",
    )
    expected_answer = Answer(
        answer="Two cards.",
        supported=True,
        citations=(1,),
        missing_information="",
    )
    active_index = Mock()
    active_index.active.return_value = (rulebook, report, settings.vector_collection)
    embeddings = Mock()
    embeddings.query.return_value = [1.0, 0.0]
    store = Mock()
    store.search_indexed.return_value = [Evidence(citation_id=1, chunk=chunk, score=0.9)]
    generator = Mock()
    generator.generate.return_value = expected_answer

    result = run_evaluation(
        [case],
        "b" * 64,
        Catalog(schema_version=1, documents=(rulebook,)),
        active_index,
        embeddings,
        store,
        generator,
        settings,
    )

    embeddings.query.assert_called_once_with(case.question)
    store.search_indexed.assert_called_once_with(
        settings.vector_collection,
        rulebook.document_id,
        report.fingerprint,
        [1.0, 0.0],
        settings.retrieval_limit,
    )
    assert [entry.args[1] for entry in generator.generate.call_args_list] == [
        "plain",
        "few_shot_citations",
    ]
    assert generator.generate.call_args_list[0].args[0] == generator.generate.call_args_list[
        1
    ].args[0]
    assert result.retrieval_hit_count == 1
    assert result.retrieval_hit_rate == 1.0
    assert result.cases[0].plain.manual_faithfulness is None
    assert result.cases[0].few_shot_citations.manual_faithfulness is None
    assert summarize_ragas(result)["plain"]["scored"] == 0
    EvaluationReport.model_validate_json(result.model_dump_json())

    ragas = pytest.importorskip("ragas")

    def fake_evaluate(**kwargs):
        samples = kwargs["dataset"].to_list()
        assert len(samples) == 2
        assert all(
            sample["retrieved_contexts"] == [chunk.contextualized_text] for sample in samples
        )
        return SimpleNamespace(
            scores=[
                {"faithfulness": 0.75, "factual_correctness(mode=f1)": 0.85},
                {
                    "faithfulness": ValueError("Judge response was truncated"),
                    "factual_correctness(mode=f1)": 0.65,
                },
            ]
        )

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    ragas_report = evaluate_with_ragas(result, settings)
    assert ragas_report.ragas_judge_model == settings.ollama_model
    assert ragas_report.cases[0].plain.ragas_faithfulness == 0.75
    assert ragas_report.cases[0].plain.ragas_factual_correctness == 0.85
    assert ragas_report.cases[0].few_shot_citations.ragas_faithfulness is None
    assert ragas_report.cases[0].few_shot_citations.ragas_factual_correctness == 0.65
    assert summarize_ragas(ragas_report)["plain"]["faithfulness"] == 0.75
    assert summarize_ragas(ragas_report)["few_shot_citations"]["scored"] == 0
    assert summarize_ragas(ragas_report)["few_shot_citations"]["factual_correctness_scored"] == 1


def test_retrieval_hit_requires_the_expected_pdf_page(
    settings: Settings, rulebook: Rulebook, chunk: Chunk
) -> None:
    two_page_rulebook = rulebook.model_copy(update={"pdf_pages": 2})
    case = EvaluationCase(
        case_id="test_case",
        document_id=rulebook.document_id,
        question="How many cards?",
        reference_answer="Two.",
        evidence_pages=(2,),
    )
    report = ParseReport(
        document_id=rulebook.document_id,
        pdf_pages=2,
        chunk_count=1,
        max_chunk_tokens=12,
        pages_without_chunks=(2,),
        picture_count=0,
        fingerprint="a" * 64,
        artifact_directory="data/parsed/test",
    )
    active_index = Mock()
    active_index.active.return_value = (two_page_rulebook, report, settings.vector_collection)
    embeddings = Mock()
    embeddings.query.return_value = [1.0, 0.0]
    store = Mock()
    store.search_indexed.return_value = [Evidence(citation_id=1, chunk=chunk, score=0.9)]
    generator = Mock()
    generator.generate.return_value = Answer(
        answer="Two cards.",
        supported=True,
        citations=(1,),
        missing_information="",
    )

    result = run_evaluation(
        [case],
        "b" * 64,
        Catalog(schema_version=1, documents=(two_page_rulebook,)),
        active_index,
        embeddings,
        store,
        generator,
        settings,
    )

    assert result.retrieval_hit_count == 0
    assert result.retrieval_hit_rate == 0.0


def test_faithfulness_summary_keeps_unreviewed_answers_unscored() -> None:
    report = EvaluationReport(
        dataset_sha256="b" * 64,
        completed_at="2026-10-31T12:00:00Z",
        collection="collection",
        embedding_model="embedding",
        ollama_model="gemma",
        retrieval_limit=6,
        context_tokens=8192,
        answer_tokens=768,
        cases=(
            {
                "case": {
                    "case_id": "test_case",
                    "document_id": "test-en",
                    "question": "Question?",
                    "reference_answer": "Answer.",
                    "evidence_pages": [1],
                },
                "retrieval_hit": True,
                "document_sha256": "a" * 64,
                "index_fingerprint": "c" * 64,
                "retrieved_evidence": [],
                "plain": {
                    "answer": {
                        "answer": "Answer.",
                        "supported": True,
                        "citations": [1],
                        "missing_information": "",
                    },
                    "manual_faithfulness": "supported",
                },
                "few_shot_citations": {
                    "answer": {
                        "answer": "Answer.",
                        "supported": True,
                        "citations": [1],
                        "missing_information": "",
                    }
                },
            },
        ),
        retrieval_hit_count=1,
        retrieval_hit_rate=1.0,
    )

    assert summarize_faithfulness(report) == {
        "plain": {
            "supported": 1,
            "partly_supported": 0,
            "unsupported": 0,
            "scored": 1,
            "unscored": 0,
        },
        "few_shot_citations": {
            "supported": 0,
            "partly_supported": 0,
            "unsupported": 0,
            "scored": 0,
            "unscored": 1,
        },
    }
