import hashlib
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import Field, model_validator

from board_game_rules.application.answering import Generator, answer_question
from board_game_rules.config import Settings
from board_game_rules.domain.models import (
    Answer,
    Catalog,
    Evidence,
    Model,
    ParseReport,
    Rulebook,
)

type Faithfulness = Literal["supported", "partly_supported", "unsupported"]


class EvaluationCase(Model):
    case_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    evidence_pages: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_pages(self) -> EvaluationCase:
        if any(page < 1 for page in self.evidence_pages):
            raise ValueError("Expected PDF pages must be positive and 1-based.")
        if len(set(self.evidence_pages)) != len(self.evidence_pages):
            raise ValueError("Expected PDF pages must be unique.")
        return self


class EvaluationDataset(Model):
    cases: tuple[EvaluationCase, ...] = Field(min_length=15, max_length=20)

    @model_validator(mode="after")
    def validate_case_ids(self) -> EvaluationDataset:
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique.")
        return self


class PromptResult(Model):
    answer: Answer
    manual_faithfulness: Faithfulness | None = None
    ragas_faithfulness: float | None = Field(default=None, ge=0, le=1)
    ragas_factual_correctness: float | None = Field(default=None, ge=0, le=1)


class EvaluatedCase(Model):
    case: EvaluationCase
    retrieval_hit: bool
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    retrieved_evidence: tuple[Evidence, ...]
    plain: PromptResult
    few_shot_citations: PromptResult


class EvaluationReport(Model):
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    completed_at: datetime
    collection: str
    embedding_model: str
    ollama_model: str
    ragas_judge_model: str | None = None
    retrieval_limit: int = Field(ge=1)
    context_tokens: int = Field(gt=0)
    answer_tokens: int = Field(gt=0)
    cases: tuple[EvaluatedCase, ...] = Field(min_length=1)
    retrieval_hit_count: int = Field(ge=0)
    retrieval_hit_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_retrieval_summary(self) -> EvaluationReport:
        hits = sum(case.retrieval_hit for case in self.cases)
        rate = round(hits / len(self.cases), 4)
        if self.retrieval_hit_count != hits or self.retrieval_hit_rate != rate:
            raise ValueError("Retrieval summary does not match the evaluated cases.")
        return self


class ActiveIndex(Protocol):
    def active(self, document_id: str) -> tuple[Rulebook, ParseReport, str] | None: ...


class QueryEmbedder(Protocol):
    def query(self, text: str) -> list[float]: ...


class EvaluationStore(Protocol):
    def search_indexed(
        self, collection: str, document_id: str, fingerprint: str, vector: list[float], limit: int
    ) -> list[Evidence]: ...


def load_dataset(path: Path) -> tuple[EvaluationDataset, str]:
    contents = path.read_bytes()
    return EvaluationDataset.model_validate_json(contents), hashlib.sha256(contents).hexdigest()


def run_evaluation(
    cases: Sequence[EvaluationCase],
    dataset_sha256: str,
    catalog: Catalog,
    active_index: ActiveIndex,
    embeddings: QueryEmbedder,
    store: EvaluationStore,
    generator: Generator,
    settings: Settings,
) -> EvaluationReport:
    results = []
    for case in cases:
        source_document = catalog.document(case.document_id)
        if any(page > source_document.pdf_pages for page in case.evidence_pages):
            raise ValueError(
                f"Expected evidence page for {case.document_id} exceeds its PDF page count."
            )
        active = active_index.active(case.document_id)
        if active is None:
            raise ValueError(
                f"Rulebook {case.document_id} is not indexed yet. "
                "Run the Prefect folder scan first."
            )
        doc, report, collection = active
        if doc.document_id != case.document_id or report.document_id != case.document_id:
            raise ValueError(f"Active index metadata does not match {case.document_id}.")
        if doc.sha256 != source_document.sha256:
            raise ValueError(
                f"Indexed PDF for {case.document_id} differs from the benchmark source."
            )
        if collection != settings.vector_collection:
            raise ValueError(
                f"Rulebook {case.document_id} needs re-indexing for the configured embedding model."
            )

        query_vector = embeddings.query(case.question)
        evidence = store.search_indexed(
            collection,
            case.document_id,
            report.fingerprint,
            query_vector,
            settings.retrieval_limit,
        )
        retrieval_hit = any(
            entry.chunk.document_id == case.document_id
            and set(entry.chunk.pdf_pages).intersection(case.evidence_pages)
            for entry in evidence
        )
        plain, plain_evidence = answer_question(
            doc, case.question, evidence, generator, settings, prompt_strategy="plain"
        )
        few_shot, few_shot_evidence = answer_question(
            doc,
            case.question,
            evidence,
            generator,
            settings,
            prompt_strategy="few_shot_citations",
        )
        if plain_evidence != few_shot_evidence:
            raise ValueError("Prompt strategies must use the same retrieved evidence.")
        results.append(
            EvaluatedCase(
                case=case,
                retrieval_hit=retrieval_hit,
                document_sha256=doc.sha256,
                index_fingerprint=report.fingerprint,
                retrieved_evidence=tuple(plain_evidence),
                plain=PromptResult(answer=plain),
                few_shot_citations=PromptResult(answer=few_shot),
            )
        )

    hit_count = sum(case.retrieval_hit for case in results)
    return EvaluationReport(
        dataset_sha256=dataset_sha256,
        completed_at=datetime.now(UTC),
        collection=settings.vector_collection,
        embedding_model=settings.embedding_model,
        ollama_model=settings.ollama_model,
        retrieval_limit=settings.retrieval_limit,
        context_tokens=settings.context_tokens,
        answer_tokens=settings.answer_tokens,
        cases=tuple(results),
        retrieval_hit_count=hit_count,
        retrieval_hit_rate=round(hit_count / len(results), 4),
    )


def summarize_faithfulness(report: EvaluationReport) -> dict[str, dict[str, int]]:
    labels: tuple[Faithfulness, ...] = ("supported", "partly_supported", "unsupported")
    summary = {}
    for strategy in ("plain", "few_shot_citations"):
        counts = {
            label: sum(
                getattr(case, strategy).manual_faithfulness == label for case in report.cases
            )
            for label in labels
        }
        scored = sum(counts.values())
        summary[strategy] = {**counts, "scored": scored, "unscored": len(report.cases) - scored}
    return summary


def ragas_samples(report: EvaluationReport) -> list[dict[str, object]]:
    samples = []
    for evaluated in report.cases:
        contexts = [item.chunk.contextualized_text for item in evaluated.retrieved_evidence]
        for result in (evaluated.plain, evaluated.few_shot_citations):
            sample: dict[str, object] = {
                "user_input": evaluated.case.question,
                "response": result.answer.answer,
                "retrieved_contexts": contexts,
                "reference": evaluated.case.reference_answer,
            }
            samples.append(sample)
    return samples


def evaluate_with_ragas(report: EvaluationReport, settings: Settings) -> EvaluationReport:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    try:
        from openai import OpenAI
        from ragas import EvaluationDataset as RagasDataset
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationResult
        from ragas.llms import llm_factory
        from ragas.metrics.base import Metric
        from ragas.metrics.collections import FactualCorrectness, Faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise ValueError(
            "Ragas evaluation dependencies are missing. Install them with "
            "`uv sync --extra evaluation`."
        ) from exc

    client = OpenAI(
        base_url=f"{str(settings.ollama_url).rstrip('/')}/v1",
        api_key="ollama",
        timeout=settings.timeout_seconds,
    )
    try:
        judge = llm_factory(
            model=settings.ollama_model,
            client=client,
            temperature=0,
            max_tokens=settings.answer_tokens,
            extra_body={
                "options": {
                    "num_ctx": settings.context_tokens,
                    "num_predict": settings.answer_tokens,
                }
            },
        )
        metrics = cast(
            list[Metric],
            [Faithfulness(llm=judge), FactualCorrectness(llm=judge, mode="f1")],
        )
        result = cast(
            EvaluationResult,
            evaluate(
                dataset=RagasDataset.from_list(ragas_samples(report)),
                metrics=metrics,
                run_config=RunConfig(
                    timeout=settings.timeout_seconds,
                    max_retries=2,
                    max_workers=1,
                ),
                raise_exceptions=True,
                show_progress=False,
            ),
        )
    finally:
        client.close()
    return _apply_ragas_scores(report, result.scores, settings.ollama_model)


def _apply_ragas_scores(
    report: EvaluationReport, scores: Sequence[dict[str, object]], judge_model: str
) -> EvaluationReport:
    scored_cases = []
    factual_score_key = "factual_correctness(mode=f1)"
    if len(scores) != len(report.cases) * 2:
        raise ValueError("Ragas returned an unexpected number of prompt scores.")
    for index, evaluated in enumerate(report.cases):
        prompt_results = []
        for prompt_index, prompt_result in enumerate(
            (evaluated.plain, evaluated.few_shot_citations)
        ):
            prompt_scores = scores[index * 2 + prompt_index]
            prompt_results.append(
                PromptResult.model_validate(
                    {
                        **prompt_result.model_dump(),
                        "ragas_faithfulness": prompt_scores["faithfulness"],
                        "ragas_factual_correctness": prompt_scores[factual_score_key],
                    }
                )
            )
        scored_cases.append(
            evaluated.model_copy(
                update={"plain": prompt_results[0], "few_shot_citations": prompt_results[1]}
            )
        )

    return report.model_copy(
        update={"cases": tuple(scored_cases), "ragas_judge_model": judge_model}
    )


def summarize_ragas(report: EvaluationReport) -> dict[str, dict[str, float | int]]:
    summary = {}
    for strategy in ("plain", "few_shot_citations"):
        results = [getattr(case, strategy) for case in report.cases]
        faithfulness = [
            result.ragas_faithfulness
            for result in results
            if result.ragas_faithfulness is not None
        ]
        factual_correctness = [
            result.ragas_factual_correctness
            for result in results
            if result.ragas_factual_correctness is not None
        ]
        summary[strategy] = {
            "faithfulness": round(sum(faithfulness) / len(faithfulness), 4)
            if faithfulness
            else 0.0,
            "factual_correctness": round(sum(factual_correctness) / len(factual_correctness), 4)
            if factual_correctness
            else 0.0,
            "scored": min(len(faithfulness), len(factual_correctness)),
            "unscored": len(report.cases) - min(len(faithfulness), len(factual_correctness)),
        }
    return summary