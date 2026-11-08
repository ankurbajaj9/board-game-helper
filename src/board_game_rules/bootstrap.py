from dataclasses import dataclass
from typing import TYPE_CHECKING

from board_game_rules.application.answering import answer_question
from board_game_rules.config import Settings
from board_game_rules.domain.models import (
    Answer,
    Catalog,
    Evidence,
    ParseReport,
    QueryObserver,
    Rulebook,
)
from board_game_rules.infrastructure.catalog import load_catalog, verify_pdf
from board_game_rules.infrastructure.ollama import Ollama
from board_game_rules.infrastructure.qdrant import Vectors
from board_game_rules.infrastructure.state import IndexState

if TYPE_CHECKING:
    from board_game_rules.infrastructure.docling import Parser
    from board_game_rules.infrastructure.embeddings import Embeddings


def catalog(settings: Settings) -> Catalog:
    original = load_catalog(settings.catalog_path, settings.project_root)
    documents = {doc.document_id: doc for doc in original.documents}
    state = IndexState(settings.state_path)
    for doc in state.documents():
        documents[doc.document_id] = doc
    for document_id, title in state.title_overrides().items():
        doc = documents.get(document_id)
        if doc is not None:
            documents[document_id] = Rulebook.model_validate(
                {**doc.model_dump(mode="json"), "title": title}
            )
    return Catalog(schema_version=1, documents=tuple(documents.values()))


def rename_document(settings: Settings, document_id: str, title: str) -> Rulebook:
    doc = catalog(settings).document(document_id)
    renamed = Rulebook.model_validate({**doc.model_dump(mode="json"), "title": title})
    IndexState(settings.state_path).set_title_override(document_id, renamed.title)
    return renamed


def verified_documents(settings: Settings) -> list[str]:
    result = []
    for doc in catalog(settings).documents:
        verify_pdf(settings.project_root, doc, settings.max_pdf_bytes)
        result.append(doc.document_id)
    return result


def ingestion_status(settings: Settings) -> list[dict[str, str | None]]:
    return IndexState(settings.state_path).attempts()


@dataclass
class Parsing:
    embeddings: Embeddings
    parser: Parser


def parsing(settings: Settings) -> Parsing:
    from board_game_rules.infrastructure.docling import Parser
    from board_game_rules.infrastructure.embeddings import Embeddings

    embeddings = Embeddings(settings)
    return Parsing(embeddings, Parser(settings, embeddings))


def local_services(
    settings: Settings, observer: QueryObserver | None = None
) -> tuple[Vectors, Ollama]:
    return Vectors(settings, observer), Ollama(settings, observer)


def ask_selected(
    settings: Settings,
    services: Parsing,
    document_ids: tuple[str, ...],
    question: str,
    observer: QueryObserver | None = None,
) -> tuple[Answer, list[Evidence], tuple[ParseReport, ...]]:
    if not question.strip():
        raise ValueError("Please enter a question.")
    if not document_ids or len(set(document_ids)) != len(document_ids):
        raise ValueError("Select at least one rulebook, without duplicates.")
    state = IndexState(settings.state_path)
    title_overrides = state.title_overrides()
    active_documents = []
    for document_id in document_ids:
        active = state.active(document_id)
        if active is None:
            raise ValueError(
                f"Rulebook {document_id} is not indexed yet. Run the Prefect folder scan first."
            )
        if active[2] != settings.vector_collection:
            raise ValueError(
                f"Rulebook {document_id} needs re-indexing for the configured embedding model."
            )
        doc = active[0]
        if title := title_overrides.get(document_id):
            doc = Rulebook.model_validate({**doc.model_dump(mode="json"), "title": title})
        active_documents.append((doc, active[1], active[2]))
    documents = tuple(active[0] for active in active_documents)
    reports = tuple(active[1] for active in active_documents)
    versions = tuple((doc.document_id, report.fingerprint) for doc, report, _ in active_documents)
    vectors, ollama = (
        local_services(settings) if observer is None else local_services(settings, observer)
    )
    try:
        if observer:
            observer("Checking Gemma model", {"model": settings.ollama_model})
        ollama.check_model()
        if observer:
            observer(
                "Embedding question",
                {"question": question, "embedding_model": settings.embedding_model},
            )
        query_vector = services.embeddings.query(question)
        evidence = (
            vectors.search_indexed(
                settings.vector_collection,
                versions[0][0],
                versions[0][1],
                query_vector,
                settings.retrieval_limit,
            )
            if len(versions) == 1
            else vectors.search_selected(
                settings.vector_collection, versions, query_vector, settings.retrieval_limit
            )
        )
        answer, evidence = answer_question(documents, question, evidence, ollama, settings)
        return answer, evidence, reports
    finally:
        vectors.close()
