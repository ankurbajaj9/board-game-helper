import fcntl
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from board_game_rules import bootstrap
from board_game_rules.application.ingestion import is_indexed, publish_vectors
from board_game_rules.config import Settings
from board_game_rules.domain.models import Catalog, Chunk, ParseReport, Rulebook
from board_game_rules.infrastructure.discovery import inspect_pdf, stable_pdfs
from board_game_rules.infrastructure.state import IndexState


@task(name="Inspect PDF", cache_policy=NO_CACHE, persist_result=False)
def inspect_source(path: Path, catalog: Catalog, settings: Settings) -> Rulebook:
    return inspect_pdf(path, catalog, settings)


@task(name="Docling parse and chunk", cache_policy=NO_CACHE, persist_result=False)
def parse_source(doc: Rulebook, services: bootstrap.Parsing) -> tuple[list[Chunk], ParseReport]:
    return services.parser.parse(doc)


@task(name="Embed chunks locally", cache_policy=NO_CACHE, persist_result=False)
def embed_chunks(chunks: list[Chunk], services: bootstrap.Parsing) -> list[list[float]]:
    return services.embeddings.passages([chunk.contextualized_text for chunk in chunks])


@task(
    name="Publish Qdrant vectors",
    retries=2,
    retry_delay_seconds=5,
    cache_policy=NO_CACHE,
    persist_result=False,
)
def publish_source(
    doc: Rulebook,
    chunks: list[Chunk],
    vectors: list[list[float]],
    report: ParseReport,
    settings: Settings,
    dimension: int,
) -> None:
    store, _ = bootstrap.local_services(settings)
    try:
        publish_vectors(
            doc,
            chunks,
            vectors,
            report,
            settings,
            dimension,
            IndexState(settings.state_path),
            store,
        )
    finally:
        store.close()


# Local model resources are not Pydantic flow parameters; inspect_source validates the PDF.
@flow(name="Ingest board game PDF", persist_result=False, validate_parameters=False)
def ingest_pdf(
    path: Path, catalog: Catalog, settings: Settings, services: bootstrap.Parsing
) -> str:
    state = IndexState(settings.state_path)
    doc = inspect_source(path, catalog, settings)
    previous = state.active(doc.document_id)
    if previous and previous[0].sha256 == doc.sha256:
        doc = doc.model_copy(update={"downloaded_at": previous[0].downloaded_at})
    store, _ = bootstrap.local_services(settings)
    try:
        if is_indexed(doc, services.parser.fingerprint(doc), settings, state, store):
            state.attempt(doc.local_path, "ready")
            return "unchanged"
    finally:
        store.close()
    state.attempt(doc.local_path, "parsing")
    chunks, report = parse_source(doc, services)
    state.attempt(doc.local_path, "embedding")
    vectors = embed_chunks(chunks, services)
    state.attempt(doc.local_path, "publishing")
    publish_source(doc, chunks, vectors, report, settings, services.embeddings.dimension)
    return "indexed"


@flow(name="Scan rulebook folder", log_prints=True, persist_result=False)
def scan_rulebooks(project_root: str) -> dict[str, int]:
    settings = Settings(project_root=project_root)
    state = IndexState(settings.state_path)
    logger = get_run_logger()
    lock_path = settings.project_root / "data" / "ingestion.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Another folder scan is still running; this scan is skipped.")
            return {"indexed": 0, "unchanged": 0, "failed": 0}
        catalog = bootstrap.catalog(settings)
        counts = {"indexed": 0, "unchanged": 0, "failed": 0}
        services = None
        for path in stable_pdfs(settings):
            local_path = path.relative_to(settings.project_root).as_posix()
            try:
                if services is None:
                    services = bootstrap.parsing(settings)
                outcome = ingest_pdf(path, catalog, settings, services)
                counts[outcome] += 1
                logger.info("%s: %s", local_path, outcome)
            except Exception as exc:  # One bad PDF must not prevent other files from ingesting.
                state.attempt(local_path, "failed", str(exc))
                counts["failed"] += 1
                logger.exception("PDF ingestion failed: %s", local_path)
        logger.info("Scan summary: %s", counts)
        if counts["failed"]:
            raise RuntimeError(
                f"{counts['failed']} PDF(s) failed; see task logs and ingestion status."
            )
        return counts
