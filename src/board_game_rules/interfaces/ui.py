"""Simple local question-and-answer screen."""

import logging
import sys
from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from board_game_rules import bootstrap
from board_game_rules.config import Settings
from board_game_rules.domain.models import Answer, Evidence, ParseReport, Rulebook

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading local embedding and parsing models...")
def parsing_services(settings: Settings) -> bootstrap.Parsing:
    return bootstrap.parsing(settings)


def render_answer(
    documents: tuple[Rulebook, ...],
    answer: Answer,
    evidence: list[Evidence],
    reports: tuple[ParseReport, ...],
) -> None:
    st.subheader("Answer")
    st.write(answer.answer)
    if not answer.supported:
        st.warning(answer.missing_information)
    cited = [entry for entry in evidence if entry.citation_id in answer.citations]
    if cited:
        st.subheader("Sources")
    for entry in cited:
        doc = next(doc for doc in documents if doc.document_id == entry.chunk.document_id)
        pages = ", ".join(str(page) for page in entry.chunk.pdf_pages)
        st.markdown(
            f"**[{entry.citation_id}] {doc.title}** — PDF pages {pages}  \nEdition: {doc.edition}"
        )
        if doc.download_url:
            st.link_button(
                f"Open publisher PDF · source {entry.citation_id}",
                f"{doc.download_url}#page={entry.chunk.pdf_pages[0]}",
            )
        else:
            st.caption(f"Local PDF: {doc.local_path} · page {entry.chunk.pdf_pages[0]}")
        with st.expander(f"Supporting passage [{entry.citation_id}]"):
            st.write(entry.chunk.text)
    with st.expander("Parsing details"):
        for report in reports:
            doc = next(doc for doc in documents if doc.document_id == report.document_id)
            st.write(f"{doc.title}: {report.pdf_pages} pages · {report.chunk_count} chunks")
            if report.pages_without_chunks:
                st.warning(
                    f"{doc.title}: pages without indexed text: {report.pages_without_chunks}"
                )
        st.caption("Extracted text is used for answers; diagrams may need manual inspection.")


def render_trace_event(
    stage: str, details: dict[str, object], container: DeltaGenerator | None = None
) -> None:
    expander = st.expander(stage) if container is None else container.expander(stage)
    with expander:
        st.json(details, expanded=2)


def main() -> None:
    st.set_page_config(page_title="Board Game Rules", page_icon="🎲", layout="centered")
    st.title("Board Game Rules")
    st.caption("Ask your rulebook. Answers use local models and include source pages.")
    try:
        settings = Settings()
        catalog = bootstrap.catalog(settings)
    except Exception as exc:  # UI boundary: log and display startup failures.
        logger.exception("Cannot load rulebook library")
        st.error(f"Cannot load the library: {exc}")
        return

    documents = {doc.document_id: doc for doc in catalog.documents}
    document_ids = tuple(
        st.multiselect(
            "Rulebooks",
            list(documents),
            default=list(documents)[:1],
            format_func=lambda doc_id: documents[doc_id].title,
        )
    )
    selected_docs = tuple(documents[doc_id] for doc_id in document_ids)
    st.caption(f"Model: {settings.ollama_model}")
    if message := st.session_state.pop("rename_message", None):
        st.success(message)
    for doc in selected_docs:
        st.caption(f"{doc.title} · Edition: {doc.edition}")
    for doc in selected_docs:
        if doc.kind == "expansion" and not set(doc.compatible_base_document_ids).intersection(
            document_ids
        ):
            st.warning(
                f"{doc.title}: compatible base-game rules are not included. "
                "Select a base manual explicitly."
            )
    if len(selected_docs) > 1:
        st.warning(
            "Select compatible editions for a combined ruleset. "
            "Unrelated manuals are searched too if selected; rules are not automatically merged."
        )

    if st.session_state.get("answer_document") != document_ids:
        st.session_state.pop("answer_result", None)
        st.session_state.pop("query_trace", None)
        st.session_state.pop("query_failed", None)
        st.session_state.answer_document = document_ids

    with st.form("ask"):
        question = st.text_input("Question", placeholder="How do we choose the first player?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        st.session_state.pop("answer_result", None)
        st.session_state.pop("query_trace", None)
        st.session_state.pop("query_failed", None)
        if not document_ids:
            st.error("Select at least one rulebook.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            st.session_state.query_trace = []
            trace = st.status("Loading local models...", expanded=True)

            def observe(stage: str, details: dict[str, object]) -> None:
                st.session_state.query_trace.append((stage, details))
                trace.update(label=stage, expanded=True)
                render_trace_event(stage, details, trace)

            try:
                services = parsing_services(settings)
                st.session_state.answer_result = bootstrap.ask_selected(
                    settings, services, document_ids, question.strip(), observer=observe
                )
                trace.update(label="Query complete", state="complete", expanded=True)
            except Exception as exc:  # UI boundary: log and display adapter failures.
                logger.exception("Rulebook question failed")
                st.session_state.query_failed = True
                trace.update(label="Query failed", state="error", expanded=True)
                st.error(f"Could not answer: {exc}")

    if not submitted and "query_trace" in st.session_state:
        failed = st.session_state.get("query_failed", False)
        with st.status(
            "Query failed" if failed else "Query complete",
            state="error" if failed else "complete",
            expanded=True,
        ):
            for stage, details in st.session_state.query_trace:
                render_trace_event(stage, details)

    if "answer_result" in st.session_state:
        render_answer(selected_docs, *st.session_state.answer_result)
    with st.expander("Library"):
        for item in catalog.documents:
            with st.form(f"rename-{item.document_id}"):
                new_title = st.text_input(
                    "Game / rulebook name",
                    value=item.title,
                    max_chars=120,
                    key=f"title-{item.document_id}",
                )
                renamed = st.form_submit_button("Save name")
            if renamed:
                try:
                    updated = bootstrap.rename_document(
                        settings, item.document_id, new_title.strip()
                    )
                    st.session_state.pop("answer_result", None)
                    st.session_state.pop("query_trace", None)
                    st.session_state.pop("query_failed", None)
                    st.session_state.rename_message = (
                        f"Saved name: {updated.title}. The next folder scan will refresh "
                        "its indexed context."
                    )
                    st.rerun()
                except (
                    Exception
                ) as exc:  # UI boundary: surface invalid names and persistence errors.
                    logger.exception("Could not rename rulebook %s", item.document_id)
                    st.error(f"Could not save name: {exc}")
            st.caption(f"{item.pdf_pages} PDF pages")
        st.caption(
            "Add an English PDF to data/rulebooks while the Prefect watcher is running. "
            "Refresh this page after indexing. Unknown manuals use the filename as their title."
        )
        for status in bootstrap.ingestion_status(settings):
            st.caption(f"{status['status']} · {status['path']}")
            if status["error"]:
                st.error(status["error"])


def launch() -> None:
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        "127.0.0.1",
        *sys.argv[1:],
    ]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
