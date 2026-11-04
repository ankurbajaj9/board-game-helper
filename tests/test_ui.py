from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.testing.v1 import AppTest

from board_game_rules import bootstrap
from board_game_rules.domain.models import Answer, Catalog, Chunk, Evidence, ParseReport, Rulebook
from board_game_rules.infrastructure.catalog import load_catalog
from board_game_rules.interfaces import ui


def run_ui() -> None:
    from board_game_rules.interfaces.ui import main

    main()


@pytest.fixture
def ask(monkeypatch) -> Mock:
    root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(root / "catalog/rulebooks.json", root)
    monkeypatch.setattr(bootstrap, "catalog", lambda settings: catalog)
    monkeypatch.setattr(bootstrap, "ingestion_status", lambda settings: [])
    monkeypatch.setattr(ui, "parsing_services", lambda settings: Mock())
    chunk = Chunk(
        document_id="catan-2025-en",
        index=0,
        text="The player with the highest roll starts.",
        contextualized_text="CATAN\nThe player with the highest roll starts.",
        pdf_pages=(5,),
        token_count=15,
    )
    report = ParseReport(
        document_id=chunk.document_id,
        pdf_pages=12,
        chunk_count=54,
        max_chunk_tokens=380,
        pages_without_chunks=(),
        picture_count=71,
        fingerprint="test",
        artifact_directory="data/parsed/test",
    )
    ask = Mock(
        return_value=(
            Answer(
                answer="Highest roll starts.",
                supported=True,
                citations=(1,),
                missing_information="",
            ),
            [Evidence(citation_id=1, chunk=chunk, score=0.8)],
            (report,),
        )
    )
    monkeypatch.setattr(bootstrap, "ask_selected", ask)
    return ask


def test_answer_and_source_are_shown(ask: Mock) -> None:
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(root / "catalog/rulebooks.json", root)
    assert app.multiselect[0].options == [doc.title for doc in catalog.documents]
    app.text_input[0].set_value(" Who starts? ")
    app.button[0].click().run()
    assert not app.exception
    assert ask.call_args.args[2:] == (("catan-2025-en",), "Who starts?")
    assert any("Highest roll starts." in item.value for item in app.markdown)
    assert any("PDF pages 5" in item.value for item in app.markdown)
    app.multiselect[0].set_value(["azul-en"]).run()
    assert not app.exception
    assert not any("Highest roll starts." in item.value for item in app.markdown)


def test_blank_question_does_not_call_models(ask: Mock) -> None:
    app = AppTest.from_function(run_ui).run()
    app.button[0].click().run()
    assert app.error[0].value == "Please enter a question."
    ask.assert_not_called()


def test_local_rulebook_has_no_generic_disclaimer(
    ask: Mock, rulebook: Rulebook, monkeypatch
) -> None:
    local_rulebook = rulebook.model_copy(update={"download_url": None, "source_page": None})
    catalog = Catalog(schema_version=1, documents=(local_rulebook,))
    monkeypatch.setattr(bootstrap, "catalog", lambda settings: catalog)

    app = AppTest.from_function(run_ui).run()

    assert not app.exception
    assert app.multiselect[0].value == [local_rulebook.document_id]
    assert not app.warning
    assert not any(
        "Searches only the selected manuals" in caption.value for caption in app.caption
    )
    ask.assert_not_called()


def test_adapter_failure_is_visible_and_clears_old_answer(ask: Mock) -> None:
    app = AppTest.from_function(run_ui).run()
    app.text_input[0].set_value("Who starts?")
    app.button[0].click().run()
    ask.side_effect = ValueError("Qdrant unavailable")
    app.button[0].click().run()
    assert not app.exception
    assert "Qdrant unavailable" in app.error[0].value
    assert "answer_result" not in app.session_state


def test_expansion_scope_is_explicit(ask: Mock) -> None:
    app = AppTest.from_function(run_ui).run()
    app.multiselect[0].set_value(["terraforming-mars-prelude-en"]).run()
    assert any("base-game rules are not included" in item.value for item in app.warning)


def test_unsupported_answer_is_not_displayed_as_success(ask: Mock) -> None:
    answer, evidence, report = ask.return_value
    ask.return_value = (
        Answer(
            answer="The manual does not specify this.",
            supported=False,
            citations=(),
            missing_information="Not covered in this rulebook.",
        ),
        evidence,
        report,
    )
    app = AppTest.from_function(run_ui).run()
    app.text_input[0].set_value("An unsupported question")
    app.button[0].click().run()
    assert not app.exception
    assert any("Not covered" in item.value for item in app.warning)


def test_trace_is_displayed_persisted_and_cleared_on_scope_change(ask: Mock, monkeypatch) -> None:
    result = ask.return_value
    updates = []
    original_update = StatusContainer.update

    def record_update(self, **kwargs):
        updates.append(kwargs)
        return original_update(self, **kwargs)

    monkeypatch.setattr(StatusContainer, "update", record_update)

    def traced_ask(*args, observer):
        observer("Querying Qdrant", {"request": {"query": [0.1, 0.2], "limit": 6}})
        observer(
            "Retrieved chunks",
            {"count": 1, "chunks": [result[1][0].model_dump(mode="json")]},
        )
        observer(
            "Asking Gemma",
            {"request": {"messages": [{"role": "user", "content": "Exact final prompt"}]}},
        )
        return result

    ask.side_effect = traced_ask
    app = AppTest.from_function(run_ui).run()
    app.text_input[0].set_value("Who starts?")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.json) == 3
    assert "Exact final prompt" in app.json[2].value
    assert len(app.session_state.query_trace) == 3
    assert len(updates) == 4
    assert all(update["expanded"] is True for update in updates)
    app.run()
    assert not app.exception
    assert len(app.json) == 3
    app.multiselect[0].set_value(["azul-en"]).run()
    assert not app.json
    assert "query_trace" not in app.session_state


def test_failed_query_keeps_partial_trace_and_next_attempt_clears_it(ask: Mock) -> None:
    def failing_ask(*args, observer):
        observer("Querying Qdrant", {"request": {"query": [0.1, 0.2]}})
        raise ValueError("Qdrant unavailable")

    ask.side_effect = failing_ask
    app = AppTest.from_function(run_ui).run()
    app.text_input[0].set_value("Who starts?")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.json) == 1
    assert app.session_state.query_failed
    app.run()
    assert len(app.json) == 1
    app.text_input[0].set_value("")
    app.button[0].click().run()
    assert not app.json
    assert "query_trace" not in app.session_state


def test_empty_selection_does_not_call_models(ask: Mock) -> None:
    app = AppTest.from_function(run_ui).run()
    app.multiselect[0].set_value([]).run()
    app.text_input[0].set_value("Who starts?")
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Select at least one rulebook."
    ask.assert_not_called()


def test_multiple_selection_maps_each_citation_to_its_manual(ask: Mock) -> None:
    answer, evidence, reports = ask.return_value
    second_chunk = evidence[0].chunk.model_copy(
        update={"document_id": "catan-cities-knights-2025-en", "pdf_pages": (3,)}
    )
    second_report = reports[0].model_copy(update={"document_id": second_chunk.document_id})
    ask.return_value = (
        answer.model_copy(update={"citations": (1, 2)}),
        [*evidence, Evidence(citation_id=2, chunk=second_chunk, score=0.7)],
        (*reports, second_report),
    )
    app = AppTest.from_function(run_ui).run()
    app.multiselect[0].set_value(["catan-2025-en", "catan-cities-knights-2025-en"]).run()
    assert not any("base-game rules are not included" in item.value for item in app.warning)
    app.text_input[0].set_value("Who starts?")
    app.button[0].click().run()
    assert not app.exception
    assert ask.call_args.args[2] == ("catan-2025-en", "catan-cities-knights-2025-en")
    assert any(
        "[2] CATAN: Cities & Knights" in item.value and "PDF pages 3" in item.value
        for item in app.markdown
    )


def test_library_form_saves_the_edited_game_name(ask: Mock, monkeypatch) -> None:
    rename = Mock()
    monkeypatch.setattr(bootstrap, "rename_document", rename)
    app = AppTest.from_function(run_ui).run()
    library = app.expander[-1]

    title_input = next(
        item for item in library.get("text_input") if item.key == "title-catan-2025-en"
    )
    title_input.set_value("My Display Name")
    save_button = next(
        item for item in library.get("form_submit_button") if item.label == "Save name"
    )
    save_button.click().run()

    assert not app.exception
    rename.assert_called_once_with(ui.Settings(), "catan-2025-en", "My Display Name")
