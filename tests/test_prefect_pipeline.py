from unittest.mock import Mock

import pytest

from board_game_rules.config import Settings
from board_game_rules.domain.models import Rulebook
from board_game_rules.infrastructure.prefect_pipeline import scan_rulebooks
from board_game_rules.infrastructure.state import IndexState


def test_scan_records_errors_and_continues_other_files(
    settings: Settings, rulebook: Rulebook, monkeypatch
) -> None:
    import board_game_rules.infrastructure.prefect_pipeline as pipeline
    from board_game_rules.domain.models import Catalog

    bad = settings.rulebooks_path / "bad.pdf"
    bad.write_bytes(b"invalid")
    known = settings.project_root / rulebook.local_path
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    monkeypatch.setattr(pipeline, "get_run_logger", lambda: Mock())
    monkeypatch.setattr(pipeline, "stable_pdfs", lambda s: [bad, known])
    monkeypatch.setattr(pipeline.bootstrap, "catalog", lambda settings: catalog)
    monkeypatch.setattr(pipeline.bootstrap, "parsing", lambda s: Mock())

    def ingest(path, *args):
        if path == bad:
            raise ValueError("Invalid PDF")
        return "unchanged"

    ingest_mock = Mock(side_effect=ingest)
    monkeypatch.setattr(pipeline, "ingest_pdf", ingest_mock)
    with pytest.raises(RuntimeError, match="1 PDF"):
        scan_rulebooks.fn(str(settings.project_root))
    assert ingest_mock.call_count == 2
    attempts = IndexState(settings.state_path).attempts()
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["error"] == "Invalid PDF"


def test_scan_supplies_renamed_metadata_to_ingestion(
    settings: Settings, rulebook: Rulebook, monkeypatch
) -> None:
    import board_game_rules.infrastructure.prefect_pipeline as pipeline
    from board_game_rules.domain.models import Catalog

    settings.catalog_path.parent.mkdir(parents=True)
    settings.catalog_path.write_text(
        Catalog(schema_version=1, documents=(rulebook,)).model_dump_json()
    )
    IndexState(settings.state_path).set_title_override(rulebook.document_id, "New Display Name")
    monkeypatch.setattr(pipeline, "get_run_logger", lambda: Mock())
    monkeypatch.setattr(
        pipeline, "stable_pdfs", lambda s: [settings.project_root / rulebook.local_path]
    )
    monkeypatch.setattr(pipeline.bootstrap, "parsing", lambda s: Mock())
    ingest = Mock(return_value="unchanged")
    monkeypatch.setattr(pipeline, "ingest_pdf", ingest)

    assert scan_rulebooks.fn(str(settings.project_root)) == {
        "indexed": 0,
        "unchanged": 1,
        "failed": 0,
    }
    passed_catalog = ingest.call_args.args[1]
    assert passed_catalog.document(rulebook.document_id).title == "New Display Name"


def test_scans_are_serialized(settings: Settings, monkeypatch) -> None:
    import fcntl

    import board_game_rules.infrastructure.prefect_pipeline as pipeline

    IndexState(settings.state_path)
    lock_path = settings.project_root / "data/ingestion.lock"
    monkeypatch.setattr(pipeline, "get_run_logger", lambda: Mock())
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        counts = scan_rulebooks.fn(str(settings.project_root))
    assert counts == {"indexed": 0, "unchanged": 0, "failed": 0}
