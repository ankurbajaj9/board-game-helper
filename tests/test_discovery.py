import os
import time
from unittest.mock import MagicMock

import pytest

from board_game_rules.config import Settings
from board_game_rules.domain.models import Catalog, Rulebook
from board_game_rules.infrastructure.discovery import inspect_pdf, stable_pdfs


def test_partial_copy_waits_until_settled(settings: Settings, rulebook: Rulebook) -> None:
    path = settings.project_root / rulebook.local_path
    assert stable_pdfs(settings) == []
    old = time.time() - settings.file_settle_seconds - 1
    os.utime(path, (old, old))
    assert stable_pdfs(settings) == [path]


def test_known_unchanged_pdf_keeps_metadata(settings: Settings, rulebook: Rulebook) -> None:
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    assert inspect_pdf(settings.project_root / rulebook.local_path, catalog, settings) == rulebook


def test_new_pdf_gets_local_identity_and_no_fake_publisher_url(
    settings: Settings, rulebook: Rulebook, monkeypatch
) -> None:
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    path = settings.rulebooks_path / "New_Game.PDF"
    path.write_bytes(b"%PDF-1.4 synthetic new rulebook")
    pdf = MagicMock()
    pdf.__enter__.return_value.__len__.return_value = 2
    monkeypatch.setattr(
        "board_game_rules.infrastructure.discovery.pdfium.PdfDocument", lambda p: pdf
    )
    doc = inspect_pdf(path, catalog, settings)
    assert doc.title == "New Game"
    assert doc.document_id.startswith("local-new-game-")
    assert doc.download_url is None
    assert doc.source_page is None
    assert doc.pdf_pages == 2
    assert doc.edition == "Unspecified local upload"


def test_bad_pdf_is_an_explicit_error(settings: Settings, rulebook: Rulebook) -> None:
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    path = settings.rulebooks_path / "bad.pdf"
    path.write_text("not a PDF")
    with pytest.raises(ValueError, match="does not contain"):
        inspect_pdf(path, catalog, settings)


def test_symlink_is_not_followed(settings: Settings, rulebook: Rulebook) -> None:
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    path = settings.rulebooks_path / "linked.pdf"
    path.symlink_to(settings.project_root / rulebook.local_path)
    with pytest.raises(ValueError, match="Symlinks"):
        inspect_pdf(path, catalog, settings)
