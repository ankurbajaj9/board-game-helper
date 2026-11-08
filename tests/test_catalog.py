from pathlib import Path

import pytest
from pydantic import ValidationError

from board_game_rules.config import Settings
from board_game_rules.domain.models import Catalog, Rulebook
from board_game_rules.infrastructure.catalog import load_catalog, rulebook_path, verify_pdf


def test_real_catalog_loads_known_rulebooks() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(root / "catalog/rulebooks.json", root)
    love_letter = catalog.document("love-letter-en")
    assert love_letter.title == "Love Letter"
    assert love_letter.pdf_pages == 16


def test_duplicate_catalog_document(rulebook: Rulebook) -> None:
    with pytest.raises(ValidationError, match="Duplicate document"):
        Catalog(schema_version=1, documents=(rulebook, rulebook))


def test_missing_expansion_base(rulebook: Rulebook) -> None:
    data = rulebook.model_dump(mode="json")
    data.update(kind="expansion", base_game_id="other", compatible_base_document_ids=["missing"])
    expansion = Rulebook.model_validate(data)
    with pytest.raises(ValidationError, match="Invalid compatible base"):
        Catalog(schema_version=1, documents=(expansion,))


def test_expansions_cannot_be_compatible_bases(rulebook: Rulebook) -> None:
    data = rulebook.model_dump(mode="json")
    data.update(kind="expansion", base_game_id="other", compatible_base_document_ids=["other-en"])
    first = Rulebook.model_validate(data)
    data.update(
        document_id="other-en",
        game_id="other",
        base_game_id="test",
        compatible_base_document_ids=["test-en"],
    )
    with pytest.raises(ValidationError, match="Invalid compatible base"):
        Catalog(schema_version=1, documents=(first, Rulebook.model_validate(data)))


@pytest.mark.parametrize("path", ["../escape.pdf", "/tmp/escape.pdf", "catalog/test.pdf"])
def test_path_escape_rejected(settings: Settings, rulebook: Rulebook, path: str) -> None:
    doc = rulebook.model_copy(update={"local_path": path})
    with pytest.raises(ValueError):
        rulebook_path(settings.project_root, doc)


def test_symlink_escape_rejected(settings: Settings, rulebook: Rulebook) -> None:
    path = settings.project_root / rulebook.local_path
    path.unlink()
    target = settings.project_root / "outside.pdf"
    target.write_bytes(b"%PDF-1.4")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="inside data/rulebooks"):
        rulebook_path(settings.project_root, rulebook)


def test_verify_signature_and_hash(settings: Settings, rulebook: Rulebook) -> None:
    path = verify_pdf(settings.project_root, rulebook, settings.max_pdf_bytes)
    path.write_bytes(b"%PDF-1.4" + b"x" * (rulebook.size_bytes - 8))
    with pytest.raises(ValueError, match="checksum"):
        verify_pdf(settings.project_root, rulebook, settings.max_pdf_bytes)
    path.write_bytes(b"x" * rulebook.size_bytes)
    with pytest.raises(ValueError, match="Not a PDF"):
        verify_pdf(settings.project_root, rulebook, settings.max_pdf_bytes)


def test_size_limit(settings: Settings, rulebook: Rulebook) -> None:
    with pytest.raises(ValueError, match="size"):
        verify_pdf(settings.project_root, rulebook, 5)


def test_unknown_catalog_document_is_rejected(rulebook: Rulebook) -> None:
    catalog = Catalog(schema_version=1, documents=(rulebook,))
    with pytest.raises(ValueError, match="Unknown"):
        catalog.document("unknown")


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_tokens": 0},
        {"timeout_seconds": 0},
        {"answer_tokens": 8192},
        {"retrieval_limit": 0},
        {"ollama_url": "not a URL"},
    ],
)
def test_invalid_settings(tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, _env_file=None, **overrides)
