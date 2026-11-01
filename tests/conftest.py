import hashlib
from pathlib import Path

import pytest

from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk, Rulebook


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, _env_file=None)


@pytest.fixture
def rulebook(settings: Settings) -> Rulebook:
    content = b"%PDF-1.4 synthetic test fixture"
    path = settings.project_root / "data" / "rulebooks" / "test.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return Rulebook(
        document_id="test-en",
        game_id="test",
        title="Test Game",
        edition="First",
        language="en",
        kind="base",
        base_game_id=None,
        publisher="Test Publisher",
        source_page="https://example.com/rules",
        download_url="https://example.com/rules.pdf",
        local_path="data/rulebooks/test.pdf",
        downloaded_at="2026-10-31",
        pdf_pages=1,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


@pytest.fixture
def chunk(rulebook: Rulebook) -> Chunk:
    return Chunk(
        document_id=rulebook.document_id,
        index=0,
        text="Each player chooses two cards.",
        contextualized_text="Test Game\nEach player chooses two cards.",
        pdf_pages=(1,),
        token_count=12,
    )
