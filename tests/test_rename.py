import pytest

from board_game_rules import bootstrap
from board_game_rules.config import Settings
from board_game_rules.domain.models import Catalog, Rulebook
from board_game_rules.infrastructure.state import IndexState


def test_renamed_document_is_persisted_and_used_by_the_catalog(
    settings: Settings, rulebook: Rulebook
) -> None:
    settings.catalog_path.parent.mkdir(parents=True)
    settings.catalog_path.write_text(
        Catalog(schema_version=1, documents=(rulebook,)).model_dump_json()
    )

    renamed = bootstrap.rename_document(settings, rulebook.document_id, "New Display Name")

    assert renamed.title == "New Display Name"
    restarted_state = IndexState(settings.state_path)
    assert restarted_state.title_overrides() == {rulebook.document_id: "New Display Name"}
    assert bootstrap.catalog(settings).document(rulebook.document_id).title == "New Display Name"
    assert (
        settings.catalog_path.read_text()
        == Catalog(schema_version=1, documents=(rulebook,)).model_dump_json()
    )


@pytest.mark.parametrize("title", ["", " " * 3, "x" * 121])
def test_invalid_document_title_is_not_persisted(
    settings: Settings, rulebook: Rulebook, title: str
) -> None:
    settings.catalog_path.parent.mkdir(parents=True)
    settings.catalog_path.write_text(
        Catalog(schema_version=1, documents=(rulebook,)).model_dump_json()
    )

    with pytest.raises(ValueError):
        bootstrap.rename_document(settings, rulebook.document_id, title)

    assert IndexState(settings.state_path).title_overrides() == {}
