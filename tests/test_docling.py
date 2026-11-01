from unittest.mock import Mock

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk, Rulebook
from board_game_rules.infrastructure.docling import Parser
from board_game_rules.infrastructure.embeddings import Embeddings


@pytest.fixture
def embeddings() -> Embeddings:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "cards": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    adapter = Embeddings.__new__(Embeddings)
    adapter.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=512)
    adapter.model_name = "synthetic"
    adapter.max_tokens = 512
    adapter.dimension = 2
    return adapter


def document(text: str, with_provenance: bool = True) -> DoclingDocument:
    doc = DoclingDocument(name="Synthetic Game")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    provenance = ProvenanceItem(
        page_no=1, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, len(text))
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=text,
        prov=provenance if with_provenance else None,
    )
    return doc


def test_hybrid_chunker_token_budget_and_provenance(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings
) -> None:
    parser = Parser(settings, embeddings)
    doc = document("Each player chooses two cards. " * 200)
    chunks = parser.chunk(doc, rulebook)
    assert len(chunks) > 1
    assert all(chunk.pdf_pages == (1,) for chunk in chunks)
    assert all(chunk.token_count <= settings.chunk_tokens for chunk in chunks)
    assert all(
        embeddings.count_tokens(chunk.contextualized_text) == chunk.token_count for chunk in chunks
    )
    assert all(rulebook.title in chunk.contextualized_text for chunk in chunks)


def test_missing_provenance_is_an_error(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings
) -> None:
    parser = Parser(settings, embeddings)
    with pytest.raises(ValueError, match="provenance"):
        parser.chunk(document("Each player chooses two cards.", False), rulebook)


def test_rules_inside_picture_are_not_dropped(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings
) -> None:
    doc = DoclingDocument(name="Illustrated Rules")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    provenance = ProvenanceItem(
        page_no=1, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, 30)
    )
    picture = doc.add_picture(prov=provenance)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Choose the first player by rolling dice.",
        prov=provenance,
        parent=picture,
    )
    chunks = Parser(settings, embeddings).chunk(doc, rulebook)
    assert any("rolling dice" in chunk.text for chunk in chunks)
    assert all(chunk.pdf_pages == (1,) for chunk in chunks)


def test_unchanged_parse_uses_cache(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings, chunk: Chunk
) -> None:
    parser = Parser(settings, embeddings)
    converter = Mock()
    converter.convert.return_value.status = ConversionStatus.SUCCESS
    converter.convert.return_value.document = document("Each player chooses two cards.")
    parser.converter = converter
    parser.chunk = Mock(return_value=[chunk])
    first, report = parser.parse(rulebook)
    second, cached = parser.parse(rulebook)
    assert first == second
    assert report == cached
    converter.convert.assert_called_once()


def test_partial_conversion_is_not_cached(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings
) -> None:
    parser = Parser(settings, embeddings)
    converter = Mock()
    converter.convert.return_value.status = ConversionStatus.PARTIAL_SUCCESS
    converter.convert.return_value.errors = ["Missing page"]
    parser.converter = converter
    with pytest.raises(ValueError, match="conversion failed"):
        parser.parse(rulebook)
    assert not settings.artifacts_path.exists()


def test_unused_enrichment_sets_do_not_change_fingerprint(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings
) -> None:
    first, second = Parser(settings, embeddings), Parser(settings, embeddings)
    # This disabled feature's supported engines serialize in hash-randomized order.
    second.options.chart_extraction_options.model_spec.supported_engines = None
    assert first.fingerprint(rulebook) == second.fingerprint(rulebook)
    second.options.do_ocr = False
    assert first.fingerprint(rulebook) != second.fingerprint(rulebook)


def test_display_title_change_requires_reingestion(
    settings: Settings, rulebook: Rulebook, embeddings: Embeddings
) -> None:
    parser = Parser(settings, embeddings)
    renamed = rulebook.model_copy(update={"title": "New Display Name"})

    assert parser.fingerprint(rulebook) != parser.fingerprint(renamed)


def test_embeddings_reject_truncation_and_invalid_vectors(embeddings: Embeddings) -> None:
    with pytest.raises(ValueError, match="truncation"):
        embeddings.validate_text("cards " * 600)
    with pytest.raises(ValueError, match="invalid vector"):
        embeddings.validate_vectors([[float("nan"), 0.0]], 1)
    with pytest.raises(ValueError, match="vector count"):
        embeddings.validate_vectors([], 1)
