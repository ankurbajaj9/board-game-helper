import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from docling.chunking import HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk, ParseReport, Rulebook
from board_game_rules.infrastructure.catalog import verify_pdf
from board_game_rules.infrastructure.embeddings import Embeddings


class RulesSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        serializer = ChunkingDocSerializer(doc=doc)
        # Rule text can be nested inside a large diagram.
        serializer.params.traverse_pictures = True
        return serializer


class Parser:
    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=RapidOcrOptions(lang=["en"]),
            document_timeout=settings.timeout_seconds,
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self.options)}
        )

    def fingerprint(self, doc: Rulebook) -> str:
        processing = {
            "schema": 2,
            "pdf_sha256": doc.sha256,
            "document": doc.model_dump(mode="json"),
            "docling": version("docling"),
            "docling_core": version("docling-core"),
            "fastembed": version("fastembed"),
            "embedding_model": self.embeddings.model_name,
            "chunk_tokens": self.settings.chunk_tokens,
            # Disabled enrichment defaults contain unordered sets in Docling.
            "options": self.options.model_dump(
                mode="json",
                exclude={
                    "chart_extraction_options",
                    "picture_description_options",
                    "picture_classification_options",
                    "code_formula_options",
                },
            ),
        }
        return hashlib.sha256(json.dumps(processing, sort_keys=True).encode()).hexdigest()

    def parse(self, doc: Rulebook) -> tuple[list[Chunk], ParseReport]:
        source = verify_pdf(self.settings.project_root, doc, self.settings.max_pdf_bytes)
        fingerprint = self.fingerprint(doc)
        target = self.settings.artifacts_path / doc.document_id / fingerprint
        if target.exists():
            report = ParseReport.model_validate_json((target / "report.json").read_text())
            chunks = [
                Chunk.model_validate(chunk)
                for chunk in json.loads((target / "chunks.json").read_text())
            ]
            if report.fingerprint != fingerprint or len(chunks) != report.chunk_count:
                raise ValueError(f"Invalid parsed cache for {doc.document_id}")
            return chunks, report

        result = self.converter.convert(source, max_file_size=self.settings.max_pdf_bytes)
        if result.status != ConversionStatus.SUCCESS:
            raise ValueError(f"Docling conversion failed: {result.status}; {result.errors}")
        document = result.document
        if len(document.pages) != doc.pdf_pages:
            raise ValueError(f"Docling page count does not match catalog for {doc.document_id}")
        chunks = self.chunk(document, doc)
        covered = {page for chunk in chunks for page in chunk.pdf_pages}
        if not covered.issubset(document.pages):
            raise ValueError("Chunk provenance refers to a nonexistent PDF page.")
        report = ParseReport(
            document_id=doc.document_id,
            pdf_pages=len(document.pages),
            chunk_count=len(chunks),
            max_chunk_tokens=max(chunk.token_count for chunk in chunks),
            pages_without_chunks=tuple(sorted(set(document.pages) - covered)),
            picture_count=len(document.pictures),
            fingerprint=fingerprint,
            artifact_directory=str(target),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent, prefix="staging-") as temporary:
            stage = Path(temporary)
            document.save_as_json(stage / "document.json")
            (stage / "chunks.json").write_text(
                json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2)
            )
            (stage / "report.json").write_text(report.model_dump_json(indent=2))
            stage.rename(target)
        return chunks, report

    def chunk(self, document: DoclingDocument, doc: Rulebook) -> list[Chunk]:
        identity = f"{doc.title}\nEdition: {doc.edition}\n"
        budget = self.settings.chunk_tokens - self.embeddings.count_tokens(identity) - 4
        if budget < 32:
            raise ValueError("Document identity leaves too little room for rule text.")
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=self.embeddings.tokenizer, max_tokens=budget),
            merge_peers=True,
            serializer_provider=RulesSerializerProvider(),
        )
        chunks: list[Chunk] = []
        for item in chunker.chunk(dl_doc=document):
            if not item.text.strip():
                continue
            if not isinstance(item.meta, DocMeta):
                raise ValueError("Docling emitted incompatible chunk metadata.")
            pages = tuple(sorted({p.page_no for entry in item.meta.doc_items for p in entry.prov}))
            if not pages:
                raise ValueError("Docling emitted rule text without PDF-page provenance.")
            text = identity + chunker.contextualize(chunk=item)
            tokens = self.embeddings.count_tokens(text)
            if tokens > self.settings.chunk_tokens:
                raise ValueError(
                    f"Contextualized chunk exceeds {self.settings.chunk_tokens} tokens; "
                    "adjust the chunk budget before indexing."
                )
            chunks.append(
                Chunk(
                    document_id=doc.document_id,
                    index=len(chunks),
                    text=item.text,
                    contextualized_text=text,
                    pdf_pages=pages,
                    headings=tuple(item.meta.headings or ()),
                    token_count=tokens,
                )
            )
        if not chunks:
            raise ValueError("Docling produced no usable chunks.")
        return chunks
