import hashlib
import re
import time
from datetime import date
from pathlib import Path

import pypdfium2 as pdfium

from board_game_rules.config import Settings
from board_game_rules.domain.models import Catalog, Rulebook
from board_game_rules.infrastructure.catalog import rulebook_path


def stable_pdfs(settings: Settings) -> list[Path]:
    settings.rulebooks_path.mkdir(parents=True, exist_ok=True)
    return [
        path
        for path in sorted(settings.rulebooks_path.rglob("*"))
        if path.suffix.lower() == ".pdf"
        and path.is_file()
        and time.time() - path.stat().st_mtime >= settings.file_settle_seconds
    ]


def inspect_pdf(path: Path, catalog: Catalog, settings: Settings) -> Rulebook:
    root = settings.project_root
    if path.is_symlink() or not path.resolve().is_relative_to(settings.rulebooks_path.resolve()):
        raise ValueError("Symlinks and paths outside the watched folder are not accepted.")
    before = path.stat()
    if before.st_size > settings.max_pdf_bytes:
        raise ValueError("PDF exceeds the configured file size limit.")
    with path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise ValueError("File does not contain a PDF.")
        file.seek(0)
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    local_path = path.relative_to(root).as_posix()
    known = next((doc for doc in catalog.documents if doc.local_path == local_path), None)
    if known and checksum == known.sha256 and before.st_size == known.size_bytes:
        return known
    with pdfium.PdfDocument(path) as pdf:
        page_count = len(pdf)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("PDF changed during inspection; it will be retried on the next scan.")
    metadata = {
        "sha256": checksum,
        "size_bytes": before.st_size,
        "pdf_pages": page_count,
        "downloaded_at": date.today(),
    }
    if known:
        return Rulebook.model_validate({**known.model_dump(), **metadata})
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "rulebook"
    path_hash = hashlib.sha256(local_path.encode()).hexdigest()[:12]
    doc = Rulebook(
        document_id=f"local-{slug[:60]}-{path_hash}",
        game_id=f"local-{slug[:60]}-{path_hash}",
        title=path.stem.replace("_", " ").replace("-", " "),
        edition="Unspecified local upload",
        language="en",
        kind="standalone",
        base_game_id=None,
        publisher="Local upload; publisher unverified",
        source_page=None,
        download_url=None,
        local_path=local_path,
        sha256=checksum,
        size_bytes=before.st_size,
        pdf_pages=page_count,
        downloaded_at=date.today(),
    )
    rulebook_path(root, doc)
    return doc
