import hashlib
from pathlib import Path

from board_game_rules.domain.models import Catalog, Rulebook


def rulebook_path(root: Path, doc: Rulebook) -> Path:
    relative = Path(doc.local_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"Rulebook path must be relative and cannot traverse parents: {doc.local_path}"
        )
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "data" / "rulebooks").resolve()):
        raise ValueError(f"Rulebook must be inside data/rulebooks: {doc.local_path}")
    return path


def load_catalog(path: Path, root: Path) -> Catalog:
    catalog = Catalog.model_validate_json(path.read_text())
    for doc in catalog.documents:
        rulebook_path(root, doc)
    return catalog


def verify_pdf(root: Path, doc: Rulebook, max_bytes: int) -> Path:
    path = rulebook_path(root, doc)
    size = path.stat().st_size
    if size > max_bytes or size != doc.size_bytes:
        raise ValueError(f"PDF size limit or catalog size mismatch: {doc.document_id}")
    with path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise ValueError(f"Not a PDF: {doc.document_id}")
        file.seek(0)
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    if checksum != doc.sha256:
        raise ValueError(f"PDF checksum mismatch: {doc.document_id}")
    return path
