from typing import Protocol

from board_game_rules.config import Settings
from board_game_rules.domain.models import Answer, Evidence, PromptStrategy, Rulebook


class Generator(Protocol):
    def generate(self, prompt: str, strategy: PromptStrategy = "plain") -> Answer: ...


def answer_question(
    doc: Rulebook | tuple[Rulebook, ...],
    query: str,
    evidence: list[Evidence],
    generator: Generator,
    settings: Settings,
    prompt_strategy: PromptStrategy = "plain",
) -> tuple[Answer, list[Evidence]]:
    if not query.strip():
        raise ValueError("Please enter a question.")
    documents = (doc,) if isinstance(doc, Rulebook) else doc
    if not documents:
        raise ValueError("Select at least one rulebook.")
    document_ids = {item.document_id for item in documents}
    if any(entry.chunk.document_id not in document_ids for entry in evidence):
        raise ValueError("Evidence must belong to the selected document.")
    if not evidence:
        return Answer(
            answer="No matching rule evidence was found.",
            supported=False,
            citations=(),
            missing_information="The selected document supplied no matching evidence.",
        ), []
    selected = evidence[: settings.retrieval_limit]
    while selected:
        prompt = build_prompt(doc, query, selected)
        # UTF-8 bytes are a conservative token bound; reserve space for the chat template.
        if len(prompt.encode("utf-8")) + 512 + settings.answer_tokens <= settings.context_tokens:
            result = generator.generate(prompt, prompt_strategy)
            allowed_ids = {entry.citation_id for entry in selected}
            if not set(result.citations).issubset(allowed_ids):
                raise ValueError("The model cited evidence that was not supplied.")
            return result, selected
        selected = selected[:-1]
    raise ValueError("Question and evidence do not fit the generation context window.")


def build_prompt(doc: Rulebook | tuple[Rulebook, ...], query: str, evidence: list[Evidence]) -> str:
    documents = (doc,) if isinstance(doc, Rulebook) else doc
    scope = "\n\n".join(
        f"Document: {item.document_id}\nGame: {item.title}\nEdition: {item.edition}"
        for item in documents
    )
    passages = "\n\n".join(
        f"[{entry.citation_id}] Document: {entry.chunk.document_id} "
        f"PDF pages {entry.chunk.pdf_pages}\n"
        f"{entry.chunk.contextualized_text}"
        for entry in evidence
    )
    return f"Selected manuals:\n{scope}\nQuestion: {query.strip()}\n\nEvidence:\n{passages}"
