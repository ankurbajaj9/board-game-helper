# Board Game Rules RAG

Local rules assistant in development using **Docling parsing + HybridChunker**, local
embeddings, **Qdrant**, and **Gemma 4 through Ollama**, with cited answers for a selected game
and edition.

## Features

- Validated PDF inputs and source/edition metadata.
- Optional source registry in [catalog/rulebooks.json](catalog/rulebooks.json).
- CLI commands to validate PDFs, parse/chunk them, and evaluate RAG answers.
- A fixed question set and CLI evaluation comparing plain and few-shot cited
  answers using the same retrieved evidence.
- Architecture, phased todo list, and acceptance tests in
  [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**The simple UI and local Prefect folder pipeline are implemented.** The UI now
queries persistent vectors rather than parsing/embedding a manual per question.
Multiple selected manuals can be searched together; automatic compatibility and
expansion-precedence resolution remain pending.

## Automatic PDF ingestion with local Prefect

Start Qdrant, the Prefect server, and the PDF watcher together:

```bash
docker compose up -d --build
```

Prefect UI: **http://127.0.0.1:4201**. The dedicated API, profiles, and SQLite
database stay under `data/prefect/`; this does not use the other project's
Prefect instance on port 4200 or Prefect Cloud.
Only localhost port 4201 is published; the watcher connects to Prefect and
Qdrant through the private Compose network. Ollama and the Streamlit UI stay
native on the Mac; Ollama retains Apple Silicon GPU acceleration.
The first watcher build installs the locked Python dependencies. Model downloads
on first ingestion are persisted in the Compose `models` volume.

Drop an English `.pdf` into **`data/rulebooks/`** (subfolders and uppercase
`.PDF` extensions are supported). The watcher polls every 30 seconds and waits
until the file's last modification is at least 10 seconds old. Prefer copying
to a `.part` filename and renaming to `.pdf` when complete.

```text
Folder scan -> inspect PDF -> Docling parse/chunk -> local embeddings
            -> versioned Qdrant upsert -> validate count -> activate in SQLite
```

Each changed file has a Prefect subflow with inspection, parsing, embedding,
and publishing tasks. Completed files with the same content, metadata, and
processing configuration are skipped. Missing Qdrant data is rebuilt.
Publishing has two bounded retries; failed files are logged and retried on the
next scan, while other files continue. Only one scan can hold the local lock.

For existing catalog PDFs, publisher/edition metadata is preserved. Unknown
files get a stable local ID and filename-based title, with **unverified English
language, edition, and publisher**. Automatic ingestion does not identify a game
or infer expansions from its name. Add accurate metadata to the source catalog
before importing when edition/expansion identity matters. Originals are not
modified and no publisher URL is fabricated.

Manual operations (the local server must be running for `scan`):

```bash
docker compose exec watcher board-game-rules-watch scan
docker compose exec watcher board-game-rules-watch status
docker compose logs -f watcher prefect
```

Failed replacements do not replace the last good index. The UI retrieves only
the active document generation, even when old/staging vectors exist. Old
generations are retained for safety; automatic garbage collection and deletion
on source-file removal are intentionally not implemented. Back up originals,
`data/parsed/`, `data/ingestion.sqlite`, and Qdrant storage together.

Use `docker compose stop watcher` to pause scanning and
`docker compose up -d watcher` to resume. Containers restart automatically when
Docker restarts unless explicitly stopped. Docker Desktop must be running.
`docker compose down` stops these services without deleting bind-mounted data
or the model volume; do not use `down -v` unless you intend to delete model caches.
This is a local single-worker polling pipeline, not an instant file event hook.
Defaults live in Settings; `BGR_SCAN_INTERVAL_SECONDS`, `BGR_FILE_SETTLE_SECONDS`,
and `BGR_PREFECT_URL` can be set in the watcher's Compose environment.
For native development, the `board-game-rules-watch server` and `watch` commands
are still available; do not run native and container watchers simultaneously.

## Simple web UI

```bash
uv run board-game-rules-ui
```

Open http://localhost:8501, select one or more rulebooks, and ask a question. Answers show
the edition, cited PDF pages, publisher PDF links, and supporting passages.
The Library expander lists manuals and ingestion errors, and lets you edit each
manual's display name. Names are saved in local SQLite state without changing the
source catalog or document IDs. The next folder scan refreshes the indexed context.
The server binds to localhost.

Clicking **Ask** shows live stages in a query trace. The top-level trace stays
expanded during processing and after completion or failure. Expand **Querying Qdrant**
to inspect the complete question vector, collection, active-document filter, and
result limit. **Retrieved chunks** includes every returned passage, page,
heading, and similarity score, not just passages cited in the answer.
**Asking Gemma** shows the exact Ollama request, including system instructions,
the final question/evidence prompt, model, output schema, and generation options.
The final prompt can contain fewer chunks when the context budget requires it.
The trace remains visible after completion or failure and resets for the next
submission or a different manual. It stays in the browser's server-side session;
it is not written to a trace log or sent to an external tracing service.

The UI searches **exactly the selected manuals** with a single query against
their active document versions. The retrieval limit is shared across the selection,
not applied per manual; not every manual is guaranteed a matching passage.
Select a compatible base manual explicitly when using expansions. Unrelated
games or editions can be selected, so prompts instruct the model to keep their
rules separate rather than invent a merged ruleset. Sources identify each manual.
Changing the selection clears the previous answer and trace. Parsed documents and
model resources are reused. **Run the folder scan before asking about a manual**;
unindexed documents return a visible error rather than triggering ingestion.
Refresh the UI after a new file is indexed to see its filename-based entry.
First-time model loading can take longer. Web uploads, automatic compatibility
validation, and expansion-precedence resolution are not implemented yet.

## Run the first slice

Run from this project directory. Python 3.14, uv, Docker, and Ollama are required.
The tested lockfile includes Docling 2.145.0, PyTorch 2.15.0, and FastEmbed 0.9.1.

```bash
uv sync
docker compose up -d
# Start Ollama if needed, then install the selected model if it is missing:
ollama pull gemma4:e2b

uv run board-game-rules list
uv run board-game-rules verify
uv run board-game-rules doctor
uv run board-game-rules parse DOCUMENT_ID
```

Replace `DOCUMENT_ID` with an ID returned by `board-game-rules list`.

## Prompt evaluation

After the benchmark manuals have been indexed, run the 20-question evaluation:

```bash
uv run board-game-rules evaluate
```

For each question, the runner retrieves evidence once and supplies those same
passages to both the plain and few-shot citation prompts. The report records
the expected document/page hit, retrieved passages, both answers, model and
retrieval settings, and an initially empty `manual_faithfulness` field. The
retrieval hit rate measures whether a returned passage contains the annotated
document and page; it does not judge answer quality.

The report is written to `data/evaluation/results.json` (ignored by Git). Review
each answer against its evidence, fill in `manual_faithfulness` as
`supported`, `partly_supported`, or `unsupported`, then summarize the labels:

```bash
uv run board-game-rules score data/evaluation/results.json
```

The question set is in [evaluation/questions.json](evaluation/questions.json);
reference answers are short paraphrases with expected PDF pages, not copied
rulebook text.

`parse` caches the DoclingDocument, chunks, and diagnostics under `data/parsed/`.
The automated tests cover answer citations and the indexed RAG request path.
Inspect parse reports for picture counts and page coverage: a successful
conversion does not prove all diagrams or rules have been understood.

Models download on first use. Parsing and generation stay local; model artifact
downloads require internet access. Defaults live in
[Settings](src/board_game_rules/config.py); override them with `BGR_` environment
variables or a local `.env` (see [.env.example](.env.example)).
Use `--root /absolute/project/path` before the command when running elsewhere.

Ollama's installed **`gemma4:e2b` (Q4_K_M)** was verified with structured output.
The configured context is 8,192 tokens, below its declared 131,072-token limit.
The application never silently switches models. Embeddings use
`BAAI/bge-small-en-v1.5` with its matching tokenizer, 384-dimensional vectors,
and a 512-token input limit. Final contextualized chunks are checked against the
configured 384-token budget, including document identity and special tokens.

Qdrant is bound to localhost on HTTP **6789** and gRPC **6790**, with a pinned
image, persistent local storage, and a readiness health check.
Stop only this project's container with `docker compose down`; its data remains.
Ollama is shared and is not started or stopped by Compose.

## Simple code separation

```text
src/board_game_rules/
  config.py          # validated settings and defaults
  bootstrap.py       # wire concrete adapters
  domain/models.py   # contracts and edition/expansion validation
  application/       # answering and index publication
  evaluation/        # fixed benchmark and prompt-comparison tooling
  infrastructure/    # document discovery, state, Prefect, and vendor adapters
  interfaces/        # CLI, watcher commands, and simple Streamlit screen
```

Keep the hierarchy shallow. Parser and chunker share one Docling adapter; there
are no empty web modules or generic repository wrappers. One local Prefect flow
coordinates folder ingestion; no extra queue framework is added. Add separation
only when implemented functionality needs it. See the
[dependency rules](IMPLEMENTATION_PLAN.md#separation-rules).

## Validation

```bash
uv run pytest
uv run ruff check src tests
uv run ty check src
uv build
```

The implementation tests unchanged scans, failed replacements, version filters,
restarted state, stable-file detection, and indexed UI queries without re-embedding.
It also tests UI answer/source display, scope changes, blank input, unsupported
answers, and visible errors. A regression test preserves extracted rules nested
inside pictures. Upstream Docling emits an OCR deprecation warning
during configuration serialization; the tests do not suppress it.

## Limitations and document rights

Page coverage and valid citations do not guarantee answer accuracy. Parsing can
flatten icon-labelled tables into unlabelled numbers, and text-only retrieval
cannot reliably interpret every diagram. Check table-sensitive rulings against
the original PDF. Edition, variant, and expansion compatibility must be explicit.

PDFs and generated data are ignored by Git and stay local with the default
configuration. Publisher availability is not a redistribution license; do not
commit rulebooks to a public repository or send their contents to third-party
models without permission.

Add documents through the watched folder and optional catalog metadata, without
game-specific code changes. Web import and metadata editing remain planned.

## AI governance, ethics, and security

- **Governance:** Evaluation reports record the question-set hash, selected model,
  embedding model, retrieval settings, and indexed document hashes. Retrieval
  hits and citations are measurable checks, not proof that an answer is correct.
  Faithfulness labels are intentionally left for a person to review; do not
  report a model-generated score as human evaluation.
- **Ethics:** Answers are limited to the selected published rules and edition,
  not house rules or general authority. The interface shows supporting passages
  and citations, and unsupported answers should remain unsupported. Use only
  documents you are permitted to process; local storage does not grant
  redistribution rights.
- **Security:** Retrieved text and questions are treated as untrusted input.
  Structured output validation and citation-ID checks reduce malformed or
  fabricated references. These controls do not make prompt injection impossible:
  inspect results from untrusted documents and verify important rulings in the
  source PDF. The default local setup is for one trusted machine, not remote or
  multi-user deployment; do not expose Qdrant or Ollama without a separate
  authentication, transport-security, and data-sharing design.

This is a student project, not a certified governance or security assessment.
