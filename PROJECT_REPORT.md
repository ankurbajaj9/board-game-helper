# Board-Game Rules RAG Project Report

## Abstract

This project implements a local retrieval-augmented generation (RAG) assistant for
answering questions about published board-game rules. PDF manuals are parsed into
page-aware chunks, embedded, and indexed in Qdrant. A local Ollama-hosted model
generates structured answers with citations to retrieved passages. The evaluation
workflow compares plain prompting with few-shot citation prompting using the
same retrieval result for each question. It measures page-level retrieval hits
and is configured to score answer faithfulness and factual correctness with Ragas.
A first run completed retrieval and answer generation for all benchmark cases;
its results and scoring limitations are summarized below.

## Problem and Objectives

The goal is to build a working chatbot over a defined collection of rulebooks and
apply course concepts in embeddings, vector retrieval, prompt engineering, and
chatbot development. The main comparison asks whether a few-shot example that
demonstrates citation behavior changes answer quality relative to a plain prompt.
Answers should be grounded in the selected manuals, cite supporting PDF pages, and
abstain when retrieved evidence is insufficient.

## Scope

The project catalog contains ten registered rulebooks. The fixed benchmark contains
20 question/reference-answer cases spanning four documents: Azul, Catan 2025,
Love Letter, and Pandemic. Each case records expected evidence pages. The user
interface can search selected manuals from the larger catalog, so the implemented
application is broader than a strictly single-document collection; the benchmark
remains a fixed, bounded subset.

The application and services are intended for local development. Public or
multi-user deployment, web import, multimodal question input, comparative
chunking/embedding experiments, automatic expansion compatibility resolution,
and expansion-precedence handling are outside the current implementation.

## Reproducing the Project

The repository is [board-game-helper](https://github.com/ankurbajaj9/board-game-helper).
The setup below assumes macOS. Install [Git](https://git-scm.com/downloads),
[uv](https://docs.astral.sh/uv/getting-started/installation/),
[Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/), and
[Ollama](https://ollama.com/download/mac). Python 3.14 is required; `uv` can install
it with `uv python install 3.14`. Start Docker Desktop and the Ollama application
before running the commands below. Alternatively, run `ollama serve` in a separate
terminal if no Ollama server is already running. Ollama listens at
`http://localhost:11434`.

```bash
git clone https://github.com/ankurbajaj9/board-game-helper.git
cd board-game-helper
uv python install 3.14
uv sync --extra evaluation
ollama pull gemma4:e2b
docker compose up -d --build
```

The PDF manuals are not included in the repository. Obtain copies you are
permitted to process and place them at these catalog paths:

```text
data/rulebooks/azul-en.pdf
data/rulebooks/catan-2025-en.pdf
data/rulebooks/love-letter-en.pdf
data/rulebooks/pandemic-en.pdf
```

The download and publisher pages are listed in
[catalog/rulebooks.json](catalog/rulebooks.json). Use the exact editions and
PDF bytes described there: the benchmark checks document hashes, and a changed
publisher PDF is not automatically an equivalent benchmark source. Do not change
the recorded hashes just to bypass a mismatch.

The
folder watcher normally indexes newly added PDFs automatically. To trigger a
scan and verify status explicitly:

```bash
docker compose exec watcher board-game-rules-watch scan
docker compose exec watcher board-game-rules-watch status
uv run board-game-rules list
uv run board-game-rules doctor
```

Wait until the services are healthy and the scan completes before evaluating.
Check startup and ingestion errors with `docker compose ps` and
`docker compose logs watcher`. The `verify` command checks every catalog manual,
not just the four benchmark PDFs; use `uv run board-game-rules verify` after
providing all ten catalog PDFs.

Qdrant is available at `http://localhost:6789`; the Prefect dashboard is at
`http://127.0.0.1:4201`. The first ingestion downloads parsing and embedding
models and may take longer than subsequent scans.

To use the question-answering interface:

```bash
uv run board-game-rules-ui
```

Open `http://localhost:8501`, select indexed manuals, and ask a question. Answers
show citations, PDF pages, and supporting passages.

After all four benchmark manuals are indexed and verified, run the evaluation
and summarize its saved report:

```bash
uv run --extra evaluation board-game-rules evaluate
uv run --extra evaluation board-game-rules score data/evaluation/results.json
```

The report is local and ignored by Git. It contains generated answers and
retrieved rulebook passages; review it locally and do not commit it or share it
without checking document rights and privacy implications. Docker/Qdrant and
Ollama must remain available during evaluation.

For automated application checks, run `uv run pytest`, `uv run ruff check src tests`,
and `uv run ty check src`. Use `uv run --extra evaluation pytest` to include the
optional Ragas integration test. Stop the project containers with
`docker compose stop`; do not delete local data or use `docker compose down -v`
when you need to preserve indexes and downloaded model caches. Ollama is a
separate service and is not stopped by Compose.

## System Design

### Ingestion and Indexing

PDFs are discovered from the local rulebook directory or catalog. A Prefect folder
watcher inspects stable files, parses them with Docling and HybridChunker, and
preserves PDF-page provenance in chunks. FastEmbed generates passage vectors using
`BAAI/bge-small-en-v1.5`; input lengths and vector dimensions are validated rather
than silently truncated. Versioned vectors are published to Qdrant, and SQLite
tracks which document generation is active. Failed replacements do not activate
incomplete data.

### Retrieval and Answer Generation

At question time, the application embeds the query and searches the active document
versions in Qdrant. The configured retrieval limit is six by default. Retrieved
passages, document identity, and edition are placed into the prompt. Ollama runs
`gemma4:e2b` and returns a structured answer containing the response, support
status, citation IDs, and missing information. Application validation rejects
citations that were not present in the supplied evidence and prevents an answer
marked supported from omitting citations.

The Streamlit interface lets users select one or more manuals and displays the
answer, cited PDF pages, source passages, and a query trace. The trace exposes the
retrieval request and model request for inspection. Retrieval is limited to the
selected active documents; the UI does not automatically infer expansion
compatibility.

### Prompt Variants

The `plain` strategy uses the standard evidence-grounded prompt. The
`few_shot_citations` strategy adds a brief example showing a concise answer with a
citation. The benchmark retrieves once per question and supplies the same evidence
to both strategies, so the comparison isolates prompting rather than retrieval
variation.

## Evaluation Method

The hand-authored benchmark is constrained to 15-20 cases and currently contains
20. Each case includes a question, short reference answer, document ID, and
expected PDF evidence pages. Before evaluation, the runner validates that source
documents are indexed, match the catalog PDF hashes, and use the configured vector
collection.

Retrieval hit rate is the proportion of cases where at least one retrieved chunk
belongs to the expected document and overlaps an expected evidence page. This is a
benchmark-specific retrieval measure; it does not establish that the generated
answer is correct.

When the optional evaluation dependencies are installed, Ragas scores each answer
for faithfulness to retrieved context and factual correctness against the reference
answer. The configured local Ollama model acts as the evaluator. Scores are stored
per prompt variant and summarized separately. Human reviewers may also label
answers as `supported`, `partly_supported`, or `unsupported`; these manual labels
remain distinct from model-generated scores.

### Running the Evaluation

After indexing the benchmark manuals and starting Qdrant and Ollama:

```bash
uv sync --extra evaluation
uv run --extra evaluation board-game-rules evaluate
```

The report is written to `data/evaluation/results.json`. Review answers and
evidence, optionally fill in the manual faithfulness labels, and summarize the
report with:

```bash
uv run board-game-rules score data/evaluation/results.json
```

## Security Considerations

Retrieved passages and user questions are treated as untrusted input in the system
prompt. Answers use a constrained schema, and citations are checked against the
retrieved evidence. A unit test verifies that the untrusted-evidence instruction
is sent to Ollama. These controls reduce malformed or fabricated citations but do
not prove resistance to prompt injection.

The optional adversarial experiment described in the proposal has not been run as
a controlled test: no malicious instruction was planted in a rulebook and compared
before and after mitigation. A complete security result would require such a
fixture, an explicit success criterion, and recorded before/after outcomes.

## Results and Limitations

The 20-question benchmark completed retrieval and generated both prompt variants
for every case. The results are saved locally in
[`data/evaluation/results.json`](data/evaluation/results.json).

| Observed output | Plain | Few-shot citations |
| --- | ---: | ---: |
| Answers marked supported by Gemma | 20/20 (100%) | 19/20 (95%) |
| Answers containing at least one citation | 20/20 (100%) | 19/20 (95%) |

Retrieval was shared between variants: 18/20 cases (90%) retrieved at least one
chunk from the annotated document and expected PDF page. There are no human
faithfulness labels yet. The `supported` and citation counts above are model
outputs, not independent judgments of correctness, and do not show that either
prompt is more accurate.

Ragas scoring did not complete. The local Gemma judge returned a truncated
structured faithfulness response at the configured output limit, so no Ragas
faithfulness or factual-correctness scores are included in the saved report. The
Ragas integration needs a judge configuration that reliably returns its
structured verdicts before those scores can be reported. The optional
prompt-injection experiment also remains unrun.

Page overlap and valid citations are useful checks, not guarantees of correctness.
Text extraction may lose meaning from icon-heavy tables, diagrams, or layout.
Important rulings should be checked against the original PDF. The local-only
configuration is not hardened for remote or multi-user access.

## Conclusion

The project demonstrates the core RAG pipeline and provides a controlled
plain-versus-few-shot comparison over a fixed question set. The first run reports
a 90% page-level retrieval hit rate and model-generated support/citation counts.
Manual answer review and a reliable Ragas judge run are still needed before drawing
conclusions about comparative answer quality. The controlled prompt-injection
experiment also remains future work.