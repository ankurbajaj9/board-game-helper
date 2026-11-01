# Board Game Rules RAG: implementation plan and todo list

## 1. Goal and current status

Build a local application that answers English board game rules questions using
publisher rulebooks, with game/edition filters and traceable PDF-page citations.
Use **Docling for both PDF parsing and native chunking**, and **Qdrant for retrieval**.
An LLM generates a grounded answer; vector search alone is not the final product.

**The CLI, simple question UI, and local Prefect folder ingestion are implemented.
The UI queries persistent active document generations for one or more explicitly
selected manuals. Automatic compatibility/expansion-precedence resolution and
a full library-management UI remain pending.**
The optional
[catalog](catalog/rulebooks.json) records sources, editions, expansion dependencies,
sizes, page counts, download dates, and SHA-256 checksums.

Validate source PDFs for signature, structure, checksums, and language. Inspect
sparse-text pages before classifying them as covers, image-only rules, or missing
content. Preserve foldout geometry and reading order rather than assuming
conventional booklet pagination. Successful conversion and page coverage do not
establish visual completeness or interpretation of every diagram.

Icon-labelled tables can be flattened into unlabelled numbers. Table structure
and unsupported-number handling need further evaluation; a valid page citation
alone does not prove that an answer is supported.

No rulebook text or PDFs should be committed to a public repository by default.
Publisher availability is not a blanket redistribution license. Local user
uploads also remain local; do not send documents to third-party models without
explicit opt-in.

## 2. Product behavior

### Ask Rules page

- Choose a game and document edition explicitly; never retrieve across all games
  silently. Choose compatible expansions separately.
- Ask a question and receive a concise answer, relevant exceptions, and citations
  showing document title, edition, section, and physical PDF page number.
- Open the local PDF at the cited page, with a publisher source link as a fallback.
  Printed page labels, when present, are additional metadata, not a replacement
  for the physical 1-based PDF page number.
- Show supporting passages separately from the generated answer.
- Say that evidence is insufficient when the manuals do not support an answer.
  Do not fill gaps with remembered rules, common house rules, or other editions.
- If a question omits a necessary player count, variant, or expansion choice,
  ask for clarification rather than inventing a ruleset.
- Keep conversation history isolated by selected ruleset. A game or edition
  change resets history; follow-up question rewriting cannot broaden its scope.

### Library page

- List games, editions, base/expansion relationships, source links, and ingestion
  states: not ingested, queued, parsing, chunking, embedding, ready, failed.
- Rename a document's display title through the UI without changing its stable
  document ID or source catalog; persist the override locally and refresh its
  contextualized index on the next folder scan.
- Add an English PDF with title, game ID, edition, and optional base-game link.
  Add future games through catalog data, not new game-specific code branches.
- Download known publisher documents, ingest/re-ingest selected documents, and
  remove indexed documents deliberately without deleting unrelated games.
- Show parse diagnostics, chunk counts, and errors. Adding a source does not make
  it searchable until ingestion passes validation and is published.

## 3. Architecture

```text
Publisher download or local upload
       |
Validated catalog entry + checksum
       |
Docling DocumentConverter
       |  DoclingDocument JSON + page/section provenance
Docling HybridChunker
       |  raw chunk + contextualize(chunk) + source metadata
Local embedding model
       |
Qdrant: board_game_rules_v1
       |
Ruleset-filtered retrieval -> optional measured reranking
       |
Bounded evidence context -> local LLM -> validated cited answer
       |
Streamlit Ask Rules / Library pages, or CLI
```

### Initial technology choices

| Component | Initial choice | Validation before implementation is locked |
|---|---|---|
| Environment | Python managed with uv | Try Python 3.14 first; verify Docling, PyTorch, tokenizer, and embedding wheels on this Mac. Use a documented compatible version if required, rather than assuming the old project's environment still works. |
| Parsing | `docling.document_converter.DocumentConverter` | Test representative PDFs for reading order, tables, scanned text, and page provenance. |
| Chunking | `docling.chunking.HybridChunker` | Use the embedding model's tokenizer and `contextualize()`; no Chonkie or hand-written word splitting. |
| Embeddings | Local FastEmbed, initial candidate `BAAI/bge-small-en-v1.5` | Verify runtime support, effective maximum input length, query/passage encoding, dimension, and measured recall. |
| Vector store | Qdrant Docker on host HTTP 6789, gRPC 6790 | Dedicated Compose project/volume, health check, pinned tested image; leave other Qdrant instances untouched. |
| Generation | Gemma 4 through Ollama, requested model name `gemma4` | Verify the exact installed/available tag, quantization, memory, context window, and structured output on this machine. Do not silently substitute another model if unavailable. |
| UI | Streamlit native navigation | Separate Ask Rules and Library pages. |
| Local operational state | SQLite | Jobs and active document versions; manifest remains the source of truth for source metadata. |
| Tests | pytest; Streamlit AppTest; real-PDF integration suite | Small synthetic fixtures in Git; publisher corpus tests are opt-in local tests. |

### Implemented first milestone (2026-10-31)

A simple local UI is now available via `uv run board-game-rules-ui`: choose one
or more manuals, ask a question, inspect citations and supporting text. It explicitly
warns about expansion scope. Its original temporary-vector workflow was replaced
by active persistent indexes when the requested Prefect pipeline was added.
This is an early usable screen, not completion of the full Phase 5 library workflow.

### Local Prefect folder pipeline

The requested orchestration is now one local Prefect 3 scan deployment, served
with a 30-second interval, plus per-PDF subflows/tasks. A separate server on port
4201 stores its data locally under `data/prefect/`; the other project on 4200 is
not used. The folder is `data/rulebooks/`, with a 10-second modification-age
settle check, PDF signature/size/page inspection, and rejection of symlinks.

Existing catalog metadata is retained; uncatalogued PDFs become local standalone
entries with filename titles and explicitly unverified language/edition/publisher.
English is the import assumption, not an automatic language-detection claim.

Persistent indexing uses a model-specific Qdrant collection, deterministic
document/fingerprint/chunk IDs, and document/version payload indexes. SQLite
publishes the active generation only after vector counts and the final source
checksum pass. Failed replacements leave the previous generation active.
Unchanged versions skip parsing/embedding. Old/staged generations are retained
and excluded by retrieval filters; garbage collection is deferred.

The UI reads active indexed metadata and vectors without re-parsing/re-embedding
on question submission. Ingestion errors appear in the Library expander and
the watcher status command. The standalone smoke command remains temporary.
See README for startup/shutdown and backup instructions.

- Python 3.14.10 works with the locked Docling 2.145.0, PyTorch 2.15.0,
  FastEmbed 0.9.1, and Qdrant client 1.20.0 on this Mac.
- Dedicated Qdrant v1.20.0 is healthy on localhost:6789/6790, isolated from the
  existing Qdrant on 6333.
- `gemma4:e2b` is installed (Q4_K_M) and produces structured answers through
  Ollama. Use an 8,192-token configured context and check its declared limit
  of 131,072. No other model was substituted.
- The CLI implements `list`, `verify`, `doctor`, `parse`, and `smoke`.
  The smoke workflow uses a unique temporary collection and deletes it on exit;
  it is not persistent or expansion-aware retrieval yet.
- Final embedding inputs are checked with the matching tokenizer, including
  headings and document identity. The embedding model has 384 dimensions and
  a 512-token limit; the configured chunk limit is 384 tokens.
- Parsed artifacts are cached by document/config/version fingerprint. A fresh
  process reuses unchanged conversion results. Disabled Docling enrichment
  options with unordered sets are excluded from the fingerprint.
The adapter enables native picture-child traversal to preserve extracted rule
text nested inside pictures; a synthetic regression test covers this behavior.
This preserves extracted text, not visual understanding of diagrams.
Page coverage alone is not an accuracy/completeness guarantee.

Smoke tests check supported answers, evidence IDs, and PDF-page references after
Qdrant retrieval. They are not a substitute for a held-out quality evaluation.
Docling currently emits an upstream OCR deprecation warning during config
serialization; it is visible rather than suppressed.

Do not add LangChain, LlamaIndex, an API server, Celery, or Kubernetes for this
single-user MVP. Use the official Docling and Qdrant APIs through small adapters.
Define defaults once in validated configuration models and derive interface
choices from shared data.

### Package hierarchy

The implemented first slice deliberately uses fewer files and levels:

```text
src/board_game_rules/
  config.py
  bootstrap.py
  domain/models.py
  application/
    answering.py
    smoke.py
  infrastructure/
    catalog.py
    docling.py
    embeddings.py
    qdrant.py
    ollama.py
  interfaces/cli.py
tests/                         # focused tests; split further when needed
```

Parser and chunker are together in `infrastructure/docling.py`; ruleset validation
is together with domain models. These share a small responsibility and do not
need extra wrappers. SQLite, durable jobs, and web packages are intentionally
not scaffolded. The outline below is a possible future split, **not a requirement
to create every file**. Keep modules together until size or responsibility
justifies separation.

```text
catalog/rulebooks.json          # source registry, already present
src/board_game_rules/
  config.py                    # validated settings; single source of defaults
  bootstrap.py                 # construct adapters and wire application services
  domain/
    models.py                  # document, ruleset, chunk, hit, answer contracts
    rulesets.py                # expansion and edition compatibility rules
  application/
    library.py                 # catalog/import/delete use cases
    ingestion.py               # reusable parse -> chunk -> embed -> publish flow
    retrieval.py               # scoped retrieval, deduplication, context budget
    answering.py               # grounded answers and citation validation
    jobs.py                    # ingestion job lifecycle and worker orchestration
  infrastructure/
    documents/
      catalog.py               # JSON catalog persistence
      files.py                 # bounded downloads/uploads and local PDF access
      docling_parser.py        # Docling conversion and artifact cache
      docling_chunker.py       # HybridChunker adapter and provenance
    embeddings/
      fastembed.py             # local passage/query embedding adapter
    storage/
      qdrant.py                # vectors, filters, indexes, and batches
      sqlite.py                # durable jobs and active document versions
    llm/
      ollama.py                # Gemma 4 transport, structured output, timeouts
  interfaces/
    cli.py                     # thin commands calling application services
    web/
      app.py                   # Streamlit entry point and navigation
      ask_rules.py             # question inputs and cited-answer presentation
      library.py               # document management and ingestion progress
      components.py            # shared UI rendering only
tests/
  unit/                        # mirror domain/application package responsibilities
  integration/                 # Docling, Qdrant, embeddings, Ollama adapters
  ui/                          # page navigation and user workflows
  fixtures/                    # synthetic PDFs, not copied publisher manuals
  evals/                       # reviewed questions and evidence references
data/                          # ignored: PDFs, parsed artifacts, SQLite, vectors
```

### Separation rules

- **Domain** contains typed contracts and ruleset invariants only; no Streamlit,
  Docling, Qdrant, Ollama, file I/O, or database imports.
- **Application** owns workflows and answer-validation policy. It uses injected
  adapters and domain contracts, never UI state or concrete vendor clients.
  Add small typed protocols alongside their consumers where needed, not an
  abstract class or repository wrapper for every function.
- **Infrastructure** adapts external libraries and persistence to domain
  contracts. It must not import interfaces or orchestrate application workflows.
- **Interfaces** handle input/output and call application services; no direct
  Qdrant queries, PDF conversion, embedding, or LLM requests in page functions.
- **Bootstrap** is the composition root: it imports concrete adapters and wires
  them into application services. CLI, web, and worker entry points reuse it.
- Keep citation validation in application answering, not in the Ollama transport.
  Keep edition compatibility in domain rulesets, not duplicated across selectors
  and vector filters. Define settings/defaults once in configuration.
- Create packages/modules as their functionality is implemented, not empty
  scaffolding. Avoid circular imports, broad re-exports, and generic `utils`
  dumping grounds. Keep parsing, indexing, and answering usable without Streamlit.

### Gemma 4 configuration

Use Ollama locally for generation, separately from the FastEmbed embedding model.
During Phase 1, inspect Ollama's installed models and verify the exact `gemma4`
tag with a real structured-response request. Record the working tag and limits
in configuration; derive the prompt budget from the configured model context
window. If the requested model is missing or unsupported, surface that blocker
instead of reverting to Qwen or another model. This was verified during Phase 1;
`gemma4:e2b` is the tested default, and the larger installed `gemma4:26b` was not
needed for the first smoke tests.

## 4. Parsing, chunking, and document provenance

1. Validate PDF signature, size limit, file readability, document identity, and
   permitted source URL before download/import. Use bounded timeouts/retries,
   reject unexpected hosts/redirects, and write downloads atomically.
2. Cache conversion by PDF SHA-256 plus parser version and parser configuration.
   Preserve the DoclingDocument JSON as the canonical parsed artifact.
3. Enable table structure extraction and appropriate English OCR through
   Docling options. Review sparse-text pages, columns, sidebars, lists, icons,
   diagrams, and foldouts. Do not silently drop image-only rules.
4. Preserve Docling item references, headings, captions, page numbers, and bounding
   boxes where available. A chunk can span multiple pages: store the set of
   supporting page numbers, not just the first page.
5. Use HybridChunker with a Hugging Face tokenizer aligned to the embedding model.
   Start with a **384-token total embedding input budget** as a tuning candidate,
   merge compatible peers, and preserve table headers when splitting tables.
6. Embed `contextualize(chunk)` enriched with a compact game/document identity.
   Retain raw chunk text separately for source display.
7. Count the **final serialized embedding input**, including identity, headings,
   table headers, special tokens, and model prefixes. Every input must fit the
   verified model limit; reject or re-chunk overflow, never silently truncate.
8. Compare against Docling HierarchicalChunker only as an evaluation baseline.
   Add no arbitrary overlap until retrieval evidence demonstrates a need; fetch
   adjacent chunks within the same document when a rule and exception separate.

Text-only RAG cannot reliably interpret every diagram or unlabeled icon.
Show the cited page and report limitations. A multimodal parser/answerer is a
later, separately evaluated enhancement, not an assumed capability.

## 5. Qdrant storage and safe ingestion

Use one versioned collection for one embedding schema, not one collection per
game. Assert the configured model dimension and cosine distance before upsert.
Changing the embedding model/dimension requires a new collection and re-index.

### Payload contract

- `document_id`, `game_id`, `base_game_id`, `kind`, `language`, `edition`
- `title`, `publisher`, `source_url`, `pdf_sha256`
- `ingestion_id`, `chunk_index`, `chunk_id`, `section_path`
- `pdf_pages`, optional printed labels, Docling item references/bounding boxes
- `text`, `contextualized_text`, `token_count`
- Parser/chunker/model versions and a processing-config fingerprint

Create payload indexes for actual filters: `document_id`, `game_id`, `language`,
`edition`, and `ingestion_id`. Store stable UUID point IDs derived from document
hash, processing fingerprint, chunk index, and chunk hash.

Ingest a document into a new staging ingestion ID; validate chunk/vector counts
and schema before marking it active. Store active IDs in a SQLite transaction.
Retrieval resolves a snapshot of active IDs and includes those IDs in its Qdrant
filter, so incomplete ingestion is never exposed. A failed replacement keeps
the previous version searchable. Retain old generations until in-flight requests
finish or a defined cleanup grace period passes.

An unchanged document/config is a no-op. Replacing one document must not recreate
the entire collection. Deletion removes only its active/staged points and state;
original PDF deletion is a separate explicit action.

## 6. Retrieval and grounded answers

1. Resolve an explicit ruleset to exact active document IDs. An expansion scope
   includes its compatible base edition; a base-only scope excludes expansions.
   Reject incompatible or unresolved base/edition combinations.
2. Start with dense retrieval, candidate top-k 20, deduplicate, and select up to
   six useful passages under the generation model's context budget. These are
   initial tuning values, defined only in configuration when implemented.
3. Preserve source diversity across base and selected expansions. An expansion
   question must not be answered from base evidence alone when an overriding
   rule exists; inspect both sources and cite the exception explicitly.
4. Add same-section neighbors when needed, within the same scoped document.
5. Establish a baseline first. If named cards/actions or exceptions are missed,
   evaluate FastEmbed BM25 sparse vectors with Qdrant reciprocal-rank fusion.
   Add a local cross-encoder reranker only if measured accuracy justifies it.
6. Do not label cosine scores as confidence probabilities. Calibrate any
   evidence threshold on validation questions and test it separately.
7. Give the LLM numbered evidence passages, ruleset metadata, and explicit
   instructions to use only that evidence. Treat user queries and PDF content
   as data, not instructions capable of changing scope or running tools.
8. Require structured output: `answer`, `supported`, `citations`, and
   `missing_information`. Validate that citation IDs belong to retrieved evidence
   and map to real pages. Unsupported responses must not masquerade as rulings.
   Citation existence alone does not prove entailment; test claim support.
9. Surface generation/network/format failures explicitly. At most one bounded
   formatting retry; otherwise show an error and supporting search results,
   clearly labeled as retrieval-only, not a successfully generated answer.

Official errata/FAQs may be added later as separate catalog documents with
source type, date, and declared precedence. Do not mix unofficial forum answers
or house rules into the official-rule corpus.

## 7. Phased todo list

Dependencies are explicit below. Phases are milestones, not estimates.

### Phase 0: input contracts and design

- [x] Define source, edition, and expansion metadata contracts.
- [x] Define PDF signature, size, page-count, and checksum validation.
- [x] Write this end-to-end architecture, todo list, and acceptance criteria.

### Phase 1: compatibility spike and project setup

Depends on Phase 0.

- [x] Create the uv package, lock tested dependencies, and verify the selected
  Python version against Docling/PyTorch/FastEmbed on this Mac.
- [x] Establish the domain/application/infrastructure/interfaces hierarchy and
  shared composition root as the first functionality is implemented.
- [x] Pin a tested Qdrant Docker image; configure HTTP 6789, gRPC 6790, persistence,
  and health checks without affecting other Compose projects.
- [ ] Review representative PDFs with covers, columns/foldouts, diagrams,
  tables, and short expansion cases.
- [x] Smoke-test local embedding length/dimension and Gemma 4 on Ollama; verify
  the exact model tag, structured output, context budget, and hardware feasibility.
- [x] Add validated settings, typed models, and catalog validation, including
  duplicate IDs, dependency cycles, edition compatibility, and safe local paths.
- [x] Establish pytest and Ruff commands and installation/startup documentation.

**Exit:** one PDF converts, chunks, embeds, and yields a structured local LLM
response; all challenging PDFs in this spike have documented parse diagnostics.

### Phase 2: reusable document ingestion

Depends on Phase 1.

- [ ] Implement bounded download/upload validation and atomic local file writes.
- [x] Implement Docling conversion, configuration fingerprints, caching, and
  explicit failure/partial-conversion reporting.
- [ ] Review representative PDFs for page coverage, tables, icons, reading order,
  and the sparse-text pages; retain diagnostics and original page references.
- [x] Implement HybridChunker contextualization and final token-budget checks.
- [ ] Implement Qdrant schema/index validation, batching, stable IDs, staging,
  active-version publication, retry safety, and document-scoped deletion.
- [ ] Add CLI `library list`, `library add`, `library download`, `ingest`, and
  `library remove` using the same services the UI will call.
- [ ] Verify no-op re-ingestion, failed replacement recovery, and unchanged
  retrieval for unrelated documents.

The local folder pipeline implements inspection, versioned publication, and
unchanged detection. Automated tests cover failed replacements and version
filtering. Full visual review and library download/upload/removal commands remain
unfinished.

**Exit:** validated documents are queryable; rerunning ingestion creates no
duplicates and cannot expose partial replacements.

### Phase 3: scoped retrieval and evaluation baseline

Depends on Phase 2.

- [ ] Implement base-plus-expansion ruleset resolution and mandatory filters.
- [ ] Implement query encoding, ranked retrieval, deduplication, neighbors, and
  page/section source objects.
- [ ] Build a manually reviewed evidence dataset: at least five answerable
  questions per evaluation ruleset, plus at least two unsupported/ambiguous cases
  per ruleset. Reserve a held-out split before tuning.
- [ ] Measure Recall@10, cross-game leakage, edition leakage, and expansion
  exception coverage before introducing hybrid search/reranking.
- [ ] Add hybrid BM25/RRF or reranking only if the baseline misses the gates;
  compare complexity, accuracy, and warm latency.

**Exit:** retrieval meets the gates below on held-out questions, with a saved
evaluation report and exact configuration.

### Implemented prompt comparison

- [x] Add a fixed 20-question benchmark with concise reference answers and
  expected document/page evidence.
- [x] Compare plain prompting with few-shot citation prompting against the same
  retrieved passages; record the retrieval hit rate and model configuration.
- [x] Save the retrieved evidence and both answers for manual
  supported/partly-supported/unsupported faithfulness labels.
- [ ] Run the benchmark against the indexed collection and manually review its
  answers before reporting a prompt winner.

### Phase 4: answer generation

Depends on Phase 3 and the local generation smoke test in Phase 1.

- [ ] Implement the Ollama adapter, timeouts, bounded evidence prompts,
  structured answers, and citation validation.
- [ ] Implement unsupported-answer handling, ruleset clarification, and
  expansion/base conflict presentation.
- [ ] Add CLI `search` for evidence and `ask` for generated rules answers.
- [ ] Evaluate claim support, citation correctness, abstention, and injected
  instructions in uploaded documents.
- [ ] Test unavailable model, malformed output, empty retrieval, and context
  overflow without success-shaped fallbacks.
- [ ] Run a controlled prompt-injection experiment using a synthetic document;
  review whether the answer remains grounded in citations or abstains.

**Exit:** generated rulings pass grounded-answer gates and visibly distinguish
unsupported questions from application errors.

### Phase 5: UI and extensible library

Depends on Phases 2 and 4.

- [ ] Add separate Ask Rules and Library pages using native Streamlit navigation.
- [ ] Add game/edition/expansion selectors derived from the catalog, cited answer
  display, local PDF-page access, and source-evidence inspection.
- [ ] Persist ingestion jobs in SQLite with one local worker and atomic job
  claiming; prevent reruns/page navigation from submitting duplicate jobs.
- [ ] Add progress/error views and worker restart recovery. Do not pretend
  Streamlit background threads are a durable job queue.
- [ ] Add PDF import and metadata editing; validate paths, size, language,
  duplicate hashes, and relationships before publishing catalog changes.
- [ ] Test adding an independent synthetic document through data alone; no module or
  game-name branch changes should be required.
- [ ] Test navigation, selection persistence, scope changes, pending ingestion,
  document deletion, and inaccessible local PDFs.

**Exit:** a user can add a game, ingest it, and ask a cited question through the
UI without editing source code.

### Phase 6: release and operations

Depends on Phases 3-5.

- [ ] Run unit tests, corpus integration tests, UI tests, and held-out evaluation.
- [ ] Document clean setup, first-run model downloads, startup/shutdown,
  re-indexing, backup/restore, source provenance, and document rights.
- [ ] Verify restart persistence and backup/restore of the catalog, SQLite,
  Docling artifacts, and Qdrant.
- [ ] Bind local services appropriately; require authentication/TLS and an
  explicit document-sharing policy before any remote deployment.
- [ ] Produce a release checklist and measured warm/cold performance report.

**Exit:** reproducible local application with documented limitations and all
acceptance gates passing.

## 8. Acceptance tests and release gates

| Area | Required outcome |
|---|---|
| Governance | Evaluation outputs record dataset, model, retrieval, and indexed-document versions; human faithfulness labels remain distinct from automated metrics. |
| Ethics and rights | Users can identify the selected edition and evidence; unsupported answers are not presented as rulings; document processing and redistribution rights are reviewed. |
| Security | Retrieved content is explicitly untrusted, citation/output validation is tested, and the synthetic prompt-injection experiment is reviewed; no claim of complete injection resistance. |
| Downloads | Valid English PDFs match declared checksums; bad/HTML/password-protected files produce explicit errors. |
| Parsing | Every physical page is accounted for in diagnostics; every rules-bearing page is represented or flagged for review. Diagrams are not assumed understood. |
| Chunking | 100% of final embedding inputs fit the actual tokenizer/model limit; provenance remains valid after split/merge. |
| Ingestion | Second unchanged run has identical IDs/counts; failed replacement preserves old answers; deleting one document leaves other games intact. |
| Retrieval | Held-out Recall@10 >= 90% for annotated evidence; 0 cross-game or disallowed-edition results. |
| Expansion scope | All dedicated base-versus-expansion tests retrieve the required overriding evidence; standalone documents never include unrelated base-game rules. |
| Citations | 100% of returned citation IDs resolve to retrieved document/page evidence; no fabricated page references. |
| Grounding | >= 90% of answerable held-out questions judged correct and supported by reviewed evidence; every material ruling has a supporting citation. |
| Abstention | >= 90% of unsupported/ambiguous cases abstain or request the missing ruleset details; no invented house rules. |
| Add-more workflow | An independent synthetic rulebook imports and becomes queryable without game-specific source-code changes. |
| UI/operations | Both pages, ingestion progress/failure, scope reset, deletion, service restart, and backup/restore pass. |
| Performance | Record cold parse/model startup and warm query timings on this Mac. Provisional targets: warm retrieval p95 <= 2 s, full answer p95 <= 30 s; revisit model choice if unmet. |

Use synthesized test documents for deterministic unit tests. Real publisher PDFs
stay outside Git. Keep evaluation expectations as short, independently written
answers and evidence references rather than reproducing full rulebook passages.
The default deployment is local and single-user; remote access requires a
separate authentication, transport-security, and data-sharing review.

## 9. Deferred scope

Not part of the initial release: arbitrary website crawling, forum knowledge,
automatic rulebook updates, remote multi-user hosting, multilingual answers,
multimodal diagram reasoning, card databases, voice input, or agent tool use.
Add them only after the core grounded-rules workflow passes evaluation.

## 10. Technical references

- [Docling installation](https://docling-project.github.io/docling/getting_started/installation/)
- [Docling native chunking concepts](https://docling-project.github.io/docling/concepts/chunking/)
- [HybridChunker example and contextualization](https://docling-project.github.io/docling/_generated/examples/hybrid_chunking/)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [FastEmbed documentation](https://qdrant.github.io/fastembed/)
- [Ollama documentation](https://docs.ollama.com/)

Publisher landing pages and exact PDF URLs are in the catalog, rather than
duplicated here as another source registry.
