# Board Game Rules RAG Report

## Purpose

This project answers questions about published board-game rules. It uses a local retrieval-augmented generation system, called RAG.

The system searches rulebook text and sends matching text to a language model. The model writes an answer with citations to PDF pages.

The project compares two prompt types. A prompt is the instruction and data sent to the model. The two types are `plain` and `few_shot_citations`.

## Project Goals

The project shows how to:

- Create text embeddings.
- Search a vector database.
- Send a prompt to a language model.
- Build a chat page that shows answers and sources.
- Compare two prompt types with a fixed question set.

An embedding is a list of numbers that represents text. Retrieval selects text that matches a question. Qdrant stores the embeddings and text chunks.

## Scope

The catalog has ten rulebooks. The benchmark has 20 questions. It covers four rulebooks: Azul, Catan 2025, Love Letter, and Pandemic.

Each question has a reference answer, a document ID, and expected PDF pages. The chat page can search one or more rulebooks from the catalog.

The system runs on a local computer. It does not support web import, image questions, remote use, or multiple users. It does not compare chunking or embedding models. It does not select compatible expansions automatically.

## Set Up

### Requirements

Use a Mac with Python 3.14. Install Git, `uv`, Docker Desktop, and Ollama:

- [Git](https://git-scm.com/downloads)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Ollama](https://ollama.com/download/mac)

### Clone and Install

```bash
git clone https://github.com/ankurbajaj9/board-game-helper.git
cd board-game-helper
uv python install 3.14
uv sync --extra evaluation
ollama pull gemma4:e2b
docker compose up -d --build
```

Start the Ollama application before you run the evaluation. If Ollama is not running, start it in a separate terminal:

```bash
ollama serve
```

### Add the Benchmark PDFs

The repository does not include the rulebook PDFs. Get copies that you have permission to use. Put each PDF at the path shown below:

```text
data/rulebooks/azul-en.pdf
data/rulebooks/catan-2025-en.pdf
data/rulebooks/love-letter-en.pdf
data/rulebooks/pandemic-en.pdf
```

The publisher pages and download links are in [catalog/rulebooks.json](catalog/rulebooks.json). Use the listed edition and PDF file. The benchmark checks each PDF's hash. A different PDF will fail this check.

### Index and Check the PDFs

The folder watcher scans for new PDFs. Run a scan when you need to start it at once:

```bash
docker compose exec watcher board-game-rules-watch scan
docker compose exec watcher board-game-rules-watch status
uv run board-game-rules list
uv run board-game-rules doctor
```

Wait for the services and scan to finish. Check their status and logs:

```bash
docker compose ps
docker compose logs watcher
```

The `verify` command checks all ten catalog PDFs. Add all ten PDFs before you run it:

```bash
uv run board-game-rules verify
```

Qdrant uses `http://localhost:6789`. The Prefect page uses `http://127.0.0.1:4201`. Ollama uses `http://localhost:11434`.

### Start the Chat Page

```bash
uv run board-game-rules-ui
```

Open `http://localhost:8501`. Select indexed rulebooks and ask a question. The page shows the answer, cited pages, and source text.

## System Design

### Add and Index PDFs

Docling reads each PDF. HybridChunker divides the text into chunks. Each chunk keeps its PDF page numbers.

FastEmbed creates embeddings with `BAAI/bge-small-en-v1.5`. The system checks text length and vector size. It does not silently shorten text.

Qdrant stores versioned vectors. SQLite records which version is active. The system does not activate a version if indexing fails.

### Find and Answer

The system creates an embedding for each question. It searches active rulebooks in Qdrant. The default search limit is six chunks.

The prompt contains the question, selected rulebooks, and retrieved text. Ollama runs `gemma4:e2b` and returns a structured answer. The answer has a support flag, citation IDs, and missing-information text.

The system rejects citation IDs that are not in the retrieved text. A supported answer must have at least one citation.

The chat page can search only the rulebooks that the user selects. It does not select expansions automatically.

### Prompt Types

The `plain` prompt gives the model the question and retrieved text.

The `few_shot_citations` prompt also gives one example of an answer with a citation.

For each benchmark question, the system retrieves text once. It gives the same text to both prompts. This keeps the retrieval result the same for each comparison.

## Evaluation Method

The benchmark has 20 hand-written questions. Each question has a reference answer and expected PDF pages.

The system checks that each source PDF is indexed, has the expected hash, and uses the configured Qdrant collection.

A retrieval hit occurs when at least one retrieved chunk comes from the expected document and includes an expected page. The retrieval hit rate is the number of hits divided by 20.

Ragas can score each answer for faithfulness and factual correctness. The local Gemma model acts as the judge. The report stores each score with its prompt type. A person can also label answer faithfulness as `supported`, `partly_supported`, or `unsupported`.

### Run the Evaluation

Index the four benchmark PDFs first. Then run:

```bash
uv run --extra evaluation board-game-rules evaluate
```

The system saves the report to `data/evaluation/results.json`. The `score` command shows Ragas scores and any manual labels:

```bash
uv run --extra evaluation board-game-rules score data/evaluation/results.json
```

The report file is local and is ignored by Git. It contains rulebook text. Do not commit or share it unless you have checked the rights to do so.

Run the project tests and code checks with:

```bash
uv run --extra evaluation pytest
uv run ruff check src tests
uv run ty check src
```

Stop the Docker services with:

```bash
docker compose stop
```

Do not use `docker compose down -v` if you need to keep the local data or model files. Ollama is not part of Docker Compose. Stop it separately when needed.

## Results

The 20-question benchmark completed retrieval and answer generation for both prompts. The saved results are in [data/evaluation/results.json](data/evaluation/results.json).

| Result | Plain | Few-shot citations |
| --- | ---: | ---: |
| Gemma marked the answer as supported | 20/20 (100%) | 19/20 (95%) |
| Answer has at least one citation | 20/20 (100%) | 19/20 (95%) |

Both prompts used the same retrieved text. The retrieval hit rate was 18/20, or 90%.

These support flags and citations came from Gemma. They are not human checks of answer quality. There are no manual faithfulness labels in the report.

### Ragas Scores

Gemma also acted as the Ragas judge. The same model wrote and scored the answers. This can cause self-evaluation bias.

| Ragas score | Plain mean | Plain scored | Few-shot mean | Few-shot scored |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness | 0.9211 | 19/20 | 1.0000 | 18/20 |
| Factual correctness, F1 | 0.5460 | 20/20 | 0.4440 | 20/20 |

The faithfulness means do not use the same questions. The judge failed to score three answers because its structured output was too long. The failed answers were `catan_city_points` for both prompts and `pandemic_outbreak_effect` for the few-shot prompt. The system records these cases as unscored. It does not record them as zero.

Few-shot has a higher mean faithfulness for its scored answers. Plain has a higher mean factual-correctness score by 0.1020. These results do not prove that one prompt is better. The benchmark is small, three scores are missing, and Gemma wrote and judged the answers.

## Security

The system treats the question and retrieved text as untrusted input. The system prompt tells Gemma to ignore instructions inside retrieved text. The system also checks the output format and citations.

These controls can reduce some errors. They cannot prove that prompt injection will fail. The project has not tested a rulebook with a planted malicious instruction. A controlled test must compare results before and after the mitigation.

## Limits and Next Steps

PDF page hits and valid citations do not prove that an answer is correct. Text extraction can lose information from icons, tables, and diagrams. Check important answers against the source PDF.

Next, a person must review the answers and add manual faithfulness labels. The Ragas test also needs more work to reduce failed judge outputs. The project must run the planned prompt-injection test before it can report security results.

The local setup is not designed for remote or multi-user access.

## Conclusion

The project has a working local RAG pipeline and chat page. It compares plain and few-shot citation prompts on 20 questions. It found a 90% page-level retrieval hit rate.

Plain had a higher factual-correctness score. Few-shot had a higher faithfulness mean for the answers that the judge scored. The judge did not score three faithfulness cases. Human review and a prompt-injection test remain necessary.
