import argparse
import json
import logging
import sys
from importlib.metadata import version
from pathlib import Path

import httpx
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from board_game_rules import bootstrap
from board_game_rules.config import Settings
from board_game_rules.evaluation import (
    EvaluationReport,
    load_dataset,
    run_evaluation,
    summarize_faithfulness,
)
from board_game_rules.infrastructure.state import IndexState


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local board game rules RAG tools")
    parser.add_argument("--root", help="Project directory containing catalog/ and data/")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List catalog document IDs and editions")
    commands.add_parser("verify", help="Verify all downloaded PDF signatures, sizes, and checksums")
    commands.add_parser("doctor", help="Check Python/package versions, Qdrant, and Gemma 4")
    parse = commands.add_parser("parse", help="Parse/chunk one PDF and cache local diagnostics")
    parse.add_argument("document_id")
    evaluate = commands.add_parser(
        "evaluate", help="Compare plain and few-shot prompts on the fixed question set"
    )
    evaluate.add_argument(
        "--questions", default="evaluation/questions.json", help="Path to the benchmark JSON"
    )
    evaluate.add_argument(
        "--output", default="data/evaluation/results.json", help="Path for the evaluation report"
    )
    score = commands.add_parser("score", help="Summarize manually reviewed answer faithfulness")
    score.add_argument("results", help="Evaluation report JSON, with manual labels filled in")
    return parser


def run(args: argparse.Namespace, settings: Settings) -> None:
    if args.command == "score":
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = settings.project_root / results_path
        report = EvaluationReport.model_validate_json(results_path.read_bytes())
        print(json.dumps(summarize_faithfulness(report), indent=2))
        return

    catalog = bootstrap.catalog(settings)
    if args.command == "list":
        for doc in catalog.documents:
            print(f"{doc.document_id}\t{doc.title}\t{doc.kind}\t{doc.edition}")
    elif args.command == "verify":
        for document_id in bootstrap.verified_documents(settings):
            print(f"{document_id}: verified")
    elif args.command == "doctor":
        vectors, ollama = bootstrap.local_services(settings)
        try:
            collections = vectors.collections()
            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "packages": {
                            name: version(name)
                            for name in ("docling", "torch", "fastembed", "qdrant-client")
                        },
                        "qdrant_collections": collections,
                        "ollama": ollama.check_model(),
                    },
                    indent=2,
                )
            )
        finally:
            vectors.close()
    elif args.command == "evaluate":
        questions_path = Path(args.questions)
        if not questions_path.is_absolute():
            questions_path = settings.project_root / questions_path
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = settings.project_root / output_path
        dataset, dataset_sha256 = load_dataset(questions_path)
        from board_game_rules.infrastructure.embeddings import Embeddings

        embeddings = Embeddings(settings)
        vectors, ollama = bootstrap.local_services(settings)
        try:
            ollama.check_model()
            report = run_evaluation(
                dataset.cases,
                dataset_sha256,
                catalog,
                IndexState(settings.state_path),
                embeddings,
                vectors,
                ollama,
                settings,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report.model_dump_json(indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "output": str(output_path),
                        "questions": len(report.cases),
                        "retrieval_hits": report.retrieval_hit_count,
                        "retrieval_hit_rate": report.retrieval_hit_rate,
                    },
                    indent=2,
                )
            )
        finally:
            vectors.close()
    else:
        doc = catalog.document(args.document_id)
        parsing = bootstrap.parsing(settings)
        chunks, report = parsing.parser.parse(doc)
        print(report.model_dump_json(indent=2))


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        overrides = {"project_root": args.root} if args.root else {}
        settings = Settings(**overrides)
        run(args, settings)
    except (
        OSError,
        ValueError,
        ValidationError,
        httpx.HTTPError,
        ResponseHandlingException,
        UnexpectedResponse,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
