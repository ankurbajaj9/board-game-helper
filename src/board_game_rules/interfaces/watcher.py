import argparse
import os
from pathlib import Path

from board_game_rules.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Prefect PDF folder pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "command",
        choices=("server", "scan", "watch", "status"),
        help="Start the local server, scan once, serve scheduled scans, or show document status",
    )
    args = parser.parse_args()
    settings = Settings(project_root=args.root)
    from urllib.parse import urlsplit

    target = urlsplit(str(settings.prefect_url))
    # Isolate this process from other projects' Prefect profiles, server, and database.
    os.environ["PREFECT_HOME"] = str(settings.project_root / "data" / "prefect")
    os.environ["PREFECT_API_URL"] = str(settings.prefect_url).rstrip("/")
    os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] = "false"
    os.environ["PREFECT_SERVER_EPHEMERAL_ENABLED"] = "false"
    os.environ["PREFECT_SERVER_DATABASE_CONNECTION_URL"] = (
        f"sqlite+aiosqlite:///{settings.project_root / 'data/prefect/prefect.db'}"
    )
    os.environ["PREFECT_SERVER_UI_API_URL"] = str(settings.prefect_url).rstrip("/")

    if args.command == "server":
        import sys

        if target.hostname not in ("localhost", "127.0.0.1") or not target.port:
            parser.error("Local Prefect server requires a localhost URL with an explicit port.")
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "prefect",
                "server",
                "start",
                "--host",
                target.hostname,
                "--port",
                str(target.port),
            ],
        )
    elif args.command == "status":
        from board_game_rules.infrastructure.state import IndexState

        for item in IndexState(settings.state_path).attempts():
            print(f"{item['status']}\t{item['path']}\t{item['error'] or ''}")
    else:
        import httpx

        try:
            response = httpx.get(f"{str(settings.prefect_url).rstrip('/')}/health", timeout=5)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            parser.exit(
                1, f"Local Prefect is not ready: {exc}\nRun board-game-rules-watch server.\n"
            )
        from board_game_rules.infrastructure.prefect_pipeline import scan_rulebooks

        if args.command == "scan":
            scan_rulebooks(str(settings.project_root))
        else:
            scan_rulebooks.serve(
                name="local-rulebook-folder",
                parameters={"project_root": str(settings.project_root)},
                interval=settings.scan_interval_seconds,
                limit=1,
                pause_on_shutdown=True,
            )


if __name__ == "__main__":
    main()
