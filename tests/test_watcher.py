import sys
from unittest.mock import Mock

import httpx

from board_game_rules.config import Settings
from board_game_rules.infrastructure import prefect_pipeline
from board_game_rules.interfaces import watcher


def test_watch_connects_to_compose_server(settings: Settings, monkeypatch) -> None:
    monkeypatch.setenv("BGR_PREFECT_URL", "http://prefect:4201/api")
    monkeypatch.setattr(
        sys, "argv", ["board-game-rules-watch", "--root", str(settings.project_root), "watch"]
    )
    for key in (
        "PREFECT_HOME",
        "PREFECT_API_URL",
        "PREFECT_SERVER_ANALYTICS_ENABLED",
        "PREFECT_SERVER_EPHEMERAL_ENABLED",
        "PREFECT_SERVER_DATABASE_CONNECTION_URL",
        "PREFECT_SERVER_UI_API_URL",
    ):
        monkeypatch.setenv(key, "")
    get = Mock(
        return_value=httpx.Response(
            200, json=True, request=httpx.Request("GET", "http://prefect:4201/api/health")
        )
    )
    monkeypatch.setattr(httpx, "get", get)
    scan = Mock()
    monkeypatch.setattr(prefect_pipeline, "scan_rulebooks", scan)

    watcher.main()

    get.assert_called_once_with("http://prefect:4201/api/health", timeout=5)
    scan.serve.assert_called_once_with(
        name="local-rulebook-folder",
        parameters={"project_root": str(settings.project_root)},
        interval=settings.scan_interval_seconds,
        limit=1,
        pause_on_shutdown=True,
    )
