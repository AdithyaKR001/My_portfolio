"""Runs the same workload against three server configs — naive (no batching,
no cache), batching-only, and batching+cache — to produce the before/after
comparison in loadtest/report.md.

Each server is launched as a subprocess, polled on /health until ready, load
tested via loadtest.run_loadtest.run(), then terminated — all from one
Python process, so it works the same on macOS, Linux, and in CI regardless
of the local `make` version (GNU Make added .ONESHELL, needed to keep a
background job's PID across recipe lines, only in 3.82; macOS ships 3.81).

Run:  python -m loadtest.compare
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from loadtest import run_loadtest

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

SCENARIOS = [
    {
        "label": "naive-no-batch-no-cache",
        "port": 8011,
        "env": {"BATCH_MAX_SIZE": "1", "BATCH_MAX_WAIT_MS": "0", "CACHE_MAX_SIZE": "0"},
    },
    {
        "label": "batching-only",
        "port": 8012,
        "env": {"BATCH_MAX_SIZE": "16", "BATCH_MAX_WAIT_MS": "20", "CACHE_MAX_SIZE": "0"},
    },
    {
        "label": "batching-plus-cache",
        "port": 8013,
        "env": {
            "BATCH_MAX_SIZE": "16",
            "BATCH_MAX_WAIT_MS": "20",
            "CACHE_MAX_SIZE": "2000",
            "CACHE_TTL_SECONDS": "300",
        },
    },
]


def _wait_healthy(url: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            httpx.get(f"{url}/health", timeout=1.0).raise_for_status()
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"server at {url} never became healthy") from last_err


def main() -> None:
    results = []
    for scenario in SCENARIOS:
        url = f"http://127.0.0.1:{scenario['port']}"
        env = {**os.environ, "BACKEND": "mock", **scenario["env"]}
        print(f"\n=== {scenario['label']} ===")

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(scenario["port"]),
                "--log-level", "warning",
            ],
            cwd=PROJECT_ROOT,
            env=env,
        )
        try:
            _wait_healthy(url)
            os.environ["MODEL_SERVER_URL"] = url
            os.environ["LOADTEST_LABEL"] = scenario["label"]
            run_loadtest.MODEL_SERVER_URL = url
            run_loadtest.LABEL = scenario["label"]
            summary = run_loadtest_sync()
            results.append(summary)
            print(json.dumps(summary, indent=2))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    report_path = HERE / "report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {report_path}")

    from loadtest.render_report import main as render_main
    render_main()


def run_loadtest_sync() -> dict:
    import asyncio

    return asyncio.run(run_loadtest.run())


if __name__ == "__main__":
    main()
