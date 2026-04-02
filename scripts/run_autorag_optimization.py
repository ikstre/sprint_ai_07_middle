"""
Run AutoRAG optimization from prepared qa/corpus parquet files.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoRAG evaluate workflow.")
    parser.add_argument("--qa-path", type=str, default="data/autorag/qa.parquet")
    parser.add_argument("--corpus-path", type=str, default="data/autorag/corpus.parquet")
    parser.add_argument("--config-path", type=str, default="configs/autorag/tutorial.yaml")
    parser.add_argument("--project-dir", type=str, default="evaluation/autorag_benchmark")
    parser.add_argument("--run-dashboard", action="store_true")
    args = parser.parse_args()

    qa_path = Path(args.qa_path)
    corpus_path = Path(args.corpus_path)
    config_path = Path(args.config_path)
    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    if not qa_path.exists():
        raise FileNotFoundError(f"qa file not found: {qa_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus file not found: {corpus_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    _run(
        [
            "autorag",
            "evaluate",
            "--qa_data_path",
            str(qa_path),
            "--corpus_data_path",
            str(corpus_path),
            "--config",
            str(config_path),
            "--project_dir",
            str(project_dir),
        ]
    )

    trial_dir = project_dir / "0"
    print("\nAutoRAG evaluation finished.")
    print(f"- expected trial dir: {trial_dir}")
    print(f"- dashboard command: autorag dashboard --trial_dir {trial_dir}")
    print(f"- API command: autorag run_api --trial_dir {trial_dir} --host 0.0.0.0 --port 8000")
    print(f"- Web command: autorag run_web --trial_path {trial_dir}")

    if args.run_dashboard and trial_dir.exists():
        _run(["autorag", "dashboard", "--trial_dir", str(trial_dir)])


if __name__ == "__main__":
    main()

