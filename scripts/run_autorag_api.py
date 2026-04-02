"""
Run AutoRAG API server from trial directory.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoRAG API server from trial directory.")
    parser.add_argument("--trial-dir", type=str, default="evaluation/autorag_benchmark/0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cmd = [
        "autorag",
        "run_api",
        "--trial_dir",
        args.trial_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

