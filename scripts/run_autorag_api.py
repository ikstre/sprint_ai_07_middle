"""
Run AutoRAG API server from trial directory.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AUTORAG_PYTHON = os.getenv("AUTORAG_PYTHON", "")
if _AUTORAG_PYTHON and Path(_AUTORAG_PYTHON).exists() and \
        sys.executable != str(Path(_AUTORAG_PYTHON).resolve()):
    print(f"[AutoRAG] 인터프리터 전환: {_AUTORAG_PYTHON}")
    result = subprocess.run([_AUTORAG_PYTHON, __file__] + sys.argv[1:])
    sys.exit(result.returncode)

try:
    import autorag  # noqa: F401
except ImportError:
    print(f"[AutoRAG] 미설치. 현재 인터프리터: {sys.executable}")
    print("설치 방법: pip install -r requirements-gemma4.txt")
    sys.exit(1)


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
