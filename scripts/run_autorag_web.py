"""
Run AutoRAG web UI from trial folder.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoRAG built-in web UI.")
    parser.add_argument("--trial-path", type=str, default="evaluation/autorag_benchmark/0")
    args = parser.parse_args()

    cmd = ["autorag", "run_web", "--trial_path", args.trial_path]
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

