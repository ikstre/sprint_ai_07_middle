"""
Run core evaluation gate as a single release check command.

Example:
  python scripts/check_release_gate.py
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def build_eval_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "scripts/run_evaluation.py",
        "--mode",
        "core",
        "--gate",
        "on",
        "--output-dir",
        args.output_dir,
    ]

    if args.gate_thresholds:
        cmd.extend(["--gate-thresholds", args.gate_thresholds])
    if args.test_limit and args.test_limit > 0:
        cmd.extend(["--test-limit", str(args.test_limit)])
    if args.judge != "auto":
        cmd.extend(["--judge", args.judge])
    if args.bertscore != "auto":
        cmd.extend(["--bertscore", args.bertscore])

    return cmd


def load_gate_report(output_dir: str) -> dict:
    path = Path(output_dir) / "gate_report_core.json"
    if not path.exists():
        raise FileNotFoundError(f"gate report not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_gate(gate_report: dict) -> tuple[bool, str]:
    configs = gate_report.get("configs", {})
    if not configs:
        return False, "No config results found in gate report."

    lines = []
    lines.append("Release Gate Summary")
    lines.append("-" * 72)
    lines.append(f"best_config: {gate_report.get('best_config')}")
    lines.append("")
    lines.append(f"{'config':<20} {'gate':<6} {'pass_count':>10} {'ratio':>8}")
    lines.append("-" * 72)

    passed_configs = []
    for label, row in configs.items():
        gate_passed = bool(row.get("gate_passed", False))
        gate_str = "PASS" if gate_passed else "FAIL"
        pass_count = f"{row.get('pass_count', 0)}/{row.get('total_count', 0)}"
        ratio = float(row.get("pass_ratio", 0.0))
        lines.append(f"{label:<20} {gate_str:<6} {pass_count:>10} {ratio:>8.2f}")
        if gate_passed:
            passed_configs.append(label)

    lines.append("-" * 72)
    if passed_configs:
        lines.append(f"RESULT: PASS (passed configs: {', '.join(passed_configs)})")
        return True, "\n".join(lines)

    lines.append("RESULT: FAIL (no config passed the core gate)")
    return False, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-command release gate runner for core metrics.")
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument(
        "--gate-thresholds",
        type=str,
        default="configs/evaluation/core_gate.default.json",
        help="JSON file with gate thresholds. Empty string to use defaults in run_evaluation.py",
    )
    parser.add_argument("--test-limit", type=int, default=0, help="Run only first N questions for quick checks")
    parser.add_argument(
        "--judge",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Override LLM judge behavior",
    )
    parser.add_argument(
        "--bertscore",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Override BERTScore behavior",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip evaluation run and only read existing gate report from output-dir",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Always exit 0 even when gate result is FAIL",
    )
    args = parser.parse_args()

    if not args.no_run:
        cmd = build_eval_command(args)
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

    gate_report = load_gate_report(args.output_dir)
    passed, summary_text = summarize_gate(gate_report)
    print(summary_text)

    if passed or args.allow_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
