"""
Run RAG evaluation over multiple retrieval configs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from configs.config import Config
from configs import paths
from src.evaluation.single_dataset import EVALUATION_QUESTIONS
from src.evaluator import RAGEvaluator
from src.rag_pipeline import RAGPipeline


CORE_METRICS = [
    "avg_elapsed_time",
    "p50_elapsed_time",
    "p95_elapsed_time",
    "avg_hit_at_5",
    "avg_ndcg_at_5",
    "avg_keyword_recall",
    "avg_field_coverage",
    "avg_grounded_token_ratio",
    "decline_accuracy",
    "avg_total_tokens",
]

DETAILED_EXTRA_METRICS = [
    "avg_bertscore_f1",
    "avg_rougeL_f1",
    "avg_meteor",
    "avg_hallucination_risk_proxy",
    "avg_llm_relevance",
    "avg_llm_accuracy",
    "avg_llm_faithfulness",
    "avg_llm_completeness",
    "avg_llm_conciseness",
]

DEFAULT_CORE_GATE_THRESHOLDS = {
    "p95_elapsed_time": {"op": "<=", "value": 12.0},
    "avg_hit_at_5": {"op": ">=", "value": 0.80},
    "avg_ndcg_at_5": {"op": ">=", "value": 0.65},
    "avg_field_coverage": {"op": ">=", "value": 0.55},
    "avg_grounded_token_ratio": {"op": ">=", "value": 0.55},
    "decline_accuracy": {"op": ">=", "value": 0.90},
}


def run_single_config(
    config: Config,
    label: str,
    output_dir: Path,
    questions: list[dict],
    use_llm_judge: bool = True,
    use_bertscore: bool = False,
    collection_name: str = "rfp_chunk600",
    skip_keys: set | None = None,
):
    print(f"\n{'=' * 56}")
    print(f"config: {label}")
    print(f"{'=' * 56}")

    pipeline = RAGPipeline(config)
    pipeline.initialize_vectorstore(collection_name=collection_name)

    evaluator = RAGEvaluator(config, generator=pipeline.generator)
    df = evaluator.run_evaluation_suite(
        questions=questions,
        use_llm_judge=use_llm_judge,
        use_bertscore=use_bertscore,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(str(output_dir / f"eval_{label}.json"))
    df.to_csv(output_dir / f"eval_{label}.csv", index=False, encoding="utf-8-sig")

    summary = evaluator.summary_report(df)
    if skip_keys:
        summary = {k: v for k, v in summary.items() if k not in skip_keys}
    with open(output_dir / f"summary_{label}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== summary: {label} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, df


def resolve_mode_flags(
    mode: str,
    judge: str,
    bertscore: str,
    no_judge: bool,
    use_bertscore_flag: bool,
) -> tuple[bool, bool]:
    if judge == "on":
        use_llm_judge = True
    elif judge == "off":
        use_llm_judge = False
    else:
        use_llm_judge = mode == "detailed"

    if no_judge:
        use_llm_judge = False

    if bertscore == "on":
        use_bertscore = True
    elif bertscore == "off":
        use_bertscore = False
    else:
        use_bertscore = mode == "detailed"

    if use_bertscore_flag:
        use_bertscore = True

    return use_llm_judge, use_bertscore


def select_questions(limit: Optional[int]) -> list[dict]:
    if not limit or limit <= 0:
        return EVALUATION_QUESTIONS
    return EVALUATION_QUESTIONS[:limit]


def print_summary_view(summary: dict, mode: str) -> None:
    print("\nSUMMARY VIEW")
    print("-" * 40)

    keys = CORE_METRICS.copy()
    if mode == "detailed":
        keys.extend(DETAILED_EXTRA_METRICS)

    for k in keys:
        if k in summary:
            v = summary[k]
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")


def save_mode_csv(df, output_dir: Path, label: str, mode: str) -> None:
    if mode == "core":
        core_columns = [
            "id",
            "category",
            "elapsed_time",
            "num_retrieved",
            "hit_at_5",
            "ndcg_at_5",
            "keyword_recall",
            "field_coverage",
            "grounded_token_ratio",
            "hallucination_risk_proxy",
            "correctly_declined",
            "total_tokens",
        ]
        cols = [c for c in core_columns if c in df.columns]
        core_df = df[cols].copy()
        core_df.to_csv(output_dir / f"eval_{label}_core.csv", index=False, encoding="utf-8-sig")
        return

    # detailed mode keeps all metrics for tuning/debug.
    df.to_csv(output_dir / f"eval_{label}_detailed.csv", index=False, encoding="utf-8-sig")


def _metric_pass(actual: float, op: str, target: float) -> bool:
    if op == ">=":
        return actual >= target
    if op == "<=":
        return actual <= target
    raise ValueError(f"Unsupported operator: {op}")


def load_gate_thresholds(path: Optional[str]) -> dict:
    if not path:
        return DEFAULT_CORE_GATE_THRESHOLDS

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"gate threshold file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_default_collection(scenario: str, collection: str) -> str:
    if collection:
        return collection
    return "rfp_chunk600_a" if scenario == "A" else "rfp_chunk600"


def parse_chunk_sizes(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def derive_collection_for_size(scenario: str, chunk_size: int) -> str:
    suffix = "_a" if scenario == "A" else ""
    return f"rfp_chunk{chunk_size}{suffix}"


def derive_output_dir_for_size(base_output_dir: Path, scenario: str, chunk_size: int, mode: str, test_limit: int) -> Path:
    prefix = scenario.lower()
    scope = "smoke" if test_limit else "full"
    return base_output_dir / f"{prefix}_chunk{chunk_size}_{scope}_{mode}"


def build_child_command(args: argparse.Namespace, chunk_size: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--scenario", args.scenario,
        "--mode", args.mode,
        "--judge", args.judge,
        "--bertscore", args.bertscore,
        "--gate", args.gate,
        "--gate-thresholds", args.gate_thresholds,
        "--output-dir", str(output_dir),
        "--collection", derive_collection_for_size(args.scenario, chunk_size),
    ]

    if args.test_limit:
        command += ["--test-limit", str(args.test_limit)]
    if args.no_judge:
        command.append("--no-judge")
    if args.use_bertscore:
        command.append("--use-bertscore")

    return command


def run_chunk_sizes_in_parallel(args: argparse.Namespace, chunk_sizes: list[int]) -> int:
    if args.collection:
        raise ValueError("--chunk-sizes 사용 시 --collection은 지정하지 마세요.")

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    max_parallel = args.max_parallel if args.max_parallel > 0 else len(chunk_sizes)
    running: list[tuple[int, Path, subprocess.Popen, object]] = []
    exit_code = 0

    def launch(size: int) -> None:
        output_dir = derive_output_dir_for_size(base_output_dir, args.scenario, size, args.mode, args.test_limit)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        cmd = build_child_command(args, size, output_dir)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        running.append((size, log_path, proc, log_fh))
        print(f"[spawn] chunk={size} pid={proc.pid} log={log_path}")

    pending = list(chunk_sizes)
    while pending or running:
        while pending and len(running) < max_parallel:
            launch(pending.pop(0))

        next_running: list[tuple[int, Path, subprocess.Popen, object]] = []
        for size, log_path, proc, log_fh in running:
            status = proc.poll()
            if status is None:
                next_running.append((size, log_path, proc, log_fh))
                continue
            log_fh.close()
            print(f"[done] chunk={size} exit_code={status} log={log_path}")
            if status != 0 and exit_code == 0:
                exit_code = status
        running = next_running

        if running:
            time.sleep(2)

    return exit_code


def build_gate_report(all_summaries: dict, thresholds: dict, mode: str) -> dict:
    report = {"mode": mode, "thresholds": thresholds, "configs": {}, "best_config": None}

    best_label = None
    best_score = -1.0

    for label, summary in all_summaries.items():
        metric_results = {}
        pass_count = 0
        total_count = 0

        for metric, rule in thresholds.items():
            op = rule["op"]
            target = float(rule["value"])
            actual_raw = summary.get(metric)
            if actual_raw is None:
                metric_results[metric] = {"status": "missing", "actual": None, "op": op, "target": target}
                continue

            actual = float(actual_raw)
            passed = _metric_pass(actual, op, target)
            metric_results[metric] = {
                "status": "pass" if passed else "fail",
                "actual": actual,
                "op": op,
                "target": target,
            }
            total_count += 1
            pass_count += 1 if passed else 0

        gate_passed = (pass_count == total_count) if total_count > 0 else False
        score = pass_count / total_count if total_count else 0.0
        report["configs"][label] = {
            "gate_passed": gate_passed,
            "pass_count": pass_count,
            "total_count": total_count,
            "pass_ratio": score,
            "metrics": metric_results,
        }

        if score > best_score:
            best_score = score
            best_label = label

    report["best_config"] = best_label
    return report


def save_gate_reports(gate_report: dict, output_dir: Path) -> None:
    mode = gate_report["mode"]
    json_path = output_dir / f"gate_report_{mode}.json"
    md_path = output_dir / f"gate_report_{mode}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, ensure_ascii=False, indent=2)

    lines = [
        f"# Gate Report ({mode})",
        "",
        f"- best_config: `{gate_report.get('best_config')}`",
        "",
        "| config | gate | pass_count | ratio |",
        "|---|---:|---:|---:|",
    ]
    for label, row in gate_report["configs"].items():
        gate_str = "PASS" if row["gate_passed"] else "FAIL"
        lines.append(f"| {label} | {gate_str} | {row['pass_count']}/{row['total_count']} | {row['pass_ratio']:.2f} |")

    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append("| metric | rule |")
    lines.append("|---|---|")
    for metric, rule in gate_report["thresholds"].items():
        lines.append(f"| {metric} | {rule['op']} {rule['value']} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nGate reports saved:\n- {json_path}\n- {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="B",
        choices=["A", "B"],
        help="평가 시나리오 선택 (A=로컬 HF, B=OpenAI API)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="core",
        choices=["core", "detailed"],
        help="core: 실사용 핵심지표 중심, detailed: 모델/튜닝 상세평가",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="LLM judge 사용 여부 (auto는 mode에 따름)",
    )
    parser.add_argument(
        "--bertscore",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="BERTScore 사용 여부 (auto는 mode에 따름)",
    )
    parser.add_argument("--test-limit", type=int, default=0, help="앞에서 N개 질문만 테스트 실행")
    parser.add_argument(
        "--gate",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="PASS/FAIL gate 적용 (auto: core=on, detailed=off)",
    )
    parser.add_argument("--gate-thresholds", type=str, default="", help="gate threshold JSON path")
    parser.add_argument("--no-judge", action="store_true", help="(legacy) Disable LLM-as-a-judge scoring")
    parser.add_argument("--use-bertscore", action="store_true", help="(legacy) Enable BERTScore")
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="",
        help="여러 chunk 크기를 병렬 실행 (쉼표 구분, 예: 600,800,1000,1200)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="chunk-sizes 병렬 실행 시 동시 실행 개수 (0=전체 동시 실행)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="평가에 사용할 ChromaDB 컬렉션 이름 (미지정 시 A=rfp_chunk600_a, B=rfp_chunk600)",
    )
    args = parser.parse_args()

    chunk_sizes = parse_chunk_sizes(args.chunk_sizes)
    if chunk_sizes:
        raise_code = run_chunk_sizes_in_parallel(args, chunk_sizes)
        sys.exit(raise_code)

    use_llm_judge, use_bertscore = resolve_mode_flags(
        mode=args.mode,
        judge=args.judge,
        bertscore=args.bertscore,
        no_judge=args.no_judge,
        use_bertscore_flag=args.use_bertscore,
    )

    questions = select_questions(args.test_limit)
    output_dir = Path(args.output_dir)
    collection_name = resolve_default_collection(args.scenario, args.collection)

    print(
        f"scenario={args.scenario} | mode={args.mode} | judge={use_llm_judge} | "
        f"bertscore={use_bertscore} | test_limit={args.test_limit or 'all'} | "
        f"collection={collection_name}"
    )

    if args.scenario == "A" and use_llm_judge and not os.getenv("OPENAI_API_KEY"):
        print("[warn] scenario A에서도 LLM judge는 OpenAI API를 사용합니다. OPENAI_API_KEY가 필요합니다.")

    configs = [
        {"label": "similarity_k5", "kwargs": {"retrieval_method": "similarity", "retrieval_top_k": 5}},
        {"label": "mmr_k5", "kwargs": {"retrieval_method": "mmr", "retrieval_top_k": 5}},
        {"label": "hybrid_k5", "kwargs": {"retrieval_method": "hybrid", "retrieval_top_k": 5}},
        {"label": "similarity_k10", "kwargs": {"retrieval_method": "similarity", "retrieval_top_k": 10}},
    ]

    skip_keys = {"avg_total_tokens"} if args.scenario == "A" else set()

    all_summaries = {}
    for cfg in configs:
        config = Config(
            scenario=args.scenario,
            metadata_csv=paths.METADATA_CSV,
            vectordb_dir=paths.VECTORDB_DIR,
            **cfg["kwargs"],
        )
        summary, df = run_single_config(
            config=config,
            label=cfg["label"],
            output_dir=output_dir,
            questions=questions,
            use_llm_judge=use_llm_judge,
            use_bertscore=use_bertscore,
            collection_name=collection_name,
            skip_keys=skip_keys,
        )
        save_mode_csv(df=df, output_dir=output_dir, label=cfg["label"], mode=args.mode)

        # Save a concise mode-specific summary file.
        mode_summary = {"mode": args.mode, "config": cfg["label"]}
        for key in CORE_METRICS + (DETAILED_EXTRA_METRICS if args.mode == "detailed" else []):
            if key in summary:
                mode_summary[key] = summary[key]
        with open(output_dir / f"summary_{cfg['label']}_{args.mode}.json", "w", encoding="utf-8") as f:
            json.dump(mode_summary, f, ensure_ascii=False, indent=2)

        print_summary_view(summary, args.mode)
        all_summaries[cfg["label"]] = summary

    use_gate = (args.mode == "core") if args.gate == "auto" else (args.gate == "on")
    if use_gate:
        thresholds = load_gate_thresholds(args.gate_thresholds)
        gate_report = build_gate_report(all_summaries=all_summaries, thresholds=thresholds, mode=args.mode)
        save_gate_reports(gate_report=gate_report, output_dir=output_dir)

    print("\n" + "=" * 96)
    print("CONFIG COMPARISON")
    print("=" * 96)
    print(
        f"{'config':<20} {'p95(s)':>8} {'hit@5':>8} {'nDCG@5':>8} {'ground':>8} {'kwRecall':>10} {'bertF1':>8}"
    )
    print("-" * 96)
    for label, s in all_summaries.items():
        p95 = s.get("p95_elapsed_time", 0.0)
        hit5 = s.get("avg_hit_at_5", 0.0)
        ndcg5 = s.get("avg_ndcg_at_5", 0.0)
        ground = s.get("avg_grounded_token_ratio", 0.0)
        kw = s.get("avg_keyword_recall", 0.0)
        bf1 = s.get("avg_bertscore_f1", 0.0)
        print(f"{label:<20} {p95:>8.2f} {hit5:>8.2f} {ndcg5:>8.2f} {ground:>8.2f} {kw:>10.2f} {bf1:>8.2f}")


if __name__ == "__main__":
    main()
