"""
Run AutoRAG optimization from prepared qa/corpus parquet files.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# flashinfer가 deprecated된 cuda.cudart / cuda.nvrtc 모듈을 사용해 발생하는 경고.
# flashinfer 업스트림 이슈 — 우리 코드에서 수정 불가, 해당 메시지만 타깃 억제.
# 다른 FutureWarning은 정상 출력됨.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The cuda.cudart module is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The cuda.nvrtc module is deprecated.*",
)
# destroy_process_group 경고: vLLM spawn 서브프로세스의 C++ NCCL 레벨 경고.
# spawned 프로세스는 환경을 새로 시작하므로 Python 코드로 수정 불가.
# 동작에 영향 없는 정보성 경고 (OS가 프로세스 종료 시 리소스 자동 회수).

load_dotenv()


def _patch_transformers_validation() -> None:
    """vLLM config 로드 시 transformers 5.x strict 검증 우회 (kanana 등)."""
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        if hasattr(LlamaConfig, "__class_validators__"):
            LlamaConfig.__class_validators__ = [
                v for v in LlamaConfig.__class_validators__
                if getattr(v, "__name__", "") != "validate_architecture"
            ]
    except Exception:
        pass


def _patch_rope_parameters() -> None:
    """transformers 4.x 환경에서 EXAONE 등 5.x 전용 RopeParameters 임포트 오류 방지.

    configuration_exaone.py가 타입 힌트용으로만 사용하므로 더미 클래스로 대체해도 무해.
    """
    try:
        import transformers.modeling_rope_utils as _rope_utils
        if not hasattr(_rope_utils, "RopeParameters"):
            class RopeParameters:
                pass
            _rope_utils.RopeParameters = RopeParameters
    except Exception:
        pass


_patch_transformers_validation()
_patch_rope_parameters()

_AUTORAG_PYTHON = os.getenv("AUTORAG_PYTHON", "")
if _AUTORAG_PYTHON and Path(_AUTORAG_PYTHON).exists() and \
        str(Path(sys.executable).resolve()) != str(Path(_AUTORAG_PYTHON).resolve()):
    print(f"[AutoRAG] 인터프리터 전환: {_AUTORAG_PYTHON}")
    result = subprocess.run([_AUTORAG_PYTHON, __file__] + sys.argv[1:])
    sys.exit(result.returncode)

try:
    import autorag  # noqa: F401
except ImportError:
    print(f"[AutoRAG] 미설치. 현재 인터프리터: {sys.executable}")
    print("설치 방법: pip install -r requirements-gemma4.txt")
    sys.exit(1)

# Patch 1: ChromaDB max_batch_size(5461) 초과 시 실패하는 버그 패치
# add_embedding이 전체 corpus를 한 번에 add → batch 분할 처리로 수정
def _patch_chroma_add_embedding():
    from autorag.vectordb.chroma import Chroma
    from typing import List

    def _batched_add_embedding(self, ids: List[str], embeddings: List[List[float]]):
        max_batch = self.client.get_max_batch_size()
        for i in range(0, len(ids), max_batch):
            self.collection.add(
                ids=ids[i : i + max_batch],
                embeddings=embeddings[i : i + max_batch],
            )

    Chroma.add_embedding = _batched_add_embedding


# Patch 2: 모듈 평가 후 GPU 메모리 명시적 해제
# del instance만으로는 __del__ 즉시 호출이 보장되지 않으므로
# gc.collect() + cuda.empty_cache()를 강제 실행해 다음 모델 로드 전 VRAM 확보
def _patch_run_evaluator():
    import gc
    from pathlib import Path
    from typing import Union

    import pandas as pd
    from autorag.schema.base import BaseModule

    original_run_evaluator = BaseModule.run_evaluator.__func__

    @classmethod
    def _run_evaluator_with_cleanup(
        cls,
        project_dir: Union[str, Path],
        previous_result: pd.DataFrame,
        *args,
        **kwargs,
    ):
        result = original_run_evaluator(cls, project_dir, previous_result, *args, **kwargs)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        return result

    BaseModule.run_evaluator = _run_evaluator_with_cleanup


# Patch 3: ChromaDB is_exist가 전체 ID를 한 번에 SQLite에 넘겨
# "too many SQL variables" 에러 발생 → 배치 분할 처리로 수정
def _patch_chroma_is_exist():
    from autorag.vectordb.chroma import Chroma
    from typing import List

    async def _batched_is_exist(self, ids: List[str]) -> List[bool]:
        batch_size = 500  # SQLite 변수 제한(999) 안전 범위
        existed: set[str] = set()
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            fetched = self.collection.get(batch, include=[])
            existed.update(fetched["ids"])
        return [x in existed for x in ids]

    Chroma.is_exist = _batched_is_exist


_patch_chroma_add_embedding()
_patch_chroma_is_exist()
_patch_run_evaluator()


# Patch 4: VectorDB 임베딩 모델 ingestion 후 즉시 GPU 해제
#
# 문제: AutoRAG start_trial 시 load_all_vectordb_from_yaml로 모든 vectordb 객체를
#       한꺼번에 생성. BaseVectorStore.__init__에서 HuggingFace 모델이 즉시 GPU에 올라감.
#       vectordb_list가 start_trial 전체 scope에 살아있어 generator 진입 시까지
#       VRAM 점유 유지. (5종 × 평균 1.26GB ≈ 6.3GB 상시 점유)
#
# 해결: vectordb_ingest_huggingface 완료 직후 embedding 모델을 None으로 해제.
#       ingestion이 끝난 vectordb는 embedding 없이도 retrieval에서 새로 생성되므로 무해.
def _patch_vectordb_ingest_cleanup():
    import gc
    import autorag.evaluator as _ev_module

    _orig_ingest = _ev_module.vectordb_ingest_huggingface

    def _ingest_and_cleanup(vectordb, corpus):
        _orig_ingest(vectordb, corpus)
        # ingestion 완료 → HuggingFace 임베딩 모델 즉시 해제
        try:
            vectordb.embedding = None
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    _ev_module.vectordb_ingest_huggingface = _ingest_and_cleanup


_patch_vectordb_ingest_cleanup()


# Patch 6: hybrid_cc 정규화 함수의 0-division RuntimeWarning 수정
#
# 문제: normalize_mm/normalize_tmm/normalize_z/normalize_dbsf 모두
#       max == min(점수가 전부 동일)이면 분모가 0이 되어 NaN 반환.
#       top_k=1이거나 BM25가 동일 점수를 반환할 때 쿼리 단위로 반복 발생.
#       NaN 점수는 fuse_per_query의 weighted_sum → 정렬 순서 불일치.
#
# 해결: 각 정규화 함수에서 분모 == 0 이면 의미 있는 상수를 반환:
#       - mm/tmm: 1.0 (모두 동일 → 동등 최고 점수)
#       - z:      0.0 (표준 편차 0 → z-score 0)
#       - dbsf:   0.5 (3-sigma 범위 중앙)
def _patch_hybrid_cc_normalize():
    import numpy as np
    import autorag.nodes.hybridretrieval.hybrid_cc as _cc

    def _normalize_mm(scores, fixed_min_value=0):
        arr = np.array(scores, dtype=float)
        max_v, min_v = np.max(arr), np.min(arr)
        denom = max_v - min_v
        return np.ones_like(arr) if denom == 0 else (arr - min_v) / denom

    def _normalize_tmm(scores, fixed_min_value):
        arr = np.array(scores, dtype=float)
        max_v = np.max(arr)
        denom = max_v - fixed_min_value
        return np.zeros_like(arr) if denom == 0 else (arr - fixed_min_value) / denom

    def _normalize_z(scores, fixed_min_value=0):
        arr = np.array(scores, dtype=float)
        std_v = np.std(arr)
        return np.zeros_like(arr) if std_v == 0 else (arr - np.mean(arr)) / std_v

    def _normalize_dbsf(scores, fixed_min_value=0):
        arr = np.array(scores, dtype=float)
        mean_v = np.mean(arr)
        std_v = np.std(arr)
        if std_v == 0:
            return np.full_like(arr, 0.5)
        min_v = mean_v - 3 * std_v
        max_v = mean_v + 3 * std_v
        return (arr - min_v) / (max_v - min_v)

    _cc.normalize_mm = _normalize_mm
    _cc.normalize_tmm = _normalize_tmm
    _cc.normalize_z = _normalize_z
    _cc.normalize_dbsf = _normalize_dbsf
    _cc.normalize_method_dict["mm"] = _normalize_mm
    _cc.normalize_method_dict["tmm"] = _normalize_tmm
    _cc.normalize_method_dict["z"] = _normalize_z
    _cc.normalize_method_dict["dbsf"] = _normalize_dbsf


_patch_hybrid_cc_normalize()


# Patch 5: summary.csv의 module_name을 모델 경로 basename으로 치환
# AutoRAG는 vLLM 모듈을 항상 "Vllm"으로 표기해 모델 구분이 불가능하므로
# 평가 완료 후 module_params의 llm 경로 basename을 module_name으로 덮어씀
def _rename_summary_module_names(trial_dir: Path) -> None:
    import pandas as pd

    for summary_csv in sorted(trial_dir.rglob("summary.csv")):
        try:
            df = pd.read_csv(summary_csv)
            if "module_name" not in df.columns or "module_params" not in df.columns:
                continue

            def _display_name(row):
                if row["module_name"] != "Vllm":
                    return row["module_name"]
                try:
                    params = (
                        ast.literal_eval(row["module_params"])
                        if isinstance(row["module_params"], str)
                        else row["module_params"]
                    )
                    llm_path = params.get("llm", "")
                    return Path(llm_path).name if llm_path else row["module_name"]
                except Exception:
                    return row["module_name"]

            df["module_name"] = df.apply(_display_name, axis=1)
            df.to_csv(summary_csv, index=False)
            print(f"  ✓ module_name 갱신: {summary_csv.relative_to(trial_dir.parent)}")
        except Exception as e:
            print(f"  ⚠ module_name 갱신 실패 ({summary_csv.name}): {e}")


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

    from autorag.evaluator import Evaluator

    print(f"$ autorag evaluate --qa_data_path {qa_path} --corpus_data_path {corpus_path} "
          f"--config {config_path} --project_dir {project_dir}")
    evaluator = Evaluator(str(qa_path), str(corpus_path), project_dir=str(project_dir))
    evaluator.start_trial(str(config_path))

    trial_dir = project_dir / "0"
    print("\nAutoRAG evaluation finished.")

    # summary.csv module_name → 모델 basename으로 치환
    if trial_dir.exists():
        print("summary.csv module_name 갱신 중...")
        _rename_summary_module_names(trial_dir)
    print(f"- expected trial dir: {trial_dir}")
    print(f"- dashboard command: autorag dashboard --trial_dir {trial_dir}")
    print(f"- API command: autorag run_api --trial_dir {trial_dir} --host 0.0.0.0 --port 8000")
    print(f"- Web command: autorag run_web --trial_path {trial_dir}")

    if args.run_dashboard and trial_dir.exists():
        _run(["autorag", "dashboard", "--trial_dir", str(trial_dir)])


if __name__ == "__main__":
    main()
