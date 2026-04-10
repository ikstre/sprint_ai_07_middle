#!/bin/bash
# Gemma4-E4B AutoRAG 실행 스크립트
#
# 모델: /srv/shared_data/models/gemma/Gemma4-E4B
#   - Gemma 4 4B dense 모델 (BF16, ~15GB)
#   - transformers 5.x + user-local vLLM 필요 (Gemma4ForConditionalGeneration 등록됨)
#
# transformers 5.x (user-local ~/.local/lib/python3.11/site-packages) 환경 필요.
# PYTHONNOUSERSITE를 해제하여 user-local 패키지가 sprint_env보다 우선 로드되도록 보장.
#
# 사용법:
#   bash scripts/run_gemma4_optimization.sh
#   또는 추가 인자:
#   bash scripts/run_gemma4_optimization.sh --qa-path custom/qa.parquet

set -e

# user-local 패키지 차단 방지 (메인 실행 스크립트와 분리 실행할 때 환경 보장)
unset PYTHONNOUSERSITE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Gemma4-E4B AutoRAG 최적화 ==="
echo "Python: $(/opt/miniconda3/envs/sprint_env/bin/python --version)"
echo "transformers: $(/opt/miniconda3/envs/sprint_env/bin/python -c 'import transformers; print(transformers.__version__, transformers.__file__)')"
echo ""

# ── GPU 메모리 확인 및 정리 ──────────────────────────────────────────────────
# Gemma4-E4B (~15GB) 로드를 위해 최소 15GB 여유 필요
MIN_FREE_GIB=15

echo "── GPU 상태 확인 ──"
nvidia-smi --query-gpu=memory.free,memory.total,memory.used --format=csv,noheader,nounits | \
  awk '{printf "  GPU 메모리: free=%.1fGiB / total=%.1fGiB (used=%.1fGiB)\n", $1/1024, $2/1024, $3/1024}'

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
FREE_GIB=$(echo "scale=2; $FREE_MIB / 1024" | bc)

if (( FREE_MIB < MIN_FREE_GIB * 1024 )); then
  echo ""
  echo "  [경고] GPU 여유 메모리 ${FREE_GIB}GiB < 필요 ${MIN_FREE_GIB}GiB"
  echo ""
  echo "  GPU 점유 프로세스:"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | \
    awk -F',' '{printf "    PID %-8s %s  (%s)\n", $1, $2, $3}' || true

  echo ""
  echo "  점유 프로세스를 종료합니다..."
  # CUDA 프로세스 PID 목록 수집 후 종료
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
  if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
      echo "    kill $PID"
      kill "$PID" 2>/dev/null || true
    done
    echo "  5초 대기 (프로세스 정리)..."
    sleep 5
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    FREE_GIB=$(echo "scale=2; $FREE_MIB / 1024" | bc)
    echo "  정리 후 여유 메모리: ${FREE_GIB}GiB"
    if (( FREE_MIB < MIN_FREE_GIB * 1024 )); then
      echo ""
      echo "  [오류] 여전히 메모리 부족 (${FREE_GIB}GiB). 수동으로 GPU 프로세스를 종료하세요:"
      echo "    nvidia-smi  # 점유 프로세스 확인"
      echo "    kill <PID>"
      exit 1
    fi
  fi
fi
echo ""

/opt/miniconda3/envs/sprint_env/bin/python "$SCRIPT_DIR/run_autorag_optimization.py" \
  --qa-path "$PROJECT_ROOT/data/autorag/qa.parquet" \
  --corpus-path "$PROJECT_ROOT/data/autorag/corpus.parquet" \
  --config-path "$PROJECT_ROOT/configs/autorag/local_gemma4.yaml" \
  --project-dir "$PROJECT_ROOT/evaluation/autorag_benchmark_gemma4" \
  "$@"

echo ""
echo "=== 완료. 결과 병합 명령어 ==="
echo "python scripts/merge_gemma4_results.py \\"
echo "  --main-dir evaluation/autorag_benchmark_local \\"
echo "  --gemma4-dir evaluation/autorag_benchmark_gemma4"
