#!/usr/bin/env bash
# Task 3 GO/NO-GO gate on a rented GPU. Designed to minimise paid minutes:
#   - does NOT reinstall CUDA torch (it is preinstalled in any PyTorch vast.ai image)
#   - installs only the few extra deps exp0 needs
#   - runs a fast 1.5B GPU smoke FIRST to prove the box works, then the 14B gate
#
# Usage on the GPU box (from the repo root):
#   bash scripts/gpu_run_exp0.sh
#
# Env knobs (all optional):
#   SKIP_SMOKE=1     skip the 1.5B smoke, go straight to the 14B
#   MODEL_14B=...     override the 14B model id
#   HF_TOKEN=...      only needed if a model is gated (these deepseek distills are public)
set -euo pipefail

# Reduce CUDA allocator fragmentation (belt-and-suspenders for the 14B on a 48GB card;
# the real memory win is freezing params in jlens_vectors so backward allocates no
# parameter-gradient buffer).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_14B="${MODEL_14B:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
MODEL_SMOKE="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

cd "$(dirname "$0")/.."   # repo root
echo "== repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?') =="

# 1. Sanity: CUDA must be visible, and torch must already see it (don't reinstall torch).
python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA not available — pick a GPU PyTorch image on vast.ai"
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  gpu={torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"gpu mem: {total/1e9:.0f} GB total, {free/1e9:.0f} GB free")
assert total/1e9 >= 40, "Need a >=40GB GPU for the 14B in bf16"
PY

# 2. Extra deps only (torch stays as-is). Pin transformers to the tested major (5.x).
pip install -q --no-input "transformers>=5,<6" accelerate numpy

mkdir -p runs/exp0

# 3. Fast GPU smoke on the 1.5B (cheap; proves the box + code + GPU path before the 14B download).
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "== 1.5B GPU smoke =="
  python exp/exp0_validation.py --model "$MODEL_SMOKE" --device cuda \
      --out runs/exp0/validation_smoke_gpu.json
fi

# 4. THE GATE: full 14B run.
echo "== 14B GATE run =="
time python exp/exp0_validation.py --model "$MODEL_14B" --device cuda \
    --out runs/exp0/validation.json

echo
echo "== DONE. Copy runs/exp0/validation.json back to your laptop and commit it. =="
echo "   scp / vast.ai 'Download' the file, then locally:"
echo "     git add runs/exp0/validation.json && git commit -m 'exp0: 14B gate result' && git push"
