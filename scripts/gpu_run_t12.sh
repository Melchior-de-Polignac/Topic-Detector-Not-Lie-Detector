#!/bin/bash
# T1.2 instrumented H1 rerun on the 14B: set-a full layer sweep + set-b robustness.
# Detached; writes /root/jspace-run/t12.log with explicit exit markers.
set -x
cd /root/jspace-run
export HF_HOME=/root/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
CACHE=runs/exp1/h1b_responses.json
mkdir -p runs/exp1

python3 exp/exp1b_h1_instrumented.py --model "$M" --device cuda \
  --layers 8,12,16,20,24,28,32,36 --primary-layer 24 --averaging-set a \
  --responses-cache "$CACHE" --out runs/exp1/h1_instrumented.json
echo "SETA_EXIT=$?"

python3 exp/exp1b_h1_instrumented.py --model "$M" --device cuda \
  --layers 24 --primary-layer 24 --averaging-set b \
  --responses-cache "$CACHE" --out runs/exp1/h1_instrumented_avgsetb.json
echo "SETB_EXIT=$?"

echo "T12_ALL_DONE_MARK"
