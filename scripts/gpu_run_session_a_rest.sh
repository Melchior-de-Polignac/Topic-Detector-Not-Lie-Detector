#!/bin/bash
# Session A remainder on the 14B: T1.6 (construct validity), T1.3 (benign-China),
# T1.4 (status-contrast D). Runs AFTER T1.2 (reuses runs/exp1/h1b_responses.json cache).
# Adapters for T1.4 must already be at runs/lora/{belief,refusal} (shipped via runpodctl).
# Detached; writes /root/jspace-run/rest.log with explicit exit markers.
set -x
cd /root/jspace-run
export HF_HOME=/root/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# T1.6 construct validity (self-contained; generates its own neutral continuations)
python3 exp/exp6_construct_validity.py --model "$M" --device cuda --layer 24 --k 1 \
  --max-new-tokens 60 --freq-vocab 150 --out runs/exp6/construct_validity.json
echo "EXP6_EXIT=$?"

# T1.3 benign-China control (reuses T1.2 cache for sensitive+control arms)
python3 exp/exp5_benign_china.py --model "$M" --device cuda --layer 24 \
  --responses-cache runs/exp1/h1b_responses.json --out runs/exp5/benign_china.json
echo "EXP5_EXIT=$?"

# T1.4 status-contrast probe D (base + belief + refusal; CPU-merges adapters)
python3 exp/exp4_status_contrast.py --model "$M" --device cuda --layer 24 \
  --belief-adapter runs/lora/belief --refusal-adapter runs/lora/refusal \
  --responses-cache runs/exp1/h1b_responses.json --out runs/exp4/status_contrast.json
echo "EXP4_EXIT=$?"

echo "SESSION_A_REST_DONE_MARK"
