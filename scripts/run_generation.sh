#!/usr/bin/env bash
# Stage 1 for every (model, dataset). Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
# run inside the project environment (see README)
export HF_HUB_OFFLINE=1
mkdir -p outputs/logs
for spec in qwen3b:4 llama3b:3 qwen7b:2 llama8b:2; do
  m=${spec%%:*}; bq=${spec##*:}
  for d in triviaqa nq popqa; do
    if [ -f "outputs/${m}__${d}/hidden.npz" ] && [ -f "outputs/${m}__${d}/.complete" ]; then
      echo "skip $m $d"; continue
    fi
    echo "start $m $d $(date +%H:%M)"
    if python src/generate_and_score.py --model "$m" --dataset "$d" --batch_q "$bq" \
         > "outputs/logs/${m}__${d}.log" 2>&1; then
      touch "outputs/${m}__${d}/.complete"
      echo "$(grep '^done' outputs/logs/${m}__${d}.log) $(date +%H:%M)"
    else
      echo "FAILED $m $d (see outputs/logs/${m}__${d}.log)"
    fi
  done
done
echo "ALL GENERATION FINISHED"
