#!/bin/bash
# Second wave: GPU-heavy stages only (generate+score+P(True) and judge labels). Clustering and
# probe fitting run locally afterwards. Starts once this pod has no generation/labelling running.
set -u
S=/mnt/shared; cd /work/wq
export HF_HOME=$S/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 WQ_OUT=$S/outputs JUDGE_BS=128
grep -q qwen14b src/common.py || echo 'MODELS.update({"qwen14b": "Qwen/Qwen2.5-14B-Instruct", "phi35": "microsoft/Phi-3.5-mini-instruct"})' >> src/common.py
sleep 5
while pgrep -f "src/generate_and_score|src/label" > /dev/null; do sleep 10; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
for spec in qwen14b:triviaqa:48 qwen14b:nq:48 qwen14b:popqa:48 phi35:triviaqa:64 phi35:nq:64 phi35:popqa:64; do
  IFS=: read m d bq <<< "$spec"; run=${m}__${d}
  mkdir $S/claims/$run 2>/dev/null || continue
  echo "$POD-X claimed $run $(date +%T)" >> $S/logs/claims.log
  { python src/generate_and_score.py --model $m --dataset $d --batch_q $bq --batch_seqs 176 \
      && touch $WQ_OUT/$run/.complete && python src/label.py --runs $run && touch $WQ_OUT/$run/.gpu_done; } \
      > $S/logs/$run.log 2>&1 || echo "FAILED" >> $S/logs/$run.log
  echo "$POD-X finished $run $(date +%T)" >> $S/logs/claims.log
done
