#!/bin/bash
# One pod = one GPU = two queue workers. Runs claim work via atomic mkdir on the shared volume.
set -u
S=/mnt/shared
export HF_HOME=$S/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6 WQ_OUT=$S/outputs JUDGE_BS=128
mkdir -p $WQ_OUT $S/claims $S/logs /work/wq/src
cp /cfg/*.py /work/wq/src/ && mv /work/wq/src/prefetch.py /work/wq/
cd /work/wq
pip install -q "transformers==5.24.0" accelerate sentence-transformers datasets pandas pyarrow \
  scikit-learn scipy tqdm > $S/logs/pip_$POD.log 2>&1 || echo "pip failed" >> $S/logs/pip_$POD.log
if [ "$POD" = "w0" ]; then
  python prefetch.py > $S/logs/prefetch.log 2>&1 && touch $S/READY
fi
while [ ! -f $S/READY ]; do sleep 5; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

pipeline() {  # $1=model $2=dataset $3=batch_q
  local run=$1__$2
  python src/generate_and_score.py --model $1 --dataset $2 --batch_q $3 --batch_seqs 176 \
    && touch $WQ_OUT/$run/.complete || { echo "GEN FAILED"; return 1; }
  python src/label.py --runs $run || echo "LABEL FAILED"
  python src/cluster.py --device cuda --runs $run || echo "CLUSTER FAILED"
  python src/build_items.py --runs $run &
  python src/build_items.py --runs $run --layer_idx 0 &
  python src/build_items.py --runs $run --layer_idx 2 &
  python src/build_items.py --runs $run --label alias &
  wait
  touch $WQ_OUT/$run/.done
}

worker() {
  for spec in llama8b:triviaqa:48 llama8b:nq:48 llama8b:popqa:48 qwen7b:triviaqa:48 qwen7b:nq:48 \
              qwen7b:popqa:48 llama3b:triviaqa:64 llama3b:nq:64 llama3b:popqa:64 qwen3b:triviaqa:64 \
              qwen3b:nq:64 qwen3b:popqa:64; do
    IFS=: read m d bq <<< "$spec"
    mkdir $S/claims/${m}__${d} 2>/dev/null || continue
    echo "$POD-$1 claimed ${m}__${d} $(date +%T)" >> $S/logs/claims.log
    pipeline $m $d $bq > $S/logs/${m}__${d}.log 2>&1
    echo "$POD-$1 finished ${m}__${d} $(date +%T)" >> $S/logs/claims.log
  done
}
worker A &
sleep 20
worker B &
wait
echo "$POD idle" >> $S/logs/claims.log
sleep infinity
