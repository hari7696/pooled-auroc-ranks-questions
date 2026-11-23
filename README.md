# Pooled AUROC Ranks Questions, Not Answers

Code and result tables for the paper *Pooled AUROC Ranks Questions, Not Answers: Within-Question Discrimination of LLM Hallucination Detectors*.

Hallucination detectors are usually benchmarked with pooled AUROC over all (question, answer) pairs. Pooled AUROC is an exact mixture of a within-question and a between-question AUROC,

    pooled AUROC = π_W · A_W + (1 − π_W) · A_B,   with π_W < 2/Q for Q questions,

so with thousands of questions it measures how well a detector ranks *questions*. Best-of-n answer selection depends only on the within-question part A_W. This repository measures both parts for 14 detectors and two baselines, on 6 open models × 3 QA datasets (396k sampled answers).

## Layout

```
src/common.py              models, datasets, prompts, label helpers (WQ_OUT overrides the output folder)
src/generate_and_score.py  stage 1: greedy answer + 10 samples, likelihood features, P(True), hidden states
src/label.py               stage 2: LLM-judge correctness labels (plus alias matching)
src/cluster.py             stage 3: DeBERTa-MNLI semantic clustering
src/build_items.py         stage 4: consistency features and cross-fitted probes -> items*.parquet
src/metrics.py             AUROC decomposition, cluster bootstrap, best-of-n, AURC
src/analyze.py             stage 5: all result tables
src/make_figures.py        paper figures from the result CSVs
src/make_tables.py         paper tables from the result CSVs
scripts/run_generation.sh  sequential stage-1 runner for a single GPU
scripts/finish_local.sh    stages 3-4 for runs whose GPU stages ran elsewhere
scripts/together/          Kubernetes manifests and worker scripts for the 8×H100 run
outputs_h100/results*/     result CSVs reported in the paper
```

The `results_L0_judge`, `results_L2_judge` and `results_L1_alias` folders are the robustness variants: probes on other hidden layers and alias-match labels instead of the LLM judge.

## Models and data

- Generators: Qwen2.5-3B/7B/14B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Phi-3.5-mini-instruct.
- Judge: Qwen2.5-7B-Instruct. NLI model: microsoft/deberta-large-mnli.
- Datasets: TriviaQA, Natural Questions, PopQA (2,000 questions each, 10 samples per question, seed 0).

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # creates .venv from pyproject.toml / uv.lock (PyTorch CUDA 12.8 wheels)
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## Reproduce

Stages 1-4 need a GPU. On an 8 GB GPU, use 4-bit loading for the 7B/8B models and small batches.

```bash
python src/generate_and_score.py --model qwen3b --dataset triviaqa --batch_q 4
python src/label.py
python src/cluster.py
python src/build_items.py
python src/build_items.py --layer_idx 0
python src/build_items.py --layer_idx 2
python src/build_items.py --label alias
```

Stage 5 and the paper assets run on a CPU in under a minute:

```bash
export WQ_OUT=outputs_h100
python src/analyze.py
for s in _L0_judge _L2_judge _L1_alias; do python src/analyze.py --suffix $s; done
python src/make_figures.py
python src/make_tables.py      # writes paper/figures and paper/tables
```

The reported runs used one 8×H100 node on Kubernetes (under an hour). Create the PVC and a ConfigMap holding `src/*.py`, `scripts/together/prefetch.py` and `scripts/together/worker.sh`, then run `kubectl apply -f scripts/together/pods.yaml` and pull the results with `scripts/together/fetch_loop.sh`.

## Citation

The paper is under review. A citation entry will be added on publication.
