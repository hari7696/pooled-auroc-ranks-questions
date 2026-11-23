"""Stage 3: semantic clustering of the answers to each question (Kuhn et al., 2023).

Two answers share a cluster if their normalised strings are equal or if an NLI model
(DeBERTa-large-MNLI) predicts entailment in both directions for "question + answer" texts.
Clusters are built greedily by comparing each answer with each cluster representative.

Usage: python src/cluster.py [--device cuda]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import NLI_MODEL, OUT, normalize, read_jsonl


class NLI:
    def __init__(self, device):
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
        if device.startswith("cuda"):
            self.model.half()
        self.device = device
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = lab["entailment"]
        self.cache: dict[tuple[str, str], bool] = {}

    @torch.no_grad()
    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        todo = [p for p in set(pairs) if p not in self.cache]
        for b in range(0, len(todo), 64):
            chunk = todo[b:b + 64]
            enc = self.tok([a for a, _ in chunk], [c for _, c in chunk], padding=True,
                           truncation=True, max_length=128, return_tensors="pt").to(self.device)
            pred = self.model(**enc).logits.argmax(-1).tolist()
            for p, y in zip(chunk, pred):
                self.cache[p] = (y == self.ent)
        return [self.cache[p] for p in pairs]

    @torch.no_grad()
    def prefill(self, rows, bs: int = 512):
        """Batch every ordered within-question pair the greedy pass could query (superset), so that
        clustering itself only hits the cache. Same result as the lazy path, far fewer GPU calls."""
        pairs = set()
        for r in rows:
            q = r["question"]
            uniq = {}
            for a in r["answers"]:
                n = normalize(a)
                if n:
                    uniq.setdefault(a, n)
            texts = list(uniq)
            for i, x in enumerate(texts):
                for j, y in enumerate(texts):
                    if i != j and uniq[x] != uniq[y]:
                        pairs.add((f"{q} {x}", f"{q} {y}"))
        todo = sorted((p for p in pairs if p not in self.cache), key=lambda p: len(p[0]) + len(p[1]))
        for b in tqdm(range(0, len(todo), bs), desc="nli-prefill"):
            chunk = todo[b:b + bs]
            enc = self.tok([a for a, _ in chunk], [c for _, c in chunk], padding=True,
                           truncation=True, max_length=128, return_tensors="pt").to(self.device)
            pred = self.model(**enc).logits.argmax(-1).tolist()
            for p, y in zip(chunk, pred):
                self.cache[p] = (y == self.ent)


def cluster_question(nli: NLI, question: str, answers: list[str]) -> list[int]:
    norm = [normalize(a) for a in answers]
    reps: list[int] = []  # index of representative answer per cluster
    cid = [-1] * len(answers)
    for i, a in enumerate(answers):
        if not norm[i]:
            cid[i] = -2  # empty answers: each is its own singleton (assigned below)
            continue
        found = -1
        for c, r in enumerate(reps):
            if norm[r] == norm[i]:
                found = c
                break
        if found < 0 and reps:
            texts = [(f"{question} {answers[r]}", f"{question} {a}") for r in reps]
            fwd = nli.entails(texts)
            bwd = nli.entails([(y, x) for x, y in texts])
            for c, (f, b) in enumerate(zip(fwd, bwd)):
                if f and b:
                    found = c
                    break
        if found < 0:
            reps.append(i)
            found = len(reps) - 1
        cid[i] = found
    nxt = len(reps)
    for i in range(len(cid)):
        if cid[i] == -2:
            cid[i] = nxt
            nxt += 1
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    runs = args.runs or sorted(p.name for p in OUT.iterdir() if (p / ".complete").exists())
    runs = [r for r in runs if not (OUT / r / "clusters.npy").exists()]
    if not runs:
        print("nothing to cluster")
        return
    nli = NLI(args.device)
    for run in runs:
        nli.cache.clear()
        rows = read_jsonl(OUT / run / "generations.jsonl")
        if args.device.startswith("cuda"):
            nli.prefill(rows)
        cl = np.array([cluster_question(nli, r["question"], r["answers"])
                       for r in tqdm(rows, desc=run)], dtype=np.int16)
        np.save(OUT / run / "clusters.npy", cl)
        ncl = np.array([len(set(c[1:])) for c in cl])
        print(f"{run}: mean clusters among samples={ncl.mean():.2f}")


if __name__ == "__main__":
    main()
