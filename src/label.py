"""Stage 2: correctness labels for every (question, answer).

Two labellers:
  * alias  : lenient normalised string match against the gold aliases;
  * judge  : Qwen2.5-7B-Instruct (4-bit) graded with references; we store the log-odds of
             "Yes" vs "No". Each unique (question, normalised answer) is judged once.

Usage: python src/label.py --runs qwen3b__triviaqa llama3b__nq ...   (default: all finished)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from common import (JUDGE_MODEL, OUT, alias_match, judge_messages, load_lm, normalize,
                    read_jsonl)


@torch.no_grad()
def judge_batch(tok, model, seqs, yes_id, no_id, bs=int(os.environ.get("JUDGE_BS", 24))):
    out = np.zeros(len(seqs), np.float32)
    order = np.argsort([len(s) for s in seqs])  # length bucketing for speed
    for b in tqdm(range(0, len(seqs), bs), desc="judge"):
        idx = order[b:b + bs]
        chunk = [seqs[i] for i in idx]
        L = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, s in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        last = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        lens = att.sum(1) - 1
        logits = model.lm_head(last[torch.arange(len(chunk)), lens]).float()
        out[idx] = (logits[:, yes_id] - logits[:, no_id]).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    runs = args.runs or sorted(p.name for p in OUT.iterdir() if (p / ".complete").exists())
    runs = [r for r in runs if not (OUT / r / "labels.npz").exists()]
    if not runs:
        print("nothing to label")
        return
    tok, model = load_lm(JUDGE_MODEL)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    for run in runs:
        rows = read_jsonl(OUT / run / "generations.jsonl")
        alias, keys = [], []
        uniq: dict[tuple, int] = {}
        seqs = []
        for qi, r in enumerate(rows):
            for a in r["answers"]:
                alias.append(alias_match(a, r["golds"]))
                key = (qi, normalize(a))
                if key not in uniq:
                    uniq[key] = len(seqs)
                    msgs = judge_messages(r["question"], r["golds"], a if a.strip() else "(no answer)")
                    seqs.append(list(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                             tokenize=True, return_dict=False)))
                keys.append(uniq[key])
        scores = judge_batch(tok, model, seqs, yes_id, no_id)
        judge = scores[np.array(keys)]
        # empty answers are incorrect by definition
        empty = np.array([not normalize(a) for r in rows for a in r["answers"]])
        judge[empty] = -20.0
        alias = np.array(alias)
        np.savez(OUT / run / "labels.npz", alias=alias, judge_logodds=judge)
        jc = judge > 0
        print(f"{run}: n={len(judge)} unique={len(seqs)} acc_alias={alias.mean():.3f} "
              f"acc_judge={jc.mean():.3f} agree={(jc == alias).mean():.3f}")


if __name__ == "__main__":
    main()
