"""Stage 1: sample answers, then score every answer with the generating model.

For each question we keep the greedy answer (index 0) and N_SAMPLES samples at T=1
(indices 1..N). For every (question, answer) we record
  * answer token log-probs and predictive entropies (teacher forcing, T=1),
  * hidden states at the last prompt token (question-level) and last answer token,
  * P(True) from a self-evaluation prompt.

Usage: python src/generate_and_score.py --model qwen3b --dataset triviaqa [--limit 200]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from common import (MAX_NEW_TOKENS, MODELS, N_SAMPLES, build_messages, load_lm,
                    load_questions, ptrue_messages, read_jsonl, run_dir, set_seed,
                    write_jsonl)


def chat_ids(tok, messages, add_generation_prompt=True):
    out = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                  tokenize=True, return_dict=False)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def clean_answer_ids(tok, ids):
    """Drop pad/eos and cut at the first newline. Returns (ids, text)."""
    stop = set(x for x in [tok.eos_token_id, tok.pad_token_id] if x is not None)
    gen_cfg_eos = getattr(tok, "_extra_eos", set())
    out = []
    for t in ids:
        if t in stop or t in gen_cfg_eos:
            break
        out.append(int(t))
    text = tok.decode(out, skip_special_tokens=True)
    if "\n" in text:
        # cut at the token where the newline first appears
        for k in range(1, len(out) + 1):
            if "\n" in tok.decode(out[:k], skip_special_tokens=True):
                out = out[: k - 1]
                break
        text = tok.decode(out, skip_special_tokens=True)
    return out, text.strip()


@torch.no_grad()
def generate(tok, model, prompts_ids, batch_q, seed):
    """Return list (per question) of list of (ids, text); index 0 greedy, then samples."""
    eos_ids = set()
    gc = model.generation_config
    for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int) else (gc.eos_token_id or [])):
        eos_ids.add(e)
    tok._extra_eos = eos_ids
    results = []
    tok.padding_side = "left"
    for b in tqdm(range(0, len(prompts_ids), batch_q), desc="generate"):
        chunk = prompts_ids[b:b + batch_q]
        enc = tok.pad({"input_ids": chunk}, return_tensors="pt").to(model.device)
        plen = enc["input_ids"].shape[1]
        common = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                      eos_token_id=list(eos_ids), stop_strings=["\n"], tokenizer=tok)
        g = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                           repetition_penalty=1.0, **common)
        torch.manual_seed(seed + b)
        s = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                           repetition_penalty=1.0, num_return_sequences=N_SAMPLES, **common)
        g = g[:, plen:].tolist()
        s = s[:, plen:].tolist()
        for i in range(len(chunk)):
            ans = [clean_answer_ids(tok, g[i])]
            ans += [clean_answer_ids(tok, s[i * N_SAMPLES + j]) for j in range(N_SAMPLES)]
            results.append(ans)
    return results


@torch.no_grad()
def teacher_force(tok, model, prompt_ids, answers, layers, batch_seqs):
    """Score answers. Returns dict of arrays aligned with the flat (q, a) list."""
    flat = []  # (qi, ai, prompt_ids, answer_ids)
    for qi, (p, ans) in enumerate(zip(prompt_ids, answers)):
        for ai, (ids, _) in enumerate(ans):
            a = ids if len(ids) > 0 else [tok.eos_token_id]
            flat.append((qi, ai, p, a))
    n = len(flat)
    H = model.config.hidden_size
    sum_lp = np.zeros(n, np.float32)
    mean_lp = np.zeros(n, np.float32)
    min_lp = np.zeros(n, np.float32)
    mean_ent = np.zeros(n, np.float32)
    max_ent = np.zeros(n, np.float32)
    ntok = np.zeros(n, np.int32)
    h_ans = np.zeros((len(layers), n, H), np.float16)
    h_q = np.zeros((len(layers), len(prompt_ids), H), np.float16)
    backbone = model.model
    head = model.lm_head
    for b in tqdm(range(0, n, batch_seqs), desc="score"):
        chunk = flat[b:b + batch_seqs]
        seqs = [p + a for (_, _, p, a) in chunk]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        out = backbone(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layers+1
        last = out.last_hidden_state
        # gather every answer-predicting position in the batch at once
        rows_i, pos_i, tgt_i, seg = [], [], [], []
        for i, (qi, ai, p, a) in enumerate(chunk):
            P, A = len(p), len(a)
            rows_i += [i] * A
            pos_i += list(range(P - 1, P + A - 1))  # positions predicting answer tokens
            tgt_i += a
            seg += [i] * A
        dev = model.device
        rows_t = torch.tensor(rows_i, device=dev)
        pos_t = torch.tensor(pos_i, device=dev)
        logp = torch.log_softmax(head(last[rows_t, pos_t]).float(), -1)
        tlp = logp[torch.arange(len(tgt_i), device=dev), torch.tensor(tgt_i, device=dev)]
        ent = -(logp.exp() * logp).sum(-1)
        tlp, ent, seg_np = tlp.cpu().numpy(), ent.cpu().numpy(), np.array(seg)
        end_pos = torch.tensor([len(p) + len(a) - 1 for (_, _, p, a) in chunk], device=dev)
        q_pos = torch.tensor([len(p) - 1 for (_, _, p, a) in chunk], device=dev)
        ar = torch.arange(len(chunk), device=dev)
        h_end = [hs[l][ar, end_pos].float().cpu().numpy() for l in layers]
        h_qs = [hs[l][ar, q_pos].float().cpu().numpy() for l in layers]
        for i, (qi, ai, p, a) in enumerate(chunk):
            k = b + i
            m = seg_np == i
            sum_lp[k] = tlp[m].sum()
            mean_lp[k] = tlp[m].mean()
            min_lp[k] = tlp[m].min()
            mean_ent[k] = ent[m].mean()
            max_ent[k] = ent[m].max()
            ntok[k] = len(a)
            for li in range(len(layers)):
                h_ans[li, k] = h_end[li][i]
                if ai == 0:
                    h_q[li, qi] = h_qs[li][i]
        del out, hs, last, logp
    return dict(sum_lp=sum_lp, mean_lp=mean_lp, min_lp=min_lp, mean_ent=mean_ent,
                max_ent=max_ent, ntok=ntok), h_ans, h_q


@torch.no_grad()
def ptrue(tok, model, questions, answers, batch_seqs):
    t_id = tok.encode("True", add_special_tokens=False)[0]
    f_id = tok.encode("False", add_special_tokens=False)[0]
    seqs = []
    for q, ans in zip(questions, answers):
        for ids, text in ans:
            seqs.append(chat_ids(tok, ptrue_messages(q["question"], text if text else "(no answer)")))
    out = np.zeros(len(seqs), np.float32)
    for b in tqdm(range(0, len(seqs), batch_seqs), desc="p_true"):
        chunk = seqs[b:b + batch_seqs]
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
        # log-odds of True vs False (avoids saturation ties of the probability)
        out[b:b + len(chunk)] = (logits[:, t_id] - logits[:, f_id]).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_q", type=int, default=8)
    ap.add_argument("--batch_seqs", type=int, default=22)
    args = ap.parse_args()
    set_seed()
    d = run_dir(args.model, args.dataset)
    qs = load_questions(args.dataset)
    if args.limit:
        qs = qs[: args.limit]
    tok, model = load_lm(MODELS[args.model])
    nl = model.config.num_hidden_layers
    layers = sorted(set([nl // 2, (3 * nl) // 4, nl]))
    prompt_ids = [chat_ids(tok, build_messages(q["question"])) for q in qs]

    t0 = time.time()
    gen_path = d / "generations.jsonl"
    if gen_path.exists() and len(read_jsonl(gen_path)) == len(qs):
        rows = read_jsonl(gen_path)
        answers = [[(r["ids"][k], r["answers"][k]) for k in range(len(r["answers"]))] for r in rows]
    else:
        answers = generate(tok, model, prompt_ids, args.batch_q, seed=1000)
        rows = [{**q, "answers": [t for _, t in ans], "ids": [i for i, _ in ans]}
                for q, ans in zip(qs, answers)]
        write_jsonl(gen_path, rows)
    t1 = time.time()
    feats, h_ans, h_q = teacher_force(tok, model, prompt_ids, answers, layers, args.batch_seqs)
    t2 = time.time()
    feats["p_true"] = ptrue(tok, model, qs, answers, args.batch_seqs * 2)
    t3 = time.time()
    np.savez(d / "features.npz", **feats)
    np.savez(d / "hidden.npz", h_ans=h_ans, h_q=h_q, layers=np.array(layers))
    print(f"done {args.model} {args.dataset}: gen {t1-t0:.0f}s score {t2-t1:.0f}s ptrue {t3-t2:.0f}s")


if __name__ == "__main__":
    main()
