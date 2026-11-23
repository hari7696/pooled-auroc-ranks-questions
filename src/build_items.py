"""Stage 4: assemble one table per run with labels and every detector's risk score.

Rows: one per (question, answer); answer index 0 = greedy, 1..N = samples.
Detector scores are oriented so that HIGHER = MORE LIKELY INCORRECT.

Probe detectors are cross-fitted with 5-fold GroupKFold over questions.
Usage: python src/build_items.py [--runs ...] [--layer_idx 1]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import N_SAMPLES, OUT, normalize, read_jsonl

K = N_SAMPLES + 1  # answers per question


def token_f1(a: str, b: str) -> float:
    ta, tb = normalize(a).split(), normalize(b).split()
    if not ta or not tb:
        return 0.0
    common = Counter(ta) & Counter(tb)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p, r = ns / len(ta), ns / len(tb)
    return 2 * p * r / (p + r)


def consistency_features(rows, clusters):
    """Answer-level: leave-one-out cluster frequency and lexical similarity to the samples.
    Question-level: discrete semantic entropy and number of clusters among samples."""
    nq = len(rows)
    freq = np.zeros((nq, K))
    lex = np.zeros((nq, K))
    se = np.zeros(nq)
    ncl = np.zeros(nq)
    for qi, (r, c) in enumerate(zip(rows, clusters)):
        samp = c[1:]
        cnt = Counter(samp.tolist())
        p = np.array(list(cnt.values()), float) / len(samp)
        se[qi] = -(p * np.log(p)).sum()
        ncl[qi] = len(cnt)
        for ai in range(K):
            others = [j for j in range(1, K) if j != ai]
            freq[qi, ai] = np.mean([c[j] == c[ai] for j in others])
            lex[qi, ai] = np.mean([token_f1(r["answers"][ai], r["answers"][j]) for j in others])
    return freq, lex, se, ncl


def _lr(n_pca: int, C: float):
    return make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=0),
                         StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def cross_fit_probes(h_ans, h_q, y, qid, mixed, n_pca=256, C=0.05, seed=0):
    """Cross-fitted probe risk scores (logits).

    h_ans: (nq, K, H) last-answer-token states; h_q: (nq, H) last-prompt-token states.
    y: (nq, K) 1 = incorrect. Training always uses the sampled answers (1..N) only.
    Returns dict of (nq, K) arrays.
    """
    nq = h_ans.shape[0]
    out = {k: np.zeros((nq, K)) for k in ["probe_answer", "probe_question", "probe_centered", "probe_twoblock"]}
    centered = h_ans - h_ans[:, 1:].mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    perm_groups = rng.permutation(nq)
    folds = GroupKFold(n_splits=5).split(np.zeros(nq), groups=perm_groups)
    for tr, te in folds:
        Xa = h_ans[tr, 1:].reshape(-1, h_ans.shape[-1])
        ya = y[tr, 1:].reshape(-1)
        # (1) answer probe: pooled training on answer states
        m = _lr(n_pca, C).fit(Xa, ya)
        out["probe_answer"][te] = m.decision_function(h_ans[te].reshape(-1, h_ans.shape[-1])).reshape(len(te), K)
        # (2) question probe: question state, label = each sample's correctness
        Xq = np.repeat(h_q[tr], N_SAMPLES, axis=0)
        mq = _lr(n_pca, C).fit(Xq, ya)
        sq = mq.decision_function(h_q[te])
        out["probe_question"][te] = np.repeat(sq[:, None], K, axis=1)
        # (3) centred probe: within-question deviations, trained on mixed questions only
        trm = tr[mixed[tr]]
        Xc = centered[trm, 1:].reshape(-1, h_ans.shape[-1])
        yc = y[trm, 1:].reshape(-1)
        mc = _lr(n_pca, C).fit(Xc, yc)
        sc = mc.decision_function(centered[te].reshape(-1, h_ans.shape[-1])).reshape(len(te), K)
        out["probe_centered"][te] = sc
        # (4) two-block: question logit + centred logit (sum of the two separately fitted parts)
        out["probe_twoblock"][te] = np.repeat(sq[:, None], K, axis=1) + sc
    return out


def question_text_probe(questions, y, seed=0):
    """Model-agnostic difficulty baseline: MiniLM embedding of the question text only."""
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    E = enc.encode(questions, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    nq = len(questions)
    out = np.zeros(nq)
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(nq), groups=rng.permutation(nq)):
        X = np.repeat(E[tr], N_SAMPLES, axis=0)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X, y[tr, 1:].reshape(-1))
        out[te] = m.decision_function(E[te])
    return np.repeat(out[:, None], K, axis=1)


def build(run: str, layer_idx: int = 1, label: str = "judge") -> pd.DataFrame:
    d = OUT / run
    rows = read_jsonl(d / "generations.jsonl")
    f = np.load(d / "features.npz")
    lab = np.load(d / "labels.npz")
    cl = np.load(d / "clusters.npy")
    nq = len(rows)
    y_judge = (lab["judge_logodds"] <= 0).astype(int).reshape(nq, K)
    y_alias = (~lab["alias"]).astype(int).reshape(nq, K)
    y = y_judge if label == "judge" else y_alias
    mixed = (y[:, 1:].sum(1) > 0) & (y[:, 1:].sum(1) < N_SAMPLES)

    R = lambda a: np.asarray(a, float).reshape(nq, K)
    det = {
        "neg_mean_logprob": -R(f["mean_lp"]),
        "neg_sum_logprob": -R(f["sum_lp"]),
        "neg_min_logprob": -R(f["min_lp"]),
        "mean_entropy": R(f["mean_ent"]),
        "max_entropy": R(f["max_ent"]),
        "neg_p_true": -R(f["p_true"]),
        "length": R(f["ntok"]),
    }
    freq, lex, se, ncl = consistency_features(rows, cl)
    det["neg_cluster_freq"] = -freq
    det["neg_lexical_sim"] = -lex
    det["semantic_entropy"] = np.repeat(se[:, None], K, 1)
    det["num_clusters"] = np.repeat(ncl[:, None], K, 1)

    hz = np.load(d / "hidden.npz")
    h_ans = hz["h_ans"][layer_idx].astype(np.float32).reshape(nq, K, -1)
    h_q = hz["h_q"][layer_idx].astype(np.float32)
    det.update(cross_fit_probes(h_ans, h_q, y, np.arange(nq), mixed))
    det["question_text_probe"] = question_text_probe([r["question"] for r in rows], y)

    qi = np.repeat(np.arange(nq), K)
    ai = np.tile(np.arange(K), nq)
    df = pd.DataFrame({"q": qi, "a": ai, "y": y.reshape(-1), "y_judge": y_judge.reshape(-1),
                       "y_alias": y_alias.reshape(-1), "cluster": cl.reshape(-1)})
    for k, v in det.items():
        df[k] = np.asarray(v, float).reshape(-1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--layer_idx", type=int, default=1)
    ap.add_argument("--label", default="judge", choices=["judge", "alias"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    runs = args.runs or sorted(p.name for p in OUT.iterdir()
                               if (p / "labels.npz").exists() and (p / "clusters.npy").exists())
    suffix = "" if (args.layer_idx == 1 and args.label == "judge") else f"_L{args.layer_idx}_{args.label}"
    for run in runs:
        path = OUT / run / f"items{suffix}.parquet"
        if path.exists() and not args.force:
            continue
        df = build(run, args.layer_idx, args.label)
        df.to_parquet(path)
        print(run, path.name, df.shape, f"err={df.y.mean():.3f}")


if __name__ == "__main__":
    main()
